import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K

# ─── XLA JIT 활성화 ─────────────────────────────────────────────────────────
# 가능한 연산들을 XLA가 컴파일하도록 설정
tf.config.optimizer.set_jit(True)
# ───────────────────────────────────────────────────────────────────────────

# 하이퍼파라미터
GAMMA, LAMBDA   = 0.99, 0.95
CLIP_EPS        = 0.2
LR              = 2.5e-4
ENTROPY_COEF    = 1e-3
VALUE_COEF      = 0.5
ROLLOUT_STEPS   = 2048
EPOCHS_UPDATE   = 3
MINIBATCH       = 256
MAX_GRAD_NORM   = 0.5


def build_ac(add_bn=False):
    """
    Actor-Critic 네트워크
    입력: (20,10) float32
    출력: logits (각 후보 보드마다 1 스칼라), value (각 후보 보드마다 1 스칼라)
    """
    inp = layers.Input((20,10), dtype=tf.float32)
    x   = layers.Reshape((20,10,1))(inp)
    for f in (32,64,64,128):
        x = layers.Conv2D(f, 3, padding='same')(x)
        if add_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)

    # Policy head
    p = layers.Dense(256, activation='relu')(x)
    logits = layers.Dense(1, name='logits')(p)    # (batch_of_candidates, 1)

    # Value head
    v  = layers.Dense(256, activation='relu')(x)
    value = layers.Dense(1, name='value')(v)      # (batch_of_candidates, 1)

    return models.Model(inp, [logits, value])


class PPOAgent:
    def __init__(self, add_bn=False):
        self.model    = build_ac(add_bn)
        self.opt      = optimizers.Adam(learning_rate=LR)
        self.clip_eps = CLIP_EPS
        self.clear()

    def clear(self):
        """
        에피소드가 끝난 뒤 버퍼 초기화
        """
        self.cands, self.idx = [], []
        self.lp_old, self.rew = [], []
        self.vals, self.dns  = [], []

    def ready(self):
        """
        버퍼에 ROLLOUT_STEPS만큼 샘플이 쌓였으면 True 반환
        """
        return len(self.rew) >= ROLLOUT_STEPS

    @tf.function
    def _select_tensor(self, boards_tf):
        """
        그래프 모드에서 동작하는 행동 선택 로직
        boards_tf: (K,20,10) float32 tensor
        반환:
          - i_tf (int32 scalar): 선택된 index
          - logp_tf (float32 scalar): 해당 index의 log-probability
          - val_tf (float32 scalar): 해당 index의 state-value
        """
        # 1) 모델을 통해 logits(lo)와 value(va) 얻기
        lo, va = self.model(boards_tf, training=False)  # lo,va: (K,1) float32
        lo = tf.squeeze(lo, axis=-1)                    # (K,) float32
        va = tf.squeeze(va, axis=-1)                    # (K,) float32

        # 2) logits에서 바로 샘플링 (tf.random.categorical)
        lo_2d = tf.expand_dims(lo, axis=0)              # (1,K)
        i_tf  = tf.random.categorical(lo_2d, num_samples=1)[0, 0]  # 스칼라 int64

        # 3) 선택된 index의 log-prob와 value를 추출
        logp_all = tf.nn.log_softmax(lo)                # (K,) float32
        logp_tf  = tf.gather(logp_all, i_tf)            # 스칼라 float32

        val_tf   = tf.gather(va, i_tf)                  # 스칼라 float32

        return tf.cast(i_tf, tf.int32), logp_tf, val_tf

    def select(self, boards: np.ndarray):
        """
        NumPy 입력을 받아, TensorFlow 그래프 모드로 처리
        boards: np.ndarray of shape (K,20,10), dtype=np.float32
        반환:
          - i (int): 선택된 후보 index
          - logp (tf.Tensor scalar): 해당 index log-probability
          - val (float): 해당 index state-value
        """
        boards_tf = tf.convert_to_tensor(boards, dtype=tf.float32)
        i_tf, logp_tf, val_tf = self._select_tensor(boards_tf)

        # tf.Tensor → Python native로 꺼내기
        try:
            i    = int(i_tf.numpy())
            logp = logp_tf  # 이미 float32 Tensor
            val  = float(val_tf.numpy())
        except Exception:
            i    = int(K.get_value(i_tf))
            logp = logp_tf
            val  = float(K.get_value(val_tf))

        return i, logp, val

    def select_action(self, boards: np.ndarray):
        return self.select(boards)

    def store(self, cands, idx, lp, r, v, d):
        """
        각 timestep별 transition을 저장
        - cands: np.ndarray of shape (K_t,20,10), dtype=np.float32
        - idx: 선택된 index (int)
        - lp: tf.Tensor float32 (log-probability)
        - r: reward (float)
        - v: state-value (float)
        - d: done flag (bool or 0/1)
        """
        self.cands.append(cands.astype(np.float32))
        self.idx.append(int(idx))

        # lp Tensor → Python float
        try:
            self.lp_old.append(float(lp.numpy()))
        except Exception:
            self.lp_old.append(float(K.get_value(lp)))

        self.rew.append(float(r))
        self.vals.append(float(v))
        self.dns.append(1.0 if d else 0.0)

    @tf.function
    def _compute_loss(self,
                      cands_rt,   # RaggedTensor[T][(K_t,20,10)]
                      idxs_tf,    # shape=(T,) int32
                      lp_old_tf,  # shape=(T,) float32
                      vals_tf,    # shape=(T,) float32
                      rews_tf,    # shape=(T+1,) float32
                      dns_tf):    # shape=(T+1,) float32
        """
        그래프 모드에서 전체 loss와 gradients를 계산
        """

        T = tf.shape(idxs_tf)[0]

        # 1) GAE 계산 (tf.while_loop)
        adv_ta = tf.TensorArray(dtype=tf.float32, size=T)
        gae = tf.constant(0.0, dtype=tf.float32)

        def cond(t, adv_ta, gae):
            return tf.greater_equal(t, 0)

        def body(t, adv_ta, gae):
            delta = rews_tf[t] + GAMMA * vals_tf[t+1] * (1.0 - dns_tf[t]) - vals_tf[t]
            gae_new = delta + GAMMA * LAMBDA * (1.0 - dns_tf[t]) * gae
            adv_ta = adv_ta.write(t, gae_new)
            return t-1, adv_ta, gae_new

        t0 = T - 1
        _, adv_ta, _ = tf.while_loop(cond, body, [t0, adv_ta, gae])
        adv = adv_ta.stack()                                # shape=(T,)
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)
        ret = adv + vals_tf                                 # shape=(T,)

        # 2) 전체 rollout에 대해 모델 연산 및 loss 누적
        total_loss = tf.constant(0.0, dtype=tf.float32)
        for t in tf.range(T):
            boards_t = cands_rt[t]                         # RaggedTensor (K_t,20,10) float32
            lo_t, va_t = self.model(boards_t, training=True)  # (K_t,1) float32
            lo_t = tf.squeeze(lo_t, axis=-1)               # (K_t,) float32
            va_t = tf.squeeze(va_t, axis=-1)               # (K_t,) float32

            # policy 손실
            logp_all = tf.nn.log_softmax(lo_t)             # (K_t,) float32
            idx_t    = idxs_tf[t]                          # int32
            lp_new   = tf.gather(logp_all, idx_t)          # 스칼라 float32

            ratio = tf.exp(lp_new - lp_old_tf[t])          # float32

            adv_t = adv[t]
            s1 = ratio * adv_t
            s2 = tf.clip_by_value(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            pol_loss = -tf.minimum(s1, s2)

            # value 손실
            val_loss = tf.square(ret[t] - tf.gather(va_t, idx_t)) * VALUE_COEF

            # entropy bonus
            ent = -tf.reduce_sum(tf.exp(logp_all) * logp_all) * ENTROPY_COEF

            total_loss += pol_loss + val_loss - ent

        # 3) gradient 계산 및 clipping
        grads = tf.gradients(total_loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)

        return total_loss, grads

    def train(self, last_value=0.0):
        """
        1) Python 리스트 버퍼를 RaggedTensor + Tensor로 변환
        2) _compute_loss() 호출 → loss, grads 계산
        3) apply_gradients()
        4) 버퍼 초기화
        """
        T = len(self.rew)

        # 1) 버퍼 → RaggedTensor/ Tensor 변환
        cands_rt  = tf.ragged.constant(self.cands, dtype=tf.float32)  # shape=(T, None, 20,10)
        idxs_tf   = tf.convert_to_tensor(self.idx,   dtype=tf.int32)  # shape=(T,)
        lp_old_tf = tf.convert_to_tensor(self.lp_old, dtype=tf.float32)  # shape=(T,)
        vals_tf   = tf.convert_to_tensor(self.vals,   dtype=tf.float32)  # shape=(T,)

        rews_tf = tf.convert_to_tensor(self.rew + [last_value], dtype=tf.float32)  # shape=(T+1,)
        dns_tf  = tf.convert_to_tensor(self.dns + [0.0],       dtype=tf.float32)   # shape=(T+1,)

        # 2) 그래프 모드에서 loss와 grads 계산
        total_loss, grads = self._compute_loss(cands_rt, idxs_tf, lp_old_tf, vals_tf, rews_tf, dns_tf)

        # 3) 최종 그래디언트 적용
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # 4) 버퍼 초기화
        self.clear()
