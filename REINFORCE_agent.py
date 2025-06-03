import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
import numpy as np
import random

class REINFORCEAgent:
    def __init__(self,
                 state_size,
                 mem_size=10000,
                 discount=0.99,
                 epsilon=1.0,
                 epsilon_min=0.02,
                 epsilon_stop_episode=500,
                 add_batch_norm=False,
                 optimizer='adam',
                 # 아래 두 인자를 새로 추가해서 받아는 주지만
                 # 실제로는 내부에서 사용하지 않도록 함
                 use_target_model=None,
                 update_target_every=None,
                 replay_start_size=None
    ):

        self.state_size = state_size
        self.discount = discount

        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode

        self.use_target_model   = use_target_model
        self.replay_start_size  = replay_start_size

        # 정책 네트워크 생성 (Conv2D → Dense → output 1)
        self.model     = self._build_model(add_batch_norm)
        self.optimizer = tf.keras.optimizers.get(optimizer)

        # 한 에피소드 동안 모을 log_prob, reward 버퍼
        self.episode_log_probs = []
        self.episode_rewards   = []

        self.episode_states  = []      # ⬅️ 새로 추가 (20×10 boards)


    def _build_model(self, add_batch_norm=False):
        """
        Builds a deeper Keras CNN model that accepts a (20, 10) input (no data changes needed),
        then automatically adds a channel dimension before applying convolutions.
        """
        model = Sequential()

        # 1) 입력을 (20, 10) 형태로 바로 받되, 내부에서 채널 차원을 추가
        #    이 Reshape 레이어가 (batch, 20, 10) → (batch, 20, 10, 1) 변환을 수행합니다.
        model.add(keras.layers.Reshape((20, 10, 1), input_shape=(20, 10)))

        # 2) 깊은 2D 컨볼루션 블록들
        # 첫 번째 블록: 32채널 Conv → BatchNorm(Optional) → ReLU
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same'))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(keras.layers.Activation('relu'))

        # 두 번째 블록: 64채널 Conv → BatchNorm(Optional) → ReLU
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(keras.layers.Activation('relu'))

        # 세 번째 블록: 64채널 Conv → BatchNorm(Optional) → ReLU
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(keras.layers.Activation('relu'))

        # 네 번째 블록: 128채널 Conv → BatchNorm(Optional) → ReLU
        model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same'))
        if add_batch_norm:
            model.add(BatchNormalization())
        model.add(keras.layers.Activation('relu'))

        # 3) 특징 맵을 일차원 벡터로 펼침
        model.add(keras.layers.Flatten())

        # 4) Fully Connected 블록들
        model.add(Dense(256, activation='relu'))
        if add_batch_norm:
            model.add(BatchNormalization())

        model.add(Dense(128, activation='relu'))
        if add_batch_norm:
            model.add(BatchNormalization())

        # 최종 스칼라 출력
        model.add(Dense(1, activation='linear'))

        return model


    def add_to_memory(self, state, reward):
        """
        state : (20,10) numpy array ─ 선택한 next_board
        reward: 해당 스텝 reward
        """
        self.episode_states.append(state.astype(np.float32))
        self.episode_rewards.append(reward)

    def random_value(self):
        return random.random()

    # ---------------- predict_value (선택) ----------------
    def predict_value(self, state):
        state = state.astype(np.float32)
        return self.model.predict(state[None, ...])[0]

    # ---------------- REINFORCEAgent.best_state ----------------
    def best_state(self, states, exploration=True):
        """
        states   : iterable of (20,10) numpy arrays  ─ 후보 보드들
        exploration=True  : ε-greedy 탐험 여부
        반환값  : (선택한_board_numpy, log_prob_tf)
        """
        # 1) 리스트 변환 & 배치 텐서 생성
        states_list = list(states)                                    # 길이 K
        states_np   = np.stack(states_list).astype(np.float32)        # (K,20,10)

        # 2) 모델 예측 → numpy (K,)
        logits_np = self.model.predict(states_np, verbose=0).ravel()  # (K,)
        
        # 3) 소프트맥스 확률
        exp_logits = np.exp(logits_np - np.max(logits_np))            # 안정적 softmax
        probs_np   = exp_logits / exp_logits.sum()                    # (K,)

        K = probs_np.shape[0]

        # 4) ε-greedy 로 인덱스 선택
        if exploration and random.random() < self.epsilon:
            idx = random.randrange(K)
        else:
            idx = np.random.choice(K, p=probs_np)                     # ← Numpy 배열이므로 OK

        # 5) 선택한 상태 & (선택 시점의) log-prob 반환
        chosen_state_np = states_list[idx]
        log_prob        = np.log(probs_np[idx] + 1e-8)                # float
        log_prob_tf     = tf.convert_to_tensor(log_prob, dtype=tf.float32)

        return chosen_state_np, log_prob_tf



    def train(self):
        """한 에피소드가 끝난 뒤 호출"""

        if not self.episode_rewards:      # 데이터가 없으면 패스
            return

        # ① 할인 누적 보상 G_t
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.discount * G
            returns.insert(0, G)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # ② 상태 텐서로 묶기
        states_tf = tf.convert_to_tensor(
            np.stack(self.episode_states), dtype=tf.float32)   # (T,20,10)

        # ③ 정책 그래디언트
        with tf.GradientTape() as tape:
            # 모델 forward ― (T,1) → squeeze → (T,)
            logits   = tf.squeeze(self.model(states_tf, training=True), axis=-1)
            log_probs = logits                                # 스칼라 logit을 log-prob 로 해석
            loss = -tf.reduce_mean(log_probs * returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # ④ ε 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        # ⑤ 버퍼 초기화
        self.episode_states.clear()
        self.episode_rewards.clear()
