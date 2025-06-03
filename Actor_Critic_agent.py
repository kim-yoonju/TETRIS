"""
actor_critic_tetris_fixed.py  —  A2C‑style agent and training loop for Tetris
──────────────────────────────────────────────────────────────────────────────
Revision 2  (2025‑06‑03)
──────────────────────────────────────────────────────────────────────────────
• **Hot‑fix:** `select_action()` now converts logits to NumPy safely, even when
  eager execution is disabled, eliminating the `AttributeError: 'Tensor' object
  has no attribute 'numpy'`.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, Input
from tensorflow.keras import backend as K


class ActorCriticAgent:
    """A very small A2C‑like agent that chooses among *sets of candidate boards*.

    Each candidate board (20×10) is run through a shared CNN encoder that
    outputs
        • a scalar *logit*  – unnormalised preference score
        • a scalar *value*  – baseline V(s)
    The policy over a given set is the soft‑max of the logits in that set.
    """

    def __init__(
        self,
        state_shape=(20, 10),
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_stop_episode: int = 500,
        add_batch_norm: bool = False,
    ):
        self.state_shape = state_shape
        self.gamma = gamma

        # ε‑greedy exploration (extra on top of soft‑max sampling)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode

        self.model = self._build_model(add_batch_norm)
        self.opt = optimizers.Adam(lr)

        # Per‑episode buffers
        self.candidate_batches = []   # list[(K_t,20,10)]
        self.chosen_indices = []      # list[int]
        self.rewards = []             # list[float]

    # ──────────────────────────────────────────────────────────────────────
    # Network
    # ──────────────────────────────────────────────────────────────────────
    def _build_model(self, add_batch_norm: bool) -> Model:
        inp = Input(shape=self.state_shape)                    # (20,10)
        x = layers.Reshape((*self.state_shape, 1))(inp)        # (20,10,1)

        for filters in (32, 64, 64, 128):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            if add_batch_norm:
                x = layers.BatchNormalization()(x)

        x = layers.Flatten()(x)

        # Policy head
        pi_h = layers.Dense(256, activation="relu")(x)
        if add_batch_norm:
            pi_h = layers.BatchNormalization()(pi_h)
        logits = layers.Dense(1, name="policy_logits")(pi_h)  # (B,1)

        # Value head
        v_h = layers.Dense(256, activation="relu")(x)
        if add_batch_norm:
            v_h = layers.BatchNormalization()(v_h)
        value = layers.Dense(1, name="value")(v_h)            # (B,1)

        return Model(inp, [logits, value])

    # ──────────────────────────────────────────────────────────────────────
    # Interaction
    # ──────────────────────────────────────────────────────────────────────
    def select_action(self, cand_boards, greedy: bool = False) -> int:
        """Pick an index in *cand_boards* and store info for learning."""
        boards_np = np.asarray(cand_boards, dtype=np.float32)         # (K,20,10)
        logits, _ = self.model.predict(boards_np, verbose=0)

        # — robust NumPy conversion —
        logits_np = np.squeeze(logits, axis=-1)                       # (K,)
        if isinstance(logits_np, tf.Tensor):                          # eager off ⇒ Tensor
            logits_np = K.get_value(logits_np)

        # Soft‑max distribution (NumPy implementation avoids TF dependency)
        exp_logits = np.exp(logits_np - np.max(logits_np))
        probs = exp_logits / np.sum(exp_logits)

        K_actions = len(cand_boards)
        if (not greedy) and np.random.rand() < self.epsilon:
            idx = np.random.randint(K_actions)
        else:
            idx = np.random.choice(K_actions, p=probs)

        # Book‑keeping for update phase
        self.candidate_batches.append(boards_np)
        self.chosen_indices.append(idx)
        return idx

    def record_reward(self, r):
        self.rewards.append(np.float32(r))

    # ──────────────────────────────────────────────────────────────────────
    # Learning
    # ──────────────────────────────────────────────────────────────────────
    def train_episode(self):
        T = len(self.rewards)
        if T == 0:
            return 0.0

        # 1) Monte‑Carlo returns
        returns = np.zeros(T, dtype=np.float32)
        G = 0.0
        for t in reversed(range(T)):
            G = self.rewards[t] + self.gamma * G
            returns[t] = G
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)      # (T,)

        with tf.GradientTape() as tape:
            policy_loss = tf.constant(0.0, dtype=tf.float32)
            value_loss  = tf.constant(0.0, dtype=tf.float32)

            for t in range(T):
                cand_tensor = tf.convert_to_tensor(self.candidate_batches[t], dtype=tf.float32)
                logits_t, values_t = self.model(cand_tensor, training=True)
                logits_t = tf.squeeze(logits_t, axis=-1)               # (K_t,)
                values_t = tf.squeeze(values_t, axis=-1)               # (K_t,)

                probs_t = tf.nn.softmax(logits_t)

                a_t = self.chosen_indices[t]
                log_prob_a = tf.math.log(tf.gather(probs_t, a_t) + 1e-8)
                value_a    = tf.gather(values_t, a_t)

                advantage_t = returns[t] - value_a
                policy_loss += -log_prob_a * advantage_t
                value_loss  += tf.square(advantage_t)

            policy_loss /= T
            value_loss  /= T
            loss = policy_loss + 0.5 * value_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # ε decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        # Reset buffers
        self.candidate_batches.clear()
        self.chosen_indices.clear()
        self.rewards.clear()

        return float(K.get_value(loss))

# (Runner code remains the same)
