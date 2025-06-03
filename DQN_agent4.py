import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from collections import deque
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation

from tensorflow.keras import layers, Model
# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.
class DQNAgent:
    def __init__(self, state_size,
                 mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0,
                 epsilon_stop_episode=500,
                 add_batch_norm=False,
                 use_target_model=False,
                 update_target_every=None,
                 loss='mse', optimizer='adam',
                 replay_start_size=None):

        self.state_shape = state_size
        self.state_size  = state_size
        self.memory      = deque(maxlen=mem_size)

        self.discount        = discount
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = (epsilon - epsilon_min) / epsilon_stop_episode

        if replay_start_size is None:
            replay_start_size = mem_size // 2
        self.replay_start_size = replay_start_size

        # 네트워크
        self.model = self._build_model(add_batch_norm)
        self.model.compile(loss=loss, optimizer=optimizer)

        # (선택) 타깃 네트워크
        self.use_target_model = use_target_model
        self.update_target_every = update_target_every
        self.target_counter = 0
        if use_target_model:
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            self.target_model.compile(loss=loss, optimizer=optimizer)


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

    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        '''Random score for a certain action'''
        return random.random()


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state)[0]

    def best_state(self, states, exploration=True):
        '''Returns the best state for a given collection of states'''
        if exploration and random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            states = np.array(list(states))
            values = self.model.predict(states)
            best_state = states[ np.argmax(values) ]

        return list(best_state)

    def train(self, memory_batch_size=32,
              training_batch_size=32, epochs=3):
        """
        한 번 호출 시 replay buffer에서 샘플을 뽑아
        'epochs' 만큼 미니배치 학습을 수행하고,
        마지막 스텝의 loss(float)를 반환.
        (학습이 일어나지 않으면 None 반환)
        """
        if len(self.memory) < max(self.replay_start_size, memory_batch_size):
            return None

        batch = random.sample(self.memory, memory_batch_size)
        states, next_states, reward, done = map(np.array, zip(*batch))

        # 타깃 Q 계산
        target_net = self.target_model if self.use_target_model else self.model
        next_qs = target_net.predict(next_states, verbose=0).flatten()
        targets = reward + (~done) * self.discount * next_qs

        history = self.model.fit(
            states, targets,
            batch_size=training_batch_size,
            epochs=epochs,
            verbose=0
        )

        # ε-greedy 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        # 타깃 네트워크 주기적 업데이트
        if self.use_target_model and self.update_target_every is not None:
            self.target_counter += 1
            if self.target_counter >= self.update_target_every:
                self.target_model.set_weights(self.model.get_weights())
                self.target_counter = 0

        return history.history['loss'][-1]  # 마지막 mini-epoch loss