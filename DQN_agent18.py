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
from tensorflow.keras import optimizers
# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.
class DQNAgent:

    '''Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    def __init__(self, state_size,
                 mem_size=10000,
                 discount=0.95,
                 epsilon=1,
                 epsilon_min=0,
                 epsilon_stop_episode=500,
                 add_batch_norm=False,
                 use_target_model=False,
                 update_target_every=None,
                 loss='mse',
                 optimizer='adam',
                 replay_start_size=None):

        # assert len(activations) == len(n_neurons) + 1
        self.state_shape = state_size
        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self._build_model(add_batch_norm)

        self.use_target_model = use_target_model
        if use_target_model:
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_counter = 0
            self.update_target_every = update_target_every

    from tensorflow.keras import optimizers

    def _build_model(self, add_batch_norm=False):
        """
        Builds a deeper Keras CNN model that accepts a (20, 10) input,
        then automatically adds a channel dimension before applying convolutions.
        마지막에 compile까지 한 번에 처리하도록 수정했습니다.
        """
        model = Sequential()

        # 1) 입력 Reshape
        model.add(keras.layers.Reshape((20, 10, 1), input_shape=(20, 10)))

        # 2) 18개의 컨볼루션 블록
        filters_list = (
                [32, 32]  # 2개 블록은 32채널
                + [64] * 4  # 다음 4개 블록은 64채널
                + [128] * 4  # 다음 4개 블록은 128채널
                + [256] * 4  # 다음 4개 블록은 256채널
                + [512] * 4  # 마지막 4개 블록은 512채널
        )
        for filt in filters_list:
            model.add(keras.layers.Conv2D(filt, kernel_size=(3, 3), padding='same'))
            if add_batch_norm:
                model.add(BatchNormalization())
            model.add(keras.layers.Activation('relu'))

        # 3) Flatten
        model.add(keras.layers.Flatten())

        # 4) Fully Connected 블록
        model.add(Dense(256, activation='relu'))
        if add_batch_norm:
            model.add(BatchNormalization())

        model.add(Dense(128, activation='relu'))
        if add_batch_norm:
            model.add(BatchNormalization())

        # 5) 최종 출력
        model.add(Dense(1, activation='linear'))

        # 6) 컴파일 (예: Adam optimizer + MSE loss)
        optimizer = optimizers.Adam(lr=1e-4)
        model.compile(optimizer=optimizer, loss='mse')

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


    def train(self, memory_batch_size=32, training_batch_size=32, epochs=3):
        '''Trains the agent'''
        n = len(self.memory)

        if n >= self.replay_start_size and n >= memory_batch_size:

            batch = random.sample(self.memory, memory_batch_size)

            states,next_states,reward,done = map(lambda x : np.array(x), zip(*batch))

            if not self.use_target_model:
                next_qs = self.model.predict(next_states).flatten()
            else:
                next_qs = self.target_model.predict(next_states).flatten()
                self.target_counter += 1

            new_q = reward + ~done * self.discount * next_qs

            # Fit the model to the given values
            self.model.fit(states, new_q, batch_size=training_batch_size, epochs=epochs, verbose=0)

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

            if (self.use_target_model
                    and self.update_target_every is not None
                    and self.target_counter >= self.update_target_every):
                self.target_model = tf.keras.models.clone_model(self.model)
                self.target_counter = 0

