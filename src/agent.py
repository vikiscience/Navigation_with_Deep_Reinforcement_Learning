# -*- coding: utf-8 -*-
import const
from src.model import DQN

import random
import numpy as np
from collections import deque


class DQNAgent:
    use_double_dqn = False
    model_path = const.model_path

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01 #0.001
        self.batch_size = 10  # 32
        self.replay_after = 10
        self.update_target_each_iter = 100

        self.model = DQN(self.num_states, self.num_actions,
                         self.learning_rate, self.batch_size)
        self.target_model = DQN(self.num_states, self.num_actions,
                         self.learning_rate, self.batch_size)
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon):
        # e-greedy policy for Q
        probs = np.full(self.num_actions, epsilon / self.num_actions)
        q_values = self.model.predict(state)[0]
        best_a = np.argmax(q_values)
        probs[best_a] += 1 - epsilon
        a = np.random.choice(range(self.num_actions), p=probs)
        return a

    def replay(self):
        # Sample a minibatch from the replay memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Calculate q values and targets (Double DQN)
        for state, action, reward, next_state, done in minibatch:
            # self.model.predict(state)
            # target vector must have shape model_batch_size(=1) x num_actions
            target = np.zeros((1, self.num_actions))
            q_value_next = self.model.predict(next_state)[0]
            best_action = np.argmax(q_value_next)  # todo - immer 2 dieselben

            if done:
                target[0][action] = reward
            else:
                ### a = self.model.predict(next_state)[0]
                #t = self.target_model.predict(next_state)[0]
                #target[0][action] = reward + self.gamma * np.amax(t)
                ### target[0][action] = reward + self.gamma * t[np.argmax(a)]
                q_value_next_target = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * q_value_next_target[best_action]

            # Perform gradient descent update
            self.model.fit(state, target, epochs=1, verbose=0)

        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def replay_minibatch(self):
        # Sample a minibatch from the replay memory
        minibatch = random.sample(self.memory, self.batch_size)

        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*minibatch))
        states_batch = states_batch.reshape(states_batch.shape[0], -1)  # infer last dim (input)
        next_states_batch = next_states_batch.reshape(next_states_batch.shape[0], -1)

        if self.use_double_dqn:
            targets_batch = self._weight_update_Double_DQN(reward_batch, next_states_batch, done_batch)
        else:
            targets_batch = self._weight_update_DQN(reward_batch, next_states_batch, done_batch)

        #targets_batch_zeros = np.zeros(shape=(self.batch_size, self.num_states, ))
        targets_batch_zeros = self.model.predict(states_batch, batch_size=self.batch_size)
        print(targets_batch_zeros.shape)
        for i, a, t in zip(range(self.batch_size), action_batch, targets_batch):
            targets_batch_zeros[i][a] = t

        #print(states_batch.shape, action_batch.shape, reward_batch.shape,
        #      next_states_batch.shape, done_batch.shape,
        #      targets_batch.shape, targets_batch_zeros.shape)

        # Perform gradient descent update
        self.model.fit(states_batch, targets_batch_zeros,
                       batch_size=self.batch_size,
                       epochs=1,
                       verbose=0)

    def load(self):
        print('Loading model from:', self.model_path)
        self.model.load_weights(str(self.model_path))

    def save(self):
        print('Saving model to:', self.model_path)
        self.model.save_weights(str(self.model_path))

    def _weight_update_DQN(self, reward_batch, next_states_batch, done_batch):
        # Calculate q values and targets (DQN with fixed Q-targets)
        q_values_next_target = self.target_model.predict(next_states_batch,
                                                         batch_size=self.batch_size)
        # Get max predicted Q values (for next states) from target model
        q_values_next_target_max = np.max(q_values_next_target, axis=1)

        # Compute Q targets for current states
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.gamma * q_values_next_target_max
        return targets_batch

    def _weight_update_Double_DQN(self, reward_batch, next_states_batch, done_batch):
        # Calculate q values and targets (Double DQN)

        # select best actions
        q_values_next = self.model.predict(next_states_batch,
                                           batch_size=self.batch_size)
        best_actions = np.argmax(q_values_next, axis=1)

        # evaluate best actions
        q_values_next_target = self.target_model.predict(next_states_batch,
                                                         batch_size=self.batch_size)
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.gamma * q_values_next_target[:, best_actions]
                        #self.gamma * q_values_next_target[np.arange(self.batch_size), best_actions]
        return targets_batch
