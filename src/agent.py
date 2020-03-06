# -*- coding: utf-8 -*-
import const
from src.model import DQN

import random
import numpy as np
from collections import deque


class DQNAgent:
    use_double_dqn = False #True todo fix error
    model_path = const.file_path_model

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory = deque(maxlen=20000) # 2000
        self.gamma = 0.95  # discount rate
        self.learning_rate = 0.0001
        self.batch_size = 64 #32
        self.replay_after = 10
        self.update_target_each_iter = 100
        self.fc1_num = 20
        self.fc2_num = 10

        self.model = DQN(self.num_states, self.num_actions,
                         self.learning_rate, self.batch_size,
                         fc1_num=self.fc1_num, fc2_num=self.fc2_num)
        self.target_model = DQN(self.num_states, self.num_actions,
                                self.learning_rate, self.batch_size,
                                fc1_num=self.fc1_num, fc2_num=self.fc2_num)
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

        # get predictions (output - all actions), one action value will be changed
        targets_batch_zeros = self.model.predict(states_batch)
        for i, a, t in zip(range(self.batch_size), action_batch, targets_batch):
            targets_batch_zeros[i][a] = t

        #print(states_batch.shape, action_batch.shape, reward_batch.shape,
        #      next_states_batch.shape, done_batch.shape,
        #      targets_batch.shape, targets_batch_zeros.shape)

        # Perform gradient descent update
        self.model.fit(states_batch, targets_batch_zeros)

    def load(self):
        print('Loading model from:', self.model_path)
        self.model.load_weights(str(self.model_path))

    def save(self):
        print('Saving model to:', self.model_path)
        self.model.save_weights(str(self.model_path))

    def _weight_update_DQN(self, reward_batch, next_states_batch, done_batch):
        # Calculate q values and targets (DQN with fixed Q-targets)
        q_values_next_target = self.target_model.predict(next_states_batch)
        # Get max predicted Q values (for next states) from target model
        q_values_next_target_max = np.max(q_values_next_target, axis=1)

        # Compute Q targets for current states
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.gamma * q_values_next_target_max
        return targets_batch

    def _weight_update_Double_DQN(self, reward_batch, next_states_batch, done_batch):
        # Calculate q values and targets (Double DQN)

        # select best actions
        q_values_next = self.model.predict(next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)

        # evaluate best actions
        q_values_next_target = self.target_model.predict(next_states_batch)
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.gamma * q_values_next_target[:, best_actions]
        return targets_batch
