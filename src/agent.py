# -*- coding: utf-8 -*-
import const
from src.model import DQN

from pathlib import Path
import random
import numpy as np
from collections import deque


class DQNAgent:
    model_path = const.file_path_model

    def __init__(self, num_states: int = const.state_size,
                 num_actions: int = const.action_size,
                 use_double_dqn: bool = const.use_double_dqn,
                 memory_size: int = const.memory_size,
                 update_target_each_iter: int = const.update_target_each_iter,
                 gamma: float = const.gamma,
                 batch_size: int = const.batch_size,
                 model_learning_rate: float = const.model_learning_rate,
                 model_fc1_num: int = const.model_fc1_num,
                 model_fc2_num: int = const.model_fc2_num
                 ):

        # agent params
        self.num_states = num_states
        self.num_actions = num_actions
        self.use_double_dqn = use_double_dqn
        self.memory = deque(maxlen=memory_size)
        self.update_target_each_iter = update_target_each_iter
        self.gamma = gamma
        self.batch_size = batch_size

        # model params
        self.model_learning_rate = model_learning_rate
        self.model_fc1_num = model_fc1_num
        self.model_fc2_num = model_fc2_num

        self.model = DQN(num_inputs=self.num_states, num_outputs=self.num_actions,
                         lr=self.model_learning_rate,
                         fc1_num=self.model_fc1_num, fc2_num=self.model_fc2_num)
        self.target_model = DQN(num_inputs=self.num_states, num_outputs=self.num_actions,
                                lr=self.model_learning_rate,
                                fc1_num=self.model_fc1_num, fc2_num=self.model_fc2_num)

    def act(self, state, eps):
        # epsilon-greedy policy for Q
        probs = np.full(self.num_actions, eps / self.num_actions)
        q_values = self.model.predict(state)
        best_a = np.argmax(q_values)
        probs[best_a] += 1 - eps
        a = np.random.choice(range(self.num_actions), p=probs)
        return a

    def do_stuff(self, state, action, reward, next_state, done, t):
        self._memorize(state, action, reward, next_state, done)

        # always replay after each time step, if the memory is large enough
        if len(self.memory) >= self.batch_size:
            # replay experience (generate random batch + fit)
            self._replay_minibatch()

        # update target model
        if t % self.update_target_each_iter == 0 or done:
            # print('>>> Updating target model')
            self._update_target_model()

    def load(self):
        const.myprint('Loading model from:', self.model_path)
        self.model.load_weights(str(self.model_path))

    def save(self):
        const.myprint('Saving model to:', self.model_path)
        self.model.save_weights(str(self.model_path))

    def set_model_path(self, i):
        p = self.model_path
        self.model_path = Path(p.parent, 'model_' + str(i) + p.suffix)

    def _memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _replay_minibatch(self):
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
        expected_batch = self.model.predict(states_batch)
        for i, a, t in zip(range(self.batch_size), action_batch, targets_batch):
            expected_batch[i][a] = t

        #print(states_batch.shape, action_batch.shape, reward_batch.shape,
        #      next_states_batch.shape, done_batch.shape,
        #      targets_batch.shape, expected_batch.shape)

        # Perform gradient descent update
        self.model.fit(states_batch, expected_batch)

    def _weight_update_DQN(self, reward_batch, next_states_batch, done_batch):
        # DQN with fixed Q-targets

        # Calculate Q values from target model for next states
        q_values_next_target = self.target_model.predict(next_states_batch)

        # Get max of these Q values
        q_values_next_target_max = np.max(q_values_next_target, axis=1)

        # Compute Q targets for current states
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.gamma * q_values_next_target_max

        return targets_batch

    def _weight_update_Double_DQN(self, reward_batch, next_states_batch, done_batch):
        # Calculate Q values and targets (Double DQN)

        # select best actions
        q_values_next = self.model.predict(next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)

        # evaluate best actions
        q_values_next_target = self.target_model.predict(next_states_batch)

        q_values_next_target_argmax = np.zeros(self.batch_size)
        for i, j in enumerate(best_actions):
            q_values_next_target_argmax[i] = q_values_next_target[i][j]  # i = batch index, j = best action

        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        self.gamma * q_values_next_target_argmax

        return targets_batch
