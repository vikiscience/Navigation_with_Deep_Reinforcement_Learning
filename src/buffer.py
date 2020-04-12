import const

from collections import deque
import numpy as np
import random


class ReplayBuffer:

    def __init__(self, memory_size: int = const.memory_size):
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def append(self, tupel):
        assert len(tupel) == 5
        # state, action, reward, next_state, done = tupel
        self.memory.append(tupel)

    def sample(self, batch_size: int):
        minibatch = random.sample(self.memory, batch_size)

        # states, actions and next_states are of shape (batch_size x n_dims)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*minibatch))

        return states_batch, action_batch, reward_batch, next_states_batch, done_batch
