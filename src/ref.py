import const

import random
import numpy as np
from collections import deque

import torch
from torch import nn
from collections import namedtuple
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update network (old: learn ?!)

###############################################################################


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

###############################################################################


class Agent:
    """Interacts with and learns from the environment."""
    model_path = const.file_path_ref_model

    def __init__(self, num_states: int = const.state_size,
                 num_actions: int = const.action_size,
                 seed=const.random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = BATCH_SIZE
        self.replay_after = BATCH_SIZE
        self.update_target_each_iter = UPDATE_EVERY
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(num_states, num_actions, seed)
        self.qnetwork_target = QNetwork(num_states, num_actions, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(num_actions, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    #def step(self, state, action, reward, next_state, done):
    #    # Save experience in replay memory
    #    self.memory.add(state, action, reward, next_state, done)
    #
    #    # Learn every UPDATE_EVERY time steps.
    #    self.t_step = (self.t_step + 1) % UPDATE_EVERY
    #    if self.t_step == 0:
    #        # If enough samples are available in memory, get random subset and learn
    #        if len(self.memory) > BATCH_SIZE:
    #            experiences = self.memory.sample()
    #            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
            action = np.int32(action)
            #print('hier', state, type(action_values.cpu().data.numpy()))
        else:
            action = random.choice(np.arange(self.num_actions))
        #print(action, type(action))
        return action

    def do_stuff(self, state, action, reward, next_state, done, t):
        self._memorize(state, action, reward, next_state, done)

        # always replay after each time step, if the memory is large enough
        if len(self.memory) >= self.batch_size:
            # replay experience (generate random batch + fit)
            self._replay_minibatch()

        if t % self.update_target_each_iter == 0 or done:
            # update target model
            # print('>>> Updating target model')
            self._update_target_model()

    def save(self, fp=None):
        if fp is None:
            fp = str(self.model_path)
        torch.save(self.qnetwork_target.state_dict(), fp)

    def load(self, fp=None):
        if fp is None:
            fp = str(self.model_path)
        self.qnetwork_local.load_state_dict(torch.load(fp))  # load the model used for inference in "act"
        self.qnetwork_local.eval()  # change the model to evaluation mode (to use only for inference)

    def _memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def _update_target_model(self):
        self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _replay_minibatch(self):
        experiences = self.memory.sample()
        self._learn(experiences, GAMMA)

    def _learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        #self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


###############################################################################


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
