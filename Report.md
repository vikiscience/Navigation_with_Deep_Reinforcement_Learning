# Learning Algorithm

The problem of navigating in a square environment and collecting bananas can be formulated as a Reinforcement Learning (RL) problem, where an Agent learns through interaction with Environment to achieve their goal - collecting maximum amount of yellow bananas and avoiding the blue ones. The Agent can observe the state which the Environment is in, and takes actions that affect this state. In turn, the Environment gives the Agent feedback ("rewards") based on the actions.

This setting can be formulated as Markov Decision Process (MDP), where:

* Action space `A` contains 4 actions (_forward_, _backward_, _left_, _right_)
* State space `S`, which is defined to be 37-dimensional
* The transition to the next state `s_{t+1}` and the resulting reward `r_{t+1}` are defined by the Environment and depend only on the current state `s_t` and the Agent's chosen action `a_t` ("one-step dynamics")
* Discount rate `gamma`, which is used by the Agent to prioritize current rewards over future rewards.

The Agent's goal is to maximize the expected discounted return (weighted sum of all future rewards per episode). In order to achieve his goal the Agent estimates the values of `(s, a)` pairs by learning the so-called _action-value function_ `Q(s, a)`. It is then used to calculate the optimal _policy_, i.e. a function that gives the best possible action for each Environment state.

In our case, the state space is continuous and high-dimensional, which means that Deep Neural Networks (DNNs) can be used to represent the action-value function `Q`. Generally, the input of a DNN is state vector, and the output is a vector of Q-values for every action.

Well-established algorithms are eg. Deep Q-Networks ([DQN](https://www.nature.com/articles/nature14236)) and Double Deep Q-Networks ([Double DQN](https://arxiv.org/abs/1509.06461)). Both are based on a Q-Learning algorithm, which updates the Q-values iteratively as folows:

`Q_new(s_t, a_t) := Q(s_t, a_t) + alpha * (r_{t+1} + gamma * max(Q(s_{t+1}, a)) - Q(s_t, a_t))`

In other words, `<Q_new> = <Q_old> + alpha * (<target> - <Q_old>)`

The Q-values are utilized by the Agent to choose his actions with _epsilon-greedy policy_, where for each state the best action according to `Q` is chosen with probability `(1 - epsilon)`, and a random action - with probability `epsilon`. Traditionally, `epsilon` is decayed after each episode during training.

## DQN

DNNs are notably unstable if trained to solve RL tasks (as reported in this [article](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf)). Causes of instability are:

* correlations in sequences of Environment observations

* small updates in action-value function can significantly change the Agent's policy and therefore impact data distribution for the next steps

* correlations between action-values and target values

To overcome these disadvantages, two techniques can be used as described below.

#### Experience replay

The Agent maintains a replay buffer of capacity `M` where he stores his previous experience in form of tuples `(s_t, a_t, r_{t+1}, s_{t+1})`. Every now and then, the Agent samples a batch of tuples randomly from buffer and uses these to update `Q`. Thus, the sequence correlation gets eliminated, and the learnt policy is more robust.

#### Fixed Q-targets

To solve a problem of "moving target", we don't change the DNN weights during training step, because they are used for estimating next best action. We achieve this by maintaining two DNNs - one is used for training, the other ("target network") is fixed and only updated with current weights after each `C` steps. 

#### Algorithm

1. Initialize: `N` - number of training episodes, replay memory with capacity `M`, two DNNs with the same architecture, `epsilon` decay strategy.

2. For each episode out of `N`:
   
   2.1. `t := 0`

   2.2. While not done:

      2.2.1. Observe `s_t`

      2.2.2. Choose `a_t` using epsilon-greedy policy w.r.t. `Q`
   
      2.2.3. Take action `a_t`, observe reward `r_{t+1}` and next state `s_{t+1}` of the Environment 
   
      2.2.4. Store tuple `(s_t, a_t, r_{t+1}, s_{t+1}, done_{t+1})` in replay memory, where `done_{t+1} = 1` if the episode ended at timestep `t+1`, else `0`
   
      2.2.5. Sample random batch of size `B` from memory
   
      2.2.6. Perform gradient descent on `(y_j - <Q_old>)` w.r.t. weights of the network `Q`, where `j` is index of a tuple `(s_{j-1}, a_{j-1}, r_j, s_j, done_j)` in the batch, and:
   
      `y_{j-1} = r_j + gamma * (1 - done_j) * max(Q_target(s_j, a))`
   
      2.2.7. Every `C` steps, update the target network `Q_target`
   
      2.2.8. `t := t + 1`


## Double DQN

Double DQN improves on DQN algorithm and calculates weight updates in step 2.2.6. as follows:

`y_{j-1} = r_j + gamma * (1 - done_j) * Q_target(s_j, argmax(Q(s_j, a)))`

Here, `Q` is used to select best action for the next state (`argmax(...)`), and `Q_target` - to evaluate this action. The motivation behind it is to prevent incidental high rewards to be propagated further by `Q`.


## Implementation

The interaction between the Agent and Environment is implemented in `algo.py`. The Agent's internal logic is placed in `agent.py`, including action selection according to epsilon-greedy policy, minibatch sampling and learning as well as updating target network. The Agent can switch between updating weights according to DQN or Double DQN (boolean hyperparameter `use_double_dqn`).

`model.py` contains the DNN itself and makes use of PyTorch. The model architecture is sequential with 3 linear neuron layers and ELU as an activation function. Here, input and output vectors are transformed to and from PyTorch tensors, accordingly. 

Finally, Grid Search is implemented in `hyperparameter_search.py` in order to select the best Agent solving the given Environment. The script also documents what hyperparameter values were tested so far. Best resulting hyperparameters are already listed in `const.py`.



## Hyperparameter optimization

As mentioned above, the current best hyperparameters of the algorithm found by Grid Search are the following:

`num_episodes = 2000`

`epsilon_0 = 0.9`

`epsilon_decay_factor = 0.95`

`epsilon_min = 0.01`

`epsilon_test = 0.05`

`use_double_dqn = False`

`memory_size = 20000`

`update_target_each_iter = 4`

`gamma = 0.95`

`batch_size = 128`

`model_learning_rate = 0.0001`

`model_fc1_neurons = 32`

`model_fc2_neurons = 16`

# Future Work

Other Deep RL algorithms can be implemented such as [Dueling DQN](https://arxiv.org/abs/1511.06581), [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), [NoisyNets](https://arxiv.org/abs/1706.10295) or a [combination](https://arxiv.org/abs/1710.02298) of them.
