import const
from src import agent, algo, hyperparameter_search, utils_env

import numpy as np
import argparse

random_seed = const.random_seed
np.random.seed(random_seed)

state_size = const.state_size
action_size = const.action_size
N = const.rolling_mean_N


def train_default_algo():
    env = utils_env.Environment()
    # use default params
    ag = agent.DQNAgent()
    al = algo.DQNAlgo(env, ag)
    al.train()


def test_default_algo(use_ref_model: bool = False):
    env = utils_env.Environment()
    # use default params
    ag = agent.DQNAgent()
    if use_ref_model:
        print('... Test the agent using reference model ...')
        ag.set_model_path('ref')
    al = algo.DQNAlgo(env, ag)
    al.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a Deep '
                                                 'Reinforcement Learning agent '
                                                 'to navigate in a Banana Environment',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--exec', choices=['train', 'test', 'grid'],
                        default='train', help='Train/test a DQN agent with default parameters or perform grid search to find the best agent')
    parser.add_argument('-r', '--use_reference_model', action="store_true", default=False,
                        help='In Test Mode, use the pretrained reference model')

    args = parser.parse_args()
    exec = args.exec
    use_ref_model = args.use_reference_model

    if exec == 'train':
        train_default_algo()
    elif exec == 'test':
        test_default_algo(use_ref_model)
    else:
        hyperparameter_search.grid_search()
