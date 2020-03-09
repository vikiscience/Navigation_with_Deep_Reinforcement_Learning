import const
from src import agent, algo, utils_env

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid

state_size = const.state_size
action_size = const.action_size
N = const.rolling_mean_N


class MyNavigator(BaseEstimator, ClassifierMixin):
    def __init__(self, num_states: int = const.state_size,
                 num_actions: int = const.action_size,
                 num_episodes: int = const.num_episodes,
                 eps_0: float = const.eps_0,
                 eps_decay_factor: float = const.eps_decay_factor,
                 eps_min: float = const.eps_min,
                 eps_test: float = const.eps_test,
                 use_double_dqn: bool = const.use_double_dqn,
                 memory_size: int = const.memory_size,
                 update_target_each_iter: int = const.update_target_each_iter,
                 gamma: float = const.gamma,
                 batch_size: int = const.batch_size,
                 model_learning_rate: float = const.model_learning_rate,
                 model_fc1_num: int = const.model_fc1_num,
                 model_fc2_num: int = const.model_fc2_num
                 ):

        # algo params
        self.num_episodes = num_episodes
        self.eps_0 = eps_0
        self.eps_decay_factor = eps_decay_factor
        self.eps_min = eps_min
        self.eps_test = eps_test

        # agent params
        self.num_states = num_states
        self.num_actions = num_actions
        self.use_double_dqn = use_double_dqn
        self.memory_size = memory_size
        self.update_target_each_iter = update_target_each_iter
        self.gamma = gamma
        self.batch_size = batch_size

        # model params
        self.model_learning_rate = model_learning_rate
        self.model_fc1_num = model_fc1_num
        self.model_fc2_num = model_fc2_num

    def fit(self, i: int, env: utils_env.Environment):
        self.ag = agent.DQNAgent(state_size, action_size,
                                 self.use_double_dqn, self.memory_size,
                                 self.update_target_each_iter, self.gamma, self.batch_size,
                                 self.model_learning_rate, self.model_fc1_num, self.model_fc2_num)
        self.ag.set_model_path(i)  # save each candidate's model separately

        self.al = algo.DQNAlgo(env, self.ag,
                               self.num_episodes, self.eps_0, self.eps_decay_factor,
                               self.eps_min, self.eps_test)
        self.al.set_image_path(i)  # save each candidate's score separately

        history = self.al.train(with_close=False)  # do not close the Env so that other agents can be trained
        score = self._get_score(history)
        return score

    def _get_score(self, hist):
        # prepare data
        x = pd.Series(hist)
        y = x.rolling(window=N).mean().iloc[N - 1:]
        if not y.empty:
            score = y.iloc[-1]
        else:
            score = 0.
        print('\n', score, hist)
        return score


def grid_search():
    env = utils_env.Environment()

    print('=' * 30, 'Grid Search', '=' * 30)

    params = {
        # 'num_episodes': [5, 10],  # test
        'batch_size': [32, 64, 128],
        'use_double_dqn': [True, False],
        'eps_decay_factor': [0.99, 0.95, 0.9],
        'gamma': [0.95, 0.9],
        'update_target_each_iter': [2, 4, 8, 16],
        'model_learning_rate': [0.001, 0.0001, 0.00001],
        'model_fc1_num': [32, 20],
        'model_fc2_num': [16, 10],
        'num_episodes': [625, 700, 1000, 2000],
        'memory_size': [20000, 40000]
    }

    grid = ParameterGrid(params)
    rf = MyNavigator()

    best_score = -10.
    best_grid = None
    best_grid_index = 0
    result_dict = {}
    key_list = list(params.keys()) + ['score']
    df = pd.DataFrame(columns=key_list)

    for i, g in enumerate(grid):
        rf.set_params(**g)
        score = rf.fit(i, env)
        result_dict[i] = {'score': score, 'grid': g}

        d = g
        d['score'] = score
        df = df.append(d, ignore_index=True)

        print('Evaluated candidate:', i, result_dict[i])
        # save if best
        if score >= best_score:
            best_score = score
            best_grid = g
            best_grid_index = i

    for k, v in result_dict.items():
        print(k, v)

    print("==> Best score:", best_score)
    print("==> Best grid:", best_grid_index, best_grid)

    print(df.pivot(index=key_list[0], columns=key_list[1], values=key_list[2]))

    env.close()  # finally, close the Env
