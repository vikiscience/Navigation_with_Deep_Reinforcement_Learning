import const
from src import utils_plot


eps_0 = 0.9
decay_factor = 0.999
eps_min = 0.01

#################################################################################


def get_glie(eps=None):
    if eps is None:
        eps_i = eps_0
    else:
        eps_i = eps * decay_factor
    return max(eps_i, eps_min)


#################################################################################


class DQNAlgo:

    def __init__(self, env, agent, num_episodes: int = const.num_episodes):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.agent = agent
        self.num_states = agent.num_states
        self.num_actions = agent.num_actions

        # algo params
        self.num_episodes = num_episodes
        self.eps_test = 0.05

    def train(self):

        history = []
        eps = None

        for e in range(self.num_episodes):
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state (s_t)
            score = 0  # initialize the score

            eps = get_glie(eps)  # decay epsilon
            done = False
            t = 0

            while not done:

                # choose a_t using epsilon-greedy policy
                action = self.agent.act(state, eps)

                # take action a_t, observe r_{t+1} and s_{t+1}
                env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished

                # Memorize new sample, replay, update target network
                self.agent.do_stuff(state, action, reward, next_state, done, t)

                state = next_state
                score += reward
                t += 1

            print('-' * 80, "\nEpisode: {}/{}, score: {}, e: {:.2}"
                  .format(e + 1, self.num_episodes, score, eps), t)
            history.append(score)

            if (e + 1) % 100 == 0 or e + 1 == self.num_episodes:
                self.agent.save()

        print('History:', history)
        utils_plot.plot_history_rolling_mean(history)
        self.env.close()

    def test(self):
        self.agent.load()

        env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = self.agent.act(state, eps=self.eps_test)  # select an action
            env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        print("Score: {}".format(score))

        self.env.close()
