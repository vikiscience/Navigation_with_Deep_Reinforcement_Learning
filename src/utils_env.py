import const

from unityagents import UnityEnvironment


class Environment(UnityEnvironment):
    def __init__(self):
        super().__init__(file_name=const.file_name_env)

    def get_info(self):
        # get the default brain
        brain_name = self.brain_names[0]
        brain = self.brains[brain_name]

        # reset the environment
        env_info = self.reset(train_mode=True)[brain_name]

        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))

        # number of actions
        action_size = brain.vector_action_space_size
        print('Number of actions:', action_size)

        # examine the state space
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

        return state_size, action_size
