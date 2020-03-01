from src import agent, algo

from unityagents import UnityEnvironment

fn = 'D:\D_Downloads\Banana_Windows_x86_64\Banana.exe'
env = UnityEnvironment(file_name=fn)


def get_info():
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

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


if __name__ == '__main__':
    state_size, action_size = get_info()

    ag = agent.DQNAgent(state_size, action_size)
    al = algo.DQNAlgo(env, ag)

    al.train()

    al.test()
