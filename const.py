from pathlib import Path

file_name_env = 'D:\D_Downloads\Banana_Windows_x86_64\Banana.exe'

model_path = Path('./models/')
output_path = Path('./output/')
file_path_model = model_path / 'model.npy'
file_path_ref_model = model_path / 'ref_model.npy'
file_path_img_score = output_path / 'score.png'

# general params
random_seed = 0xABCD
rolling_mean_N = 100
state_size = 37
action_size = 4
episode_length = 300
verbose = False

# algo params
num_episodes = 2000
eps_0 = 0.9
eps_decay_factor = 0.999
eps_min = 0.01
eps_test = 0.05

# agent params
use_double_dqn = True #False
memory_size = 20000
update_target_each_iter = 4
gamma = 0.95  # discount rate
batch_size = 64

# model params
model_learning_rate = 0.0001
model_fc1_num = 32
model_fc2_num = 16


def myprint(*args):
    if verbose:
        print(args)
