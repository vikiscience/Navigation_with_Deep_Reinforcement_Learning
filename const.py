from pathlib import Path

model_path = Path('./models/')
output_path = Path('./output/')
file_path_model = model_path / 'model.npy'
file_path_img_score = output_path / 'score.png'

random_seed = 0xABCD
num_episodes = 200 #1000
rolling_mean_N = 100
