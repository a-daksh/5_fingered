from stable_baselines3 import SAC
import os
from grasp_env import GraspEnv

import warnings
warnings.filterwarnings("ignore")

models_dir = "models/SAC"
logdir = "logs"

if not os.path.exists(models_dir): 
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = GraspEnv()
# env = gym.make('LunarLander-v2')
env.reset()

model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000
for i in range(1,500):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC")
    model.save(f"{models_dir}P/{TIMESTEPS*i}")

env.close()

