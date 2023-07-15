import gymnasium as gym
from stable_baselines3 import SAC
from grasp_env import GraspEnv

import warnings
warnings.filterwarnings("ignore")

env = GraspEnv()
env.reset() 
models_dir = "DR_500/models/SAC"
model_path=f"{models_dir}/466000"
# NDR 350000
# DR 466000
model=SAC.load(model_path,env=env)

episodes=50

for ep in range(episodes):

    obs=env.reset()

    truncate=False
    terminate=False

    while not truncate:
        env.render()
        obs,reward,terminate,truncate,_=env.step(env.action_space.sample())
    
env.close()
