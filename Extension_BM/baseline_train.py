import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO, DDPG, A2C
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM')
from beam_management_env_letter import BeamManagementEnv
from util_functions import SaveOnBestTrainingRewardCallback

# %%
model_type = "PPO"
model_class = getattr(sys.modules[__name__], model_type)
env = gym.make("BeamManagementEnv-v2",  if_training=True)
check_env(env, warn=True)
log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(env.unwrapped.N_UTs) +"ue" + "lr2e-4/" 

os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

model = model_class("MlpPolicy", env, learning_rate=3e-4, verbose=1)
timesteps = 2e4


# %%
################################### Learn a model #################################
# Create the callback: check every episode
callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)


model.learn(total_timesteps=timesteps, callback=callback)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO BeamManagement-v2")
# plt.show()

model.save(log_dir + "best_model")
print(log_dir)

#%%
monitor_file = log_dir + "monitor.csv"
monitor_data = pd.read_csv(monitor_file, skiprows=1)
fig,ax = plt.subplots()
ax.plot(monitor_data['r'])
plt.show()