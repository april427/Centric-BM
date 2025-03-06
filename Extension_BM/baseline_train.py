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
import torch.nn as nn
import argparse


import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM')
from beam_management_env_letter import BeamManagementEnv
from util_functions import SaveOnBestTrainingRewardCallback

# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_UTs', type=int, default=2, help='Number of UEs')
    parser.add_argument('--Nt', type=int, default=16, help='Number of antennas')
    parser.add_argument('--model', type=str, default='PPO', help='Model to train')
    parser.add_argument('--timesteps', type=int, default=2e4, help='Number of timesteps to train')
    parser.add_argument('--if_training', type=bool, default=True, help='Training or testing')
    parser.add_argument('--if_debugging', type=bool, default=False, help='If in debugging mode')
    parser.add_argument('--Ptdbm', type=int, default=15, help='Transmit power in dbm')
    parser.add_argument('--EnvID', type=str, default='BeamManagementEnv-v3', help='Environment ID')
    args, unknown = parser.parse_known_args()
    return args

# %%

if __name__ == "__main__":
    args = parse_args()
    model_type = args.model
    model_class = getattr(sys.modules[__name__], model_type)
    env = gym.make("BeamManagementEnv-v3",  args = args)
    # check_env(env, warn=True)
    log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(env.unwrapped.N_UTs) +"ue" + str(env.unwrapped.number_of_antennas)+ "Nt/" 

    policy_kwargs = {
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "activation_fn": nn.ReLU
    }

    # %%
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    model = model_class("MlpPolicy", env, policy_kwargs=policy_kwargs, ent_coef=1e-4, learning_rate=1e-4, verbose=1)
    timesteps = args.timesteps


    # %%
    ################################### Learn a model #################################
    # Create the callback: check every episode
    callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)


    model.learn(total_timesteps=timesteps, callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO BeamManagement-v3")
    # plt.show()

    model.save(log_dir + "best_model")
    print(log_dir)
    env.close()
    #%%
    monitor_file = log_dir + "monitor.csv"
    monitor_data = pd.read_csv(monitor_file, skiprows=1)

    window_size = 100
    moving_average = monitor_data['r'].rolling(window=window_size).mean()

    fig, ax = plt.subplots()
    # ax.plot(monitor_data['r'], label='Reward')
    ax.plot(moving_average, label='Moving Average', color='orange')
    ax.legend()
    ax.grid()
    
    plt.show()
# %%
