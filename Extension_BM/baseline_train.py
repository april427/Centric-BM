import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_device
import pandas as pd
import torch.nn as nn

import torch


import sys
# sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beam_management_env_letter import BeamManagementEnv
from util_functions import SaveOnBestTrainingRewardCallback
from parse_args import parse_args

# %%

if __name__ == "__main__":
    
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # print(f"CUDA device count: {torch.cuda.device_count()}")
    # if torch.cuda.is_available():
    #     print(f"Current CUDA device: {torch.cuda.current_device()}")
    #     print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    args = parse_args()
    model_type = args.model
    model_class = getattr(sys.modules[__name__], model_type)
    env = gym.make(args.EnvID,  args = args)
    # check_env(env, warn=True)
    log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(args.N_UTs) +"ue" + str(args.Nt)+ "Nt_" + args.cdir +"/"

    policy_kwargs = {
        "net_arch": dict(pi=[128,128], vf=[128, 128]),
        "activation_fn": nn.ReLU
    }

    # %%
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir) 

    model_path = log_dir + "best_model.zip"
    if os.path.exists(model_path):
        model = model_class.load(model_path, env=env, learning_rate=lambda progress: 1e-4 + (5e-4 - 1e-4) * progress)
    else:
        model = model_class(   
                "MlpPolicy", 
                env, 
                device="cpu", 
                policy_kwargs=policy_kwargs, 
                ent_coef=1e-2, 
                learning_rate=lambda progress: 1e-4 + (3e-3 - 1e-4) * progress, 
                verbose=1)

    timesteps = args.timesteps


    # %%
    ################################### Learn a model #################################
    # Create the callback: check every episode
    callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)

    print(get_device())
    model.learn(total_timesteps=timesteps, callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO BeamManagement-v4")
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
# Append data from one CSV to another
# def append_csv(source_file, target_file):
#     source_data = pd.read_csv(source_file)
#     target_data = pd.read_csv(target_file, skiprows = 1)
    
#     combined_data = pd.concat([source_data, target_data])
#     combined_data.to_csv(target_file, index=False)

# # Example usage
# source_file = 'RL_mimo_ofdm/monitor_0.csv'
# target_file = 'RL_mimo_ofdm/monitor.csv'
# append_csv(source_file, target_file)
# %%
