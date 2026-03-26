import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter

from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_device
import pandas as pd
import torch.nn as nn

import torch


import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beam_management_env_letter import BeamManagementEnv
from util_functions import SaveOnBestTrainingRewardCallback
from parse_args import parse_args

# %%

if __name__ == "__main__":

    args = parse_args()
    model_type = args.model
    model_class = getattr(sys.modules[__name__], model_type)
    log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(args.N_UTs) +"ue" + str(args.Nt)+ "Nt_" + args.cdir +"/"
    os.makedirs(log_dir, exist_ok=True)

    num_cpu = 4
    if num_cpu == 1:
        from stable_baselines3.common.monitor import Monitor
        env = gym.make(args.EnvID,  args = args)
        env = Monitor(env, log_dir)
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.vec_env import VecMonitor
        env = SubprocVecEnv([lambda: gym.make(args.EnvID, args=args) for i in range(num_cpu)])
        env = VecMonitor(env, log_dir)
    # check_env(env, warn=True)

    # Optional observation and reward normalization
    if args.norm_obs or args.norm_reward:
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize(
            env,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
            clip_obs=args.clip_obs,
            clip_reward=args.clip_reward,
        )
    
    act_fn = nn.Tanh if args.activation == 'Tanh' else nn.ReLU
    h = args.net_arch_size
    policy_kwargs = {
        "net_arch": dict(pi=[h, h], vf=[h, h]),
        "activation_fn": act_fn
    }

    def linear_schedule(initial_value, final_value):
        return lambda progress: final_value + (initial_value - final_value) * progress

    # %%

    model_path = log_dir + "best_model.zip"
    if os.path.exists(model_path):
        if model_type == "A2C":
            model = model_class.load(
                model_path,
                env=env,
                learning_rate=linear_schedule(1e-3, 1e-4)
            )
        else:
            model = model_class.load(
                model_path,
                env=env,
                learning_rate=linear_schedule(args.ppo_lr_start, args.ppo_lr_end)
            )
    else:
        if model_type == "A2C":
            model = model_class(
                "MlpPolicy",
                env,
                device="cpu",
                policy_kwargs=policy_kwargs,
                n_steps=args.n_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                rms_prop_eps=1e-5,
                learning_rate=linear_schedule(args.lr_start, args.lr_end),
                verbose=1
            )
        else:
            model = model_class(
                "MlpPolicy",
                env,
                device="cpu",
                policy_kwargs=policy_kwargs,
                n_steps=args.ppo_n_steps,
                batch_size=args.ppo_batch_size,
                n_epochs=args.ppo_n_epochs,
                gamma=args.ppo_gamma,
                gae_lambda=args.ppo_gae_lambda,
                clip_range=linear_schedule(args.ppo_clip_range, args.ppo_clip_range),
                ent_coef=args.ppo_ent_coef,
                vf_coef=args.ppo_vf_coef,
                max_grad_norm=args.ppo_max_grad_norm,
                learning_rate=linear_schedule(args.ppo_lr_start, args.ppo_lr_end),
                verbose=1
            )

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
    # Save VecNormalize stats so they can be loaded during evaluation
    if args.norm_obs or args.norm_reward:
        env.save(log_dir + "vec_normalize.pkl")
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
