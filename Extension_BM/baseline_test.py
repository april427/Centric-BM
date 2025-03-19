#%%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO, DDPG, A2C
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.callbacks import BaseCallback
import time
import argparse
import itertools

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
from beam_management_env_letter import BeamManagementEnv
import pandas as pd
from baseline_train import parse_args

if __name__ == "__main__":
    args = parse_args()
    model_type = args.model
    model_class = getattr(sys.modules[__name__], model_type)

    args.cdir = 'buffer'
    args.if_training=False
    # args.if_debugging = True
    args.dt = 0.1

    log_dir = log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(args.N_UTs) +"ue" + str(args.Nt)+ "Nt_" + args.cdir +"/" 


    
    env = gym.make(args.EnvID,  args = args)
    print("N_UTs=", args.N_UTs)
    print(log_dir)

    # %%
    ################################## Evaluate ####################################
    model = model_class.load(log_dir + "best_model")

    num_episodes = 10
    num_steps = 40 if args.if_training else 100 
    reward_done = []
    reward_list = []
    throughput = []
    latency = []
    error_range = np.zeros((num_episodes, args.N_UTs))
    error_speed = np.zeros((num_episodes, args.N_UTs))
    BER  = np.zeros((num_episodes, args.N_UTs))

    for ep in range(1, num_episodes+1):
        # st = time.time()
        obs,_ = env.reset()
        # et = time.time()
        # print("reset time: ", et-st)
        reward_episode = 0
        er = np.zeros(args.N_UTs)
        ev = np.zeros(args.N_UTs)
        ber = np.zeros(args.N_UTs)

        for t in range(num_steps):
            action, _states = model.predict(obs)
            # st1 = time.time()
            obs, rewards, dones, _, info = env.step(action)
            # et1 = time.time()
            # print("step time: ", et1-st1)
            reward_episode += rewards
            er += info['nmse_R']
            ev += info['nmse_v']
            #ber += info['BER']
            latency.append(info['latency'])

            if dones:
                reward_done.append(reward_episode)
                print("all transmitted")
                break

            # estimation_error.append(info['est_error'])

        error_range[ep-1,] = er/(t+1)
        error_speed[ep-1,] = ev/(t+1)
        #BER[ep-1,] = ber/(t+1)

        if t == num_steps - 1:  
            reward_list.append(reward_episode)       
        print(reward_episode)
        thp = np.sum(info['transmitted_packets'])/ info['total_traffic']
        print(info['transmitted_packets'])
        throughput.append(thp)
        
        print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)


    thp_cdf = np.sort(throughput)
    yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)

    # 
    figure_handle = plt.figure()
    ax = figure_handle.add_subplot()
    ax.plot(thp_cdf, yvals, 'o')
    ax.set_xlabel('Throughput')
    ax.set_ylabel('CDF')

    figure_handle = plt.figure()
    ax1 = figure_handle.add_subplot()
    for i in range(args.N_UTs):
        ax1.plot(error_range[:,i], '-o')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Normalized Error of Range Estimation')
    plt.show()

    figure_handle = plt.figure()
    ax1 = figure_handle.add_subplot()
    for i in range(args.N_UTs):
        ax1.plot(error_speed[:,i],'--x')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Normalized Error of Speed Estimation')
    plt.show()

    figure_handle = plt.figure()
    ax2 = figure_handle.add_subplot()
    ax2.hist(list(itertools.chain(*latency)))
    ax2.set_xticks(range(5)) 
    ax2.set_xticklabels(range(5))


# %%
# Save throughput to a CSV file


# df = pd.read_csv(log_dir+"latency.csv")
# df = df.drop(rows_to_delete)
# df['benchmark'] = latency
# df.to_csv(log_dir+"latency.csv", index=False)
# latency_df = pd.DataFrame({'baseline': list(itertools.chain(*latency))})
# latency_df.to_csv(log_dir+'latency.csv', index=False, mode='a')

# monitor_file = log_dir + "monitor.csv"
# monitor_data = pd.read_csv(monitor_file, skiprows=1)

# print(monitor_data.head())

# # Plotting the 'r' column from the monitor.csv file
# plt.figure()
# plt.plot(monitor_data['r'])
# plt.xlabel('Timesteps')
# plt.ylabel('Reward')
# plt.title('Reward over Time')
# plt.show()
# %%
