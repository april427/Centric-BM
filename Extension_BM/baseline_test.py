#%%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO, DDPG, A2C
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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

    args.cdir = 'modified_r_s'
    args.if_training=False
    # args.if_debugging = True
    args.dt = 0.2

    log_dir = log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(args.N_UTs) +"ue" + str(args.Nt)+ "Nt_" + args.cdir +"/" 
    monitor_file = log_dir + "monitor.csv"
    monitor_data = pd.read_csv(monitor_file, skiprows=1)

    window_size = 1000
    moving_average = monitor_data['r'].rolling(window=window_size).mean()

    fig, ax = plt.subplots()
    # ax.plot(monitor_data['r'], label='Reward')
    ax.plot(moving_average, label='Moving Average', color='orange')
    ax.legend()
    ax.grid()
    
    plt.show()
    
    env = gym.make(args.EnvID,  args = args)
    print("N_UTs=", args.N_UTs)
    print(log_dir)

    # %%
    ################################## Evaluate ####################################
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize.load("vec_normalize.pkl", env)
    # env.training = False  # Don't update stats during evaluation
    # env.norm_reward = False  # No need to normalize rewards during evaluation
    
    model = model_class.load(log_dir + "best_model", env=env)

    num_episodes = 10
    num_steps = 40
    reward_done = []
    reward_list = []
    throughput = []
    latency = []
    error_range = np.zeros((num_episodes, args.N_UTs))
    error_speed = np.zeros((num_episodes, args.N_UTs))
    nmseR_allsteps = []
    nmseV_allsteps = []
    # BER  = np.zeros((num_episodes, args.N_UTs))

    for ep in range(1, num_episodes+1):
        # st = time.time()
        obs,_ = env.reset()
        # et = time.time()
        # print("reset time: ", et-st)
        reward_episode = 0
        er = np.zeros((args.N_UTs, num_steps))
        ev = np.zeros((args.N_UTs, num_steps))
        # ber = np.zeros(args.N_UTs)

        for t in range(num_steps):
            action, _states = model.predict(obs)
            # st1 = time.time()
            obs, rewards, dones, _, info = env.step(action)
            # et1 = time.time()
            # print("step time: ", et1-st1)
            reward_episode += rewards
            er[:,t] = info['nmse_R']
            ev[:,t] = info['nmse_v']
            #ber += info['BER']
            latency.append(info['latency'])

            if dones:
                reward_done.append(reward_episode)
                break

        error_range[ep-1,] = np.sum(er, axis=1)/(t+1)
        error_speed[ep-1,] = np.sum(ev, axis=1)/(t+1)
        #BER[ep-1,] = ber/(t+1)

        nmseR_allsteps.append(er)
        nmseV_allsteps.append(ev)

        if t == num_steps - 1:  
            reward_list.append(reward_episode)       
        print(reward_episode)
        thp = np.sum(info['transmitted_packets'])/ info['total_traffic']
        print(info['transmitted_packets'])
        throughput.append(thp)
        
        print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)


    thp_cdf = np.sort(throughput)
    yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)

    # Reward
    figure_handle = plt.figure()
    ax = figure_handle.add_subplot()
    ax.plot(reward_list, '-.') 
    # throughput CDF 
    figure_handle = plt.figure()
    ax = figure_handle.add_subplot()
    ax.plot(thp_cdf, yvals, 'o')
    ax.set_xlabel('Throughput')
    ax.set_ylabel('CDF')

    # figure_handle = plt.figure()
    # ax1 = figure_handle.add_subplot()
    # for i in range(args.N_UTs):
    #     ax1.plot(error_range[:,i], '-o')
    # ax1.set_xlabel('Episodes')
    # ax1.set_ylabel('Normalized Error of Range Estimation')
    # plt.show()

    # figure_handle = plt.figure()
    # ax1 = figure_handle.add_subplot()
    # for i in range(args.N_UTs):
    #     ax1.plot(error_speed[:,i],'--x')
    # ax1.set_xlabel('Episodes')
    # ax1.set_ylabel('Normalized Error of Speed Estimation')
    # plt.show()

    # latency
    latency_flat = list(itertools.chain(*latency))
    figure_handle = plt.figure()
    ax2 = figure_handle.add_subplot()
    counts, bins, patches = ax2.hist(latency_flat, bins=range(7), density=True, alpha=0.7)
    bin_centers = np.arange(6)
    ax2.cla()  # Clear the previous plot
    ax2.bar(bin_centers, counts, width=0.7,align = 'center', alpha=0.7)
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(range(6))
    ax2.set_xlabel('Latency')
    ax2.set_ylabel('Probability')
    plt.show()

    # CDF of normalized error
    figure_handle = plt.figure()
    ax3 = figure_handle.add_subplot()
    nmseR_allsteps = np.array(nmseR_allsteps)
    nmseV_allsteps = np.array(nmseV_allsteps)
    for u in range(args.N_UTs):
        
        temp = nmseR_allsteps[:,u,:].flatten()
        er_cdf = np.sort(temp)
        yvals = np.arange(len(er_cdf))/float(len(er_cdf)-1)

        ax3.plot(er_cdf, yvals, '-')

    ax3.set_xlabel('Normalized Error of Range Estimation')
    ax3.set_ylabel('CDF')
    plt.show()


    figure_handle = plt.figure()
    ax4 = figure_handle.add_subplot()
    for u in range(args.N_UTs):
        temp = nmseV_allsteps[:,u,:].flatten()
        ev_cdf = np.sort(temp)
        yvals = np.arange(len(ev_cdf))/float(len(ev_cdf)-1)
        ax4.plot(ev_cdf, yvals, '-')
    ax4.set_xlabel('Normalized Error of Speed Estimation')
    ax4.set_ylabel('CDF')
    plt.show()

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
