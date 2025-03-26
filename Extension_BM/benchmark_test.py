import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import itertools

from stable_baselines3 import  PPO, DDPG
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import time

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
from beam_management_env_letter import BeamManagementEnv

from baseline_train import parse_args

args = parse_args()
# args.dt = 0.1
args.if_training=False
env = gym.make(args.EnvID,  args=args)
# check_env(env)
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

    obs,_ = env.reset()
    reward_episode = 0
    er = np.zeros((args.N_UTs, num_steps))
    ev = np.zeros((args.N_UTs, num_steps))
    # ber = np.zeros(args.N_UTs)

    for t in range(num_steps):
        # print(action)
        
        _, rewards, dones, _, info = env.step('estimated')
        
        reward_episode += rewards
        er[:,t] = info['nmse_R']
        ev[:,t] = info['nmse_v']
        #ber += info['BER']

        if dones:
            reward_done.append(reward_episode)
            break
        latency.append(info['latency'])

    error_range[ep-1,] = np.sum(er, axis=1)/(t+1)
    error_speed[ep-1,] = np.sum(ev, axis=1)/(t+1)
    # BER[ep-1,] = ber/(t+1)

    nmseR_allsteps.append(er)
    nmseV_allsteps.append(ev)


    reward_list.append(reward_episode)       
    
    thp = info['transmitted_packets']/ info['total_traffic']
    print(info['buffers'])
    throughput.append(thp)
    
    print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)

    print("Reward: ", reward_episode)


thp_cdf = np.sort(throughput)
yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)

# %%
# Plotting
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
ax.plot(reward_list, '-.') 
#
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
ax.plot(thp_cdf, yvals, '-')
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
