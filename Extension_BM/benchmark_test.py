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
args.dt = 0.1
args.if_training=True
env = gym.make(args.EnvID,  args=args)
# check_env(env)
num_episodes = 100
num_steps = 100
reward_done = []
reward_list = []
throughput = []
latency = []
error_range = np.zeros((num_episodes, args.N_UTs))
error_speed = np.zeros((num_episodes, args.N_UTs))
# BER  = np.zeros((num_episodes, args.N_UTs))

for ep in range(1, num_episodes+1):

    obs,_ = env.reset()
    reward_episode = 0
    er = np.zeros(args.N_UTs)
    ev = np.zeros(args.N_UTs)
    # ber = np.zeros(args.N_UTs)

    for t in range(num_steps):
        # print(action)
        
        _, rewards, dones, _, info = env.step('greedy')
        
        reward_episode += rewards
        er += info['nmse_R']
        ev += info['nmse_v']
        #ber += info['BER']

        if dones:
            reward_done.append(reward_episode)
            print("all transmitted")
            break
        latency.append(info['latency'])

    error_range[ep-1,] = er/(t+1)
    error_speed[ep-1,] = ev/(t+1)
    # BER[ep-1,] = ber/(t+1)


    reward_list.append(reward_episode)       
    
    thp = info['transmitted_packets']/ info['total_traffic']
    print(info['buffer'])
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
