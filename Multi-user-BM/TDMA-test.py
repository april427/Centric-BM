# %%
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Multi-user-BM/')

from beam_management_env_letter import BeamManagementEnv
from plot_functions import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
experiment_number = random.randint(0, 1000000000)

# %%
def N_TDMA(N, N_UEs): # S - C - C... 
    b = 5
    indices = []
    a = 2*np.ones(N_UEs, dtype = int)
    ind = sum([a[i]*b**(N_UEs-i-1) for i in range(N_UEs)])
    indices.append(ind)
    # indices.extend([np.zeros(N_UEs, dtype = int)])

    for _ in range(N):
        a = np.zeros(N_UEs, dtype = int)
        ind = sum([a[i]*b**(N_UEs-i-1) for i in range(N_UEs)]) 
        indices.append(ind)

    return indices

# %%
env = gym.make('BeamManagementEnv-v1')
num_episodes = 100
episode_durations = []
reward_episode = []
num_sensing_slot = []
throughput = []
action_list = []
num_steps = 100
action = 0
counter = 0
N=25
list_actions = N_TDMA(N, env.unwrapped.N_UEs)
sum_reward = []
th_episode = []

reward_list = []
reward_done = []

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    #print("Episode: " + str(i_episode), flush=True)
    state, _ = env.reset()
    total_packets = sum(env.unwrapped.datapool)
    state = torch.tensor(state, dtype=torch.float32)
    position_ue_list = []

    l_a = 0
    reward_episode = 0
    for t in range(num_steps):
        action = int(list_actions[l_a])
        action_list.append(action)
        observation, reward, terminated,_, info = env.step(action)
        # print(observation)
        position_ue_list.append(info['ue_position'])
        reward_episode += reward


        l_a += 1
        if l_a >= len(list_actions):
            l_a = 0

        if terminated:
            episode_durations.append(t + 1)
            reward_done.append(reward_episode)
            break

    reward_list.append(reward_episode)

    thp = 1 - np.sum(env.unwrapped.datapool)/ total_packets
    throughput.append(thp)
    num_sensing_slot.append(info['num_sensing_slots'])
    #sensing_slots_ratio.append(info['num_sensing_slots']/(t+1))

    print("Episode : {} \t Throughput: {}".format(i_episode, round(thp,2)), flush = True)

print('Complete')
# print(np.mean(mean_packet_rate_episode))
# print(np.mean(sum_reward))
# print(np.mean(th_episode))

# %%
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
tdma_thp_cdf = np.sort(throughput)
yvals = np.arange(len(tdma_thp_cdf))/float(len(tdma_thp_cdf)-1)
ax.plot(tdma_thp_cdf, yvals, '-')
ax.set_xlabel('Throughput')
ax.set_ylabel('CDF')
ax.legend(['{}-TDMA'.format(N)])




np.savetxt('Multi-user-BM/results_bm/Thp_' + str(N) + 'TDMA', throughput)



#%%
# fig,ax = plt.subplots()
# #ax.plot(reward_done)
# ax.plot(reward_list)
# plot_column_from_csv('Multi-user-BM/saved_policy/PPO_logs/PPO_BeamManagementEnv-v1_log_9.csv', 'reward')

# fig.show()
# %%
# fig, ax = plt.subplots()
# for N in [1,3,20,25,30]:
#     th = np.loadtxt('Multi-user-BM/results_bm/Thp_' + str(N) + 'TDMA')
#     th_cdf = np.sort(th)
#     yvals = np.arange(len(th))/float(len(th)-1)
#     ax.plot(th_cdf, yvals, '-')
# ax.legend(['1-TDMA','3-TDMA','20-TDMA', '20-TDMA', '30-TDMA'])

# %%
