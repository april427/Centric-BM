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

from beam_management_env_letter import BeamManagementEnv

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
num_episodes = 1000
episode_durations = []
reward_episode = []
num_sensing_slot = []
throughput = []
action_list = []
num_steps = 100
action = 0
counter = 0
N=6
list_actions = N_TDMA(N, env.unwrapped.N_UEs)
sum_reward = []
th_episode = []
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    print("Episode: " + str(i_episode), flush=True)
    state = env.reset()
    total_packets = sum(env.unwrapped.datapool)
    state = torch.tensor(state, dtype=torch.float32)
    position_ue_list = []
    reward_list = []
    l_a = 0
    for t in range(num_steps):
        action = int(list_actions[l_a])
        action_list.append(action)
        observation, reward, terminated, info = env.step(action)
        # print(observation)
        position_ue_list.append(info['ue_position'])
        reward_list.append(reward)
        reward = torch.tensor([reward])
        done = terminated

        l_a += 1
        if l_a >= len(list_actions):
            l_a = 0

        if done:
            episode_durations.append(t + 1)
            reward_episode.append(np.mean(reward_list))
            break
    
    throughput.append(1 - np.sum(env.unwrapped.datapool)/ total_packets)
    num_sensing_slot.append(info['num_sensing_slots'])
    #sensing_slots_ratio.append(info['num_sensing_slots']/(t+1))

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

# print("Total lost traffic DL and UL: " + str(sum(info['lost_traffic'])))
# print("Total traffic DL and UL: " + str(sum(info['total_traffic'])))
# print("Total lost traffic DL and UL for 3 users:\n" + str(info['lost_traffic'] / info['total_traffic']))
# print("Average PER: " + str(np.mean(info['lost_traffic'] / info['total_traffic'])))
# print("Total Throughput: " + str(1-sum(info['failure_type']/sum(sum(info['total_traffic'])))))
# print("(%) of failure types (Buffer, Beam, Blockage) over failed traffic: " + str(info['failure_type']/sum(info['failure_type'])))
# print("(%) of failure types (Buffer, Beam, Blockage) over total traffic: " + str(info['failure_type']/sum(sum(info['total_traffic']))))
# print("(%) of drops over the total traffic: " + str(sum(info['failure_type']/sum(sum(info['total_traffic'])))))
# print("Lost packets by type: " + str(info['failure_type']))


np.savetxt('results_bm/Thp_' + str(N) + 'TDMA', throughput)



# %%
