# %%
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
#from collections import namedtuple, deque
from itertools import count
import numpy as np

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Multi-user-BM/')

from beam_management_env_letter import BeamManagementEnv
from channel_functions import *
from plot_functions import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)


# %%
env = gym.make('BeamManagementEnv-v1', reward_inx = 0, if_training=False)
num_episodes = 1000
episode_durations = []
reward_done= []
num_sensing_slot = []
sensing_slots_ratio = []
throughput = []

num_steps = 100

counter = 0
ji = 1

sum_reward = []
th_episode = []
reward_list = []
estimation_error = []
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, _ = env.reset()
    total_packets = env.unwrapped.total_traffic_episode
    #state = torch.tensor(state, dtype=torch.float32)
    position_ue_list = []
    reward_episode = 0

    action = np.zeros((env.unwrapped.N_UEs))
    noted = []
   
    for t in range(num_steps):

        observation, reward, terminated,_, info = env.step(action)
        # print(observation)
        position_ue_list.append(info['ue_position'])
        reward_episode += reward

        done = terminated

        if ji == 1: 
        # set the action for the next step
            action_array = (info['est_error'] > (env.unwrapped.beamwidth/2)) #np.ones((env.unwrapped.N_UEs)) ) #*random.randint(1,4)
        
        elif ji == 2: # true crlb
            action_array = (np.sqrt(info['crlb']) > (env.unwrapped.beamwidth/2 ) ) *5 #*random.randint(1,4)
        
        elif ji==3: # crlb estimates
            action_array = (np.sqrt(info['crlb_aodest']) > env.unwrapped.beamwidth/2 ) *  5 
        
        elif ji==4:  # genie beams 
            action_array = np.array(7*np.ones(env.unwrapped.N_UEs))
        else: 
            raise ValueError('Invalid criterion')

        action = action_array # action space is multiDiscrete
        # action = 0  
        # for i in range(env.unwrapped.N_UEs):
        #     action += action_array[i]*2**(env.unwrapped.N_UEs - i - 1)
        
        for i in range(env.unwrapped.N_UEs):
                if info['buffer'][i] == 0 and i not in noted:
                    noted.append(i)
                    sensing_slots_ratio.append(info['num_sensing_slots'][i]/(t+1))

        if done:
            episode_durations.append(t + 1)
            reward_done.append(reward_episode)
            break

        estimation_error.append(info['est_error'])

    reward_list.append(reward_episode)

    thp = 1 - np.sum(env.unwrapped.datapool)/ total_packets
    throughput.append(thp)

    
    # print("Episode : {} \t Throughput: {}".format(i_episode, round(thp,2)), flush = True)

print('Complete')
# print(np.mean(mean_packet_rate_episode))
# print(np.mean(sum_reward))
# print(np.mean(th_episode))

# plot_road_trraffic(env.unwrapped.road_matrix, env.unwrapped.N_UEs, position_ue_list)
# %%
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
tdma_thp_cdf = np.sort(throughput)
yvals = np.arange(len(tdma_thp_cdf))/float(len(tdma_thp_cdf)-1)
ax.plot(tdma_thp_cdf, yvals, '-')
ax.set_xlabel('Throughput')
ax.set_ylabel('CDF')
ax.legend(['Heuristic Env'])

#%%
#fig,ax = plt.subplots()
# #ax.plot(reward_done)
# ax.plot(reward_list)
# fig.show()


# er = np.concatenate(estimation_error).ravel().tolist()
# data_cdf = np.sort(er)
# yvals = np.arange(len(data_cdf))/float(len(data_cdf)-1)
# ax.plot(data_cdf, yvals, '-')

# %%

# np.savetxt('Multi-user-BM/results_bm/Thp_heuristic_crlb_est'+'_2UE_rician_v=30', throughput)
# np.savetxt('Multi-user-BM/results_bm/SR_heuristic_crlb_est'+'_2UE_rician_v=30', sensing_slots_ratio)



# np.savetxt('Multi-user-BM/results_bm/Er__rician' + 'Heuristic_crlb_aod_est', estimation_error)


# %%
