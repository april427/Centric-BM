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
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM_buffer_test/')
from beam_management_env_letter import BeamManagementEnv

from baseline_train import parse_args
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap

args = parse_args()
args.if_training=False
# args.if_debugging=True
env = gym.make(args.EnvID,  args=args)
# check_env(env)
num_episodes = 500
num_steps = 40
reward_done = []
reward_list = []
throughput = []
latency = []
error_range = np.zeros((num_episodes, args.N_UTs))
error_speed = np.zeros((num_episodes, args.N_UTs))
error_angle = np.zeros((num_episodes, args.N_UTs))
nmseR_allsteps = []
nmseV_allsteps = []
nmseA_allsteps = []

spectrum_efficiency = []
method_name = 'greedy'

for ep in range(1, num_episodes+1):

    obs,_ = env.reset()
    reward_episode = 0
    er = np.zeros((args.N_UTs, num_steps))
    ev = np.zeros((args.N_UTs, num_steps))
    ea = np.zeros((args.N_UTs, num_steps))

    genie_beams = []
    beams = []

    for t in range(num_steps):
        # print(action)
        
        _, rewards, dones, _, info = env.step(method_name)
        
        reward_episode += rewards
        er[:,t] = info['nmse_R']
        ev[:,t] = info['nmse_v']
        ea[:,t] = info['angle_error']
        
        spectrum_efficiency.append(info['SE'])

        genie_beams.append(info['genie_beams'])
        beams.append(info['allocated_beams'].copy())

        if dones:
            reward_done.append(reward_episode)
            break
        latency.append(info['latency'])

    error_range[ep-1,] = np.mean(er, axis=1)/(t+1)
    error_speed[ep-1,] = np.sum(ev, axis=1)/(t+1)
    error_angle[ep-1,] = np.sum(ea, axis=1)/(t+1)

    nmseR_allsteps.append(er)
    nmseV_allsteps.append(ev)
    nmseA_allsteps.append(ea)

    reward_list.append(reward_episode)       
    
    thp = info['transmitted_packets']/ info['total_traffic']
    # print(info['buffers'])
    throughput.append(thp)
    
    print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)

    print("Reward: ", reward_episode)


# %%
#################################  PLOTS  ##############################

thp_cdf = np.sort(throughput)
yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)

# rewrd of test
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
ax.plot(reward_list, '-.') 

# throughput CDF
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
ax.plot(thp_cdf, yvals, '-')
ax.set_xlabel('Throughput')
ax.set_ylabel('CDF')
plt.show()

# CDF of spectrum efficiency
spectrum_efficiency = np.array(spectrum_efficiency)
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
se_cdf = np.sort(spectrum_efficiency[:,0])
yvals = np.arange(len(se_cdf))/float(len(se_cdf)-1)
ax.plot(se_cdf, yvals, '-')

se_cdf = np.sort(spectrum_efficiency[:,1])
yvals = np.arange(len(se_cdf))/float(len(se_cdf)-1)
ax.plot(se_cdf, yvals, '-')
ax.set_xlabel('Spectrum Efficiency')
ax.set_ylabel('CDF')
plt.show()

# # Estimation of Range and Speed
# figure_handle = plt.figure()
# ax1 = figure_handle.add_subplot()
# for i in range(args.N_UTs):
#     ax1.plot(error_range[:,i], '-o')
# ax1.set_xlabel('Episodes')
# ax1.set_ylabel('Normalized Error of Range Estimation')
# plt.show()

# figure_handle = plt.figure()
# ax12 = figure_handle.add_subplot()
# for i in range(args.N_UTs):
#     ax12.plot(error_speed[:,i],'--x')
# ax12.set_xlabel('Episodes')
# ax12.set_ylabel('Normalized Error of Speed Estimation')
# plt.show()

figure_handle = plt.figure()
ax13 = figure_handle.add_subplot()
for i in range(args.N_UTs):
    ax13.plot(error_angle[:,i],'-*')
ax13.set_xlabel('Episodes')
ax13.set_ylabel('Error of Angle Estimation')
plt.show()

# Latency
latency_flat = list(itertools.chain(*latency))
latency_flat = [x for x in latency_flat if x != -1]
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

figure_handle = plt.figure()
ax5 = figure_handle.add_subplot()
nmseA_allsteps = np.array(nmseA_allsteps)
for u in range(args.N_UTs):
    temp = nmseA_allsteps[:,u,:].flatten()
    ea_cdf = np.sort(temp)
    yvals = np.arange(len(ea_cdf))/float(len(ea_cdf)-1)
    ax5.plot(ea_cdf, yvals, '-')
ax5.set_xlabel('Error of Angle Estimation')
ax5.set_ylabel('CDF')

# %%
num_steps_range = range(num_steps)
# Create a 2D array for heatmap
heatmap_data = np.zeros((args.Nt, num_steps))

for t in num_steps_range:
    #for u in range(args.N_UTs):
        
    heatmap_data[:, t] = np.array(beams[t])  # Increment the count for the allocated beam
    #heatmap_data[genie_beams[:][t], t]  = np.array([0.5, 1.5]) 

# Define discrete levels for the color bar
# levels = [0, 1, 2, 3, 4, 5]  # Adjust levels as needed

# Add a legend for the levels
levels = [0, 1,  2, 3]
cmap = plt.get_cmap("Paired", len(levels) - 1)  # Use a colormap with discrete intervals
norm = BoundaryNorm(levels, cmap.N, clip=True)
legend_labels = ['Idle', 'UE 0 beams', 'UE 1 beams']
custom_cmap = ListedColormap(cmap(np.linspace(0, 1, len(levels) - 1)))

# Plot heatmap with discrete color bar
figure_handle = plt.figure()
ax7 = figure_handle.add_subplot()
sns.heatmap(heatmap_data, cmap=cmap, norm=norm, ax=ax7, cbar = False)
ax7.set_xlabel('Time Index')
ax7.set_ylabel('Beam Steps')

# Scatter plot for genie beams
for t in num_steps_range:
    for u in range(args.N_UTs):
        if t == 0:
            ax7.scatter(t + 0.5, genie_beams[t][u] + 0.5, color='white' if u == 0 else 'blue', s=20, label='UE' + str(u))
        else:
            ax7.scatter(t + 0.5, genie_beams[t][u] + 0.5, color='white' if u == 0 else 'blue', s=20)
ax7.legend(loc='upper right', fontsize=8, markerscale=1.5)

# Create a color bar with custom labels
colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax7, ticks=[(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)])
colorbar.ax.set_yticklabels(legend_labels)
plt.show()

# %%
#################################  SAVE FILES  ##############################

# log_dir = "RL_mimo_ofdm/"  + "PPO_"+ str(args.N_UTs) +"ue" + str(args.Nt)+ "Nt_buffer/NEW/" + method_name
# # Create directory if it does not exist
# os.makedirs(log_dir, exist_ok=True)

# # Save throughput
# throughput_df = pd.DataFrame(throughput, columns=['Throughput'])
# throughput_df.to_csv(os.path.join(log_dir, 'throughput.csv'), index=False)

# # Save nmseR_allsteps
# temp = np.zeros((num_episodes * num_steps,args.N_UTs))
# nmseR_allsteps = np.array(nmseR_allsteps)
# temp[:,0] = nmseR_allsteps[:,0,:].flatten()
# temp[:,1] = nmseR_allsteps[:,1,:].flatten()
# nmseR_allsteps_df = pd.DataFrame(temp, columns=['UE1', 'UE2'])
# nmseR_allsteps_df.to_csv(os.path.join(log_dir, 'nmseR_allsteps.csv'), index=False)

# # Save nmseV_allsteps
# nmseV_allsteps = np.array(nmseV_allsteps)
# temp[:,0] = nmseV_allsteps[:,0,:].flatten()
# temp[:,1] = nmseV_allsteps[:,1,:].flatten()
# nmseV_allsteps_df = pd.DataFrame(temp, columns=['UE1', 'UE2'])
# nmseV_allsteps_df.to_csv(os.path.join(log_dir, 'nmseV_allsteps.csv'), index=False)

# # Save nmseA_allsteps
# nmseA_allsteps = np.array(nmseA_allsteps)
# temp[:,0] = nmseA_allsteps[:,0,:].flatten()
# temp[:,1] = nmseA_allsteps[:,1,:].flatten()
# nmseA_allsteps_df = pd.DataFrame(temp, columns=['UE1', 'UE2'])
# nmseA_allsteps_df.to_csv(os.path.join(log_dir, 'nmseA_allsteps.csv'), index=False)


# # Save latency
# latency_flat = list(itertools.chain(*latency))
# latency_df = pd.DataFrame(latency_flat, columns=['Latency'])
# latency_df.to_csv(os.path.join(log_dir, 'latency.csv'), index=False)

# # save Reward
# reward = pd.DataFrame(reward_list, columns=['Reward'])
# reward.to_csv(os.path.join(log_dir, 'reward.csv'), index=False)

# # save spectral efficiency
# se = pd.DataFrame(spectrum_efficiency, columns=['SE1', 'SE2'])
# se.to_csv(os.path.join(log_dir, 'SE.csv'), index=False)


# %%
