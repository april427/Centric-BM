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

import sys, os
# sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from beam_management_env_letter import BeamManagementEnv
import pandas as pd
from baseline_train import parse_args
import seaborn as sns
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap

if __name__ == "__main__":
    args = parse_args()
    model_type = args.model
    model_class = getattr(sys.modules[__name__], model_type)
    args.N_UTs = 2
    args.cdir = 'sAs_newB_waitT_newSense_128_lrscheduling'
    args.if_training=False
    args.if_debugging = True
    args.dt = 0.2

    log_dir = log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(args.N_UTs) +"ue" + str(args.Nt)+ "Nt_" + args.cdir +"/" 
    monitor_file = log_dir + "monitor.csv"
    # monitor_data = pd.read_csv(monitor_file, skiprows=0)

    # window_size = 1000
    # moving_average = monitor_data['r'].rolling(window=window_size).mean()

    # fig, ax = plt.subplots()
    # ax.plot(moving_average, label='Moving Average', color='orange')
    # ax.legend()
    # ax.grid()
    
    # plt.show()
    
    env = gym.make(args.EnvID,  args = args)
    print("N_UTs=", args.N_UTs)
    print(log_dir)

    # %%
    ################################## Evaluate ####################################
    
    model = model_class.load(log_dir + "best_model", env=env)

    num_episodes = 1
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

    for ep in range(1, num_episodes+1):
        obs,_ = env.reset()
        reward_episode = 0
        er = np.zeros((args.N_UTs, num_steps))
        ev = np.zeros((args.N_UTs, num_steps))
        ea = np.zeros((args.N_UTs, num_steps))

        ut_loc = np.zeros((num_steps+1, args.N_UTs, 3))
        ut_loc[0,] = env.unwrapped.ut_loc[0,:,:]

        genie_beams = []
        beams = []

        for t in range(num_steps):
            action, _states = model.predict(obs)
            
            obs, rewards, dones, _, info = env.step(action=action)
            
            reward_episode += rewards
            er[:,t] = info['nmse_R']
            ev[:,t] = info['nmse_v']
            ea[:,t] = info['angle_error']

            latency.append(info['latency'])
            spectrum_efficiency.append(info['SE'])

            genie_beams.append(info['genie_beams'])
            beams.append(info['allocated_beams'].copy())

            ut_loc[t+1,] = env.unwrapped.ut_loc[0,:,:]

            if dones:
                reward_done.append(reward_episode)
                break

        error_range[ep-1,] = np.sum(er, axis=1)/(t+1)
        error_speed[ep-1,] = np.sum(ev, axis=1)/(t+1)
        error_angle[ep-1,] = np.sum(ea, axis=1)/(t+1)

        nmseR_allsteps.append(er)
        nmseV_allsteps.append(ev)
        nmseA_allsteps.append(ea)

        if t == num_steps - 1:  
            reward_list.append(reward_episode)       
        print(reward_episode)
        thp = np.sum(info['transmitted_packets'])/ info['total_traffic']
        print(info['transmitted_packets'])
        throughput.append(thp)
        
        print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)


    thp_cdf = np.sort(throughput)
    yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)

# %%
################################## PLOTS ####################################
# Reward
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
ax.plot(reward_list, '-.') 

# throughput CDF 
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
ax.plot(thp_cdf, yvals, '-')
ax.set_xlabel('Throughput')
ax.set_ylabel('CDF')

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

# # Estimation of episodes
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

# Latency
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
plt.show()

# %%
# Plotting the trajectory of UEs
figure_handle = plt.figure()
bs_loc = env.unwrapped.bs_loc[0,:,:]
ax6 = figure_handle.add_subplot()
ax6.scatter(bs_loc[:,0], bs_loc[:,1], marker="o", color='red', label='BS')
for i in range(args.N_UTs):
    ue_pos_x = ut_loc[:,i,0]
    ue_pos_y = ut_loc[:,i,1]
    du = np.diff(ue_pos_x)
    dv = np.diff(ue_pos_y)
    pos_x = ue_pos_x[:-1] + du/2
    pos_y = ue_pos_y[:-1] + dv/2
    norm = np.sqrt(du**2+dv**2) 
    ax6.plot(ue_pos_x,ue_pos_y, marker="o")
    ax6.quiver(pos_x, pos_y, du/norm, dv/norm, angles="xy", zorder=5, pivot="mid") 
    # hold the plot   
    # ax6.plot(ut_loc[:,i,0], ut_loc[:,i,1], ut_loc[:,i,2], label='UE' + str(i))
ax6.set_xlabel('X-axis')
ax6.set_ylabel('Y-axis')
ax6.set_title('Trajectory of UEs')
ax6.legend()
plt.show()


# %%                  
###           Plot Allocated Beams and Genie Beams varying with num_steps
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
            ax7.scatter(t + 0.5, genie_beams[t][u] + 0.5, color='darkorange' if u == 0 else 'blue', s=20, label='UE' + str(u))
        else:
            ax7.scatter(t + 0.5, genie_beams[t][u] + 0.5, color='darkorange' if u == 0 else 'blue', s=20)
ax7.legend(loc='upper right', fontsize=8, markerscale=1.5)

# Create a color bar with custom labels
colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax7, ticks=[(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)])
colorbar.ax.set_yticklabels(legend_labels)
# Set grid lines in between two integers
ax7.set_xticks(np.arange(-0.5, num_steps, 1), minor=True)
ax7.set_yticks(np.arange(-0.5, args.Nt, 1), minor=True)
ax7.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)
ax7.tick_params(which='minor', bottom=False, left=False)
ax7.grid(True, axis='both', linestyle='--', linewidth=0.5, which='major')
plt.show()


# # Dynamically set levels, colors, and labels based on number of users
# levels = [0] + [i + 1 for i in range(args.N_UTs)]  # 0: Idle, 1: UE0, 2: UE1, ...
# cmap = plt.get_cmap("Paired", len(levels) - 1)
# norm = BoundaryNorm(levels, cmap.N, clip=True)
# legend_labels = ['Idle'] + [f'UE {i} beams' for i in range(args.N_UTs)]
# custom_cmap = ListedColormap(cmap(np.linspace(0, 1, len(levels) - 1)))

# figure_handle = plt.figure()
# ax7 = figure_handle.add_subplot()
# sns.heatmap(heatmap_data, cmap=cmap, norm=norm, ax=ax7, cbar=False)
# ax7.set_xlabel('Time Index')
# ax7.set_ylabel('Beam Steps')
# ax7.grid(True, axis='both', linestyle='--', linewidth=0.5, which='major')

# # Scatter plot for genie beams, with unique color per user
# colors = plt.cm.get_cmap('tab10', args.N_UTs)
# for t in num_steps_range:
#     for u in range(args.N_UTs):
#         if t == 0:
#             ax7.scatter(t + 0.5, genie_beams[t][u] + 0.5, color=colors(u), s=20, label=f'UE{u}')
#         else:
#             ax7.scatter(t + 0.5, genie_beams[t][u] + 0.5, color=colors(u), s=20)
# handles, labels = ax7.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax7.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, markerscale=1.5)

# colorbar = plt.colorbar(
#     plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
#     ax=ax7,
#     ticks=[(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]
# )
# colorbar.ax.set_yticklabels(legend_labels)
# plt.show()

# %%                                        
#####################            SAVE DATA 
##################### estimated, # trained, # genie, # random
# log_dir_data =log_dir + "/genie/"
# # # Create directory if it does not exist
# os.makedirs(log_dir_data, exist_ok=True)

# # # Save throughput
# throughput_df = pd.DataFrame(throughput, columns=['Throughput'])
# throughput_df.to_csv(os.path.join(log_dir_data, 'throughput.csv'), index=False)

# # Save nmseR_allsteps
# temp = np.zeros((num_episodes * num_steps,args.N_UTs))
# nmseR_allsteps = np.array(nmseR_allsteps)
# temp[:,0] = nmseR_allsteps[:,0,:].flatten()
# temp[:,1] = nmseR_allsteps[:,1,:].flatten()
# nmseR_allsteps_df = pd.DataFrame(temp, columns=['UE1', 'UE2'])
# nmseR_allsteps_df.to_csv(os.path.join(log_dir_data, 'nmseR_allsteps.csv'), index=False)

# # Save nmseV_allsteps
# nmseV_allsteps = np.array(nmseV_allsteps)
# temp[:,0] = nmseV_allsteps[:,0,:].flatten()
# temp[:,1] = nmseV_allsteps[:,1,:].flatten()
# nmseV_allsteps_df = pd.DataFrame(temp, columns=['UE1', 'UE2'])
# nmseV_allsteps_df.to_csv(os.path.join(log_dir_data, 'nmseV_allsteps.csv'), index=False)

# #Save nmseA_allsteps
# nmseA_allsteps = np.array(nmseA_allsteps)
# temp[:,0] = nmseA_allsteps[:,0,:].flatten()
# temp[:,1] = nmseA_allsteps[:,1,:].flatten()
# nmseA_allsteps_df = pd.DataFrame(temp, columns=['UE1', 'UE2'])
# nmseA_allsteps_df.to_csv(os.path.join(log_dir_data, 'nmseA_allsteps.csv'), index=False)

# # Save latency
# latency_flat = list(itertools.chain(*latency))
# latency_df = pd.DataFrame(latency_flat, columns=['Latency'])
# latency_df.to_csv(os.path.join(log_dir_data, 'latency.csv'), index=False)

# reward = pd.DataFrame(reward_list, columns=['Reward'])
# reward.to_csv(os.path.join(log_dir_data, 'reward.csv'), index=False)

# # save spectral efficiency
# se = pd.DataFrame(spectrum_efficiency, columns=['SE1', 'SE2'])
# se.to_csv(os.path.join(log_dir_data, 'SE.csv'), index=False)

