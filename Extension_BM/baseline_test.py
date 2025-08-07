#%%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO, DDPG, A2C
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time

import itertools

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from beam_management_env_letter import BeamManagementEnv
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from parse_args import parse_args
from channel_functions import base, N_TDMA

if __name__ == "__main__":
    args = parse_args()
    model_type = args.model
    model_class = getattr(sys.modules[__name__], model_type)
    # args.N_UTs = 2
    args.if_training=False
    # args.if_debugging = True
    method = args.eval_method #'heuristic'#

    log_dir =  "RL_mimo_ofdm/" + model_type + "_"+ str(args.N_UTs) +"ue" + str(args.Nt)+ "Nt_" + args.cdir +"/" 
    # monitor_file = log_dir + "monitor_0.csv"
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

    #### N-TDMA method ###
    if method.endswith('TDMA'):
        # Generate the N-TDMA actions
        nn = int(method.split('-')[0])
        list_actions = N_TDMA(nn, args.N_UTs)
        for i in range(len(list_actions)):
            list_actions[i] = base(2, list_actions[i], args.N_UTs)
        list_actions = (np.array(list_actions) * 8 + 1).tolist()

    # %%
    ################################## Evaluate ####################################
    
    model = model_class.load(log_dir + "best_model", env=env)

    num_episodes = args.num_eval_episodes
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
        l_a = 0
        buffer_info = []

        for t in range(num_steps):
            if method == 'trained':
                action, _states = model.predict(obs)
                obs, rewards, dones, _, info = env.step(action=action)
            elif method.endswith('TDMA'):
                action = list_actions[l_a]
                obs, rewards, dones, _, info = env.step(action=action)
                l_a += 1
                if l_a >= len(list_actions):
                    l_a = 0
            elif method == 'heuristic':
                if t==0:
                    action = np.ones((env.unwrapped.N_UTs))
                else:
                    action =  (info['angle_error'] > (env.unwrapped.beamwidth/2/(np.pi/env.unwrapped.number_of_antennas))) * 9 + 1
                obs, rewards, dones, _, info = env.step(action=action)  
                
            else:
                obs, rewards, dones, _, info = env.step(action=method)
            
            reward_episode += rewards
            er[:,t] = info['nmse_R']
            ev[:,t] = info['nmse_v']
            ea[:,t] = info['angle_error']

            latency.append(info['latency'])
            spectrum_efficiency.append(info['SE'])

            genie_beams.append(info['genie_beams'])
            beams.append(info['allocated_beams'].copy())

            ut_loc[t+1,] = env.unwrapped.ut_loc[0,:,:]

            buffer_info.append(info['buffers'])

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



# %%
################################## PLOTS ####################################
    if args.if_plot:
        from matplotlib.colors import ListedColormap
        my_colors = ['#B2CCF3', '#FF983B', '#AF5124', '#83CA27']
        my_cmap = ListedColormap(my_colors)
        if args.N_UTs == 2:
            colors = ["#DB6C04", "#089D03"]
            markers = ['o', '^']
        elif args.N_UTs == 3:
            colors = ["#DB6C04","#600202EF", "#089D03"]
            markers = ['o', '*', '^']

        # colors = ['orange', 'red', 'green']
        labels = ['UE1', 'UE2', 'UE3']
        if args.num_eval_episodes > 1:
            # Reward
            figure_handle = plt.figure()
            ax = figure_handle.add_subplot()
            ax.plot(reward_list, '-.') 

            # throughput CDF 
            thp_cdf = np.sort(throughput)
            yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)
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
            ax.plot(se_cdf, yvals, '-', color = 'red', label='UE1')

            se_cdf = np.sort(spectrum_efficiency[:,1])
            yvals = np.arange(len(se_cdf))/float(len(se_cdf)-1)
            ax.plot(se_cdf, yvals, '-', color = 'grey', label='UE2')
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

            ax3.plot(er_cdf, yvals, '-', color=colors[u], label=labels[u])

        ax3.set_xlabel('Normalized Error of Range Estimation')
        ax3.set_ylabel('CDF')
        plt.show()


        figure_handle = plt.figure()
        ax4 = figure_handle.add_subplot()
        for u in range(args.N_UTs):
            temp = nmseV_allsteps[:,u,:].flatten()
            ev_cdf = np.sort(temp)
            yvals = np.arange(len(ev_cdf))/float(len(ev_cdf)-1)
            ax4.plot(ev_cdf, yvals, '-', color=colors[u], label=labels[u])
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
            ax5.plot(ea_cdf, yvals, '-', color=colors[u], label=labels[u])
        ax5.set_xlabel('Error of Angle Estimation')
        ax5.set_ylabel('CDF')
        plt.show()

        # %%
        # Plot four figures in a single figure with four subplots for easier viewing
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # 1. NMSE of Range Estimation per User
        nmseR_allsteps = np.array(nmseR_allsteps)
        for i in range(args.N_UTs):
            axs[0, 0].plot(nmseR_allsteps[0, i, :], color=colors[i], label=labels[i])
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Normalized Error of Range Estimation')
        axs[0, 0].set_title('NMSE of Range Estimation per User')
        axs[0, 0].legend()

        # 2. NMSE of Speed Estimation per User
        nmseV_allsteps = np.array(nmseV_allsteps)
        for i in range(args.N_UTs):
            axs[0, 1].plot(nmseV_allsteps[0, i, :], color=colors[i], label=labels[i])
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Normalized Error of Speed Estimation')
        axs[0, 1].set_title('NMSE of Speed Estimation per User')
        axs[0, 1].legend()

        # 3. Trajectory of UEs
        bs_loc = env.unwrapped.bs_loc[0,:,:]
        axs[1, 0].scatter(bs_loc[:,0], bs_loc[:,1], marker="^", color='green', label='BS')
        
        # Clutter location
        faic_list = [np.pi/15*1, np.pi/14*3, np.pi/17*9, np.pi/17*15]  # Example: two clutters with different angles
        Rdc_list = [13, 20, 29, 19]  
        
        for faic, Rdc in zip(faic_list, Rdc_list):
            clutter_x = bs_loc[0,0] + Rdc * np.cos(faic)
            clutter_y = bs_loc[0,1] + Rdc * np.sin(faic)
            axs[1, 0].scatter(clutter_x, clutter_y, marker="x", color='black', label='Clutter')
        # Avoid duplicate legend entries for clutters
        handles, labels_ = axs[1, 0].get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        axs[1, 0].legend(by_label.values(), by_label.keys())

        for i in range(args.N_UTs):
            ue_pos_x = ut_loc[:,i,0]
            ue_pos_y = ut_loc[:,i,1]
            du = np.diff(ue_pos_x)
            dv = np.diff(ue_pos_y)
            pos_x = ue_pos_x[:-1] + du/2
            pos_y = ue_pos_y[:-1] + dv/2
            norm = np.sqrt(du**2+dv**2) 
            axs[1, 0].plot(ue_pos_x, ue_pos_y, marker="o", markersize=5, color=colors[i], label=f'UE {i+1}', zorder=3)
            axs[1, 0].quiver(pos_x, pos_y, du/norm, dv/norm, angles="xy", zorder=1, pivot="mid")
        axs[1, 0].set_xlabel('X-axis')
        axs[1, 0].set_ylabel('Y-axis')
        axs[1, 0].set_title('Trajectory of UEs')
        axs[1, 0].legend()

        # 4. Allocated Beams and Genie Beams heatmap
        num_steps_range = range(num_steps)
        heatmap_data = np.zeros((args.Nt, num_steps))
        for t in num_steps_range:
            heatmap_data[:, t] = np.array(beams[t])
        levels = [0] + [i + 1 for i in range(args.N_UTs+1)]
        # cmap = plt.get_cmap("Paired", len(levels) - 1)
        norm = BoundaryNorm(levels, my_cmap.N, clip=True)
        legend_labels = ['Idle'] + [f'UE {i+1} beams' for i in range(args.N_UTs)]
        # custom_cmap = ListedColormap(cmap(np.linspace(0, 1, len(levels) - 1)))
        sns.heatmap(heatmap_data, cmap=my_cmap, norm=norm, ax=axs[1, 1], cbar=False)
        axs[1, 1].set_xlabel('Time Index')
        axs[1, 1].set_ylabel('Beam Steps')
        axs[1, 1].set_title('Allocated Beams and Genie Beams')
        axs[1, 1].grid(True, axis='both', linestyle='--', linewidth=0.5, which='major')
        # colors = plt.cm.get_cmap('Set1', args.N_UTs)
        for t in num_steps_range:
            for u in range(args.N_UTs):
                ttemp = buffer_info[t][u]
                buffer_val = np.sum(ttemp[ttemp>=0], dtype = np.int32)  # buffer_info: shape [num_steps, N_UTs]
                if t == 0:
                    axs[1, 1].scatter(t + 0.5, genie_beams[t][u] + 0.5, color=colors[u], marker=markers[u], s=20, label=f'UE{u+1}')
                else:
                    axs[1, 1].scatter(t + 0.5, genie_beams[t][u] + 0.5, color=colors[u], marker=markers[u], s=20)
                    # axs[1, 1].scatter(t + 0.5, genie_beams[t][u] + 0.5, 
                            # color=colors[u], marker=markers[u], s=20 + buffer_val*10, label=f'UE{u+1}')
                axs[1, 1].text(t + 0.5, genie_beams[t][u] +0.5, f'{int(buffer_val)}', 
                        color='black', fontsize=8, ha='center', va='center')
        
        
        buffer_sizes = [0, 1, 2,3,4]  # Adjust to your buffer scale
        legend_handles = [
            Line2D([0], [0], marker=markers[0], color='w', label='Low Buffer', markerfacecolor=colors[0], markersize=np.sqrt(20 + buffer_sizes[0]*10)),
            Line2D([0], [0], marker=markers[0], color='w', label='Medium Buffer', markerfacecolor=colors[0], markersize=np.sqrt(20 + buffer_sizes[1]*10)),
            Line2D([0], [0], marker=markers[0], color='w', label='High Buffer', markerfacecolor=colors[0], markersize=np.sqrt(20 + buffer_sizes[2]*10)),
        ]

        # Add to your legend
        handles, labels_ = axs[1, 1].get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        all_handles = list(by_label.values()) + legend_handles
        all_labels = list(by_label.keys()) + ['Low Buffer', 'Medium Buffer', 'High Buffer']

        axs[1, 1].legend(all_handles, all_labels, loc='upper right', fontsize=8, markerscale=1.5)
        
        handles, labels_ = axs[1, 1].get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        axs[1, 1].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, markerscale=1.5)
        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=my_cmap),
            ax=axs[1, 1],
            ticks=[(levels[i] + levels[i + 1]) / 2 for i in range(len(levels)-1 )]
        )
        colorbar.ax.set_yticklabels(legend_labels)


        plt.tight_layout()
        plt.show()


        ############################ Single figure for heatmap ########################
        num_steps_range = range(num_steps)
        # Create a 2D array for heatmap
        heatmap_data = np.zeros((args.Nt, num_steps))

        for t in num_steps_range:
            #for u in range(args.N_UTs):
                
            heatmap_data[:, t] = np.array(beams[t])  # Increment the count for the allocated beam
            #heatmap_data[genie_beams[:][t], t]  = np.array([0.5, 1.5]) 


        # # Dynamically set levels, colors, and labels based on number of users
        levels = [0] + [i + 1 for i in range(args.N_UTs+1)]  # 0: Idle, 1: UE0, 2: UE1, ...
        # cmap = plt.get_cmap("Paired", len(levels) - 1)
        norm = BoundaryNorm(levels, my_cmap.N, clip=True)
        legend_labels = ['Idle'] + [f'UE {i+1} beams' for i in range(args.N_UTs)]
        # custom_cmap = ListedColormap(cmap(np.linspace(0, 1, len(levels) - 1)))

        figure_handle = plt.figure()
        ax7 = figure_handle.add_subplot()
        sns.heatmap(heatmap_data, cmap=my_cmap, norm=norm, ax=ax7, cbar=False)
        ax7.set_xlabel('Time Index')
        ax7.set_ylabel('Beam Steps')
        ax7.grid(True, axis='both', linestyle='--', linewidth=0.5, which='major')

        # Scatter plot for genie beams, with unique color per user
        # colors = plt.cm.get_cmap('Set1', args.N_UTs)
        
        import matplotlib.colors as mcolors
        
        def desaturate_color(color, factor):
            """Blend the color with white by the given factor (0=no change, 1=white)."""
            rgb = np.array(mcolors.to_rgb(color))
            white = np.array([1, 1, 1])
            return tuple(rgb + (white - rgb) * factor)
        
        for t in num_steps_range:
            for u in range(args.N_UTs):
                if t == 0:
                    buffer_val = 0
                else:
                    ttemp = buffer_info[t][u]
                    buffer_val = np.sum(ttemp[ttemp>=0], dtype = np.int32)  # buffer_info: shape [num_steps, N_UTs]
                # Desaturate color based on buffer_val (cap factor between 0 and 0.8 for visibility)
                desat_factor = min(buffer_val / 12.0, 0.8)
                scatter_color = desaturate_color(colors[u], desat_factor)
                ax7.scatter(t + 0.5, genie_beams[t][u] + 0.5, marker=markers[u], color=scatter_color, s=20 + buffer_val*15, label=f'UE{u+1}' if t == 0 else None)
                ax7.text(t + 0.5, genie_beams[t][u] +0.5, f'{int(buffer_val)}', 
                        color='black', fontsize=8, ha='center', va='center')
        handles, labels = ax7.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax7.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, markerscale=1.5)

        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=my_cmap),
            ax=ax7,
            ticks=[(levels[i] + levels[i + 1]) / 2 for i in range(len(levels)-1 )]
        )
        colorbar.ax.set_yticklabels(legend_labels)
        plt.show()

# %%                                        
################################          SAVE DATA 
# ##################### estimated, # trained, # genie, # random
    if args.if_log:
        log_dir_data =log_dir + "/" + method +"/"
        # # Create directory if it does not exist
        os.makedirs(log_dir_data, exist_ok=True)

        # # Save throughput
        throughput_df = pd.DataFrame(throughput, columns=['Throughput'])
        throughput_df.to_csv(os.path.join(log_dir_data, 'throughput.csv'), index=False)

        # Dynamically set column names according to N_UTs
        ue_columns = [f'UE{i+1}' for i in range(args.N_UTs)]
        se_columns = [f'SE{i+1}' for i in range(args.N_UTs)]

        # Save nmseR_allsteps
        temp = np.zeros((num_episodes * num_steps, args.N_UTs))
        nmseR_allsteps = np.array(nmseR_allsteps)
        for i in range(args.N_UTs):
            temp[:, i] = nmseR_allsteps[:, i, :].flatten()
        nmseR_allsteps_df = pd.DataFrame(temp, columns=ue_columns)
        nmseR_allsteps_df.to_csv(os.path.join(log_dir_data, 'nmseR_allsteps.csv'), index=False)

        # Save nmseV_allsteps
        nmseV_allsteps = np.array(nmseV_allsteps)
        for i in range(args.N_UTs):
            temp[:, i] = nmseV_allsteps[:, i, :].flatten()
        nmseV_allsteps_df = pd.DataFrame(temp, columns=ue_columns)
        nmseV_allsteps_df.to_csv(os.path.join(log_dir_data, 'nmseV_allsteps.csv'), index=False)

        # Save nmseA_allsteps
        nmseA_allsteps = np.array(nmseA_allsteps)
        for i in range(args.N_UTs):
            temp[:, i] = nmseA_allsteps[:, i, :].flatten()
        nmseA_allsteps_df = pd.DataFrame(temp, columns=ue_columns)
        nmseA_allsteps_df.to_csv(os.path.join(log_dir_data, 'nmseA_allsteps.csv'), index=False)

        # Save latency
        latency_flat = list(itertools.chain(*latency))
        latency_df = pd.DataFrame(latency_flat, columns=['Latency'])
        latency_df.to_csv(os.path.join(log_dir_data, 'latency.csv'), index=False)

        reward = pd.DataFrame(reward_list, columns=['Reward'])
        reward.to_csv(os.path.join(log_dir_data, 'reward.csv'), index=False)

        # Save spectral efficiency
        se = pd.DataFrame(spectrum_efficiency, columns=se_columns)
        se.to_csv(os.path.join(log_dir_data, 'SE.csv'), index=False)


# %%   

# if num_episodes == 1:
#     figure_handle = plt.figure()
#     ax = figure_handle.add_subplot()
#     nmseR_allsteps = np.array(nmseR_allsteps)
#     for i in range(args.N_UTs):
#         ax.plot(nmseR_allsteps[0, i, :], color=color[i], label=labels[i])
#     ax.set_xlabel('Time Step')
#     ax.set_ylabel('Normalized Error of Range Estimation')
#     ax.set_title('NMSE of Range Estimation per User')
#     ax.legend()
#     plt.show()

#     figure_handle = plt.figure()
#     ax = figure_handle.add_subplot()
#     nmseV_allsteps = np.array(nmseV_allsteps)
#     for i in range(args.N_UTs):
#         ax.plot(nmseV_allsteps[0, i, :], color=color[i], label=labels[i])
#     ax.set_xlabel('Time Step')
#     ax.set_ylabel('Normalized Error of Speed Estimation')
#     ax.set_title('NMSE of Speed Estimation per User')
#     ax.legend()
#     plt.show()

# # Plotting the trajectory of UEs
# figure_handle = plt.figure()
# bs_loc = env.unwrapped.bs_loc[0,:,:]
# ax6 = figure_handle.add_subplot()
# ax6.scatter(bs_loc[:,0], bs_loc[:,1], marker="^", color='green', label='BS')
# markcolor = ['red' , 'grey']
# for i in range(args.N_UTs):
#     ue_pos_x = ut_loc[:,i,0]
#     ue_pos_y = ut_loc[:,i,1]
#     du = np.diff(ue_pos_x)
#     dv = np.diff(ue_pos_y)
#     pos_x = ue_pos_x[:-1] + du/2
#     pos_y = ue_pos_y[:-1] + dv/2
#     norm = np.sqrt(du**2+dv**2) 
#     ax6.plot(ue_pos_x,ue_pos_y, marker="o", markersize = 5, color = markcolor[i], label = f'UE {i}', zorder=3)
#     ax6.quiver(pos_x, pos_y, du/norm, dv/norm, angles="xy", zorder=1, pivot="mid") 
#     # hold the plot   
#     # ax6.plot(ut_loc[:,i,0], ut_loc[:,i,1], ut_loc[:,i,2], label='UE' + str(i))
# ax6.set_xlabel('X-axis')
# ax6.set_ylabel('Y-axis')
# ax6.set_title('Trajectory of UEs')
# ax6.legend()
# plt.show()


# %%
