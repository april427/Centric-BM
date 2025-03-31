import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO, DQN
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import seaborn as sns
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Multi-user-BM/')
from beam_management_env_letter import BeamManagementEnv

model_type = "PPO"
model_class = getattr(sys.modules[__name__], model_type)

rinx = 1
env = gym.make("BeamManagementEnv-v1", reward_inx=rinx, if_training=False)
#check_env(env)

#log_dir = "baseline_result/" + model_type + "_"+ str(env.unwrapped.N_UEs) +"ue_" + str(rinx) + "reward_inx/"
log_dir = "baseline_result/Model_EUCNC_PPO_2ue_1reward_inx/"
print("N_UEs=", env.unwrapped.N_UEs, " reward_inx=", rinx)

sensing_slots_ratio = []
#env = Monitor(env, log_dir)
# %%
################################## Evaluate ####################################
figure_handle = plt.figure()
ax = figure_handle.add_subplot()

for ri in range(5,6):
#   log_dir = "baseline_result/" + model_type + "_reward_inx_" + str(ri) + "/"
  model = PPO.load(log_dir + "BeamManagementEnv")
  #model = PPO.load("PPO_BeamManagementEnv-v1")

  num_episodes = 1
  num_steps = 100
  reward_done = []
  reward_list = []
  throughput = []
  estimation_error = []

  for ep in range(1, num_episodes+1):

      genie_beams = []
      beams = []
        
      state = env.reset()
      current_ep_reward = 0
      obs,_ = env.reset()
      reward_episode = 0
      noted = []

      for t in range(num_steps):
          action, _states = model.predict(obs)
          # print(action)
          obs, rewards, dones, _, info = env.step(action)
          reward_episode += rewards
          
          genie_beams.append(info['genie_beams'])
          beams.append(info['allocated_beams'].copy())


          for i in range(env.unwrapped.N_UEs):
                if info['buffer'][i] == 0 and i not in noted:
                    noted.append(i)
                    sensing_slots_ratio.append(info['num_sensing_slots'][i]/(t+1))

          if dones:
              reward_done.append(reward_episode)
              print("all transmitted")
              break
          
          estimation_error.append(info['est_error'])

      
      if t == num_steps - 1:  
          reward_list.append(reward_episode)       
      
      thp = 1 - np.sum(info['buffer'])/ info['total_traffic']
      throughput.append(thp)
      
      #sensing_slots_ratio.append(info['num_sensing_slots']/(t+1))

      print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)


  thp_cdf = np.sort(throughput)
  yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)
  ax.plot(thp_cdf, yvals, 'o')
  ax.set_xlabel('Throughput')
  ax.set_ylabel('CDF')

# plt.show()

# %%
# np.savetxt('Multi-user-BM/results_bm/Thp_' + 'PPO_rician'+ "_"+ str(env.unwrapped.N_UEs) +"ue_" + str(rinx) + "reward_inx_v=20", throughput)

# np.savetxt('Multi-user-BM/results_bm/SR_' + 'PPO_rician' + "_"+ str(env.unwrapped.N_UEs) +"ue_" + str(rinx) + "reward_inx_v=20", sensing_slots_ratio)

# model.save(log_dir + "BeamManagementEnv-v1_1")
# %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# log_dir = "baseline_result/" + 'PPO' + "_2" +"ue_" + "1reward_inx_allinits/"

# np.savetxt('log_dir/genie_aid', reward_list)
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
 # Plot allocated beams and genie beams varying with num_steps
num_steps_range = range(t)
# Create a 2D array for heatmap
heatmap_data = np.zeros((env.unwrapped.number_of_antennas, t))

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
    for u in range(env.unwrapped.N_UEs):
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
