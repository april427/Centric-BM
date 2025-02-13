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

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Multi-user-BM/')
from beam_management_env_letter import BeamManagementEnv

model_type = "PPO"
model_class = getattr(sys.modules[__name__], model_type)

rinx = 1
env = gym.make("BeamManagementEnv-v1", reward_inx=rinx, if_training=False)
#check_env(env)

log_dir = "baseline_result/" + model_type + "_"+ str(env.unwrapped.N_UEs) +"ue_" + str(rinx) + "reward_inx/"
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

  num_episodes = 1000
  num_steps = 100
  reward_done = []
  reward_list = []
  throughput = []
  estimation_error = []

  for ep in range(1, num_episodes+1):

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

plt.show()

# %%
# np.savetxt('Multi-user-BM/results_bm/Thp_' + 'PPO_rician'+ "_"+ str(env.unwrapped.N_UEs) +"ue_" + str(rinx) + "reward_inx_v=20", throughput)

# np.savetxt('Multi-user-BM/results_bm/SR_' + 'PPO_rician' + "_"+ str(env.unwrapped.N_UEs) +"ue_" + str(rinx) + "reward_inx_v=20", sensing_slots_ratio)

# model.save(log_dir + "BeamManagementEnv-v1_1")
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_dir = "baseline_result/" + 'PPO' + "_2" +"ue_" + "1reward_inx_allinits/"

np.savetxt('log_dir/genie_aid', reward_list)
monitor_file = log_dir + "monitor.csv"
monitor_data = pd.read_csv(monitor_file, skiprows=1)

print(monitor_data.head())

# Plotting the 'r' column from the monitor.csv file
plt.figure()
plt.plot(monitor_data['r'])
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.title('Reward over Time')
plt.show()
# %%
