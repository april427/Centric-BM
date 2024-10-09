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

rinx = 0
env = gym.make("BeamManagementEnv-v1", reward_inx=rinx, if_training=False)
#check_env(env)

log_dir = "baseline_result/" + model_type + "_"+ str(env.unwrapped.N_UEs) +"ue_" + str(rinx) + "reward_inx/"

#env = Monitor(env, log_dir)
# %%
################################## Evaluate ####################################
figure_handle = plt.figure()
ax = figure_handle.add_subplot()

for ri in range(5,6):
#   log_dir = "baseline_result/" + model_type + "_reward_inx_" + str(ri) + "/"
  model = PPO.load(log_dir + "BeamManagementEnv")
# model = PPO.load("PPO_BeamManagementEnv-v1_1")

  num_episodes = 100
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

      for t in range(num_steps):
          action, _states = model.predict(obs)
          obs, rewards, dones, _, info = env.step(action)
          reward_episode += rewards

          if dones:
              reward_done.append(reward_episode)
              print("all transmitted")
              break
          
          estimation_error.append(info['est_error'])
      
      if t == num_steps - 1:  
          reward_list.append(reward_episode)       
      
      thp = 1 - np.sum(info['buffer'])/ info['total_traffic']
      throughput.append(thp)

      print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)


  thp_cdf = np.sort(throughput)
  yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)
  ax.plot(thp_cdf, yvals, '-')
  ax.set_xlabel('Throughput')
  ax.set_ylabel('CDF')
ax.legend(['0','1','2','3','4','5'])
plt.show()
# %%
# np.savetxt('Multi-user-BM/results_bm/Thp_' + 'PPO_rician', throughput)

# np.savetxt('Multi-user-BM/results_bm/Er_' + 'PPO_rician', estimation_error)

# model.save(log_dir + "BeamManagementEnv-v1_1")
# %%