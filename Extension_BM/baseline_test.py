import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO, DDPG
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Extension_BM/')
from beam_management_env_letter import BeamManagementEnv

model_type = "DDPG"
model_class = getattr(sys.modules[__name__], model_type)


env = gym.make("BeamManagementEnv-v2",  if_training=False)
#check_env(env)

log_dir = "RL_mimo_ofdm/" + model_type + "_"+ str(env.unwrapped.N_UTs) +"ue/" 
print("N_UTs=", env.unwrapped.N_UTs)

# %%
################################## Evaluate ####################################
figure_handle = plt.figure()
ax = figure_handle.add_subplot()


model = model_class.load(log_dir + "best_model")

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

        # for i in range(env.unwrapped.N_UTs):
        #     if info['buffer'][i] == 0 and i not in noted:
        #         noted.append(i)

        if dones:
            reward_done.append(reward_episode)
            print("all transmitted")
            break
        
        # estimation_error.append(info['est_error'])

    
    if t == num_steps - 1:  
        reward_list.append(reward_episode)       
    
    thp = 1 - np.sum(info['buffer'])/ info['total_traffic']
    print(info['buffer'])
    throughput.append(thp)
    
    print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)


thp_cdf = np.sort(throughput)
yvals = np.arange(len(thp_cdf))/float(len(thp_cdf)-1)
ax.plot(thp_cdf, yvals, 'o')
ax.set_xlabel('Throughput')
ax.set_ylabel('CDF')

plt.show()

# %%
