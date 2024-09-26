# %% 
import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG, PPO, A2C, DQN
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Multi-user-BM/')
from beam_management_env_letter import BeamManagementEnv

# Parallel environments
# %%
env = gym.make("BeamManagementEnv-v1")
#env = make_env("BeamManagementEnv-v1", n_envs=1)
check_env(env)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e6)

#del model # remove to demonstrate saving and loading
# %%
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# %%
num_episodes = 100
num_steps = 100
reward_done = []
reward_list = []
throughput = []

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
    
    if t == num_steps - 1:  
        reward_list.append(reward_episode)       
    
    thp = 1 - np.sum(info['buffer'])/ info['total_traffic']
    throughput.append(thp)

    print("Episode : {} \t Throughput: {}".format(ep, round(thp,2)), flush = True)

# %%
import matplotlib.pyplot as plt
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
tdma_thp_cdf = np.sort(throughput)
yvals = np.arange(len(tdma_thp_cdf))/float(len(tdma_thp_cdf)-1)
ax.plot(tdma_thp_cdf, yvals, '-')
ax.set_xlabel('Throughput')
ax.set_ylabel('CDF')
ax.legend('Heuristic Env')
# %%

model.save("DQN_BeamManagementEnv-v1")