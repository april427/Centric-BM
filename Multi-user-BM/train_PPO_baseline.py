import gymnasium as gym
import os
import torch
from datetime import datetime

import numpy as np
from beam_management_env_letter import BeamManagementEnv

from stable_baselines3 import PPO

has_continuous_action_space = False
max_ep_len = 1000  # max timesteps in one episode
max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)  # save model frequency (in num timesteps)

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4  # update policy every n timesteps
K_epochs = 80  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

random_seed = 0  # set random seed if required (0 = no random seed)
#####################################################

env_name = "BeamManagementEnv-v1"
print("training environment name : " + env_name)
env = gym.make(env_name)

model = PPO("MlpPolicy", 
            env, 
            learning_rate=lr_actor, 
            n_steps=2048, batch_size=64, 
            n_epochs=K_epochs, gamma=gamma, 
            gae_lambda=0.95, clip_range=eps_clip, clip_range_vf=None, normalize_advantage=True, 
            ent_coef=0.0, vf_coef=0.5, 
            max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=None, rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, tensorboard_log=None,  verbose=1, device='auto', _init_setup_model=True)

model.learn(total_timesteps=25000)

model.save("ppo_multiuser_bm")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_multiuser_bm")

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

###################### logging ######################

    #### log files for multiple runs are NOT overwritten
log_dir = "baseline_PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#### get number of log files in log directory
run_num = 1
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)
#####################################################

################### checkpointing ###################
run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

directory = "baseline_PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)
#####################################################

############# print all hyperparameters #############
print("--------------------------------------------------------------------------------------------")
print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)
print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
print("--------------------------------------------------------------------------------------------")
print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)
print("has continuous action space : ", has_continuous_action_space)
print("--------------------------------------------------------------------------------------------")
if has_continuous_action_space:
    print("Initializing a continuous action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("starting std of action distribution : ", action_std)
    print("decay rate of std of action distribution : ", action_std_decay_rate)
    print("minimum std of action distribution : ", min_action_std)
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
else:
    print("Initializing a discrete action space policy")
print("--------------------------------------------------------------------------------------------")
print("PPO update frequency : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("--------------------------------------------------------------------------------------------")
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)
if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
#####################################################

print("============================================================================================")

################# training procedure ################
# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

# logging file
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
