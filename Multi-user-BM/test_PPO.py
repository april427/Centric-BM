import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gymnasium as gym

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/Multi-user-BM/')
from beam_management_env_letter import BeamManagementEnv
# import roboschool

from PPO import PPO

import matplotlib.pyplot as plt


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "BeamManagementEnv-v1"
    has_continuous_action_space = False
    max_ep_len = 100           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 500   # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    throughput = np.zeros((total_test_episodes))

    #####################################################

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_dim = env.action_space.nvec.sum()
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    
    run_num_pretrained = 10     #### set this to load a particular checkpoint num

    ############# write the test data into csv #######
    log_dir = "Multi-user-BM/saved_policy/PPO_logs" 
    log_f_name = log_dir + "/TEST_PPO_" + env_name + "_log_" + str(run_num_pretrained)+ ".csv"
    log_f = open(log_f_name, "w+")
    log_f.write('episode,reward, total_packet_num, transmitted_packets\n')

    directory = "Multi-user-BM/saved_policy/PPO_preTrained" + '/' 
    #directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # checkpoint_path = directory + 'PPO_BeamManagementEnv-v1_WINNER.pth'
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    transmitted_packet_rate = []
    sensing_slots_ratio = []
    reward_episode = []
    action_list = []
    square_error = np.zeros((6,total_test_episodes,max_ep_len, env.unwrapped.N_UEs))
    sinr = np.zeros((6,total_test_episodes,max_ep_len, env.unwrapped.N_UEs))

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state,_ = env.reset()
        total_num_packet = np.sum(env.datapool)
        reward_list = []
        square_error_ep = np.zeros((max_ep_len, env.unwrapped.N_UEs))
        sinr_ep = np.zeros((max_ep_len, env.unwrapped.N_UEs))
        pos_inx = env.unwrapped.dice

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            # print(action)
            action_list.append(action)
            state, reward, done, _, info = env.step(action)
            ep_reward += reward
            reward_list.append(reward)
            square_error_ep[t-1,] = info['est_error']
            sinr_ep[t-1,] = info['sinr']

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                reward_episode.append(np.mean(reward_list))
                print("all transmitted")

        throughput[ep-1] = (total_num_packet-np.sum(env.datapool))/total_num_packet

        log_f.write('{},{},{},{}\n'.format(ep, ep_reward, total_num_packet, total_num_packet-np.sum(env.datapool)))
        log_f.flush()

        test_running_reward +=  ep_reward
        print('Episode: {} \t Reward: {} \t Throughput: {}'.format(ep, round(ep_reward, 2), round(throughput[ep-1], 2)))
        ep_reward = 0

        sensing_slots_ratio.append(info['num_sensing_slots']/t)
        square_error[pos_inx, ep-1,] = square_error_ep
        sinr[pos_inx, ep-1] = sinr_ep

        # clear buffer
        ppo_agent.buffer.clear()

    square_error = np.mean(square_error, axis=1)
    sinr = np.mean(sinr,axis =1)
    mse = square_error.reshape(-1, square_error.shape[-1])
    sinr_avg = sinr.reshape(-1, sinr.shape[-1])

    log_f.close()
    env.close()

    print("============================================================================================")
    print("============================================================================================")
    throughput_cdf = np.sort(throughput)
    yvals = np.arange(len(throughput))/float(len(throughput)-1)
    plt.plot(throughput_cdf, yvals, '-')

    # np.savetxt('Multi-user-BM/results_bm/Thp_PPO_{}'.format(run_num_pretrained), sensing_slots_ratio)

    # np.savetxt('Multi-user-BM/results_bm/mse_{}'.format(run_num_pretrained), mse)
    # np.savetxt('Multi-user-BM/results_bm/sinr_{}'.format(run_num_pretrained), sinr_avg)
    


if __name__ == '__main__':

    test()