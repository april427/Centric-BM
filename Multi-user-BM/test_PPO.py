import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gymnasium as gym
from beam_management_env_letter import BeamManagementEnv
# import roboschool

from PPO import PPO
from torch.utils.tensorboard import SummaryWriter

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

    total_test_episodes = 1000   # total num of testing episodes

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
    
    run_num_pretrained = 1     #### set this to load a particular checkpoint num

    ############# write the test data into Tensorboard and csv #######
    writer = SummaryWriter(f"runs/{'TEST_' + env_name}_{run_num_pretrained}")

    log_dir = "PPO_logs" + '/' + env_name + '/'
    log_f_name = log_dir + "/TEST_PPO_" + env_name + "_log_" + str(run_num_pretrained)+ ".csv"
    log_f = open(log_f_name, "w+")
    log_f.write('episode,reward, total_packet_num, transmitted_packets\n')

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # checkpoint_path = directory + 'PPO_BeamManagementEnv-v1_WINNER.pth'
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    transmitted_packet_rate = []
    th_episode = []
    reward_episode = []
    action_list = []

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()
        total_num_packet = np.sum(env.datapool)
        reward_list = []

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            # print(action)
            action_list.append(action)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            reward_list.append(reward)

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                reward_episode.append(np.mean(reward_list))
                print("all transmitted")

        throughput[ep-1] = (total_num_packet-np.sum(env.datapool))/total_num_packet
        writer.add_scalar("Packet transmission rate", throughput[ep-1], ep)
        writer.add_scalar("Test reward", reward, ep)

        log_f.write('{},{},{},{}\n'.format(ep, ep_reward, total_num_packet, total_num_packet-np.sum(env.datapool)))
        logf.flush()

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    log_f.close()
    env.close()

    print("============================================================================================")
    # print("Total lost traffic DL and UL: " + str(sum(info['lost_traffic'])))
    # print("Total traffic DL and UL: " + str(sum(info['total_traffic'])))
    # print("Total lost traffic DL and UL for 3 users:\n" + str(info['lost_traffic'] / info['total_traffic']))
    # print("Average PER: " + str(np.mean(info['lost_traffic'] / info['total_traffic'])))
    # print("Total Throughput: " + str(1 - sum(info['failure_type'] / sum(sum(info['total_traffic'])))))
    # print("(%) of failure types (Buffer, Beam, Blockage) over failed traffic: " + str(
    #     info['failure_type'] / sum(info['failure_type'])))
    # print("(%) of failure types (Buffer, Beam, Blockage) over total traffic: " + str(
    #     info['failure_type'] / sum(sum(info['total_traffic']))))
    # print("(%) of drops over the total traffic: " + str(sum(info['failure_type'] / sum(sum(info['total_traffic'])))))
    # print("Lost packets by type: " + str(info['failure_type']))
    print("============================================================================================")
    throughput_cdf = np.sort(throughput)
    yvals = np.arange(len(throughput))/float(len(throughput)-1)
    plt.plot(throughput_cdf, yvals, '-')

    # np.savetxt('results_beam_tracking/reward_episode_PPO_cttc', reward_episode)
    # np.savetxt('results_beam_tracking/mean_packet_rate_episode_PPO_cttc', mean_packet_rate_episode)
    # np.savetxt('results_beam_tracking/action_list_PPO_cttc', action_list)
    # np.savetxt('results_beam_tracking/failure_type_PPO_cttc', info['failure_type'])
    # np.savetxt('results_beam_tracking/th_PPO_cttc', th_episode)

if __name__ == '__main__':

    test()