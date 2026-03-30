#!/bin/bash
# DQN Hyperparameter Tuning Experiments
# Run each experiment one at a time. Comment/uncomment as needed.
# Each run uses a different --cdir so logs stay separate.

cd "$(dirname "$0")"

COMMON="python baseline_train.py --model DQN --N_UTs 2 --Nt 16"

###############################################################################
# Phase 1: Baseline and normalization
###############################################################################

# Run 1 — DQN baseline (no VecNormalize)
# $COMMON --cdir dqn_run1_baseline \
#   --norm_obs False --norm_reward False \
#   --timesteps 400000

# Run 2 — DQN + VecNormalize (obs + reward)
# $COMMON --cdir dqn_run2_vecnorm_obs_rew \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --timesteps 400000

###############################################################################
# Phase 2: Core DQN tuning
###############################################################################

# Run 3 — More stable default DQN setup
$COMMON --cdir dqn_run3_default_stable \
  --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
  --dqn_buffer_size 100000 --dqn_learning_starts 5000 \
  --dqn_batch_size 256 --dqn_train_freq 4 --dqn_gradient_steps 1 \
  --dqn_target_update_interval 1000 --dqn_gamma 0.99 \
  --dqn_exploration_fraction 0.25 --dqn_exploration_initial_eps 1.0 --dqn_exploration_final_eps 0.05 \
  --dqn_lr_start 1e-3 --dqn_lr_end 1e-4 \
  --net_arch_size 128 --activation ReLU \
  --initial_train 1 --if_rician 1 \
  --timesteps 1500000

# Run 4 — Smaller network + lower LR
# $COMMON --cdir dqn_run4_small_net_low_lr \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --dqn_learning_starts 5000 --dqn_batch_size 128 \
#   --dqn_lr_start 5e-4 --dqn_lr_end 1e-4 \
#   --net_arch_size 64 --activation Tanh \
#   --timesteps 800000
