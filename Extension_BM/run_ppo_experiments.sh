#!/bin/bash
# PPO Hyperparameter Tuning Experiments
# Run each experiment one at a time. Comment/uncomment as needed.
# Each run uses a different --cdir so logs stay separate.

cd "$(dirname "$0")"

COMMON="python baseline_train.py --model PPO --N_UTs 2 --Nt 16"

###############################################################################
# Phase 1: Baseline and normalization
###############################################################################

# Run 1 — PPO baseline: no VecNormalize, current PPO defaults from baseline_train.py
# $COMMON --cdir ppo_run1_baseline \
#   --timesteps 400000

# Run 2 — PPO with VecNormalize on observations + rewards
# $COMMON --cdir ppo_run2_vecnorm_obs_rew \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --timesteps 400000

###############################################################################
# Phase 2: Rollout / batch tuning
###############################################################################

# Run 3 — Smaller rollout: n_steps=1024, batch_size=64
# $COMMON --cdir ppo_run3_nsteps1024 \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --ppo_n_steps 1024 --ppo_batch_size 64 --initial_train 0\
#   --timesteps 400000

# # Run 4 — Larger batch and more epochs: n_steps=2048, batch_size=128, n_epochs=20
# $COMMON --cdir ppo_run4_large_batch_more_epochs \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --ppo_n_steps 2048 --ppo_batch_size 128 --ppo_n_epochs 20 \
#   --timesteps 400000

# ###############################################################################
# # Phase 3: Learning rate / clipping / network tuning
# ###############################################################################

# # Run 5 — Lower learning rate, smaller clip range, slightly lower entropy
# $COMMON --cdir ppo_run5_low_lr_clip \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --ppo_n_steps 1024 --ppo_batch_size 64 --ppo_n_epochs 10 \
#   --ppo_clip_range 0.15 --ppo_ent_coef 0.005 \
#   --ppo_lr_start 2e-4 --ppo_lr_end 5e-5 \
#   --timesteps 400000

# # Run 6 — Combined tuned PPO: smaller network + Tanh + more epochs
# $COMMON --cdir ppo_run6_combined \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --ppo_n_steps 1024 --ppo_batch_size 64 --ppo_n_epochs 20 \
#   --ppo_clip_range 0.15 --ppo_ent_coef 0.005 \
#   --ppo_gamma 0.99 --ppo_gae_lambda 0.95 \
#   --ppo_lr_start 2e-4 --ppo_lr_end 5e-5 \
#   --net_arch_size 64 --activation Tanh \
#   --timesteps 400000

$COMMON --cdir ppo_run7_clutters_sep \
  --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
  --ppo_n_steps 1024 --ppo_batch_size 64 --initial_train 1 --if_rician 1\
  --timesteps 1500000