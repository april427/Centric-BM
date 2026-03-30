#!/bin/bash
# A2C Hyperparameter Tuning Experiments
# Run each experiment one at a time. Comment/uncomment as needed.
# Each run uses a different --cdir so logs stay separate.

cd "$(dirname "$0")"

COMMON="python baseline_train.py --model A2C --N_UTs 2 --Nt 16"

# ###############################################################################
# # Phase 1: Normalization (highest expected impact)
# ###############################################################################

# # Run 1 — VecNormalize: observation + reward normalization
# # All other A2C params are the current defaults (n_steps=50, gamma=0.98, lr 7e-4→1e-4)
# $COMMON --cdir run1_vecnorm_obs_rew \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --timesteps 400000

# # Run 2 — VecNormalize: reward normalization only
# $COMMON --cdir run2_vecnorm_rew \
#   --norm_obs False --norm_reward True --clip_reward 10 \
#   --timesteps 400000

# ###############################################################################
# # Phase 2: n_steps tuning (use VecNormalize from Phase 1)
# ###############################################################################

# Run 3 — n_steps=40 (match episode length)
$COMMON --cdir run3_nsteps40 \
  --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
  --n_steps 40 --initial_train 1 --if_rician 1\
  --timesteps 1500000

# # Run 4 — n_steps=80 (2 full episodes per rollout)
# $COMMON --cdir run4_nsteps80 \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --n_steps 80 \
#   --timesteps 400000

# # Run 5 — n_steps=120 (3 full episodes per rollout)
# $COMMON --cdir run5_nsteps120 \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --n_steps 120 \
#   --timesteps 400000

###############################################################################
# Phase 3: Learning rate, network architecture, combined
###############################################################################

# Run 6 — Lower learning rate (3e-4 → 5e-5) + best n_steps from Phase 2
# $COMMON --cdir run6_low_lr \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --n_steps 40 \
#   --lr_start 3e-4 --lr_end 5e-5 \
#   --timesteps 400000

# # Run 7 — Smaller network [64,64] + Tanh (matches working PPO architecture)
# $COMMON --cdir run7_small_net_tanh \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --n_steps 40 \
#   --net_arch_size 64 --activation Tanh \
#   --timesteps 400000

# # Run 8 — Combined best: normalization + n_steps=40 + low LR + small net + tuned coefficients + longer training
# $COMMON --cdir run8_combined \
#   --norm_obs True --norm_reward True --clip_obs 10 --clip_reward 10 \
#   --n_steps 40 \
#   --lr_start 3e-4 --lr_end 5e-5 \
#   --net_arch_size 64 --activation Tanh \
#   --gamma 0.99 --ent_coef 0.005 --vf_coef 0.25 \
#   --timesteps 1000000
