import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_UTs', type=int, default=2, help='Number of UEs')
    parser.add_argument('--Nt', type=int, default=16, help='Number of antennas')
    parser.add_argument('--model', type=str, default='PPO', help='Model to train (e.g., PPO, A2C)')
    parser.add_argument('--timesteps', type=int, default=4e5, help='Number of timesteps to train')
    parser.add_argument('--if_training', type=str2bool, default=True, help='Training or testing')
    parser.add_argument('--if_debugging', type=str2bool, default=False, help='If in debugging mode')
    parser.add_argument('--Ptdbm', type=int, default=7, help='Transmit power in dbm')
    parser.add_argument('--EnvID', type=str, default='BeamManagementEnv-v4', help='Environment ID')

    parser.add_argument('--cdir', type=str, default= '6a_crlb_clutters', help='Log directory')

    parser.add_argument('--dt', type=float, default=0.2, help='Time step')
    parser.add_argument('--if_log', type=str2bool, default=False, help='If log the results')
    parser.add_argument('--num_eval_episodes', type=int, default=1, help='Number of evaluation episodes')

    parser.add_argument('--eval_method', type=str, default='estimated', help='Evaluation method: trained, genie, estimated')
    parser.add_argument('--if_rician', type=str2bool, default=False, help='If Rician channel model is used')
    parser.add_argument('--if_plot', type=str2bool, default=True, help='If plot the results')
    parser.add_argument('--reuse_beams', type=int, default=1, help='1: sweep 1 beam or 2: multiple beams, reuse is buffer priority')

    parser.add_argument('--ut_v', type=float, default=6, help='UE velocity in m/s')
    parser.add_argument('--arrival_prob', type=float, default=0.9, help='UE packet arrival probability')
    parser.add_argument('--buffer_size', type=int,  default=4, help='Buffer size for each UE')

    # A2C hyperparameter tuning arguments
    parser.add_argument('--n_steps', type=int, default=40, help='Number of steps per rollout (A2C)')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--lr_start', type=float, default=7e-4, help='Initial learning rate')
    parser.add_argument('--lr_end', type=float, default=1e-4, help='Final learning rate')
    parser.add_argument('--net_arch_size', type=int, default=128, help='Hidden layer size (e.g., 64 or 128)')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation function: ReLU or Tanh')
    parser.add_argument('--norm_obs', type=str2bool, default=True, help='Normalize observations with VecNormalize')
    parser.add_argument('--norm_reward', type=str2bool, default=True, help='Normalize rewards with VecNormalize')
    parser.add_argument('--clip_obs', type=float, default=10.0, help='Observation clipping value for VecNormalize')
    parser.add_argument('--clip_reward', type=float, default=10.0, help='Reward clipping value for VecNormalize')

    # PPO hyperparameter tuning arguments
    parser.add_argument('--ppo_n_steps', type=int, default=2048, help='Number of steps per rollout (PPO)')
    parser.add_argument('--ppo_batch_size', type=int, default=64, help='Mini-batch size for PPO updates')
    parser.add_argument('--ppo_n_epochs', type=int, default=10, help='Number of epochs per PPO update')
    parser.add_argument('--ppo_gamma', type=float, default=0.99, help='Discount factor for PPO')
    parser.add_argument('--ppo_gae_lambda', type=float, default=0.95, help='GAE lambda for PPO')
    parser.add_argument('--ppo_clip_range', type=float, default=0.2, help='Clipping range for PPO')
    parser.add_argument('--ppo_ent_coef', type=float, default=0.01, help='Entropy coefficient for PPO')
    parser.add_argument('--ppo_vf_coef', type=float, default=0.5, help='Value function coefficient for PPO')
    parser.add_argument('--ppo_max_grad_norm', type=float, default=0.5, help='Max gradient norm for PPO')
    parser.add_argument('--ppo_lr_start', type=float, default=3e-4, help='Initial learning rate for PPO')
    parser.add_argument('--ppo_lr_end', type=float, default=1e-4, help='Final learning rate for PPO')

    args, unknown = parser.parse_known_args()
    return args
