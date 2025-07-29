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
    parser.add_argument('--model', type=str, default='PPO', help='Model to train')
    parser.add_argument('--timesteps', type=int, default=4e5, help='Number of timesteps to train')
    parser.add_argument('--if_training', type=str2bool, default=True, help='Training or testing')
    parser.add_argument('--if_debugging', type=str2bool, default=False, help='If in debugging mode')
    parser.add_argument('--Ptdbm', type=int, default=7, help='Transmit power in dbm')
    parser.add_argument('--EnvID', type=str, default='BeamManagementEnv-v8', help='Environment ID')
    parser.add_argument('--cdir', type=str, default= 'pow', help='Log directory')
    parser.add_argument('--dt', type=float, default=0.2, help='Time step')
    parser.add_argument('--if_log', type=str2bool, default=False, help='If log the results')
    parser.add_argument('--num_eval_episodes', type=int, default=1, help='Number of evaluation episodes')

    parser.add_argument('--eval_method', type=str, default='trained', help='Evaluation method: trained, genie, estimated')
    parser.add_argument('--if_rician', type=str2bool, default=False, help='If Rician channel model is used')
    parser.add_argument('--if_plot', type=str2bool, default=True, help='If plot the results')
    
    args, unknown = parser.parse_known_args()
    return args
