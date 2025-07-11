
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_UTs', type=int, default=3, help='Number of UEs')
    parser.add_argument('--Nt', type=int, default=16, help='Number of antennas')
    parser.add_argument('--model', type=str, default='PPO', help='Model to train')
    parser.add_argument('--timesteps', type=int, default=4e5, help='Number of timesteps to train')
    parser.add_argument('--if_training', type=bool, default=True, help='Training or testing')
    parser.add_argument('--if_debugging', type=bool, default=False, help='If in debugging mode')
    parser.add_argument('--Ptdbm', type=int, default=7, help='Transmit power in dbm')
    parser.add_argument('--EnvID', type=str, default='BeamManagementEnv-v4', help='Environment ID')
    parser.add_argument('--cdir', type=str, default= 'logEstTh_beamAct', help='Log directory')
    parser.add_argument('--dt', type=float, default=0.2, help='Time step')
    parser.add_argument('--if_log', type=bool, default=False, help='If log the results')
    parser.add_argument('--num_eval_episodes', type=int, default=1, help='Number of evaluation episodes')

    parser.add_argument('--eval_method', type=str, default='trained', help='Evaluation method: trained, greedy, estimated')
    
    args, unknown = parser.parse_known_args()
    return args
