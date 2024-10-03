# %% 
import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt

from stable_baselines3 import  PPO, DQN
#from stable_baselines3.common.env_util import make_env
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import argparse

import sys
sys.path.append('/Users/xiyu/Documents/AAU/CENTRIC_codes/PPO_beam_management/bl_multiDiscrete/')
from beam_management_env_letter import BeamManagementEnv

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def main(args):
# Create log dir
  log_dir = args.model + "/"
  os.makedirs(log_dir, exist_ok=True)

  env = gym.make(args.env)
  #check_env(env)
  env = Monitor(env, log_dir)

  ################### Learn a model ###################
  # Create the callback: check every 1000 steps
  callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

  model_class = getattr(sys.modules[__name__], args.model)
  model = model_class(args.policy, env, verbose=1)
  timesteps = args.total_timesteps
  model.learn(total_timesteps=timesteps, callback=callback)

  plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO BeamManagement-v1")
  plt.show()
  model.save(log_dir + "BeamManagement")
  

# %%
def test(args):
# model = PPO.load("PPO_BeamManagementEnv-v1_1")
# model = DQN.load("DQN_BeamManagementEnv-v1")
#model = DQN.load("baseline/tmp_dqn/best_model")
  num_episodes = 100
  num_steps = 100
  reward_done = []
  reward_list = []
  throughput = []

  env = gym.make(args.env)
  model_class = getattr(sys.modules[__name__], args.model)
  model = model_class.load(args.log_dir + "BeamManagement")

  for ep in range(1, num_episodes+1):

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
  
  return throughput

# %%

# %%
# np.savetxt('Multi-user-BM/results_bm/Thp_' + 'DQN', throughput)
# model.save(log_dir + "DQN_BeamManagementEnv-v1_1")
# %%
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", default="MlpPolicy")
  parser.add_argument("--env", default="BeamManagementEnv-v1")
  parser.add_argument("--total_timesteps", default=1e4)
  parser.add_argument("--model", default="PPO")
  parser.add_argument("--log_dir", default="bl_multiDiscrete/tmp_dqn_v20/")
  parser.add_argument("--init_posi", default=np.random.randint(0,5), help="Initial vehicle positions: default --6 fixed options for training. When =-1 --random initial positions for testing")
  parser.add_argument("--action_space")

  args = parser.parse_args()
  main(args)

  throughput = test(args)
  
  figure_handle = plt.figure()
  ax = figure_handle.add_subplot()
  tdma_thp_cdf = np.sort(throughput)
  yvals = np.arange(len(tdma_thp_cdf))/float(len(tdma_thp_cdf)-1)
  ax.plot(tdma_thp_cdf, yvals, '-')
  ax.set_xlabel('Throughput')
  ax.set_ylabel('CDF')
  ax.legend('Heuristic Env')
# %%
