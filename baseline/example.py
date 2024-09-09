import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from beam_management_env_letter import BeamManagementEnv

# Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)
env = BeamManagementEnv()
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2500)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs,_ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render("human")