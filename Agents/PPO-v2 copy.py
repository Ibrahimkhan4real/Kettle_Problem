# ppo_kettle_train.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os

# Import your environment
from Kettle_environment_updated_reward_fun import SimpleKettleEnv  # Ensure this file exists and contains your env class

# Optional: Register environment if you want to use string IDs
from gymnasium.envs.registration import register

register(
    id='SimpleKettle-v0',
    entry_point='Kettle_environment_updated_reward_fun:SimpleKettleEnv',
)

# === Instantiate Environment ===
env = gym.make('SimpleKettle-v0')

# Optional: check if environment follows Gym API
check_env(env, warn=True)

# === Create PPO Model ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,  # Set to 1 to see training progress
    tensorboard_log="./ppo_kettle_tensorboard/"
)

total_timesteps = 2000_000  # Total timesteps for training

# === Train the Model ===
model.learn(total_timesteps)

# === Save the Model ===
os.makedirs("ppo_kettle_model", exist_ok=True)
env_file_name = SimpleKettleEnv.__module__
model.save(f"ppo_kettle_model/ppo_kettle_model_{total_timesteps}_{env_file_name}")

# === Evaluate the Model ===
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
