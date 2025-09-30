# ppo_kettle_train.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
import os

# Import your environment
from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv  # Ensure this file exists and contains your env class

# Optional: Register environment if you want to use string IDs
from gymnasium.envs.registration import register

register(
    id='SimpleKettle-v0',
    entry_point='Kettle_environment_updated_reward_fun_continuous:SimpleKettleEnv',
)

# === Instantiate Environment ===
env = Monitor(gym.make('SimpleKettle-v0'))

# Optional: check if environment follows Gym API
check_env(env, warn=True)

# === Create PPO Model ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    ent_coef=0.015,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    tensorboard_log="./ppo_kettle_tensorboard/"
)

total_timesteps = 2000  # Total timesteps for training

# === Train the Model ===
model.learn(total_timesteps)

# === Save the Model ===
os.makedirs("ppo_kettle_model", exist_ok=True)
env_file_name = SimpleKettleEnv.__module__
model.save(f"ppo_kettle_model/ppo_kettle_model_{total_timesteps}_{env_file_name}")

# === Evaluate the Model ===
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# === VISUALISE THE TRAINED AGENT ===
print("Running visualisation...")

# Turn on interactive plotting
plt.ion()

# Create a figure and two subplots (one for temperature, one for action)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Trained PPO Agent Performance")

# --- Run one episode with the trained model ---
obs, info = env.reset()
terminated, truncated = False, False

# Lists to store data for plotting
temps_history = [obs[0]] # Initial temperature
actions_history = []
time_steps = [0]

while not (terminated or truncated):
    # Get action from the trained model (deterministic for evaluation)
    action, _states = model.predict(obs, deterministic=True)
    
    # Perform the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Store data
    temps_history.append(obs[0])
    actions_history.append(action) # Action is a continuous value (power)
    time_steps.append(len(time_steps))

    # --- Update the plots in real-time ---
    
    # Clear previous plots
    axs[0].clear()
    axs[1].clear()
    
    # Plot temperature
    axs[0].plot(time_steps, temps_history, 'r-')
    axs[0].set_ylabel("Temperature (°C)")
    axs[0].grid(True)
    
    # Plot action (power)
    # Use step plot for clearer visualisation of discrete-time actions
    axs[1].step(time_steps[1:], actions_history, 'b-')
    axs[1].set_ylabel("Power (W)")
    axs[1].set_xlabel("Time (steps)")
    axs[1].set_ylim([-100, 3100]) # Set y-axis limits for actions
    axs[1].grid(True)

    # Pause to allow the plot to redraw
    plt.pause(0.01)

print("Visualisation finished.")

# Turn off interactive mode and show final plot
plt.ioff()
plt.show()