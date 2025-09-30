# compare_ppo_sac_kettle.py

import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Import the provided environment
# Adjust this import path to where SimpleKettleEnv is defined
# e.g., from kettle_env import SimpleKettleEnv
from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv  # rename as needed

# -----------------------------
# Config
# -----------------------------
SEED = 42
TOTAL_TIMESTEPS = 50_000
ROLLING_STD_WINDOW = 20
LOG_DIR = "sb3_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(log_subdir):
    env = SimpleKettleEnv()
    # Wrap with Monitor to capture episode rewards/lengths for plotting
    env = Monitor(env, filename=os.path.join(LOG_DIR, log_subdir))
    env.reset(seed=SEED)
    return env

def rolling_std(data, window):
    arr = np.array(data, dtype=np.float32)
    std_vals = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        std_vals.append(np.std(arr[start:i+1]))
    return np.array(std_vals)

# -----------------------------
# Train PPO
# -----------------------------
env_ppo = make_env("ppo")
ppo_model = PPO("MlpPolicy", env_ppo, seed=SEED, verbose=1)
ppo_model.learn(total_timesteps=TOTAL_TIMESTEPS)
ppo_rewards = env_ppo.get_episode_rewards()  # list of per-episode returns
env_ppo.close()

# -----------------------------
# Train SAC
# -----------------------------
env_sac = make_env("sac")
sac_model = SAC("MlpPolicy", env_sac, seed=SEED, verbose=1)
sac_model.learn(total_timesteps=TOTAL_TIMESTEPS)
sac_rewards = env_sac.get_episode_rewards()
env_sac.close()

# Save rewards to CSV for further analysis
np.savetxt(os.path.join(LOG_DIR, "ppo_episode_rewards.csv"), np.array(ppo_rewards), delimiter=",")
np.savetxt(os.path.join(LOG_DIR, "sac_episode_rewards.csv"), np.array(sac_rewards), delimiter=",")

# -----------------------------
# Plot 1: Reward per episode (training)
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(ppo_rewards)) + 1, ppo_rewards, label="PPO")
plt.plot(np.arange(len(sac_rewards)) + 1, sac_rewards, label="SAC")
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("Training: Episode Return vs Episode")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "episode_return_curve.png"), dpi=150)

# -----------------------------
# Plot 2: Boxplot of episode rewards (training stability)
# -----------------------------
plt.figure(figsize=(8, 5))
plt.boxplot([ppo_rewards, sac_rewards], labels=["PPO", "SAC"], showmeans=True)
plt.ylabel("Episode Return")
plt.title("Training: Episode Return Distribution (Stability)")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "episode_return_boxplot.png"), dpi=150)

# -----------------------------
# Plot 3: Rolling std of rewards (bar graph)
# -----------------------------
ppo_std = rolling_std(ppo_rewards, ROLLING_STD_WINDOW)
sac_std = rolling_std(sac_rewards, ROLLING_STD_WINDOW)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes.bar(np.arange(len(ppo_std)) + 1, ppo_std, color="tab:blue")
axes.set_title(f"PPO Rolling Std (window={ROLLING_STD_WINDOW})")
axes.set_ylabel("Std of Return")

axes[22].bar(np.arange(len(sac_std)) + 1, sac_std, color="tab:orange")
axes[22].set_title(f"SAC Rolling Std (window={ROLLING_STD_WINDOW})")
axes[22].set_xlabel("Episode")
axes[22].set_ylabel("Std of Return")

fig.suptitle("Training: Rolling Standard Deviation of Episode Returns (Bar Graph)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(LOG_DIR, "rolling_std_bar.png"), dpi=150)

# -----------------------------
# Post-training evaluation: 1 episode temperature traces
# -----------------------------
def rollout_temperature_trace(model, env, deterministic=True, max_steps=1_000):
    obs, info = env.reset(seed=SEED)
    temps = []
    steps = []
    t = 0
    while t < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        # Prefer env info 'temperature'; fallback to obs
        temp = info.get("temperature", float(obs))
        temps.append(temp)
        steps.append(t)
        t += 1
        if terminated or truncated:
            break
    return np.array(steps), np.array(temps)

eval_env_ppo = SimpleKettleEnv()
eval_env_sac = SimpleKettleEnv()
ppo_steps, ppo_temps = rollout_temperature_trace(ppo_model, eval_env_ppo)
sac_steps, sac_temps = rollout_temperature_trace(sac_model, eval_env_sac)
eval_env_ppo.close()
eval_env_sac.close()

plt.figure(figsize=(10, 5))
plt.plot(ppo_steps, ppo_temps, label="PPO")
plt.plot(sac_steps, sac_temps, label="SAC")
plt.xlabel("Timestep")
plt.ylabel("Temperature (°C)")
plt.title("Post-Training Evaluation (1 Episode): Temperature vs Timestep")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "temperature_over_time_eval.png"), dpi=150)

print("Plots saved to:", LOG_DIR)
