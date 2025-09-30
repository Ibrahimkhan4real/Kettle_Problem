import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Import your environment
from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv   # <-- Make sure kettle_env.py contains your fixed env


# ===== Custom Callback to log rewards =====
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Each step, check if an episode finished
        if "episode" in self.locals:
            r = self.locals["episode"]["r"]
            self.episode_rewards.append(r)
        return True


# ===== Helper: Train function =====
def train_agent(agent_class, env_id, timesteps=20_00, seed=42):
    env = make_vec_env(env_id, n_envs=1, seed=seed)
    callback = RewardLoggerCallback()
    model = agent_class("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps, callback=callback)
    env.close()
    return model, callback.episode_rewards


# ===== Register custom env with gym =====
gym.register(id="SimpleKettle-v0", entry_point=SimpleKettleEnv)


# ===== Train PPO and SAC =====
ppo_model, ppo_rewards = train_agent(PPO, "SimpleKettle-v0")
sac_model, sac_rewards = train_agent(SAC, "SimpleKettle-v0")

# Align lengths (if one ended early)
min_len = min(len(ppo_rewards), len(sac_rewards))
ppo_rewards = ppo_rewards[:min_len]
sac_rewards = sac_rewards[:min_len]

# ===== Plot 1: Reward per episode =====
plt.figure(figsize=(10, 6))
plt.plot(ppo_rewards, label="PPO", alpha=0.7)
plt.plot(sac_rewards, label="SAC", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode: PPO vs SAC")
plt.legend()
plt.grid()
plt.show()

# ===== Plot 2: Boxplot of rewards =====
plt.figure(figsize=(8, 6))
plt.boxplot([ppo_rewards, sac_rewards], labels=["PPO", "SAC"])
plt.ylabel("Reward")
plt.title("Reward Distribution (Learning Stability)")
plt.grid()
plt.show()

# ===== Plot 3: Std deviation bar graph =====
ppo_std = np.std(ppo_rewards)
sac_std = np.std(sac_rewards)

plt.figure(figsize=(6, 6))
plt.bar(["PPO", "SAC"], [ppo_std, sac_std], color=["blue", "orange"])
plt.ylabel("Standard Deviation of Rewards")
plt.title("Reward Std. Dev. per Algorithm")
plt.show()

# ===== Evaluation: run 1 episode and log temperature =====
def evaluate_episode(model, env_id, seed=42):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    temps = []
    steps = []
    done, truncated = False, False
    step = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        temps.append(info["temperature"])
        steps.append(step)
        step += 1
    env.close()
    return steps, temps


ppo_steps, ppo_temps = evaluate_episode(ppo_model, "SimpleKettle-v0")
sac_steps, sac_temps = evaluate_episode(sac_model, "SimpleKettle-v0")

# ===== Plot 4: Temperature over time (Evaluation) =====
plt.figure(figsize=(10, 6))
plt.plot(ppo_steps, ppo_temps, label="PPO")
plt.plot(sac_steps, sac_temps, label="SAC")
plt.axhline(100, color="red", linestyle="--", label="Target Temp (100°C)")
plt.xlabel("Timestep")
plt.ylabel("Temperature (°C)")
plt.title("Evaluation Episode: Temperature vs Timestep")
plt.legend()
plt.grid()
plt.show()
