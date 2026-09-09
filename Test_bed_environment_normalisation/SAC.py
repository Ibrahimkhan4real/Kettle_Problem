import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv


# ===== Custom Callback to log episode rewards =====
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Episode info lives inside infos (set by Monitor wrapper)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def smooth(values, window=15):
    """Running-average smoothing for noisy reward curves."""
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ===== Register custom env =====
gym.register(id="SimpleKettle-v0", entry_point=SimpleKettleEnv)


# ===== Train function =====
def train_agent(agent_class, env_id, timesteps=100_000, seed=42, **kwargs):
    env = make_vec_env(env_id, n_envs=1, seed=seed)
    callback = RewardLoggerCallback()
    model = agent_class("MlpPolicy", env, verbose=1, seed=seed, **kwargs)
    model.learn(total_timesteps=timesteps, callback=callback)
    env.close()
    return model, callback.episode_rewards


# ===== Train PPO =====
# n_steps=200 aligns rollout buffer with the episode length (200 steps).
# ent_coef adds entropy bonus to encourage exploration early on.
print("=" * 50)
print("Training PPO...")
print("=" * 50)
ppo_model, ppo_rewards = train_agent(
    PPO,
    "SimpleKettle-v0",
    timesteps=100_000,
    n_steps=200,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
)

# ===== Train SAC =====
# learning_starts lets the replay buffer warm up with random actions first.
# buffer_size and batch_size are tuned for a 200-step episode environment.
print("=" * 50)
print("Training SAC...")
print("=" * 50)
sac_model, sac_rewards = train_agent(
    SAC,
    "SimpleKettle-v0",
    timesteps=100_000,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",
)

# Save trained models
ppo_model.save("ppo_kettle")
sac_model.save("sac_kettle")
print("\nModels saved: ppo_kettle.zip, sac_kettle.zip")

# Print summary stats
ppo_mean, ppo_std = np.mean(ppo_rewards), np.std(ppo_rewards)
sac_mean, sac_std = np.mean(sac_rewards), np.std(sac_rewards)
print(f"\nPPO — episodes: {len(ppo_rewards):4d} | mean reward: {ppo_mean:8.4f} | std: {ppo_std:.4f}")
print(f"SAC — episodes: {len(sac_rewards):4d} | mean reward: {sac_mean:8.4f} | std: {sac_std:.4f}")


# ===== Plot 1: Smoothed reward per episode =====
ppo_sm = smooth(ppo_rewards)
sac_sm = smooth(sac_rewards)

plt.figure(figsize=(12, 6))
plt.plot(ppo_rewards, alpha=0.2, color="steelblue")
plt.plot(sac_rewards, alpha=0.2, color="darkorange")
plt.plot(range(len(ppo_sm)), ppo_sm, label="PPO (smoothed)", color="steelblue", linewidth=2)
plt.plot(range(len(sac_sm)), sac_sm, label="SAC (smoothed)", color="darkorange", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode: PPO vs SAC")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("reward_per_episode.png", dpi=150)
plt.show()

# ===== Plot 2: Boxplot =====
plt.figure(figsize=(8, 6))
plt.boxplot([ppo_rewards, sac_rewards], labels=["PPO", "SAC"], patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="red", linewidth=2))
plt.ylabel("Reward")
plt.title("Reward Distribution (Learning Stability)")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("reward_boxplot.png", dpi=150)
plt.show()

# ===== Plot 3: Mean ± Std bar graph =====
plt.figure(figsize=(6, 6))
bars = plt.bar(["PPO", "SAC"], [ppo_mean, sac_mean],
               yerr=[ppo_std, sac_std],
               color=["steelblue", "darkorange"],
               capsize=8, alpha=0.85)
plt.ylabel("Mean Reward ± Std Dev")
plt.title("Mean Reward per Algorithm")
plt.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("reward_mean_std.png", dpi=150)
plt.show()


# ===== Evaluate one deterministic episode =====
def evaluate_episode(model, env_id, seed=42):
    env = Monitor(gym.make(env_id))
    obs, _ = env.reset(seed=seed)
    temps, powers, steps = [], [], []
    done, truncated = False, False
    step = 0
    total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        temps.append(info["temperature"])
        powers.append(info["power_used"])
        steps.append(step)
        total_reward += reward
        step += 1
    env.close()
    return steps, temps, powers, total_reward


print("\n--- Evaluating PPO ---")
ppo_steps, ppo_temps, ppo_powers, ppo_total = evaluate_episode(ppo_model, "SimpleKettle-v0")
print(f"  Steps: {len(ppo_steps)} | Final temp: {ppo_temps[-1]:.2f}°C | Total reward: {ppo_total:.4f}")

print("--- Evaluating SAC ---")
sac_steps, sac_temps, sac_powers, sac_total = evaluate_episode(sac_model, "SimpleKettle-v0")
print(f"  Steps: {len(sac_steps)} | Final temp: {sac_temps[-1]:.2f}°C | Total reward: {sac_total:.4f}")


# ===== Plot 4: Temperature and Power over time =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ppo_steps, ppo_temps, label="PPO", color="steelblue", linewidth=2)
axes[0].plot(sac_steps, sac_temps, label="SAC", color="darkorange", linewidth=2)
axes[0].axhline(100, color="red", linestyle="--", label="Target (100°C)")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("Evaluation: Temperature vs Timestep")
axes[0].legend()
axes[0].grid(alpha=0.4)

axes[1].plot(ppo_steps, ppo_powers, label="PPO", color="steelblue", linewidth=2)
axes[1].plot(sac_steps, sac_powers, label="SAC", color="darkorange", linewidth=2)
axes[1].axhline(3000, color="red", linestyle="--", alpha=0.5, label="Max power (3000 W)")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Power (W)")
axes[1].set_title("Evaluation: Power Usage vs Timestep")
axes[1].legend()
axes[1].grid(alpha=0.4)

plt.tight_layout()
plt.savefig("evaluation.png", dpi=150)
plt.show()
