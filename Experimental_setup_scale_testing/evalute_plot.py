import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from Kettle_environment_updated_reward_fun import SimpleKettleEnv
import seaborn as sns

os.makedirs("ppo_kettle_models", exist_ok=True)
os.makedirs("ppo_kettle_plots", exist_ok=True)

scales = [1, 5, 10]
total_timesteps = 3_000_000  # reduce for faster runs while comparing

training_curves = {}
boxplot_data = {}
temp_curves = {}
train_stats = {}
eval_stats = {}

for scale in scales:
    print(f"\n=== Training with alpha*{scale}, beta*{scale} ===")
    env = SimpleKettleEnv(alpha=0.000001 * scale, beta=0.01 * scale)
    env = Monitor(env)

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model.save(f"ppo_kettle_models/ppo_alpha{scale}_beta{scale}")

    # --- Training curve ---
    rewards = env.get_episode_rewards()
    training_curves[scale] = rewards

    # Training statistics (std + CV)
    rewards_array = np.array(rewards)
    mean_training = np.mean(rewards_array)
    std_training = np.std(rewards_array)
    cv_training = std_training / mean_training if mean_training != 0 else np.nan
    train_stats[scale] = {"mean": mean_training, "std": std_training, "cv": cv_training}
    print(f"[Training Curve] Scale {scale}: Mean={mean_training:.2f}, "
          f"Std={std_training:.2f}, CV={cv_training:.2f}")

    # --- Evaluation (boxplot data) ---
    eval_env = SimpleKettleEnv(alpha=0.000001 * scale, beta=0.01 * scale)
    rewards_eval, lengths_eval = evaluate_policy(
        model, eval_env, n_eval_episodes=30, return_episode_rewards=True
    )
    boxplot_data[scale] = rewards

    # Evaluation statistics (std + CV)
    # mean_eval = np.mean(rewards_eval)
    # std_eval = np.std(rewards_eval)
    # cv_eval = std_eval / mean_eval if mean_eval != 0 else np.nan
    # eval_stats[scale] = {"mean": mean_eval, "std": std_eval, "cv": cv_eval}
    # print(f"[Evaluation] Scale {scale}: Mean={mean_eval:.2f}, "
    #       f"Std={std_eval:.2f}, CV={cv_eval:.2f}")

    # --- Single run temperature curve ---
    run_env = SimpleKettleEnv(alpha=0.000001 * scale, beta=0.01 * scale)
    obs, _ = run_env.reset()
    temps = []
    for _ in range(run_env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = run_env.step(action)
        temps.append(obs[0])
        if done:
            break
    temp_curves[scale] = temps

# === Plot Training Curves ===
plt.figure(figsize=(8,5))
for scale, rewards in training_curves.items():
    plt.plot(range(1, len(rewards)+1), rewards, label=f"scale {scale}")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Training Reward vs. Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_kettle_plots/training_curves_compare.png")

# === Plot Boxplot Comparison ===
plt.figure(figsize=(7,6))
data = [boxplot_data[s] for s in scales]
sns.violinplot(data=data)
sns.swarmplot(data=data, color=".25", size=3)
plt.xticks(range(len(scales)), [f"{s}x" for s in scales])
plt.ylabel("Episode Reward")
plt.title("Final Performance Distribution (Violin + Points)")
plt.tight_layout()
plt.savefig("ppo_kettle_plots/violin_compare.png")

plt.ylabel("Episode Reward")
plt.title("Final Performance Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_kettle_plots/boxplot_compare.png")

# === Plot Temperature Curves ===
plt.figure(figsize=(8,5))
for scale, temps in temp_curves.items():
    plt.plot(range(len(temps)), temps, label=f"scale {scale}")
plt.xlabel("Time Step")
plt.ylabel("Kettle Temperature (°C)")
plt.title("Single-Run Temperature Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_kettle_plots/temperature_curves_compare.png")

# === Plot Training Stats (Std + CV) ===
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.bar([str(s) for s in scales], [train_stats[s]["std"] for s in scales])
plt.title("Std of Training Rewards")
plt.xlabel("Scale")
plt.ylabel("Standard Deviation")

plt.subplot(1,2,2)
plt.bar([str(s) for s in scales], [train_stats[s]["cv"] for s in scales])
plt.title("CV of Training Rewards")
plt.xlabel("Scale")
plt.ylabel("Coefficient of Variation")

plt.tight_layout()
plt.savefig("ppo_kettle_plots/training_std_cv.png")

# === Plot Evaluation Stats (Std + CV) ===
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.bar([str(s) for s in scales], [eval_stats[s]["std"] for s in scales])
plt.title("Std of Evaluation Rewards")
plt.xlabel("Scale")
plt.ylabel("Standard Deviation")

plt.subplot(1,2,2)
plt.bar([str(s) for s in scales], [eval_stats[s]["cv"] for s in scales])
plt.title("CV of Evaluation Rewards")
plt.xlabel("Scale")
plt.ylabel("Coefficient of Variation")

plt.tight_layout()
plt.savefig("ppo_kettle_plots/eval_std_cv.png")

print("✅ Models, statistics, and plots saved in 'ppo_kettle_models/' and 'ppo_kettle_plots/'")
