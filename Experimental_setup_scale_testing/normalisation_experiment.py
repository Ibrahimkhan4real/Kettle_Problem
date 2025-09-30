import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Corrected import to match the provided environment file name
from Kettle_environment_updated_reward_fun import SimpleKettleEnv, SimpleKettleEnvUnscaled100, SimpleKettleEnvUnscaled10000

# --- Configuration ---
TOTAL_TIMESTEPS = 40_000  # Increased for more meaningful learning curves
EVAL_EPISODES = 100
LOG_DIR = "./ppo_kettle_tensorboard/"
MODEL_DIR = "ppo_kettle_models/"
PLOT_DIR = "plots/"
SMOOTHING_FACTOR = 0.9
AVG_EPISODE_LENGTH = 200

# --- PPO Hyperparameters ---
PPO_PARAMS = {
    "policy": "MlpPolicy",
    "ent_coef": 0.015,
    "verbose": 1,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.999,
    "tensorboard_log": LOG_DIR
}

class TrainingCallback(BaseCallback):
    """Tracks episode rewards and lengths during training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []
        
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        # Access the monitor wrapper to get episode statistics
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'episode_returns') and hasattr(env, 'episode_lengths'):
                    # Get completed episodes since last check
                    if len(env.episode_returns) > 0:
                        for r, l in zip(env.episode_returns, env.episode_lengths):
                            self.episode_rewards.append(r)
                            self.episode_lengths.append(l)
                            self.episode_timesteps.append(self.num_timesteps)
                        # Clear the monitor's buffers
                        env.episode_returns = []
                        env.episode_lengths = []
    
    def _on_step(self) -> bool:
        return True
    

        
        # Calculate rolling mean every few episodes
        window = 5
        timesteps = []
        mean_rewards = []
        
        for i in range(window, len(self.episode_rewards), 2):
            timesteps.append(self.episode_timesteps[i])
            mean_rewards.append(np.mean(self.episode_rewards[i-window:i]))
        
        return timesteps, mean_rewards

def train_agent(env_class, model_name, tensorboard_log_name):
    """Trains a PPO agent and returns the model path and reward history."""
    print(f"\n--- Training {tensorboard_log_name} Agent ---")
    
    # Wrap environment with Monitor to track episodes
    env = Monitor(env_class())
    
    model = PPO(env=env, **PPO_PARAMS)
    
    callback = TrainingCallback(verbose=1)
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=tensorboard_log_name, callback=callback)
    
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    
    # Get learning curve data
    timesteps, rewards = callback.get_learning_curve_data()
    
    
    print(f"Training complete. Collected {len(callback.episode_rewards)} episodes, {len(timesteps)} data points")
    
    if model.logger:
        model.logger.close()
    
    return model_path, timesteps, rewards

def evaluate_and_collect_data(model_path, env_class, n_episodes):
    """Evaluates a trained model and collects performance data."""
    print(f"--- Evaluating {model_path} ---")
    env = env_class()
    model = PPO.load(model_path, env=env)
    
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 1000:  # Safety limit
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
        episode_rewards.append(total_reward)
        
        if (ep + 1) % 20 == 0:
            print(f"  Evaluated {ep + 1}/{n_episodes} episodes")
    
    # Run one final deterministic episode for temperature profile
    obs, _ = env.reset()
    done = False
    temperature_profile = [obs[0]]
    steps = 0
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        temperature_profile.append(obs[0])
        steps += 1
    
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    return episode_rewards, temperature_profile

def plot_analytical_results(normalised_data, unscaled_100_data, unscaled_10000_data, plot_dir):
    """Generates and saves the boxplot and temp profile."""
    print("\n--- Generating Analytical Plots ---")
    normalised_rewards, normalised_temps = normalised_data
    unscaled_100_rewards, unscaled_100_temps = unscaled_100_data
    unscaled_10000_rewards, unscaled_10000_temps = unscaled_10000_data
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # PLOT 1: Box Plot
    plt.figure(figsize=(12, 7))
    data_to_plot = [
        normalised_rewards, 
        [r / 100.0 for r in unscaled_100_rewards],
        [r / 10000.0 for r in unscaled_10000_rewards]
    ]
    sns.boxplot(data=data_to_plot, palette="viridis", width=0.5)
    plt.xticks([0, 1, 2], ['Normalised', 'Unscaled (x100)', 'Unscaled (x10,000)'])
    plt.ylabel("Final Episode Reward (Normalised)")
    plt.title("Comparison of Final Performance Distribution Across Reward Scales")
    plot_path = os.path.join(plot_dir, "performance_boxplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to {plot_path}")
    plt.close()

    # PLOT 2: Temperature Profile
    plt.figure(figsize=(12, 7))
    plt.plot(normalised_temps, label="Normalised Agent", color='royalblue', linewidth=2.5)
    plt.plot(unscaled_100_temps, label="Unscaled Agent (x100)", color='coral', linestyle='--', linewidth=2)
    plt.plot(unscaled_10000_temps, label="Unscaled Agent (x10,000)", color='forestgreen', linestyle='-.', linewidth=2)
    plt.axhline(y=100, color='r', linestyle=':', label='Target Temperature (100°C)')
    plt.xlabel("Time (steps)")
    plt.ylabel("Kettle Temperature (°C)")
    plt.title("Learned Temperature Control Profile Comparison")
    plt.legend(frameon=True, loc='lower right')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plot_path = os.path.join(plot_dir, "temperature_profile.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Temperature plot saved to {plot_path}")
    plt.close()

def plot_learning_curves(normalised_history, unscaled_100_history, unscaled_10000_history, plot_dir):
    """Uses data from the custom callback to plot learning curves."""
    print("\n--- Generating Learning Curve Plot ---")

    runs = {
        "Normalised": {"history": normalised_history, "scale": 1.0, "color": "royalblue"},
        "Unscaled (x100)": {"history": unscaled_100_history, "scale": 100.0, "color": "coral"},
        "Unscaled (x10,000)": {"history": unscaled_10000_history, "scale": 10000.0, "color": "forestgreen"}
    }
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    has_data = False
    
    for label, info in runs.items():
        timesteps, rewards_raw = info["history"]
        
        print(f"  {label}: {len(timesteps)} data points")
        
        if not timesteps or not rewards_raw:
            print(f"    Warning: No data for {label}")
            continue
        
        has_data = True
        episodes = [t / AVG_EPISODE_LENGTH for t in timesteps]
        rewards_scaled = [r / info["scale"] for r in rewards_raw]
        
        # Direct plot without excessive smoothing
        plt.plot(episodes, rewards_scaled, label=label, color=info['color'], 
                linewidth=2, alpha=0.7, marker='o', markersize=4)
        
        # Add a smoothed line if we have enough points
        if len(episodes) > 3:
            df = pd.DataFrame({'episodes': episodes, 'rewards': rewards_scaled})
            df['smoothed'] = df['rewards'].rolling(window=min(3, len(episodes)//2), 
                                                   center=True, min_periods=1).mean()
            plt.plot(df['episodes'], df['smoothed'], color=info['color'], 
                    linewidth=3, alpha=0.9)

    if not has_data:
        # Create a dummy plot with text
        plt.text(0.5, 0.5, 'No training data collected\nTry increasing TOTAL_TIMESTEPS', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    plt.title("Mean Episode Reward During Training")
    plt.xlabel("Approximate Episodes")
    plt.ylabel("Mean Episode Reward (Normalised)")
    plt.legend(frameon=True, loc='lower right')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Learning curve plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Starting PPO Kettle Training Experiment")
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"PPO n_steps: {PPO_PARAMS['n_steps']}")
    print(f"Expected updates: {TOTAL_TIMESTEPS // PPO_PARAMS['n_steps']}")

    # --- Train All Agents ---
    normalised_path, norm_t, norm_r = train_agent(SimpleKettleEnv, "ppo_kettle_normalised", "Normalised")
    unscaled100_path, un100_t, un100_r = train_agent(SimpleKettleEnvUnscaled100, "ppo_kettle_unscaled_x100", "Unscaled_x100")
    unscaled10000_path, un10k_t, un10k_r = train_agent(SimpleKettleEnvUnscaled10000, "ppo_kettle_unscaled_x10000", "Unscaled_x10000")

    # Debug output
    print("\n--- Training Data Summary ---")
    print(f"Normalised: {len(norm_t)} points, rewards: {norm_r[:5] if norm_r else 'None'}")
    print(f"Unscaled x100: {len(un100_t)} points, rewards: {un100_r[:5] if un100_r else 'None'}")
    print(f"Unscaled x10000: {len(un10k_t)} points, rewards: {un10k_r[:5] if un10k_r else 'None'}")

    # --- Evaluate and Collect Data ---
    normalised_data = evaluate_and_collect_data(normalised_path, SimpleKettleEnv, EVAL_EPISODES)
    unscaled_100_data = evaluate_and_collect_data(unscaled100_path, SimpleKettleEnvUnscaled100, EVAL_EPISODES)
    unscaled_10000_data = evaluate_and_collect_data(unscaled10000_path, SimpleKettleEnvUnscaled10000, EVAL_EPISODES)
    
    # --- Generate and Save ALL Plots ---
    plot_analytical_results(normalised_data, unscaled_100_data, unscaled_10000_data, PLOT_DIR)
    plot_learning_curves((norm_t, norm_r), (un100_t, un100_r), (un10k_t, un10k_r), PLOT_DIR)

    print("\n--- Experiment Complete ---")
    print(f"\nALL PLOTS SAVED in the '{PLOT_DIR}' directory.")