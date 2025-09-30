import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

# Import your fixed environment
from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv  # Replace with your actual module name

class EpisodeRewardCallback(BaseCallback):
    """Callback to track episode rewards during training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            # Get episode info from the monitor
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_reward = self.locals['infos'][0]['episode']['r']
                episode_length = self.locals['infos'][0]['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
        return True

def make_monitored_env():
    """Create a monitored environment for proper episode tracking"""
    env = SimpleKettleEnv()
    env = Monitor(env)
    return env

def train_models(total_timesteps=50000):
    """Train both PPO and SAC models"""
    
    # Create environments with monitoring
    env_ppo = DummyVecEnv([make_monitored_env])
    env_sac = DummyVecEnv([make_monitored_env])
    
    # Initialize callbacks
    callback_ppo = EpisodeRewardCallback()
    callback_sac = EpisodeRewardCallback()
    
    print("Training PPO...")
    model_ppo = PPO('MlpPolicy', env_ppo, verbose=1)
    model_ppo.learn(total_timesteps=total_timesteps, callback=callback_ppo)
    
    print("\nTraining SAC...")
    model_sac = SAC('MlpPolicy', env_sac, verbose=1)
    model_sac.learn(total_timesteps=total_timesteps, callback=callback_sac)
    
    # Save models
    os.makedirs("models", exist_ok=True)
    model_ppo.save("models/ppo_kettle")
    model_sac.save("models/sac_kettle")
    
    return model_ppo, model_sac, callback_ppo.episode_rewards, callback_sac.episode_rewards

def plot_training_comparison(ppo_rewards, sac_rewards):
    """Create comprehensive training comparison plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Episode rewards over time
    ax1.plot(range(len(ppo_rewards)), ppo_rewards, label='PPO', alpha=0.7, color='blue')
    ax1.plot(range(len(sac_rewards)), sac_rewards, label='SAC', alpha=0.7, color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards Over Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Boxplot for stability comparison
    reward_data = [ppo_rewards, sac_rewards]
    ax2.boxplot(reward_data, labels=['PPO', 'SAC'])
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Distribution (Training Stability)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Standard deviation over training (bar graph)
    window_size = max(10, min(len(ppo_rewards), len(sac_rewards)) // 20)
    
    # Calculate rolling standard deviation
    ppo_rolling_std = pd.Series(ppo_rewards).rolling(window=window_size).std().fillna(0)
    sac_rolling_std = pd.Series(sac_rewards).rolling(window=window_size).std().fillna(0)
    
    # Create bar positions
    episodes_ppo = np.arange(len(ppo_rolling_std))
    episodes_sac = np.arange(len(sac_rolling_std))
    
    width = 0.4
    ax3.bar(episodes_ppo - width/2, ppo_rolling_std, width=width, alpha=0.7, label='PPO', color='blue')
    ax3.bar(episodes_sac + width/2, sac_rolling_std, width=width, alpha=0.7, label='SAC', color='red')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title(f'Rolling Standard Deviation (Window={window_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Moving average for clearer trends
    ppo_ma = pd.Series(ppo_rewards).rolling(window=window_size).mean()
    sac_ma = pd.Series(sac_rewards).rolling(window=window_size).mean()
    
    ax4.plot(range(len(ppo_ma)), ppo_ma, label='PPO (Moving Avg)', color='blue', linewidth=2)
    ax4.plot(range(len(sac_ma)), sac_ma, label='SAC (Moving Avg)', color='red', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.set_title(f'Moving Average Rewards (Window={window_size})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_temperature_control(model_ppo, model_sac):
    """Evaluate both models and plot temperature control performance"""
    
    results = {}
    
    for name, model in [('PPO', model_ppo), ('SAC', model_sac)]:
        # Create fresh environment for evaluation
        eval_env = SimpleKettleEnv()
        
        obs, _ = eval_env.reset()
        temperatures = [eval_env.temp]
        actions = []
        rewards = []
        
        total_reward = 0
        
        print(f"\nEvaluating {name}...")
        for step in range(eval_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action[0])
            obs, reward, done, _, info = eval_env.step(action)
            temperatures.append(eval_env.temp)
            rewards.append(reward)
            total_reward += reward
            
            if step < 5 or step % 50 == 0:  # Debug output
                print(f"  Step {step}: Action={action[0]:.1f}, Temp={eval_env.temp:.1f}, Reward={reward:.4f}")
            
            if done:
                print(f"  {name} episode ended at step {step}")
                break
        
        results[name] = {
            'temperatures': temperatures,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'final_temp': temperatures[-1],
            'steps': len(actions)
        }
        
        print(f"  {name} completed {len(actions)} steps, final temp: {temperatures[-1]:.1f}°C")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Temperature profiles
    for name, data in results.items():
        timesteps = range(len(data['temperatures']))
        color = 'blue' if name == 'PPO' else 'red'
        ax1.plot(timesteps, data['temperatures'], label=f'{name} (Steps: {data["steps"]})', linewidth=2, color=color)
    
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.8, label='Target Temperature')
    ax1.axvline(x=200, color='orange', linestyle='--', alpha=0.8, label='Target Time')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Control Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Control actions - ensure both have same length for comparison
    max_steps = max(len(data['actions']) for data in results.values())
    for name, data in results.items():
        timesteps = range(len(data['actions']))
        color = 'blue' if name == 'PPO' else 'red'
        ax2.plot(timesteps, data['actions'], label=f'{name} Actions', linewidth=2, color=color)
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Control Actions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_steps)
    
    plt.tight_layout()
    plt.savefig('results/temperature_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def print_training_statistics(ppo_rewards, sac_rewards):
    """Print comprehensive training statistics"""
    
    print("\n" + "="*60)
    print("TRAINING STATISTICS COMPARISON")
    print("="*60)
    
    stats = {}
    for name, rewards in [('PPO', ppo_rewards), ('SAC', sac_rewards)]:
        stats[name] = {
            'episodes': len(rewards),
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'final_10_avg': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        }
    
    for name, s in stats.items():
        print(f"\n{name} Results:")
        print(f"  Episodes: {s['episodes']}")
        print(f"  Mean Reward: {s['mean']:.4f}")
        print(f"  Std Reward: {s['std']:.4f}")
        print(f"  Min Reward: {s['min']:.4f}")
        print(f"  Max Reward: {s['max']:.4f}")
        print(f"  Final 10 Episodes Avg: {s['final_10_avg']:.4f}")
    
    print("="*60)
    
    return stats

def print_evaluation_results(eval_results):
    """Print evaluation results"""
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for name, results in eval_results.items():
        print(f"\n{name} Evaluation:")
        print(f"  Total Reward: {results['total_reward']:.4f}")
        print(f"  Final Temperature: {results['final_temp']:.2f}°C")
        print(f"  Steps Completed: {results['steps']}")
        print(f"  Target Achievement: {abs(results['final_temp'] - 100):.2f}°C from target")
    
    print("="*60)

def main():
    """Main execution function"""
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    print("Starting PPO vs SAC comparison on Kettle Environment")
    print("="*60)
    
    # Train models
    model_ppo, model_sac, ppo_rewards, sac_rewards = train_models(total_timesteps=5000)
    
    # Print training statistics
    training_stats = print_training_statistics(ppo_rewards, sac_rewards)
    
    # Plot training comparison
    print("\nGenerating training comparison plots...")
    plot_training_comparison(ppo_rewards, sac_rewards)
    
    # Evaluate models
    print("Evaluating trained models...")
    eval_results = evaluate_temperature_control(model_ppo, model_sac)
    
    # Print evaluation results
    print_evaluation_results(eval_results)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- results/training_comparison.png")
    print("- results/temperature_evaluation.png")
    print("- models/ppo_kettle.zip")
    print("- models/sac_kettle.zip")

if __name__ == "__main__":
    main()