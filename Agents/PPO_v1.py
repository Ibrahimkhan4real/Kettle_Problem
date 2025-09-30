import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from kettle_dynamic_env_v26 import KettleEnv
import gymnasium as gym
import torch
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PureEconomicCallback(BaseCallback):
    """Callback focused on pure economic efficiency without shaping"""
    
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=30, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Metrics storage
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_energy_efficiency = []
        self.evaluations_goal_rate = []
        self.evaluations_energy_used = []
        self.evaluations_goal_timing = []
        self.evaluations_diversity = []
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Track best performance
        self.best_efficiency = 0.0
        self.best_energy_used = float('inf')
        self.episodes_evaluated = 0
        
    def _on_step(self) -> bool:
        # Collect episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()
            
        return True
    
    def _evaluate_agent(self):
        """Evaluate agent focusing on diversity and efficiency"""
        episode_rewards = []
        energy_efficiencies = []
        goal_achievements = []
        energy_used_list = []
        goal_steps = []
        
        for episode in range(self.n_eval_episodes):
            # Use different random seeds to encourage diversity
            obs = self.eval_env.reset(seed=self.episodes_evaluated + episode)[0]
            done = False
            episode_reward = 0
            
            while not done:
                # Add small noise during evaluation to check policy robustness
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = done or truncated
            
            episode_rewards.append(episode_reward)
            energy_efficiencies.append(info['energy_efficiency'])
            goal_achievements.append(info['goal_achieved'])
            energy_used_list.append(info['total_energy_Wh'])
            
            if info['goal_achieved']:
                goal_steps.append(info.get('first_goal_step', 250))
        
        self.episodes_evaluated += self.n_eval_episodes
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        mean_efficiency = np.mean(energy_efficiencies)
        goal_rate = np.mean(goal_achievements)
        mean_energy = np.mean(energy_used_list)
        mean_goal_timing = np.mean(goal_steps) if goal_steps else 250
        
        # Calculate diversity (standard deviation of key metrics)
        efficiency_std = np.std(energy_efficiencies)
        energy_std = np.std(energy_used_list)
        diversity_score = efficiency_std + energy_std
        
        # Store results
        self.evaluations_timesteps.append(self.n_calls)
        self.evaluations_results.append(mean_reward)
        self.evaluations_energy_efficiency.append(mean_efficiency)
        self.evaluations_goal_rate.append(goal_rate)
        self.evaluations_energy_used.append(mean_energy)
        self.evaluations_goal_timing.append(mean_goal_timing)
        self.evaluations_diversity.append(diversity_score)
        
        # Track best performance
        if mean_efficiency > self.best_efficiency and goal_rate > 0.8:
            self.best_efficiency = mean_efficiency
        if mean_energy < self.best_energy_used and goal_rate > 0.8:
            self.best_energy_used = mean_energy
        
        if self.verbose > 0:
            print(f"\nEcon Eval at {self.n_calls:,} steps:")
            print(f"  Goal rate: {goal_rate:.2%}")
            print(f"  Mean efficiency: {mean_efficiency:.4f} (best: {self.best_efficiency:.4f})")
            print(f"  Energy used: {mean_energy:.2f} Wh (best: {self.best_energy_used:.2f} Wh)")
            print(f"  Goal timing: {mean_goal_timing:.1f} steps")
            print(f"  Policy diversity: {diversity_score:.4f}")

class PureEconomicKettleTrainer:
    """Training system using pure economic incentives without reward shaping"""
    
    def __init__(self, env_kwargs=None):
        """Initialize with pure economic incentives - no reward shaping"""
        if env_kwargs is None:
            # PURE ECONOMIC APPROACH - disable reward shaping, use only base rewards
            env_kwargs = {
                'enable_reward_shaping': False,  # DISABLE reward shaping completely
                'gamma': 0.99,
                'goal_reward': 5000.0,             # Very high goal reward
                'energy_cost_per_kJ': 10.0,        # EXTREME energy cost (100x original)
                'efficiency_bonus_scale': 8000.0,  # MASSIVE efficiency bonus (40x original)
            }
        
        self.env_kwargs = env_kwargs
        self.model = None
        self.callback = None
        self.train_env = None
        self.eval_env = None
        
    def create_environments(self):
        """Create environments for pure economic learning"""
        # Use more environments with different seeds for diversity
        self.train_env = make_vec_env(
            lambda: Monitor(KettleEnv(**self.env_kwargs)), 
            n_envs=16,  # Even more parallel environments
            seed=None   # Different seeds for diversity
        )
        
        # Evaluation environment
        self.eval_env = KettleEnv(**self.env_kwargs)
        
    def create_agent(self, policy_kwargs=None):
        """Create PPO agent optimized for pure economic learning"""
        if policy_kwargs is None:
            # Larger networks with dropout for better generalization
            policy_kwargs = {
                'net_arch': [dict(pi=[1024, 512, 512, 256, 128], vf=[1024, 512, 512, 256, 128])],
                'activation_fn': torch.nn.ReLU,  # Back to ReLU for stability
            }
        
        # PPO hyperparameters optimized for economic incentive learning
        self.model = PPO(
            'MlpPolicy',
            self.train_env,
            learning_rate=1e-4,        # Lower learning rate for stability
            n_steps=4096,              # Long rollouts for temporal credit assignment
            batch_size=1024,           # Large batches for stable updates
            n_epochs=15,               # Moderate epochs to prevent overfitting
            gamma=0.999,               # Very high discount for long-term thinking
            gae_lambda=0.95,           # Standard GAE
            clip_range=0.15,           # Smaller clipping for stability
            ent_coef=0.15,             # Very high entropy for exploration
            vf_coef=0.5,               # Standard value function coefficient
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./pure_economic_ppo_logs/",
            device='auto'
        )
        
    def train(self, total_timesteps=3000000, eval_freq=10000, early_stopping_efficiency=0.92):
        """Train with pure economic incentives"""
        print("Starting pure economic incentive training...")
        print("REWARD SHAPING DISABLED - Using only base economic rewards")
        print(f"Target timesteps: {total_timesteps:,}")
        print(f"Evaluation frequency: {eval_freq:,}")
        
        # Environment info
        theoretical_min = self.eval_env.theoretical_min_energy / 3600
        print(f"Theoretical minimum energy: {theoretical_min:.2f} Wh")
        print(f"Energy cost per kJ: {self.env_kwargs['energy_cost_per_kJ']} (100x original)")
        print(f"Efficiency bonus scale: {self.env_kwargs['efficiency_bonus_scale']} (40x original)")
        print(f"Goal reward: {self.env_kwargs['goal_reward']}")
        
        # Create callback
        self.callback = PureEconomicCallback(
            self.eval_env, 
            eval_freq=eval_freq, 
            n_eval_episodes=50  # More episodes for better statistics
        )
        
        # Custom early stopping based on efficiency improvement
        class EconomicStoppingCallback(BaseCallback):
            def __init__(self, efficiency_threshold=0.92, patience=5):
                super().__init__()
                self.efficiency_threshold = efficiency_threshold
                self.patience = patience
                self.high_efficiency_count = 0
                self.no_improvement_count = 0
                self.last_best_efficiency = 0.0
                
            def _on_step(self) -> bool:
                if hasattr(self.parent, 'callback') and self.parent.callback.evaluations_energy_efficiency:
                    latest_efficiency = self.parent.callback.evaluations_energy_efficiency[-1]
                    latest_goal_rate = self.parent.callback.evaluations_goal_rate[-1]
                    
                    # Check for high efficiency
                    if latest_efficiency >= self.efficiency_threshold and latest_goal_rate >= 0.9:
                        self.high_efficiency_count += 1
                        if self.high_efficiency_count >= self.patience:
                            print(f"\nEarly stopping: Achieved {latest_efficiency:.4f} efficiency!")
                            return False
                    else:
                        self.high_efficiency_count = 0
                    
                    # Check for lack of improvement
                    if latest_efficiency <= self.last_best_efficiency + 0.001:
                        self.no_improvement_count += 1
                        if self.no_improvement_count >= 8:  # 8 evaluations without improvement
                            print(f"\nEarly stopping: No efficiency improvement for 8 evaluations")
                            return False
                    else:
                        self.no_improvement_count = 0
                        self.last_best_efficiency = max(self.last_best_efficiency, latest_efficiency)
                        
                return True
        
        economic_stopper = EconomicStoppingCallback(early_stopping_efficiency, patience=3)
        
        # Combined callbacks
        from stable_baselines3.common.callbacks import CallbackList
        callback_list = CallbackList([self.callback, economic_stopper])
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        print("Training completed!")
        if self.callback.evaluations_energy_efficiency:
            best_eff = max(self.callback.evaluations_energy_efficiency)
            print(f"Best efficiency achieved: {best_eff:.4f}")
            print(f"Best energy usage: {self.callback.best_energy_used:.2f} Wh")
        
    def evaluate_agent(self, n_episodes=200, render=False):
        """Evaluate agent with focus on diversity and efficiency"""
        print(f"Evaluating pure economic agent over {n_episodes} episodes...")
        
        results = {
            'episode_rewards': [],
            'goal_achieved': [],
            'energy_efficiency': [],
            'energy_used_Wh': [],
            'final_temperature': [],
            'goal_step': [],
            'temperature_profiles': [],
            'power_profiles': [],
            'steps_taken': [],
            'base_rewards': [],
            'goal_timing_error': []
        }
        
        theoretical_min = self.eval_env.theoretical_min_energy / 3600
        
        for episode in range(n_episodes):
            # Use different seeds to test policy robustness
            obs = self.eval_env.reset(seed=episode * 42)[0]
            done = False
            episode_reward = 0
            
            # Track episode data
            temps = [self.eval_env.current_temp]
            powers = [0]
            
            while not done:
                # Use stochastic policy for evaluation to test robustness
                action, _ = self.model.predict(obs, deterministic=(episode < 5))  # First 5 deterministic for rendering
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                # Store trajectory data
                temps.append(info['temperature'])
                powers.append(info['power_W'])
                
                if render and episode < 3:
                    self.eval_env.render()
            
            # Calculate additional metrics
            goal_timing_error = abs(info.get('first_goal_step', 250) - 200) if info['goal_achieved'] else 50
            
            # Store results
            results['episode_rewards'].append(episode_reward)
            results['goal_achieved'].append(info['goal_achieved'])
            results['energy_efficiency'].append(info['energy_efficiency'])
            results['energy_used_Wh'].append(info['total_energy_Wh'])
            results['final_temperature'].append(info['temperature'])
            results['goal_step'].append(info.get('first_goal_step', None))
            results['temperature_profiles'].append(temps)
            results['power_profiles'].append(powers)
            results['steps_taken'].append(info['current_step'])
            results['base_rewards'].append(info['base_reward'])
            results['goal_timing_error'].append(goal_timing_error)
        
        return results
    
    def analyze_results(self, results):
        """Analyze pure economic learning results"""
        print("\n" + "="*80)
        print("PURE ECONOMIC INCENTIVE ANALYSIS")
        print("="*80)
        
        theoretical_min = self.eval_env.theoretical_min_energy / 3600
        
        # Check for diversity (major issue in previous run)
        unique_efficiencies = len(set(np.round(results['energy_efficiency'], 4)))
        unique_energies = len(set(np.round(results['energy_used_Wh'], 2)))
        
        print(f"POLICY DIVERSITY CHECK:")
        print(f"Unique efficiency values: {unique_efficiencies}/{len(results['energy_efficiency'])}")
        print(f"Unique energy usage values: {unique_energies}/{len(results['energy_used_Wh'])}")
        print(f"Efficiency std dev: {np.std(results['energy_efficiency']):.6f}")
        print(f"Energy usage std dev: {np.std(results['energy_used_Wh']):.4f}")
        
        # Basic statistics
        print(f"\nBASIC PERFORMANCE:")
        print(f"Episodes: {len(results['episode_rewards'])}")
        print(f"Goal Achievement Rate: {np.mean(results['goal_achieved']):.2%}")
        print(f"Mean Energy Efficiency: {np.mean(results['energy_efficiency']):.4f} ± {np.std(results['energy_efficiency']):.4f}")
        print(f"Best Energy Efficiency: {np.max(results['energy_efficiency']):.4f}")
        print(f"Worst Energy Efficiency: {np.min(results['energy_efficiency']):.4f}")
        print(f"Mean Energy Used: {np.mean(results['energy_used_Wh']):.2f} Wh ± {np.std(results['energy_used_Wh']):.2f} Wh")
        
        # Successful episodes analysis
        successful_episodes = [i for i, achieved in enumerate(results['goal_achieved']) if achieved]
        if successful_episodes:
            successful_effs = [results['energy_efficiency'][i] for i in successful_episodes]
            successful_energies = [results['energy_used_Wh'][i] for i in successful_episodes]
            successful_timing = [results['goal_timing_error'][i] for i in successful_episodes]
            
            print(f"\nSUCCESSFUL EPISODES ANALYSIS:")
            print(f"Count: {len(successful_episodes)}")
            print(f"Mean Efficiency: {np.mean(successful_effs):.4f}")
            print(f"Best Efficiency: {np.max(successful_effs):.4f}")
            print(f"Mean Energy: {np.mean(successful_energies):.2f} Wh")
            print(f"Best Energy: {np.min(successful_energies):.2f} Wh")
            print(f"Mean Timing Error: {np.mean(successful_timing):.1f} steps")
            
            # High efficiency analysis
            high_eff_episodes = [i for i in successful_episodes if results['energy_efficiency'][i] >= 0.85]
            if high_eff_episodes:
                print(f"\nHIGH EFFICIENCY EPISODES (≥85%):")
                print(f"Count: {len(high_eff_episodes)}")
                high_eff_energies = [results['energy_used_Wh'][i] for i in high_eff_episodes]
                print(f"Mean Energy: {np.mean(high_eff_energies):.2f} Wh")
        
        # Energy comparison
        print(f"\nENERGY COMPARISON:")
        print(f"Theoretical Minimum: {theoretical_min:.2f} Wh")
        if successful_episodes:
            energy_waste = np.mean(successful_energies) - theoretical_min
            best_waste = np.min(successful_energies) - theoretical_min
            print(f"Average Energy Waste: {energy_waste:.2f} Wh ({energy_waste/theoretical_min*100:.1f}%)")
            print(f"Best Energy Waste: {best_waste:.2f} Wh ({best_waste/theoretical_min*100:.1f}%)")
        
        return results
    
    def plot_pure_economic_analysis(self, results):
        """Plot analysis focused on pure economic learning"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pure Economic Incentive Learning Analysis', fontsize=16, fontweight='bold')
        
        theoretical_min = self.eval_env.theoretical_min_energy / 3600
        
        # Plot 1: Energy Efficiency Distribution
        axes[0, 0].hist(results['energy_efficiency'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(np.mean(results['energy_efficiency']), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(results["energy_efficiency"]):.4f}')
        axes[0, 0].axvline(0.90, color='gold', linestyle=':', linewidth=2, label='90% Target')
        axes[0, 0].set_title('Energy Efficiency Distribution')
        axes[0, 0].set_xlabel('Energy Efficiency')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Energy Usage Distribution
        axes[0, 1].hist(results['energy_used_Wh'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 1].axvline(np.mean(results['energy_used_Wh']), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(results["energy_used_Wh"]):.2f} Wh')
        axes[0, 1].axvline(theoretical_min, color='orange', linestyle=':', linewidth=2, 
                          label=f'Theoretical Min: {theoretical_min:.2f} Wh')
        axes[0, 1].set_title('Energy Usage Distribution')
        axes[0, 1].set_xlabel('Energy Used (Wh)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Goal Timing Distribution
        successful_episodes = [i for i, achieved in enumerate(results['goal_achieved']) if achieved]
        if successful_episodes:
            successful_steps = [results['goal_step'][i] for i in successful_episodes if results['goal_step'][i] is not None]
            if successful_steps:
                axes[0, 2].hist(successful_steps, bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes[0, 2].axvline(200, color='red', linestyle='--', linewidth=2, label='Target: Step 200')
                axes[0, 2].axvspan(185, 215, alpha=0.2, color='green', label='Acceptable Range')
                axes[0, 2].set_title('Goal Achievement Timing')
                axes[0, 2].set_xlabel('Goal Achievement Step')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Efficiency vs Energy Scatter
        goal_colors = ['red' if not achieved else 'green' for achieved in results['goal_achieved']]
        axes[1, 0].scatter(results['energy_used_Wh'], results['energy_efficiency'], 
                          c=goal_colors, alpha=0.6, s=30)
        axes[1, 0].axhline(0.90, color='gold', linestyle='--', alpha=0.7, linewidth=2, label='90% Target')
        axes[1, 0].axvline(theoretical_min, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Theoretical Min')
        axes[1, 0].set_title('Energy Usage vs Efficiency')
        axes[1, 0].set_xlabel('Energy Used (Wh)')
        axes[1, 0].set_ylabel('Energy Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Efficiency over episodes (check for learning)
        episodes = range(len(results['energy_efficiency']))
        axes[1, 1].plot(episodes, results['energy_efficiency'], alpha=0.6, linewidth=1, color='green')
        window = min(20, len(results['energy_efficiency']) // 10)
        if window > 1:
            rolling_eff = pd.Series(results['energy_efficiency']).rolling(window=window).mean()
            axes[1, 1].plot(episodes, rolling_eff, color='red', linewidth=3, label=f'Rolling Avg ({window})')
            axes[1, 1].legend()
        axes[1, 1].axhline(0.90, color='gold', linestyle='--', alpha=0.7, label='90% Target')
        axes[1, 1].set_title('Efficiency Over Episodes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Energy Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Policy diversity visualization
        axes[1, 2].scatter(range(len(results['energy_efficiency'])), results['energy_efficiency'], 
                          alpha=0.6, s=20, c=goal_colors)
        axes[1, 2].set_title('Policy Diversity Check')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Energy Efficiency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Main execution function
def main():
    """Pure economic incentive training"""
    print("="*80)
    print("PURE ECONOMIC INCENTIVE PPO (NO REWARD SHAPING)")
    print("="*80)
    
    # Initialize trainer
    trainer = PureEconomicKettleTrainer()
    
    # Display configuration
    print("Configuration (NO REWARD SHAPING):")
    for key, value in trainer.env_kwargs.items():
        print(f"  {key}: {value}")
    
    # Setup
    trainer.create_environments()
    trainer.create_agent()
    
    # Display economic incentives
    theoretical_min = trainer.eval_env.theoretical_min_energy / 3600
    print(f"\nEconomic Structure:")
    print(f"  Theoretical minimum energy: {theoretical_min:.2f} Wh")
    print(f"  Energy cost: 10 per kJ (extreme)")
    print(f"  Goal reward: 5000 (high)")
    print(f"  Efficiency bonus: 8000 scale (massive)")
    print(f"  NO reward shaping - pure economic discovery")
    
    # Train with pure economic incentives
    trainer.train(
        total_timesteps=3000000,  # Longer training for economic discovery
        eval_freq=10000,
        early_stopping_efficiency=0.90
    )
    
    # Save model
    model_path = "pure_economic_ppo_model"
    trainer.model.save(model_path)
    print(f"Model saved as '{model_path}'")
    
    # Evaluate
    results = trainer.evaluate_agent(n_episodes=200, render=False)
    trainer.analyze_results(results)
    
    # Plot analysis
    trainer.plot_pure_economic_analysis(results)
    
    # Final summary
    successful_episodes = [i for i, achieved in enumerate(results['goal_achieved']) if achieved]
    if successful_episodes:
        best_efficiency = max([results['energy_efficiency'][i] for i in successful_episodes])
        mean_efficiency = np.mean([results['energy_efficiency'][i] for i in successful_episodes])
        efficiency_std = np.std([results['energy_efficiency'][i] for i in successful_episodes])
        
        print(f"\nFINAL PURE ECONOMIC SUMMARY:")
        print(f"="*50)
        print(f"Best efficiency: {best_efficiency:.4f}")
        print(f"Mean efficiency: {mean_efficiency:.4f} ± {efficiency_std:.4f}")
        print(f"Policy diversity: {efficiency_std > 0.001}")
        
        # Check if we broke the local optimum
        unique_efficiencies = len(set(np.round([results['energy_efficiency'][i] for i in successful_episodes], 4)))
        print(f"Unique strategies found: {unique_efficiencies}")
    
    print("\n" + "="*80)
    print("PURE ECONOMIC TRAINING COMPLETE!")
    print("="*80)
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()