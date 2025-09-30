"""
Main Experiment Runner for MCTS Kettle Control

This script orchestrates the complete experimental pipeline:
1. Run MCTS episodes with different configurations
2. Save detailed results (CSV files)
3. Generate comprehensive visualizations
4. Create analysis reports
5. Support multiple runs with result aggregation

Usage:
    python main_experiment_runner.py --episodes 5 --iterations 3000
    python main_experiment_runner.py --config quick_test
    python main_experiment_runner.py --config full_experiment
"""

import argparse
import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from Kettle_env_v1 import KettleEnv
from MCTS_RACE_v1 import KettleMCTS, run_mcts_episode, run_mcts_episode_with_shaping, save_episode_results
from visualise import visualize_results

# Experiment configurations
EXPERIMENT_CONFIGS = {
    'quick_test': {
        'episodes': 1,
        'iterations_per_action': 1000,
        'description': 'Quick test with 3 episodes and 1000 iterations per action'
    },
    'medium_test': {
        'episodes': 1,
        'iterations_per_action': 3000,
        'description': 'Medium test with 5 episodes and 3000 iterations per action'
    },
    'full_experiment': {
        'episodes': 1,
        'iterations_per_action': 5000,
        'description': 'Full experiment with 10 episodes and 5000 iterations per action'
    },
    'deep_search': {
        'episodes': 1,
        'iterations_per_action': 10000,
        'description': 'Deep search with 5 episodes and 10000 iterations per action'
    },
    'shaping_comparison': {
        'episodes': 2,
        'iterations_per_action': 3000,
        'description': 'Compare reward shaping vs no shaping (1 episodes each)'
    }
}


class ExperimentRunner:
    """Main class for running MCTS kettle control experiments"""
    
    def __init__(self, base_results_dir="experiments"):
        self.base_results_dir = base_results_dir
        self.experiment_start_time = None
        self.results_summary = []
    
    def create_experiment_directory(self, config_name, custom_name=None):
        """Create directory for experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if custom_name:
            exp_name = f"{custom_name}_{timestamp}"
        else:
            exp_name = f"{config_name}_{timestamp}"
        
        exp_dir = os.path.join(self.base_results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(exp_dir, "individual_episodes"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "analysis"), exist_ok=True)
        
        return exp_dir
    
    def save_experiment_config(self, exp_dir, config, args):
        """Save experiment configuration"""
        config_data = {
            'experiment_config': config,
            'command_line_args': vars(args),
            'timestamp': datetime.now().isoformat(),
            'environment_params': {
                'target_temp': 100.0,
                'target_deadline_step': 200,
                'temp_tolerance': 2.5,
                'time_tolerance': 15,
                'max_steps': 250
            }
        }
        
        config_file = os.path.join(exp_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def run_single_episode(self, episode_num, iterations_per_action, exp_dir, enable_shaping=True):
        """Run a single MCTS episode"""
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_num}")
        print(f"Reward Shaping: {'ENABLED' if enable_shaping else 'DISABLED'}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run episode with shaping configuration
        episode_data, iteration_stats, action_stats = run_mcts_episode_with_shaping(
            iterations_per_action=iterations_per_action,
            enable_reward_shaping=enable_shaping,
            save_results=False  # We'll save manually
        )
        
        episode_time = time.time() - start_time
        
        # Save episode results
        episode_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = os.path.join(exp_dir, "individual_episodes", f"episode_{episode_num:02d}_{episode_timestamp}")
        os.makedirs(episode_dir, exist_ok=True)
        
        save_episode_results(episode_data, iteration_stats, action_stats, episode_timestamp)
        
        # Move saved results to episode directory
        temp_results_dir = f"results/mcts_kettle_{episode_timestamp}"
        if os.path.exists(temp_results_dir):
            import shutil
            for file in os.listdir(temp_results_dir):
                shutil.move(os.path.join(temp_results_dir, file), 
                          os.path.join(episode_dir, file))
            os.rmdir(temp_results_dir)
            if os.path.exists("results") and not os.listdir("results"):
                os.rmdir("results")
        
        # Create visualizations for this episode
        visualize_results(episode_data, iteration_stats, action_stats, episode_dir)
        
        # Extract key metrics
        df = pd.DataFrame(episode_data)
        episode_summary = {
            'episode': episode_num,
            'duration_seconds': episode_time,
            'goal_achieved': df['goal_achieved'].iloc[-1],
            'final_temperature': df['temperatures'].iloc[-1],
            'final_energy_Wh': df['energy_consumed_Wh'].iloc[-1],
            'energy_efficiency': df['energy_efficiency'].iloc[-1],
            'total_reward': df['cumulative_rewards'].iloc[-1],
            'steps_taken': len(df) - 1,
            'episode_dir': episode_dir
        }
        
        # Add strategy metrics
        actions = [a for a in df['actions'] if a is not None]
        if actions:
            heating_actions = sum(1 for i in range(1, len(df)) if df['powers'].iloc[i] > 0)
            episode_summary['heating_actions'] = heating_actions
            episode_summary['heating_percentage'] = 100 * heating_actions / len(actions)
            
            # Find first heating
            first_heat = None
            for i in range(1, len(df)):
                if df['powers'].iloc[i] > 0:
                    first_heat = i
                    break
            episode_summary['first_heating_step'] = first_heat
            episode_summary['heating_delay'] = first_heat if first_heat else 0
        
        # MCTS performance
        if iteration_stats:
            iter_df = pd.DataFrame(iteration_stats)
            episode_summary['avg_goals_found'] = iter_df['goal_found_count'].mean()
            episode_summary['max_goals_found'] = iter_df['goal_found_count'].max()
            episode_summary['avg_rollout_reward'] = iter_df['avg_reward'].mean()
        
        print(f"\nEpisode {episode_num} Summary:")
        print(f"  Goal achieved: {'YES' if episode_summary['goal_achieved'] else 'NO'}")
        print(f"  Final temperature: {episode_summary['final_temperature']:.1f}°C")
        print(f"  Energy efficiency: {episode_summary['energy_efficiency']:.1%}")
        print(f"  Duration: {episode_time:.1f} seconds")
        print(f"  Results saved to: {episode_dir}")
        
        return episode_summary
    
    def run_experiment(self, config_name, episodes=None, iterations_per_action=None, 
                      custom_name=None, args=None):
        """Run complete experiment with multiple episodes"""
        
        self.experiment_start_time = time.time()
        
        # Get configuration
        if config_name in EXPERIMENT_CONFIGS:
            config = EXPERIMENT_CONFIGS[config_name].copy()
        else:
            config = {'episodes': 5, 'iterations_per_action': 3000, 'description': 'Custom configuration'}
        
        # Override with provided parameters
        if episodes is not None:
            config['episodes'] = episodes
        if iterations_per_action is not None:
            config['iterations_per_action'] = iterations_per_action
        
        # Reward shaping configuration
        enable_shaping = getattr(args, 'reward_shaping', True) if args else True
        compare_shaping = (config_name == 'shaping_comparison') if args else False
        
        print("="*80)
        print("MCTS KETTLE CONTROL EXPERIMENT")
        print("="*80)
        print(f"Configuration: {config_name}")
        print(f"Description: {config.get('description', 'No description')}")
        print(f"Episodes: {config['episodes']}")
        print(f"MCTS iterations per action: {config['iterations_per_action']}")
        
        if compare_shaping:
            print(f"Reward shaping comparison: {config['episodes']//2} episodes each (ON/OFF)")
        else:
            print(f"Reward shaping: {'ENABLED' if enable_shaping else 'DISABLED'}")
            
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Create experiment directory
        exp_dir = self.create_experiment_directory(config_name, custom_name)
        self.save_experiment_config(exp_dir, config, args)
        
        print(f"Experiment directory: {exp_dir}")
        
        # Run episodes
        episode_summaries = []
        
        for episode_num in range(1, config['episodes'] + 1):
            try:
                # Determine shaping for this episode
                if compare_shaping:
                    episode_shaping = episode_num <= config['episodes'] // 2
                else:
                    episode_shaping = enable_shaping
                
                episode_summary = self.run_single_episode(
                    episode_num, 
                    config['iterations_per_action'], 
                    exp_dir,
                    enable_shaping=episode_shaping
                )
                episode_summary['shaping_enabled'] = episode_shaping
                episode_summaries.append(episode_summary)
                
                # Save intermediate results
                self.save_experiment_summary(exp_dir, episode_summaries, config, partial=True)
                
            except Exception as e:
                print(f"Error in episode {episode_num}: {str(e)}")
                print("Continuing with next episode...")
                continue
        
        # Create aggregate analysis
        self.create_aggregate_analysis(exp_dir, episode_summaries, config)
        
        # Save final summary
        self.save_experiment_summary(exp_dir, episode_summaries, config, partial=False)
        
        experiment_time = time.time() - self.experiment_start_time
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED")
        print(f"Total time: {experiment_time:.1f} seconds ({experiment_time/60:.1f} minutes)")
        print(f"Results saved to: {exp_dir}")
        print(f"{'='*80}")
        
        return exp_dir, episode_summaries
    
    def save_experiment_summary(self, exp_dir, episode_summaries, config, partial=False):
        """Save experiment summary to CSV and JSON"""
        
        if not episode_summaries:
            return
        
        # Save episode summaries as CSV
        df_summary = pd.DataFrame(episode_summaries)
        summary_file = os.path.join(exp_dir, "experiment_summary.csv")
        df_summary.to_csv(summary_file, index=False)
        
        # Calculate aggregate statistics
        successful_episodes = df_summary[df_summary['goal_achieved'] == True]
        
        aggregate_stats = {
            'experiment_config': config,
            'total_episodes': len(episode_summaries),
            'successful_episodes': len(successful_episodes),
            'success_rate': len(successful_episodes) / len(episode_summaries) if episode_summaries else 0,
            'completed': not partial,
            'experiment_duration_seconds': time.time() - self.experiment_start_time if self.experiment_start_time else 0
        }
        
        if len(successful_episodes) > 0:
            aggregate_stats.update({
                'avg_energy_efficiency': successful_episodes['energy_efficiency'].mean(),
                'std_energy_efficiency': successful_episodes['energy_efficiency'].std(),
                'avg_energy_consumption_Wh': successful_episodes['final_energy_Wh'].mean(),
                'avg_heating_percentage': successful_episodes['heating_percentage'].mean(),
                'avg_first_heating_step': successful_episodes['first_heating_step'].mean(),
                'avg_total_reward': successful_episodes['total_reward'].mean()
            })
        
        if len(df_summary) > 0:
            aggregate_stats.update({
                'overall_avg_energy_efficiency': df_summary['energy_efficiency'].mean(),
                'overall_success_temp_achievement': sum(abs(df_summary['final_temperature'] - 100) <= 2.5),
                'avg_episode_duration_seconds': df_summary['duration_seconds'].mean()
            })
        
        # Save aggregate statistics
        stats_file = os.path.join(exp_dir, "aggregate_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(aggregate_stats, f, indent=2)
        
        # Print summary
        if not partial:
            print(f"\nEXPERIMENT SUMMARY:")
            print(f"Success rate: {aggregate_stats['success_rate']:.1%} ({aggregate_stats['successful_episodes']}/{aggregate_stats['total_episodes']})")
            if len(successful_episodes) > 0:
                print(f"Average energy efficiency (successful): {aggregate_stats['avg_energy_efficiency']:.1%}")
                print(f"Average first heating step: {aggregate_stats['avg_first_heating_step']:.1f}")
                print(f"Average heating percentage: {aggregate_stats['avg_heating_percentage']:.1f}%")
    
    def create_aggregate_analysis(self, exp_dir, episode_summaries, config):
        """Create aggregate analysis across all episodes"""
        
        if not episode_summaries:
            return
        
        # Create aggregate plots
        self.create_aggregate_plots(exp_dir, episode_summaries)
        
        # Create comparative analysis
        self.create_comparative_analysis(exp_dir, episode_summaries)
        
        # Generate aggregate report
        self.generate_aggregate_report(exp_dir, episode_summaries, config)
    
    def create_aggregate_plots(self, exp_dir, episode_summaries):
        """Create aggregate visualization across episodes"""
        
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(episode_summaries)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Success rate
        ax1 = axes[0, 0]
        success_counts = df['goal_achieved'].value_counts()
        colors = ['red', 'green']
        success_counts.plot(kind='pie', ax=ax1, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Goal Achievement Rate', fontweight='bold')
        ax1.set_ylabel('')
        
        # Energy efficiency distribution
        ax2 = axes[0, 1]
        ax2.hist(df['energy_efficiency'] * 100, bins=10, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=90, color='green', linestyle='--', label='Excellent (90%)')
        ax2.axvline(x=80, color='orange', linestyle='--', label='Good (80%)')
        ax2.set_xlabel('Energy Efficiency (%)')
        ax2.set_ylabel('Number of Episodes')
        ax2.set_title('Energy Efficiency Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # First heating step distribution
        ax3 = axes[0, 2]
        if 'first_heating_step' in df.columns:
            first_heat_clean = df['first_heating_step'].dropna()
            if len(first_heat_clean) > 0:
                ax3.hist(first_heat_clean, bins=10, alpha=0.7, color='red', edgecolor='black')
                ax3.axvline(x=88, color='green', linestyle='--', label='Theoretical Optimal (~88)')
                ax3.axvline(x=150, color='orange', linestyle=':', label='Good Delay (150+)')
                ax3.set_xlabel('First Heating Step')
                ax3.set_ylabel('Number of Episodes')
                ax3.set_title('Heating Delay Strategy', fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Episode performance over time
        ax4 = axes[1, 0]
        episodes = df['episode']
        ax4.plot(episodes, df['energy_efficiency'] * 100, 'bo-', label='Energy Efficiency')
        ax4.axhline(y=90, color='green', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Episode Number')
        ax4.set_ylabel('Energy Efficiency (%)')
        ax4.set_title('Learning Progress', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Energy vs Temperature achievement
        ax5 = axes[1, 1]
        temp_error = abs(df['final_temperature'] - 100)
        colors = ['green' if achieved else 'red' for achieved in df['goal_achieved']]
        scatter = ax5.scatter(df['final_energy_Wh'], temp_error, c=colors, alpha=0.7, s=100)
        ax5.axhline(y=2.5, color='orange', linestyle='--', label='Temp Tolerance')
        ax5.set_xlabel('Energy Consumption (Wh)')
        ax5.set_ylabel('Temperature Error (°C)')
        ax5.set_title('Energy vs Accuracy Trade-off', fontweight='bold')
        ax5.legend(['Failed', 'Succeeded'], loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # MCTS performance metrics
        ax6 = axes[1, 2]
        if 'avg_goals_found' in df.columns:
            goals_clean = df['avg_goals_found'].dropna()
            rewards_clean = df['avg_rollout_reward'].dropna()
            
            if len(goals_clean) > 0 and len(rewards_clean) > 0:
                ax6_twin = ax6.twinx()
                
                line1 = ax6.plot(episodes[:len(goals_clean)], goals_clean, 'b-o', label='Avg Goals Found')
                line2 = ax6_twin.plot(episodes[:len(rewards_clean)], rewards_clean, 'r-s', label='Avg Rollout Reward')
                
                ax6.set_xlabel('Episode Number')
                ax6.set_ylabel('Average Goals Found', color='blue')
                ax6_twin.set_ylabel('Average Rollout Reward', color='red')
                ax6.set_title('MCTS Search Quality', fontweight='bold')
                
                # Combine legends
                lines1, labels1 = ax6.get_legend_handles_labels()
                lines2, labels2 = ax6_twin.get_legend_handles_labels()
                ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(exp_dir, "plots", "aggregate_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Aggregate plots saved to: {plot_file}")
    
    def create_comparative_analysis(self, exp_dir, episode_summaries):
        """Create comparative analysis between successful and failed episodes"""
        
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(episode_summaries)
        successful = df[df['goal_achieved'] == True]
        failed = df[df['goal_achieved'] == False]
        
        if len(successful) == 0 or len(failed) == 0:
            print("Cannot create comparative analysis: need both successful and failed episodes")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Energy efficiency comparison
        ax1 = axes[0, 0]
        data = [successful['energy_efficiency'] * 100, failed['energy_efficiency'] * 100]
        labels = ['Successful', 'Failed']
        box_plot = ax1.boxplot(data, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('green')
        box_plot['boxes'][1].set_facecolor('red')
        ax1.set_ylabel('Energy Efficiency (%)')
        ax1.set_title('Energy Efficiency: Success vs Failure', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # First heating step comparison
        ax2 = axes[0, 1]
        if 'first_heating_step' in df.columns:
            succ_heat = successful['first_heating_step'].dropna()
            fail_heat = failed['first_heating_step'].dropna()
            
            if len(succ_heat) > 0 and len(fail_heat) > 0:
                data = [succ_heat, fail_heat]
                box_plot = ax2.boxplot(data, labels=labels, patch_artist=True)
                box_plot['boxes'][0].set_facecolor('green')
                box_plot['boxes'][1].set_facecolor('red')
                ax2.axhline(y=88, color='orange', linestyle='--', label='Optimal (~88)')
                ax2.set_ylabel('First Heating Step')
                ax2.set_title('Heating Strategy: Success vs Failure', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Energy consumption comparison
        ax3 = axes[1, 0]
        data = [successful['final_energy_Wh'], failed['final_energy_Wh']]
        box_plot = ax3.boxplot(data, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('green')
        box_plot['boxes'][1].set_facecolor('red')
        
        theoretical_min = 1.0 * 4184 * 80 / 3600
        ax3.axhline(y=theoretical_min, color='blue', linestyle='--', 
                   label=f'Theoretical Min ({theoretical_min:.1f}Wh)')
        ax3.set_ylabel('Energy Consumption (Wh)')
        ax3.set_title('Energy Usage: Success vs Failure', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # MCTS performance comparison
        ax4 = axes[1, 1]
        if 'avg_goals_found' in df.columns:
            succ_goals = successful['avg_goals_found'].dropna()
            fail_goals = failed['avg_goals_found'].dropna()
            
            if len(succ_goals) > 0 and len(fail_goals) > 0:
                data = [succ_goals, fail_goals]
                box_plot = ax4.boxplot(data, labels=labels, patch_artist=True)
                box_plot['boxes'][0].set_facecolor('green')
                box_plot['boxes'][1].set_facecolor('red')
                ax4.set_ylabel('Average Goals Found by MCTS')
                ax4.set_title('MCTS Performance: Success vs Failure', fontweight='bold')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(exp_dir, "plots", "comparative_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparative analysis saved to: {plot_file}")
    
    def generate_aggregate_report(self, exp_dir, episode_summaries, config):
        """Generate comprehensive aggregate report"""
        
        df = pd.DataFrame(episode_summaries)
        
        report_file = os.path.join(exp_dir, "analysis", "aggregate_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("MCTS KETTLE CONTROL - AGGREGATE EXPERIMENT REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Experiment overview
            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 25 + "\n")
            f.write(f"Configuration: {config}\n")
            f.write(f"Total episodes: {len(episode_summaries)}\n")
            f.write(f"Experiment duration: {(time.time() - self.experiment_start_time)/60:.1f} minutes\n")
            f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Success analysis
            successful = df[df['goal_achieved'] == True]
            failed = df[df['goal_achieved'] == False]
            
            f.write("SUCCESS ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Successful episodes: {len(successful)}/{len(df)} ({len(successful)/len(df):.1%})\n")
            f.write(f"Failed episodes: {len(failed)}\n\n")
            
            if len(successful) > 0:
                f.write("SUCCESSFUL EPISODES ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average energy efficiency: {successful['energy_efficiency'].mean():.1%} ± {successful['energy_efficiency'].std():.1%}\n")
                f.write(f"Average energy consumption: {successful['final_energy_Wh'].mean():.1f} ± {successful['final_energy_Wh'].std():.1f} Wh\n")
                f.write(f"Energy efficiency range: {successful['energy_efficiency'].min():.1%} - {successful['energy_efficiency'].max():.1%}\n")
                
                if 'first_heating_step' in successful.columns:
                    first_heat_clean = successful['first_heating_step'].dropna()
                    if len(first_heat_clean) > 0:
                        f.write(f"Average first heating step: {first_heat_clean.mean():.1f} ± {first_heat_clean.std():.1f}\n")
                        f.write(f"First heating range: {first_heat_clean.min():.0f} - {first_heat_clean.max():.0f}\n")
                        
                        # Strategy assessment
                        delayed_heating = sum(first_heat_clean > 150)
                        optimal_heating = sum(first_heat_clean > 80)
                        
                        f.write(f"Episodes with delayed heating (>150): {delayed_heating}/{len(first_heat_clean)}\n")
                        f.write(f"Episodes with good timing (>80): {optimal_heating}/{len(first_heat_clean)}\n")
                
                f.write(f"Average total reward: {successful['total_reward'].mean():.1f}\n\n")
            
            # Overall performance metrics
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average energy efficiency (all): {df['energy_efficiency'].mean():.1%}\n")
            f.write(f"Average final temperature: {df['final_temperature'].mean():.1f}°C\n")
            f.write(f"Temperature within tolerance: {sum(abs(df['final_temperature'] - 100) <= 2.5)}/{len(df)}\n")
            f.write(f"Average energy consumption: {df['final_energy_Wh'].mean():.1f} Wh\n")
            
            theoretical_min = 1.0 * 4184 * 80 / 3600
            f.write(f"Theoretical minimum energy: {theoretical_min:.1f} Wh\n")
            f.write(f"Average excess energy: {df['final_energy_Wh'].mean() - theoretical_min:.1f} Wh\n\n")
            
            # MCTS performance
            if 'avg_goals_found' in df.columns:
                f.write("MCTS PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                goals_clean = df['avg_goals_found'].dropna()
                rewards_clean = df['avg_rollout_reward'].dropna()
                
                if len(goals_clean) > 0:
                    f.write(f"Average goals found per episode: {goals_clean.mean():.1f} ± {goals_clean.std():.1f}\n")
                    f.write(f"Maximum goals found: {goals_clean.max():.0f}\n")
                    f.write(f"Episodes with >100 goals found: {sum(goals_clean > 100)}/{len(goals_clean)}\n")
                
                if len(rewards_clean) > 0:
                    f.write(f"Average rollout reward: {rewards_clean.mean():.1f} ± {rewards_clean.std():.1f}\n")
                
                f.write("\n")
            
            # Strategy analysis
            f.write("STRATEGY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            if 'heating_percentage' in df.columns:
                heating_clean = df['heating_percentage'].dropna()
                if len(heating_clean) > 0:
                    f.write(f"Average heating percentage: {heating_clean.mean():.1f}% ± {heating_clean.std():.1f}%\n")
                    
                    efficient_episodes = sum((heating_clean < 50) & (df['goal_achieved'] == True))
                    f.write(f"Efficient episodes (<50% heating): {efficient_episodes}/{len(df)}\n")
            
            # Learning progression
            if len(df) >= 3:
                f.write("\nLEARNING PROGRESSION\n")
                f.write("-" * 25 + "\n")
                
                early_episodes = df.iloc[:len(df)//3]
                late_episodes = df.iloc[2*len(df)//3:]
                
                early_success = early_episodes['goal_achieved'].mean()
                late_success = late_episodes['goal_achieved'].mean()
                
                f.write(f"Early success rate: {early_success:.1%}\n")
                f.write(f"Late success rate: {late_success:.1%}\n")
                
                if late_success > early_success:
                    f.write("✓ SUCCESS RATE IMPROVED OVER TIME\n")
                else:
                    f.write("⚠ NO CLEAR IMPROVEMENT IN SUCCESS RATE\n")
                
                early_efficiency = early_episodes['energy_efficiency'].mean()
                late_efficiency = late_episodes['energy_efficiency'].mean()
                
                f.write(f"Early efficiency: {early_efficiency:.1%}\n")
                f.write(f"Late efficiency: {late_efficiency:.1%}\n")
                
                if late_efficiency > early_efficiency * 1.05:
                    f.write("✓ ENERGY EFFICIENCY IMPROVED\n")
                else:
                    f.write("⚠ NO CLEAR EFFICIENCY IMPROVEMENT\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            success_rate = len(successful) / len(df)
            if success_rate >= 0.8:
                f.write("✓ EXCELLENT SUCCESS RATE (≥80%)\n")
            elif success_rate >= 0.6:
                f.write("✓ GOOD SUCCESS RATE (≥60%)\n")
            else:
                f.write("⚠ LOW SUCCESS RATE (<60%)\n")
                f.write("• Consider increasing MCTS iterations per action\n")
                f.write("• Check if environment parameters are appropriate\n")
            
            if len(successful) > 0:
                avg_efficiency = successful['energy_efficiency'].mean()
                if avg_efficiency >= 0.9:
                    f.write("✓ EXCELLENT ENERGY EFFICIENCY (≥90%)\n")
                elif avg_efficiency >= 0.8:
                    f.write("✓ GOOD ENERGY EFFICIENCY (≥80%)\n")
                else:
                    f.write("⚠ MODERATE ENERGY EFFICIENCY (<80%)\n")
                    f.write("• MCTS should learn to delay heating longer\n")
                    f.write("• Consider adjusting reward structure for energy efficiency\n")
            
            f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Aggregate report saved to: {report_file}")


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='MCTS Kettle Control Experiment Runner')
    
    # Configuration options
    parser.add_argument('--config', type=str, default='medium_test',
                       choices=list(EXPERIMENT_CONFIGS.keys()),
                       help='Predefined experiment configuration')
    
    # Override options
    parser.add_argument('--episodes', type=int, help='Number of episodes to run')
    parser.add_argument('--iterations', type=int, help='MCTS iterations per action')
    parser.add_argument('--name', type=str, help='Custom experiment name')
    
    # Reward shaping options
    parser.add_argument('--no-reward-shaping', dest='reward_shaping', action='store_false',
                       help='Disable reward shaping (default: enabled)')
    parser.add_argument('--reward-shaping', dest='reward_shaping', action='store_true', 
                       default=True, help='Enable reward shaping (default)')
    
    # Other options
    parser.add_argument('--results-dir', type=str, default='experiments',
                       help='Base directory for results')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations and exit')
    
    args = parser.parse_args()
    
    # List configurations
    if args.list_configs:
        print("Available experiment configurations:")
        for name, config in EXPERIMENT_CONFIGS.items():
            print(f"  {name}: {config['description']}")
        return
    
    # Create experiment runner
    runner = ExperimentRunner(base_results_dir=args.results_dir)
    
    # Run experiment
    try:
        exp_dir, summaries = runner.run_experiment(
            config_name=args.config,
            episodes=args.episodes,
            iterations_per_action=args.iterations,
            custom_name=args.name,
            args=args
        )
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"Results available at: {exp_dir}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()