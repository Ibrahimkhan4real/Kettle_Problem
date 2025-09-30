# utils/visualization_utils.py
"""
Visualization utilities for MCTS experiment results.
Generates comprehensive plots and charts for analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

from models.base_mcts import EpisodeResult

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class VisualizationUtils:
    """Utilities for creating visualizations of experiment results"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.plots_dir = self.base_dir / "visualizations"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.plots_dir / "comparative_plots").mkdir(exist_ok=True)
        (self.plots_dir / "individual_analysis").mkdir(exist_ok=True)
        (self.plots_dir / "dashboard").mkdir(exist_ok=True)
        
        # Visualization settings
        self.colors = {
            'MCTS-RAVE': '#1f77b4',
            'UCT': '#ff7f0e', 
            'MCTS-NoRAVE': '#2ca02c',
            'heating': '#d62728',
            'coasting': '#9467bd',
            'target': '#8c564b',
            'deadline': '#e377c2'
        }
        
        self.figsize_single = (12, 8)
        self.figsize_multi = (15, 10)
        self.dpi = 300
    
    def generate_experiment_plots(self, 
                                episode_results: List[EpisodeResult], 
                                output_dir: Path,
                                experiment_name: str):
        """Generate all plots for a single experiment"""
        
        if not episode_results:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Single episode detailed analysis (first successful episode or first episode)
        target_episode = self._find_representative_episode(episode_results)
        if target_episode:
            self.plot_detailed_episode(target_episode, output_dir / "detailed_episode.png")
        
        # Multi-episode overview
        self.plot_episode_overview(episode_results, output_dir / "episode_overview.png")
        
        # Performance metrics
        self.plot_performance_metrics(episode_results, output_dir / "performance_metrics.png")
        
        # Energy efficiency analysis
        self.plot_energy_analysis(episode_results, output_dir / "energy_analysis.png")
        
        # Control strategy analysis
        self.plot_control_strategy(episode_results, output_dir / "control_strategy.png")
        
        # MCTS statistics
        self.plot_mcts_statistics(episode_results, output_dir / "mcts_statistics.png")
        
        # Temperature trajectories comparison
        self.plot_temperature_trajectories(episode_results, output_dir / "temperature_trajectories.png")
        
        # Interactive dashboard
        self.create_interactive_dashboard(episode_results, output_dir / "interactive_dashboard.html", experiment_name)
    
    def plot_detailed_episode(self, episode_result: EpisodeResult, output_path: Path):
        """Create detailed plot for a single episode"""
        
        fig, axes = plt.subplots(5, 1, figsize=(15, 18))
        
        steps = range(len(episode_result.temperatures))
        action_steps = range(1, len(episode_result.actions) + 1)
        
        # 1. Temperature trajectory
        ax = axes[0]
        ax.plot(steps, episode_result.temperatures, 'b-', linewidth=2.5, label='Temperature')
        ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Target (100°C)')
        ax.axhline(y=97.5, color='r', linestyle=':', alpha=0.7, label='Tolerance')
        ax.axhline(y=102.5, color='r', linestyle=':', alpha=0.7)
        ax.axvline(x=200, color='m', linestyle='--', linewidth=2, label='Deadline')
        ax.fill_between([185, 215], 0, 150, color='m', alpha=0.1, label='Time window')
        
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Episode {episode_result.episode_id}: Temperature Control\n'
                     f'Goal: {"✓" if episode_result.goal_achieved else "✗"}, '
                     f'Final: {episode_result.final_temperature:.1f}°C, '
                     f'Efficiency: {episode_result.energy_efficiency:.1%}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(250, len(episode_result.temperatures)))
        
        # 2. Actions with energy annotation
        ax = axes[1]
        if episode_result.actions:
            # Assume actions are [0, 3000] W
            action_powers = [0 if a == 0 else 3000 for a in episode_result.actions]
            ax.step(action_steps, action_powers, 'g-', where='post', linewidth=2.5, label='Power')
            
            # Highlight energy-saving periods
            for i in range(len(episode_result.actions)):
                if (action_powers[i] == 0 and i > 0 and 
                    episode_result.temperatures[i] > 95):
                    ax.axvspan(i+1, i+2, alpha=0.2, color='green', label='Energy saving' if i == 0 else '')
        
        ax.set_ylabel('Power (W)')
        ax.set_title('Heating Power Applied')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(250, len(episode_result.actions)))
        
        # 3. Rewards
        ax = axes[2]
        if episode_result.rewards:
            colors = ['red' if r < 0 else 'green' for r in episode_result.rewards]
            bars = ax.bar(action_steps, episode_result.rewards, color=colors, alpha=0.7)
            
            # Highlight significant rewards
            for i, r in enumerate(episode_result.rewards):
                if r > 100:
                    bars[i].set_alpha(1.0)
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(2)
        
        ax.set_ylabel('Reward')
        ax.set_title('Reward per Step (Red: penalties, Green: bonuses)')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(250, len(episode_result.rewards)))
        
        # 4. Cumulative energy consumption
        ax = axes[3]
        if episode_result.actions:
            energy_watts = [0 if a == 0 else 3000 for a in episode_result.actions]
            cumulative_energy_wh = np.cumsum(energy_watts) / 3600
            
            ax.plot(action_steps, cumulative_energy_wh, 'orange', linewidth=2.5, label='Actual')
            
            # Theoretical minimum
            theoretical_min = 1.0 * 4184 * 80 / 3600  # 93 Wh
            ax.axhline(y=theoretical_min, color='green', linestyle='--', 
                      label=f'Theoretical min ({theoretical_min:.1f} Wh)')
            
            ax.fill_between(action_steps, theoretical_min, cumulative_energy_wh,
                           where=(cumulative_energy_wh > theoretical_min),
                           color='red', alpha=0.2, label='Excess energy')
        
        ax.set_ylabel('Energy (Wh)')
        ax.set_title('Cumulative Energy Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(250, len(episode_result.actions)))
        
        # 5. Energy efficiency over time
        ax = axes[4]
        if episode_result.actions:
            theoretical_min = 1.0 * 4184 * 80 / 3600
            efficiency = theoretical_min / np.maximum(cumulative_energy_wh, 1)
            efficiency = np.clip(efficiency, 0, 1.1) * 100
            
            ax.plot(action_steps, efficiency, 'purple', linewidth=2.5)
            ax.axhline(y=100, color='green', linestyle='--', label='Perfect efficiency')
            ax.axhline(y=90, color='orange', linestyle=':', label='90% efficiency')
            ax.fill_between(action_steps, 0, efficiency, alpha=0.3, color='purple')
        
        ax.set_ylabel('Efficiency (%)')
        ax.set_xlabel('Time Step')
        ax.set_title('Energy Efficiency Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(250, len(episode_result.actions)))
        ax.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_episode_overview(self, episode_results: List[EpisodeResult], output_path: Path):
        """Create overview plot of all episodes"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Extract data
        episodes = [r.episode_id for r in episode_results]
        final_temps = [r.final_temperature for r in episode_results]
        energies = [r.energy_consumed_wh for r in episode_results]
        efficiencies = [r.energy_efficiency * 100 for r in episode_results]
        rewards = [r.total_reward for r in episode_results]
        goal_achieved = [r.goal_achieved for r in episode_results]
        
        # Color code by success
        colors = ['green' if success else 'red' for success in goal_achieved]
        
        # 1. Final temperatures
        ax = axes[0, 0]
        ax.scatter(episodes, final_temps, c=colors, alpha=0.7, s=60)
        ax.axhline(y=100, color='blue', linestyle='--', label='Target')
        ax.axhspan(97.5, 102.5, alpha=0.2, color='blue', label='Tolerance')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Final Temperature (°C)')
        ax.set_title('Final Temperature by Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Energy consumption
        ax = axes[0, 1]
        ax.scatter(episodes, energies, c=colors, alpha=0.7, s=60)
        theoretical_min = 1.0 * 4184 * 80 / 3600
        ax.axhline(y=theoretical_min, color='green', linestyle='--', 
                  label=f'Theoretical min ({theoretical_min:.1f} Wh)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Consumed (Wh)')
        ax.set_title('Energy Consumption by Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Energy efficiency
        ax = axes[1, 0]
        ax.scatter(episodes, efficiencies, c=colors, alpha=0.7, s=60)
        ax.axhline(y=100, color='green', linestyle='--', label='Perfect efficiency')
        ax.axhline(y=90, color='orange', linestyle=':', label='90% target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Energy Efficiency by Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Total reward
        ax = axes[1, 1]
        ax.scatter(episodes, rewards, c=colors, alpha=0.7, s=60)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Total Reward by Episode')
        ax.grid(True, alpha=0.3)
        
        # Add success rate annotation
        success_rate = np.mean(goal_achieved) * 100
        fig.suptitle(f'Episode Overview - Success Rate: {success_rate:.1f}% '
                     f'({sum(goal_achieved)}/{len(goal_achieved)} episodes)',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_performance_metrics(self, episode_results: List[EpisodeResult], output_path: Path):
        """Create performance metrics visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data
        goal_achieved = [r.goal_achieved for r in episode_results]
        final_temps = [r.final_temperature for r in episode_results]
        temp_errors = [abs(r.final_temperature - 100) for r in episode_results]
        energies = [r.energy_consumed_wh for r in episode_results]
        efficiencies = [r.energy_efficiency * 100 for r in episode_results]
        rewards = [r.total_reward for r in episode_results]
        
        # 1. Success rate pie chart
        ax = axes[0, 0]
        success_count = sum(goal_achieved)
        failure_count = len(goal_achieved) - success_count
        
        if success_count > 0 or failure_count > 0:
            ax.pie([success_count, failure_count], 
                   labels=[f'Success ({success_count})', f'Failure ({failure_count})'],
                   colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax.set_title('Goal Achievement Rate')
        
        # 2. Temperature error histogram
        ax = axes[0, 1]
        ax.hist(temp_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=2.5, color='red', linestyle='--', label='Tolerance limit')
        ax.set_xlabel('Temperature Error (°C)')
        ax.set_ylabel('Frequency')
        ax.set_title('Temperature Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Energy efficiency histogram
        ax = axes[0, 2]
        ax.hist(efficiencies, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=90, color='orange', linestyle='--', label='90% target')
        ax.axvline(x=np.mean(efficiencies), color='red', linestyle='-', 
                  label=f'Mean ({np.mean(efficiencies):.1f}%)')
        ax.set_xlabel('Energy Efficiency (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Energy Efficiency Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Energy vs Efficiency scatter
        ax = axes[1, 0]
        colors = ['green' if success else 'red' for success in goal_achieved]
        ax.scatter(energies, efficiencies, c=colors, alpha=0.7)
        
        # Add trend line
        if len(energies) > 1 and len(efficiencies) > 1:
            z = np.polyfit(energies, efficiencies, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(energies), max(energies), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Trend')
        
        ax.set_xlabel('Energy Consumed (Wh)')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Energy Consumption vs Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Reward vs Performance
        ax = axes[1, 1]
        ax.scatter(efficiencies, rewards, c=colors, alpha=0.7)
        ax.set_xlabel('Energy Efficiency (%)')
        ax.set_ylabel('Total Reward')
        ax.set_title('Efficiency vs Reward')
        ax.grid(True, alpha=0.3)
        
        # 6. Performance summary box plot
        ax = axes[1, 2]
        data_to_plot = [
            [e for e, g in zip(efficiencies, goal_achieved) if g],  # Successful
            [e for e, g in zip(efficiencies, goal_achieved) if not g]  # Failed
        ]
        
        labels = ['Successful', 'Failed']
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('green')
        if len(box_plot['boxes']) > 1:
            box_plot['boxes'][1].set_facecolor('red')
        
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Efficiency by Success Status')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_energy_analysis(self, episode_results: List[EpisodeResult], output_path: Path):
        """Create detailed energy analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Calculate energy metrics for each episode
        energy_data = []
        for result in episode_results:
            if result.actions:
                heating_actions = sum(1 for a in result.actions if a == 1)
                heating_percentage = (heating_actions / len(result.actions)) * 100
                
                # Find heating periods
                first_heat = None
                last_heat = None
                for i, action in enumerate(result.actions):
                    if action == 1:
                        if first_heat is None:
                            first_heat = i + 1
                        last_heat = i + 1
                
                energy_data.append({
                    'episode_id': result.episode_id,
                    'energy_consumed': result.energy_consumed_wh,
                    'efficiency': result.energy_efficiency * 100,
                    'heating_percentage': heating_percentage,
                    'goal_achieved': result.goal_achieved,
                    'first_heating_step': first_heat or 0,
                    'heating_duration': (last_heat - first_heat + 1) if first_heat and last_heat else 0
                })
        
        df = pd.DataFrame(energy_data)
        
        # 1. Energy consumption by success
        ax = axes[0, 0]
        successful = df[df['goal_achieved'] == True]['energy_consumed']
        failed = df[df['goal_achieved'] == False]['energy_consumed']
        
        bins = np.linspace(df['energy_consumed'].min(), df['energy_consumed'].max(), 20)
        ax.hist(successful, bins=bins, alpha=0.7, color='green', label='Successful', edgecolor='black')
        ax.hist(failed, bins=bins, alpha=0.7, color='red', label='Failed', edgecolor='black')
        
        theoretical_min = 1.0 * 4184 * 80 / 3600
        ax.axvline(x=theoretical_min, color='blue', linestyle='--', 
                  label=f'Theoretical min ({theoretical_min:.1f} Wh)')
        
        ax.set_xlabel('Energy Consumed (Wh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Energy Consumption Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Heating strategy analysis
        ax = axes[0, 1]
        ax.scatter(df['first_heating_step'], df['efficiency'], 
                  c=df['goal_achieved'].map({True: 'green', False: 'red'}), alpha=0.7)
        ax.set_xlabel('First Heating Step')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Heating Start Time vs Efficiency')
        ax.axvline(x=88, color='orange', linestyle='--', label='Optimal start (~88)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Heating percentage vs efficiency
        ax = axes[1, 0]
        ax.scatter(df['heating_percentage'], df['efficiency'],
                  c=df['goal_achieved'].map({True: 'green', False: 'red'}), alpha=0.7)
        ax.set_xlabel('Heating Percentage (%)')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Heating Duration vs Efficiency')
        ax.grid(True, alpha=0.3)
        
        # 4. Efficiency categories
        ax = axes[1, 1]
        
        # Categorize efficiency
        categories = []
        for eff in df['efficiency']:
            if eff >= 90:
                categories.append('Excellent (≥90%)')
            elif eff >= 80:
                categories.append('Good (≥80%)')
            elif eff >= 60:
                categories.append('Fair (≥60%)')
            else:
                categories.append('Poor (<60%)')
        
        category_counts = pd.Series(categories).value_counts()
        colors_cat = ['darkgreen', 'green', 'orange', 'red']
        
        wedges, texts, autotexts = ax.pie(category_counts.values, 
                                         labels=category_counts.index,
                                         colors=colors_cat[:len(category_counts)],
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Energy Efficiency Categories')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_control_strategy(self, episode_results: List[EpisodeResult], output_path: Path):
        """Analyze and visualize control strategies"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Analyze strategies across episodes
        strategy_data = []
        for result in episode_results:
            if result.actions and result.temperatures:
                # Detect strategy type
                first_heat = None
                for i, action in enumerate(result.actions):
                    if action == 1:
                        first_heat = i + 1
                        break
                
                heating_actions = sum(1 for a in result.actions if a == 1)
                heating_pct = (heating_actions / len(result.actions)) * 100
                
                # Strategy classification
                if first_heat is None:
                    strategy = "No heating"
                elif first_heat > 50:
                    strategy = "Delayed heating"
                elif heating_pct > 70:
                    strategy = "Continuous heating"
                else:
                    strategy = "Heat and coast"
                
                strategy_data.append({
                    'episode_id': result.episode_id,
                    'strategy': strategy,
                    'goal_achieved': result.goal_achieved,
                    'efficiency': result.energy_efficiency * 100,
                    'first_heating_step': first_heat or 0,
                    'heating_percentage': heating_pct
                })
        
        df = pd.DataFrame(strategy_data)
        
        # 1. Strategy distribution
        ax = axes[0, 0]
        strategy_counts = df['strategy'].value_counts()
        bars = ax.bar(strategy_counts.index, strategy_counts.values, 
                     color=['blue', 'green', 'orange', 'red'][:len(strategy_counts)])
        ax.set_ylabel('Number of Episodes')
        ax.set_title('Control Strategy Distribution')
        ax.tick_params(axis='x', rotation=45)
        
        # Add success rate annotations
        for i, (strategy, bar) in enumerate(zip(strategy_counts.index, bars)):
            strategy_episodes = df[df['strategy'] == strategy]
            success_rate = strategy_episodes['goal_achieved'].mean() * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{success_rate:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Strategy efficiency
        ax = axes[0, 1]
        
        strategies = df['strategy'].unique()
        efficiency_by_strategy = [df[df['strategy'] == s]['efficiency'].values for s in strategies]
        
        box_plot = ax.boxplot(efficiency_by_strategy, labels=strategies, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors[:len(strategies)]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Efficiency by Strategy')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 3. Representative trajectories
        ax = axes[1, 0]
        
        # Plot one representative trajectory for each strategy
        for strategy in strategies[:4]:  # Limit to 4 strategies
            strategy_episodes = df[df['strategy'] == strategy]
            if not strategy_episodes.empty:
                # Find best performing episode of this strategy
                best_episode_id = strategy_episodes.loc[strategy_episodes['efficiency'].idxmax(), 'episode_id']
                best_result = next(r for r in episode_results if r.episode_id == best_episode_id)
                
                steps = range(len(best_result.temperatures))
                ax.plot(steps, best_result.temperatures, label=f'{strategy} (Ep {best_episode_id})',
                       linewidth=2, alpha=0.8)
        
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Target')
        ax.axvline(x=200, color='purple', linestyle='--', alpha=0.7, label='Deadline')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Representative Temperature Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Strategy timing analysis
        ax = axes[1, 1]
        
        # Scatter plot of first heating step vs efficiency, colored by strategy
        strategy_colors = {'Delayed heating': 'green', 'Heat and coast': 'blue', 
                          'Continuous heating': 'orange', 'No heating': 'red'}
        
        for strategy in strategies:
            strategy_data_filtered = df[df['strategy'] == strategy]
            if not strategy_data_filtered.empty:
                ax.scatter(strategy_data_filtered['first_heating_step'], 
                          strategy_data_filtered['efficiency'],
                          c=strategy_colors.get(strategy, 'gray'), 
                          label=strategy, alpha=0.7, s=60)
        
        ax.set_xlabel('First Heating Step')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Heating Timing vs Efficiency by Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_mcts_statistics(self, episode_results: List[EpisodeResult], output_path: Path):
        """Visualize MCTS performance statistics"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract MCTS data
        mcts_data = []
        for result in episode_results:
            mcts_data.append({
                'episode_id': result.episode_id,
                'total_iterations': result.mcts_stats.total_iterations,
                'total_time': result.mcts_stats.total_time,
                'tree_size': result.mcts_stats.tree_size,
                'max_depth': result.mcts_stats.max_depth,
                'rollouts': result.mcts_stats.rollouts_performed,
                'nodes_expanded': result.mcts_stats.nodes_expanded,
                'goal_achieved': result.goal_achieved,
                'efficiency': result.energy_efficiency * 100,
                'iterations_per_second': result.mcts_stats.total_iterations / max(0.001, result.mcts_stats.total_time)
            })
        
        df = pd.DataFrame(mcts_data)
        colors = ['green' if g else 'red' for g in df['goal_achieved']]
        
        # 1. Iterations vs Performance
        ax = axes[0, 0]
        ax.scatter(df['total_iterations'], df['efficiency'], c=colors, alpha=0.7)
        ax.set_xlabel('Total MCTS Iterations')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('MCTS Iterations vs Performance')
        ax.grid(True, alpha=0.3)
        
        # 2. Tree size distribution
        ax = axes[0, 1]
        ax.hist(df['tree_size'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=df['tree_size'].mean(), color='red', linestyle='--', 
                  label=f'Mean ({df["tree_size"].mean():.0f})')
        ax.set_xlabel('Tree Size (nodes)')
        ax.set_ylabel('Frequency')
        ax.set_title('MCTS Tree Size Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Max depth vs efficiency
        ax = axes[0, 2]
        ax.scatter(df['max_depth'], df['efficiency'], c=colors, alpha=0.7)
        ax.set_xlabel('Maximum Tree Depth')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Tree Depth vs Performance')
        ax.grid(True, alpha=0.3)
        
        # 4. Computational efficiency
        ax = axes[1, 0]
        ax.scatter(df['iterations_per_second'], df['efficiency'], c=colors, alpha=0.7)
        ax.set_xlabel('MCTS Iterations per Second')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Computational Efficiency vs Performance')
        ax.grid(True, alpha=0.3)
        
        # 5. Search effort distribution
        ax = axes[1, 1]
        successful = df[df['goal_achieved'] == True]['total_time']
        failed = df[df['goal_achieved'] == False]['total_time']
        
        bins = np.linspace(df['total_time'].min(), df['total_time'].max(), 20)
        ax.hist(successful, bins=bins, alpha=0.7, color='green', label='Successful', edgecolor='black')
        ax.hist(failed, bins=bins, alpha=0.7, color='red', label='Failed', edgecolor='black')
        
        ax.set_xlabel('Total MCTS Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Search Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. MCTS metrics correlation
        ax = axes[1, 2]
        
        # Correlation heatmap of MCTS metrics
        mcts_metrics = ['total_iterations', 'tree_size', 'max_depth', 'rollouts', 'efficiency']
        corr_data = df[mcts_metrics].corr()
        
        im = ax.imshow(corr_data.values, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(mcts_metrics)))
        ax.set_yticks(range(len(mcts_metrics)))
        ax.set_xticklabels(mcts_metrics, rotation=45)
        ax.set_yticklabels(mcts_metrics)
        
        # Add correlation values
        for i in range(len(mcts_metrics)):
            for j in range(len(mcts_metrics)):
                ax.text(j, i, f'{corr_data.values[i, j]:.2f}', 
                       ha='center', va='center', color='white' if abs(corr_data.values[i, j]) > 0.5 else 'black')
        
        ax.set_title('MCTS Metrics Correlation')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_temperature_trajectories(self, episode_results: List[EpisodeResult], output_path: Path):
        """Plot temperature trajectories for all episodes"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # 1. All trajectories
        ax = axes[0, 0]
        
        successful_episodes = [r for r in episode_results if r.goal_achieved]
        failed_episodes = [r for r in episode_results if not r.goal_achieved]
        
        # Plot successful episodes
        for result in successful_episodes[:10]:  # Limit to 10 for clarity
            steps = range(len(result.temperatures))
            ax.plot(steps, result.temperatures, 'g-', alpha=0.3, linewidth=1)
        
        # Plot failed episodes
        for result in failed_episodes[:10]:  # Limit to 10 for clarity
            steps = range(len(result.temperatures))
            ax.plot(steps, result.temperatures, 'r-', alpha=0.3, linewidth=1)
        
        # Reference lines
        ax.axhline(y=100, color='blue', linestyle='--', linewidth=2, label='Target')
        ax.axhspan(97.5, 102.5, alpha=0.2, color='blue', label='Tolerance')
        ax.axvline(x=200, color='purple', linestyle='--', linewidth=2, label='Deadline')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('All Temperature Trajectories\n(Green: Successful, Red: Failed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 250)
        
        # 2. Average trajectories
        ax = axes[0, 1]
        
        if successful_episodes:
            # Calculate average successful trajectory
            max_len = max(len(r.temperatures) for r in successful_episodes)
            temp_matrix = np.full((len(successful_episodes), max_len), np.nan)
            
            for i, result in enumerate(successful_episodes):
                temp_matrix[i, :len(result.temperatures)] = result.temperatures
            
            avg_temp = np.nanmean(temp_matrix, axis=0)
            std_temp = np.nanstd(temp_matrix, axis=0)
            steps = range(len(avg_temp))
            
            ax.plot(steps, avg_temp, 'g-', linewidth=3, label='Successful (avg)')
            ax.fill_between(steps, avg_temp - std_temp, avg_temp + std_temp, 
                           color='green', alpha=0.2, label='±1 std')
        
        if failed_episodes:
            # Calculate average failed trajectory
            max_len = max(len(r.temperatures) for r in failed_episodes)
            temp_matrix = np.full((len(failed_episodes), max_len), np.nan)
            
            for i, result in enumerate(failed_episodes):
                temp_matrix[i, :len(result.temperatures)] = result.temperatures
            
            avg_temp = np.nanmean(temp_matrix, axis=0)
            std_temp = np.nanstd(temp_matrix, axis=0)
            steps = range(len(avg_temp))
            
            ax.plot(steps, avg_temp, 'r-', linewidth=3, label='Failed (avg)')
            ax.fill_between(steps, avg_temp - std_temp, avg_temp + std_temp, 
                           color='red', alpha=0.2, label='±1 std')
        
        ax.axhline(y=100, color='blue', linestyle='--', linewidth=2, label='Target')
        ax.axvline(x=200, color='purple', linestyle='--', linewidth=2, label='Deadline')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Average Temperature Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 250)
        
        # 3. Temperature at deadline
        ax = axes[1, 0]
        
        deadline_temps = []
        goal_status = []
        
        for result in episode_results:
            if len(result.temperatures) > 200:
                deadline_temps.append(result.temperatures[200])
                goal_status.append(result.goal_achieved)
            elif len(result.temperatures) > 0:
                deadline_temps.append(result.temperatures[-1])
                goal_status.append(result.goal_achieved)
        
        if deadline_temps:
            colors = ['green' if g else 'red' for g in goal_status]
            ax.scatter(range(len(deadline_temps)), deadline_temps, c=colors, alpha=0.7)
            ax.axhline(y=100, color='blue', linestyle='--', label='Target')
            ax.axhspan(97.5, 102.5, alpha=0.2, color='blue', label='Tolerance')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Temperature at Deadline (°C)')
            ax.set_title('Temperature at Deadline Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Temperature evolution phases
        ax = axes[1, 1]
        
        # Analyze heating phases
        phase_data = {'Heating': [], 'Coasting': [], 'Maintaining': []}
        
        for result in episode_results:
            if result.actions and result.temperatures:
                for i, (temp, action) in enumerate(zip(result.temperatures[1:], result.actions)):
                    if action == 1:  # Heating
                        phase_data['Heating'].append(temp)
                    elif temp > 98:  # Near target
                        phase_data['Maintaining'].append(temp)
                    else:  # Coasting
                        phase_data['Coasting'].append(temp)
        
        # Box plot of temperatures in each phase
        data_to_plot = [phase_data['Heating'], phase_data['Coasting'], phase_data['Maintaining']]
        labels = ['Heating', 'Coasting', 'Maintaining']
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['red', 'blue', 'green']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=100, color='purple', linestyle='--', label='Target')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Distribution by Control Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, episode_results: List[EpisodeResult], 
                                   output_path: Path, experiment_name: str):
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Temperature Trajectories', 'Energy Efficiency',
                           'Performance Metrics', 'MCTS Statistics',
                           'Control Actions', 'Episode Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Prepare data
        episodes = list(range(len(episode_results)))
        
        # 1. Temperature trajectories (first few episodes)
        for i, result in enumerate(episode_results[:5]):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.temperatures))),
                    y=result.temperatures,
                    mode='lines',
                    name=f'Episode {i}',
                    line=dict(color='green' if result.goal_achieved else 'red', width=2),
                    hovertemplate='Step: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Target line
        fig.add_hline(y=100, line_dash="dash", line_color="blue", row=1, col=1)
        fig.add_vline(x=200, line_dash="dash", line_color="purple", row=1, col=1)
        
        # 2. Energy efficiency
        efficiencies = [r.energy_efficiency * 100 for r in episode_results]
        colors = ['green' if r.goal_achieved else 'red' for r in episode_results]
        
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=efficiencies,
                mode='markers',
                name='Energy Efficiency',
                marker=dict(color=colors, size=8),
                hovertemplate='Episode: %{x}<br>Efficiency: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Performance metrics
        final_temps = [r.final_temperature for r in episode_results]
        total_rewards = [r.total_reward for r in episode_results]
        
        fig.add_trace(
            go.Bar(
                x=episodes,
                y=final_temps,
                name='Final Temperature',
                marker_color='lightblue',
                hovertemplate='Episode: %{x}<br>Final Temp: %{y:.1f}°C<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. MCTS statistics
        tree_sizes = [r.mcts_stats.tree_size for r in episode_results]
        
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=tree_sizes,
                mode='lines+markers',
                name='Tree Size',
                marker=dict(color='orange'),
                hovertemplate='Episode: %{x}<br>Tree Size: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Control actions (for first episode with dual y-axis)
        if episode_results:
            result = episode_results[0]
            if result.actions:
                action_powers = [0 if a == 0 else 3000 for a in result.actions]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(result.temperatures))),
                        y=result.temperatures,
                        mode='lines',
                        name='Temperature',
                        line=dict(color='blue', width=2),
                        hovertemplate='Step: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(action_powers) + 1)),
                        y=action_powers,
                        mode='lines',
                        name='Power',
                        line=dict(color='red', width=2),
                        yaxis='y2',
                        hovertemplate='Step: %{x}<br>Power: %{y}W<extra></extra>'
                    ),
                    row=3, col=1, secondary_y=True
                )
        
        # 6. Episode comparison
        energy_consumed = [r.energy_consumed_wh for r in episode_results]
        
        fig.add_trace(
            go.Scatter(
                x=energy_consumed,
                y=efficiencies,
                mode='markers',
                name='Energy vs Efficiency',
                marker=dict(
                    color=colors,
                    size=[10 if g else 6 for g in [r.goal_achieved for r in episode_results]],
                    symbol=['circle' if g else 'x' for g in [r.goal_achieved for r in episode_results]]
                ),
                hovertemplate='Energy: %{x:.1f}Wh<br>Efficiency: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"{experiment_name} - Interactive Dashboard",
            showlegend=True,
            height=1200,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time Step", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_yaxes(title_text="Final Temperature (°C)", row=2, col=1)
        
        fig.update_xaxes(title_text="Episode", row=2, col=2)
        fig.update_yaxes(title_text="Tree Size", row=2, col=2)
        
        fig.update_xaxes(title_text="Time Step", row=3, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
        fig.update_yaxes(title_text="Power (W)", row=3, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="Energy Consumed (Wh)", row=3, col=2)
        fig.update_yaxes(title_text="Efficiency (%)", row=3, col=2)
        
        # Save interactive plot
        fig.write_html(str(output_path))
    
    def generate_comparison_plots(self, results: List[Dict[str, Any]], 
                                output_dir: Path, algorithms: List[str], 
                                environments: List[str]):
        """Generate comparative plots across algorithms and environments"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Organize data by algorithm and environment
        organized_data = {}
        for result in results:
            alg = result['config'].algorithm.algorithm_name
            env_name = result['config'].experiment_name.split('_')[-1]
            
            key = f"{alg}_{env_name}"
            if key not in organized_data:
                organized_data[key] = []
            
            organized_data[key].extend(result['results'])
        
        # Performance comparison
        self._plot_algorithm_comparison(organized_data, algorithms, environments, 
                                      output_dir / "algorithm_comparison.png")
        
        # Statistical comparison
        self._plot_statistical_comparison(organized_data, algorithms, environments,
                                        output_dir / "statistical_comparison.png")
        
        # Environment sensitivity
        self._plot_environment_sensitivity(organized_data, algorithms, environments,
                                         output_dir / "environment_sensitivity.png")
    
    def generate_parameter_plots(self, results: List[Dict[str, Any]], 
                               output_dir: Path, parameter_ranges: Dict[str, List[Any]]):
        """Generate parameter sensitivity plots"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for param_name in parameter_ranges.keys():
            self._plot_parameter_sensitivity(results, param_name, 
                                           output_dir / f"parameter_{param_name}.png")
    
    def _find_representative_episode(self, episode_results: List[EpisodeResult]) -> Optional[EpisodeResult]:
        """Find a representative episode for detailed analysis"""
        
        # Prefer successful episodes
        successful = [r for r in episode_results if r.goal_achieved]
        if successful:
            # Find the one with median efficiency
            efficiencies = [r.energy_efficiency for r in successful]
            median_idx = np.argsort(efficiencies)[len(efficiencies) // 2]
            return successful[median_idx]
        
        # If no successful episodes, return first episode
        return episode_results[0] if episode_results else None
    
    def _plot_algorithm_comparison(self, organized_data: Dict[str, List[EpisodeResult]], 
                                 algorithms: List[str], environments: List[str], 
                                 output_path: Path):
        """Plot algorithm performance comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Prepare data for plotting
        comparison_data = []
        for alg in algorithms:
            for env in environments:
                key = f"{alg}_{env}"
                if key in organized_data:
                    episodes = organized_data[key]
                    
                    comparison_data.append({
                        'algorithm': alg,
                        'environment': env,
                        'success_rate': np.mean([e.goal_achieved for e in episodes]) * 100,
                        'avg_efficiency': np.mean([e.energy_efficiency for e in episodes]) * 100,
                        'avg_reward': np.mean([e.total_reward for e in episodes]),
                        'avg_energy': np.mean([e.energy_consumed_wh for e in episodes])
                    })
        
        df = pd.DataFrame(comparison_data)
        
        # 1. Success rate comparison
        ax = axes[0, 0]
        pivot_success = df.pivot(index='algorithm', columns='environment', values='success_rate')
        sns.heatmap(pivot_success, annot=True, fmt='.1f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Success Rate (%)'})
        ax.set_title('Goal Achievement Rate by Algorithm and Environment')
        
        # 2. Energy efficiency comparison
        ax = axes[0, 1]
        pivot_efficiency = df.pivot(index='algorithm', columns='environment', values='avg_efficiency')
        sns.heatmap(pivot_efficiency, annot=True, fmt='.1f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Efficiency (%)'})
        ax.set_title('Average Energy Efficiency by Algorithm and Environment')
        
        # 3. Box plot comparison
        ax = axes[1, 0]
        
        # Combine all episode data with labels
        plot_data = []
        for alg in algorithms:
            for env in environments:
                key = f"{alg}_{env}"
                if key in organized_data:
                    episodes = organized_data[key]
                    for episode in episodes:
                        plot_data.append({
                            'Algorithm': alg,
                            'Environment': env,
                            'Efficiency': episode.energy_efficiency * 100,
                            'Label': f"{alg}\n{env}"
                        })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(data=plot_df, x='Algorithm', y='Efficiency', hue='Environment', ax=ax)
            ax.set_title('Energy Efficiency Distribution')
            ax.set_ylabel('Energy Efficiency (%)')
            ax.tick_params(axis='x', rotation=45)
        
        # 4. Performance ranking
        ax = axes[1, 1]
        
        # Calculate overall scores (weighted combination)
        df['overall_score'] = (0.4 * df['success_rate'] + 
                              0.3 * df['avg_efficiency'] + 
                              0.2 * (df['avg_reward'] / df['avg_reward'].max() * 100) +
                              0.1 * (100 - df['avg_energy'] / df['avg_energy'].max() * 100))
        
        # Sort by overall score
        df_sorted = df.sort_values('overall_score', ascending=True)
        
        bars = ax.barh(range(len(df_sorted)), df_sorted['overall_score'])
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"{row['algorithm']}\n{row['environment']}" 
                           for _, row in df_sorted.iterrows()])
        ax.set_xlabel('Overall Performance Score')
        ax.set_title('Overall Performance Ranking')
        
        # Color bars based on score
        for i, bar in enumerate(bars):
            if df_sorted.iloc[i]['overall_score'] > 80:
                bar.set_color('green')
            elif df_sorted.iloc[i]['overall_score'] > 60:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_comparison(self, organized_data: Dict[str, List[EpisodeResult]], 
                                   algorithms: List[str], environments: List[str], 
                                   output_path: Path):
        """Plot statistical comparison with confidence intervals"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Prepare statistical data
        stats_data = []
        for alg in algorithms:
            for env in environments:
                key = f"{alg}_{env}"
                if key in organized_data:
                    episodes = organized_data[key]
                    
                    efficiencies = [e.energy_efficiency * 100 for e in episodes]
                    success_rates = [e.goal_achieved for e in episodes]
                    
                    stats_data.append({
                        'algorithm': alg,
                        'environment': env,
                        'efficiency_mean': np.mean(efficiencies),
                        'efficiency_std': np.std(efficiencies),
                        'efficiency_ci': 1.96 * np.std(efficiencies) / np.sqrt(len(efficiencies)),
                        'success_mean': np.mean(success_rates) * 100,
                        'success_ci': 1.96 * np.sqrt(np.mean(success_rates) * (1 - np.mean(success_rates)) / len(success_rates)) * 100,
                        'n_episodes': len(episodes)
                    })
        
        df = pd.DataFrame(stats_data)
        
        # 1. Efficiency with confidence intervals
        ax = axes[0, 0]
        
        x_pos = np.arange(len(df))
        bars = ax.bar(x_pos, df['efficiency_mean'], yerr=df['efficiency_ci'], 
                     capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_xlabel('Algorithm - Environment')
        ax.set_ylabel('Energy Efficiency (%)')
        ax.set_title('Energy Efficiency with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{row['algorithm']}\n{row['environment']}" 
                           for _, row in df.iterrows()], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Success rate with confidence intervals
        ax = axes[0, 1]
        
        bars = ax.bar(x_pos, df['success_mean'], yerr=df['success_ci'], 
                     capsize=5, alpha=0.7, color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('Algorithm - Environment')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{row['algorithm']}\n{row['environment']}" 
                           for _, row in df.iterrows()], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
        
        # 3. Effect sizes (if multiple algorithms)
        ax = axes[1, 0]
        
        if len(algorithms) > 1:
            # Calculate pairwise effect sizes
            effect_sizes = []
            comparisons = []
            
            for env in environments:
                env_data = {}
                for alg in algorithms:
                    key = f"{alg}_{env}"
                    if key in organized_data:
                        env_data[alg] = [e.energy_efficiency * 100 for e in organized_data[key]]
                
                # Compare first algorithm with others
                if len(env_data) > 1:
                    base_alg = algorithms[0]
                    if base_alg in env_data:
                        for other_alg in algorithms[1:]:
                            if other_alg in env_data:
                                # Cohen's d
                                mean1, mean2 = np.mean(env_data[base_alg]), np.mean(env_data[other_alg])
                                std1, std2 = np.std(env_data[base_alg]), np.std(env_data[other_alg])
                                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                                
                                cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
                                
                                effect_sizes.append(cohens_d)
                                comparisons.append(f"{other_alg} vs {base_alg}\n({env})")
            
            if effect_sizes:
                colors = ['green' if es > 0 else 'red' for es in effect_sizes]
                bars = ax.barh(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.7)
                ax.set_yticks(range(len(effect_sizes)))
                ax.set_yticklabels(comparisons)
                ax.set_xlabel("Cohen's d (Effect Size)")
                ax.set_title('Effect Sizes for Algorithm Comparisons')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
                ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
                ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.5)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Effect size analysis\nrequires multiple algorithms', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Effect Size Analysis')
        
        # 4. Sample size and power analysis
        ax = axes[1, 1]
        
        # Plot sample sizes
        bars = ax.bar(x_pos, df['n_episodes'], alpha=0.7, color='coral', edgecolor='black')
        
        # Add power lines (rough estimates)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Low power (n=10)')
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Medium power (n=20)')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='High power (n=30)')
        
        ax.set_xlabel('Algorithm - Environment')
        ax.set_ylabel('Number of Episodes')
        ax.set_title('Sample Sizes for Statistical Power')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{row['algorithm']}\n{row['environment']}" 
                           for _, row in df.iterrows()], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_environment_sensitivity(self, organized_data: Dict[str, List[EpisodeResult]], 
                                    algorithms: List[str], environments: List[str], 
                                    output_path: Path):
        """Plot algorithm sensitivity to different environments"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Calculate performance drops/gains across environments
        sensitivity_data = []
        
        for alg in algorithms:
            alg_performances = {}
            for env in environments:
                key = f"{alg}_{env}"
                if key in organized_data:
                    episodes = organized_data[key]
                    alg_performances[env] = {
                        'efficiency': np.mean([e.energy_efficiency * 100 for e in episodes]),
                        'success_rate': np.mean([e.goal_achieved for e in episodes]) * 100
                    }
            
            # Calculate relative performance (normalized to standard environment)
            if 'standard' in alg_performances:
                base_efficiency = alg_performances['standard']['efficiency']
                base_success = alg_performances['standard']['success_rate']
                
                for env, perf in alg_performances.items():
                    sensitivity_data.append({
                        'algorithm': alg,
                        'environment': env,
                        'efficiency_relative': (perf['efficiency'] / base_efficiency - 1) * 100 if base_efficiency > 0 else 0,
                        'success_relative': (perf['success_rate'] / base_success - 1) * 100 if base_success > 0 else 0,
                        'efficiency_absolute': perf['efficiency'],
                        'success_absolute': perf['success_rate']
                    })
        
        if sensitivity_data:
            df = pd.DataFrame(sensitivity_data)
            
            # 1. Relative efficiency change
            ax = axes[0, 0]
            pivot_eff = df.pivot(index='algorithm', columns='environment', values='efficiency_relative')
            sns.heatmap(pivot_eff, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': 'Efficiency Change (%)'})
            ax.set_title('Relative Efficiency Change from Standard Environment')
            
            # 2. Relative success rate change
            ax = axes[0, 1]
            pivot_success = df.pivot(index='algorithm', columns='environment', values='success_relative')
            sns.heatmap(pivot_success, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': 'Success Rate Change (%)'})
            ax.set_title('Relative Success Rate Change from Standard Environment')
            
            # 3. Algorithm robustness (variance across environments)
            ax = axes[1, 0]
            
            robustness_data = []
            for alg in algorithms:
                alg_data = df[df['algorithm'] == alg]
                if not alg_data.empty:
                    efficiency_var = np.var(alg_data['efficiency_absolute'])
                    success_var = np.var(alg_data['success_absolute'])
                    
                    robustness_data.append({
                        'algorithm': alg,
                        'efficiency_variance': efficiency_var,
                        'success_variance': success_var,
                        'robustness_score': 100 - (efficiency_var + success_var / 100)  # Higher is more robust
                    })
            
            if robustness_data:
                rob_df = pd.DataFrame(robustness_data)
                bars = ax.bar(rob_df['algorithm'], rob_df['robustness_score'], 
                             color='lightblue', edgecolor='black', alpha=0.7)
                ax.set_ylabel('Robustness Score')
                ax.set_title('Algorithm Robustness Across Environments\n(Higher = More Consistent)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Color code bars
                for i, bar in enumerate(bars):
                    score = rob_df.iloc[i]['robustness_score']
                    if score > 80:
                        bar.set_color('green')
                    elif score > 60:
                        bar.set_color('yellow')
                    else:
                        bar.set_color('red')
            
            # 4. Environment difficulty ranking
            ax = axes[1, 1]
            
            env_difficulty = []
            for env in environments:
                env_data = df[df['environment'] == env]
                if not env_data.empty:
                    avg_efficiency = np.mean(env_data['efficiency_absolute'])
                    avg_success = np.mean(env_data['success_absolute'])
                    
                    # Difficulty score (lower performance = higher difficulty)
                    difficulty = 100 - (0.6 * avg_efficiency + 0.4 * avg_success)
                    
                    env_difficulty.append({
                        'environment': env,
                        'difficulty_score': difficulty,
                        'avg_efficiency': avg_efficiency,
                        'avg_success': avg_success
                    })
            
            if env_difficulty:
                env_df = pd.DataFrame(env_difficulty)
                env_df = env_df.sort_values('difficulty_score', ascending=True)
                
                bars = ax.barh(env_df['environment'], env_df['difficulty_score'],
                              color='salmon', edgecolor='black', alpha=0.7)
                ax.set_xlabel('Difficulty Score')
                ax.set_title('Environment Difficulty Ranking\n(Higher = More Challenging)')
                ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, results: List[Dict[str, Any]], 
                                  parameter_name: str, output_path: Path):
        """Plot parameter sensitivity analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_multi)
        
        # Extract parameter values and performance metrics
        param_data = []
        for result in results:
            config = result['config']
            
            # Extract parameter value
            param_value = None
            if hasattr(config.algorithm, parameter_name):
                param_value = getattr(config.algorithm, parameter_name)
            elif hasattr(config.environment, parameter_name):
                param_value = getattr(config.environment, parameter_name)
            elif hasattr(config, parameter_name):
                param_value = getattr(config, parameter_name)
            
            if param_value is not None:
                episodes = result['results']
                
                param_data.append({
                    'parameter_value': param_value,
                    'avg_efficiency': np.mean([e.energy_efficiency * 100 for e in episodes]),
                    'success_rate': np.mean([e.goal_achieved for e in episodes]) * 100,
                    'avg_reward': np.mean([e.total_reward for e in episodes]),
                    'std_efficiency': np.std([e.energy_efficiency * 100 for e in episodes]),
                    'n_episodes': len(episodes)
                })
        
        if not param_data:
            # Create empty plot with message
            fig.text(0.5, 0.5, f'No data found for parameter: {parameter_name}', 
                    ha='center', va='center', fontsize=16)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return
        
        df = pd.DataFrame(param_data)
        df = df.sort_values('parameter_value')
        
        # 1. Parameter vs Efficiency
        ax = axes[0, 0]
        ax.plot(df['parameter_value'], df['avg_efficiency'], 'o-', linewidth=2, markersize=8)
        ax.fill_between(df['parameter_value'], 
                       df['avg_efficiency'] - df['std_efficiency'],
                       df['avg_efficiency'] + df['std_efficiency'], 
                       alpha=0.3)
        ax.set_xlabel(parameter_name)
        ax.set_ylabel('Average Energy Efficiency (%)')
        ax.set_title(f'Energy Efficiency vs {parameter_name}')
        ax.grid(True, alpha=0.3)
        
        # 2. Parameter vs Success Rate
        ax = axes[0, 1]
        ax.plot(df['parameter_value'], df['success_rate'], 'o-', color='green', 
               linewidth=2, markersize=8)
        ax.set_xlabel(parameter_name)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'Success Rate vs {parameter_name}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # 3. Parameter vs Reward
        ax = axes[1, 0]
        ax.plot(df['parameter_value'], df['avg_reward'], 'o-', color='orange', 
               linewidth=2, markersize=8)
        ax.set_xlabel(parameter_name)
        ax.set_ylabel('Average Total Reward')
        ax.set_title(f'Total Reward vs {parameter_name}')
        ax.grid(True, alpha=0.3)
        
        # 4. Sensitivity summary
        ax = axes[1, 1]
        
        # Calculate correlation coefficients
        correlations = {
            'Efficiency': np.corrcoef(df['parameter_value'], df['avg_efficiency'])[0, 1],
            'Success Rate': np.corrcoef(df['parameter_value'], df['success_rate'])[0, 1],
            'Reward': np.corrcoef(df['parameter_value'], df['avg_reward'])[0, 1]
        }
        
        # Bar plot of correlations
        metrics = list(correlations.keys())
        corr_values = list(correlations.values())
        colors = ['green' if abs(c) > 0.5 else 'orange' if abs(c) > 0.3 else 'red' 
                 for c in corr_values]
        
        bars = ax.bar(metrics, corr_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title(f'Parameter Sensitivity Summary\n{parameter_name}')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Strong positive')
        ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5, label='Strong negative')
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add correlation values on bars
        for bar, corr in zip(bars, corr_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                   f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()