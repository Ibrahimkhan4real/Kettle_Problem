"""
Comprehensive comparison of MCTS, MCTS-RAVE, and Bang-Bang control
for the energy-efficient kettle control problem.

This script:
1. Runs multiple episodes of each algorithm
2. Collects performance metrics
3. Generates detailed visualizations
4. Saves results for further analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime
import os
from collections import defaultdict

# Import the algorithms
from simple_MCTS_env1 import PlainMCTSAgent
from MCTS_RAVE_v22 import MCTSRAVEAgent
from bang_bang_approach_v1 import BangBangAgent

# Import the environment
from kettle_dynamic_env_v24 import KettleEnv


def run_experiments(n_episodes=5, iterations_per_action=2000, save_dir='results'):
    """Run experiments comparing all algorithms"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize results storage
    results = {
        'plain_mcts': [],
        'mcts_rave': [],
        'bang_bang_optimal': [],
        'bang_bang_time_aware': [],
        'metadata': {
            'n_episodes': n_episodes,
            'iterations_per_action': iterations_per_action,
            'timestamp': timestamp
        }
    }
    
    # Create agents
    agents = {
        'plain_mcts': PlainMCTSAgent(iterations_per_action=iterations_per_action),
        'mcts_rave': MCTSRAVEAgent(iterations_per_action=iterations_per_action),
        'bang_bang_optimal': BangBangAgent(control_type='optimal'),
        'bang_bang_time_aware': BangBangAgent(control_type='time_aware')
    }
    
    # Run experiments
    for agent_name, agent in agents.items():
        print(f"\n{'='*70}")
        print(f"Running experiments for: {agent_name}")
        print(f"{'='*70}")
        
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            
            # Create environment
            env = KettleEnv(
                initial_temp=20.0,
                ambient_temp=25.0,
                max_steps=250,
                target_deadline_step=200
            )
            
            # Run episode
            trajectory = agent.run_episode(env, render=False, verbose=(episode == 0))
            
            # Store results
            results[agent_name].append(trajectory)
            
            # Quick summary
            if episode > 0:  # Don't print for first episode (already verbose)
                print(f"  Reward: {trajectory['total_reward']:.1f}, "
                      f"Energy: {trajectory['final_energy_Wh']:.1f} Wh, "
                      f"Goal: {'Yes' if trajectory['goal_achieved'] else 'No'}")
    
    # Save raw results
    with open(os.path.join(save_dir, f'results_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary statistics
    summary = compute_summary_statistics(results)
    with open(os.path.join(save_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")
    
    return results


def compute_summary_statistics(results):
    """Compute summary statistics for all algorithms"""
    summary = {}
    
    for agent_name, trajectories in results.items():
        if agent_name == 'metadata':
            continue
            
        # Extract metrics
        rewards = [t['total_reward'] for t in trajectories]
        energies = [t['final_energy_Wh'] for t in trajectories]
        goals = [t['goal_achieved'] for t in trajectories]
        final_temps = [t['final_temperature'] for t in trajectories]
        episode_lengths = [t['episode_length'] for t in trajectories]
        
        # Compute efficiency
        theoretical_min = 93.0  # Wh
        efficiencies = [theoretical_min / e * 100 if e > 0 else 0 for e in energies]
        
        # Store statistics
        summary[agent_name] = {
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards))
            },
            'energy_Wh': {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies))
            },
            'efficiency_percent': {
                'mean': float(np.mean(efficiencies)),
                'std': float(np.std(efficiencies)),
                'min': float(np.min(efficiencies)),
                'max': float(np.max(efficiencies))
            },
            'goal_achievement_rate': float(np.mean(goals)),
            'final_temperature': {
                'mean': float(np.mean(final_temps)),
                'std': float(np.std(final_temps))
            },
            'episode_length': {
                'mean': float(np.mean(episode_lengths)),
                'std': float(np.std(episode_lengths))
            }
        }
    
    return summary


def generate_visualizations(results, save_dir='figures'):
    """Generate comprehensive visualizations"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme
    colors = {
        'plain_mcts': '#1f77b4',      # Blue
        'mcts_rave': '#2ca02c',        # Green
        'bang_bang_optimal': '#ff7f0e', # Orange
        'bang_bang_time_aware': '#d62728' # Red
    }
    
    labels = {
        'plain_mcts': 'MCTS',
        'mcts_rave': 'MCTS-RAVE',
        'bang_bang_optimal': 'Bang-Bang (Optimal)',
        'bang_bang_time_aware': 'Bang-Bang (Time-Aware)'
    }
    
    # 1. Performance Comparison Box Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prepare data
    agent_names = ['plain_mcts', 'mcts_rave', 'bang_bang_optimal', 'bang_bang_time_aware']
    
    # Total Reward
    ax = axes[0, 0]
    reward_data = [[t['total_reward'] for t in results[agent]] for agent in agent_names]
    bp = ax.boxplot(reward_data, labels=[labels[a] for a in agent_names], patch_artist=True)
    for patch, agent in zip(bp['boxes'], agent_names):
        patch.set_facecolor(colors[agent])
        patch.set_alpha(0.7)
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward Distribution')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Energy Consumption
    ax = axes[0, 1]
    energy_data = [[t['final_energy_Wh'] for t in results[agent]] for agent in agent_names]
    bp = ax.boxplot(energy_data, labels=[labels[a] for a in agent_names], patch_artist=True)
    for patch, agent in zip(bp['boxes'], agent_names):
        patch.set_facecolor(colors[agent])
        patch.set_alpha(0.7)
    ax.axhline(y=93, color='green', linestyle='--', label='Theoretical Min')
    ax.set_ylabel('Energy Consumption (Wh)')
    ax.set_title('Energy Usage Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Energy Efficiency
    ax = axes[1, 0]
    efficiency_data = [[93.0/t['final_energy_Wh']*100 if t['final_energy_Wh'] > 0 else 0 
                        for t in results[agent]] for agent in agent_names]
    bp = ax.boxplot(efficiency_data, labels=[labels[a] for a in agent_names], patch_artist=True)
    for patch, agent in zip(bp['boxes'], agent_names):
        patch.set_facecolor(colors[agent])
        patch.set_alpha(0.7)
    ax.axhline(y=90, color='green', linestyle='--', label='90% Target')
    ax.set_ylabel('Energy Efficiency (%)')
    ax.set_title('Energy Efficiency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Goal Achievement Rate
    ax = axes[1, 1]
    goal_rates = [np.mean([t['goal_achieved'] for t in results[agent]]) * 100 
                  for agent in agent_names]
    bars = ax.bar(range(len(agent_names)), goal_rates, 
                   color=[colors[a] for a in agent_names], alpha=0.7)
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels([labels[a] for a in agent_names], rotation=45)
    ax.set_ylabel('Goal Achievement Rate (%)')
    ax.set_title('Success Rate')
    ax.set_ylim(0, 105)
    for i, v in enumerate(goal_rates):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Representative Trajectories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select representative trajectory (best performing) for each algorithm
    for idx, agent in enumerate(agent_names):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Find best trajectory
        best_traj = max(results[agent], key=lambda t: t['total_reward'])
        
        # Plot temperature
        ax.plot(best_traj['steps'], best_traj['temperatures'], 
                color=colors[agent], linewidth=2, label='Temperature')
        
        # Add action shading
        for i, action in enumerate(best_traj['actions']):
            if action == 1:  # Heating
                ax.axvspan(i, i+1, alpha=0.2, color='orange', ymin=0, ymax=0.7)
        
        # Add reference lines
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Target')
        ax.axhline(y=97.5, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(y=102.5, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=200, color='magenta', linestyle='--', alpha=0.5, label='Deadline')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{labels[agent]} - Best Run\n'
                     f'Energy: {best_traj["final_energy_Wh"]:.1f} Wh, '
                     f'Reward: {best_traj["total_reward"]:.0f}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 250)
        ax.set_ylim(15, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'representative_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Control Strategy Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, agent in enumerate(agent_names):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Analyze all trajectories
        first_heat_times = []
        heating_durations = []
        heating_patterns = []
        
        for traj in results[agent]:
            # Find first heating time
            first_heat = None
            last_heat = None
            for i, action in enumerate(traj['actions']):
                if action == 1:
                    if first_heat is None:
                        first_heat = i
                    last_heat = i
            
            if first_heat is not None:
                first_heat_times.append(first_heat)
                heating_durations.append(last_heat - first_heat + 1)
                
                # Create heating pattern (binary array)
                pattern = np.zeros(250)
                pattern[np.array(traj['steps'][:-1])[np.array(traj['actions']) == 1]] = 1
                heating_patterns.append(pattern)
        
        # Plot average heating pattern
        if heating_patterns:
            avg_pattern = np.mean(heating_patterns, axis=0)
            x = np.arange(250)
            ax.fill_between(x, 0, avg_pattern, color=colors[agent], alpha=0.7)
            ax.plot(x, avg_pattern, color=colors[agent], linewidth=2)
        
        ax.axvline(x=200, color='magenta', linestyle='--', alpha=0.5, label='Deadline')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Heating Probability')
        ax.set_title(f'{labels[agent]}\n'
                     f'Avg First Heat: {np.mean(first_heat_times):.0f}, '
                     f'Avg Duration: {np.mean(heating_durations):.0f}')
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'control_strategies.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Learning Curves (for MCTS methods)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cumulative rewards over episodes
    for agent in ['plain_mcts', 'mcts_rave']:
        rewards = [t['total_reward'] for t in results[agent]]
        cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax1.plot(range(1, len(rewards) + 1), cumulative_avg, 
                marker='o', label=labels[agent], color=colors[agent], linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Total Reward')
    ax1.set_title('Learning Progress: Average Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Energy efficiency over episodes
    for agent in ['plain_mcts', 'mcts_rave']:
        efficiencies = [93.0/t['final_energy_Wh']*100 if t['final_energy_Wh'] > 0 else 0 
                       for t in results[agent]]
        ax2.plot(range(1, len(efficiencies) + 1), efficiencies, 
                marker='o', label=labels[agent], color=colors[agent], linewidth=2)
    
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Energy Efficiency (%)')
    ax2.set_title('Learning Progress: Energy Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Statistical Summary Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    summary = compute_summary_statistics(results)
    
    headers = ['Algorithm', 'Avg Reward', 'Avg Energy (Wh)', 'Efficiency (%)', 
               'Success Rate (%)', 'First Heat Step']
    
    table_data = []
    for agent in agent_names:
        # Calculate average first heating step
        first_heats = []
        for traj in results[agent]:
            for i, action in enumerate(traj['actions']):
                if action == 1:
                    first_heats.append(i + 1)
                    break
        
        avg_first_heat = np.mean(first_heats) if first_heats else 0
        
        row = [
            labels[agent],
            f"{summary[agent]['reward']['mean']:.0f} ± {summary[agent]['reward']['std']:.0f}",
            f"{summary[agent]['energy_Wh']['mean']:.1f} ± {summary[agent]['energy_Wh']['std']:.1f}",
            f"{summary[agent]['efficiency_percent']['mean']:.1f} ± {summary[agent]['efficiency_percent']['std']:.1f}",
            f"{summary[agent]['goal_achievement_rate']*100:.0f}",
            f"{avg_first_heat:.0f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    for i, agent in enumerate(agent_names):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors[agent])
            table[(i+1, j)].set_alpha(0.3)
    
    plt.title('Performance Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {save_dir}")


def main():
    """Main comparison script"""
    print("="*70)
    print("Energy-Efficient Kettle Control: Algorithm Comparison")
    print("="*70)
    print("\nAlgorithms to compare:")
    print("1. Plain MCTS (without RAVE)")
    print("2. MCTS with RAVE")
    print("3. Bang-Bang Control (Optimal variant)")
    print("4. Bang-Bang Control (Time-aware variant)")
    print("\nEnvironment: Kettle with policy-invariant reward shaping")
    print("="*70)
    
    # Run experiments
    n_episodes = 10
    iterations = 5000  # Increase for better MCTS performance
    
    print(f"\nRunning {n_episodes} episodes per algorithm...")
    print(f"MCTS iterations per action: {iterations}")
    
    results = run_experiments(n_episodes=n_episodes, 
                            iterations_per_action=iterations,
                            save_dir='experiment_results')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(results, save_dir='experiment_figures')
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    summary = compute_summary_statistics(results)
    
    print(f"\n{'Algorithm':<25} {'Avg Reward':>12} {'Avg Energy':>12} "
          f"{'Efficiency':>12} {'Success':>10}")
    print("-" * 75)
    
    for agent in ['plain_mcts', 'mcts_rave', 'bang_bang_optimal', 'bang_bang_time_aware']:
        s = summary[agent]
        print(f"{agent:<25} {s['reward']['mean']:>12.0f} "
              f"{s['energy_Wh']['mean']:>10.1f} Wh "
              f"{s['efficiency_percent']['mean']:>11.1f}% "
              f"{s['goal_achievement_rate']*100:>9.0f}%")
    
    print("\nExperiment complete! Check the following directories:")
    print("- experiment_results/ : Raw data and statistics")
    print("- experiment_figures/ : Visualizations and plots")


if __name__ == "__main__":
    main()