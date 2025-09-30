import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_comprehensive_plots(episode_data, iteration_stats, action_stats, save_dir):
    """Create comprehensive visualization of MCTS kettle control results"""
    
    # Convert to DataFrames for easier manipulation
    df_episode = pd.DataFrame(episode_data)
    df_iteration = pd.DataFrame(iteration_stats) if iteration_stats else pd.DataFrame()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: Temperature Trajectory
    ax1 = plt.subplot(6, 2, 1)
    steps = df_episode['steps']
    temps = df_episode['temperatures']
    
    plt.plot(steps, temps, 'b-', linewidth=2.5, label='Temperature')
    plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Target (100°C)')
    plt.axhline(y=97.5, color='red', linestyle=':', alpha=0.7, label='Tolerance (±2.5°C)')
    plt.axhline(y=102.5, color='red', linestyle=':', alpha=0.7)
    plt.axvline(x=200, color='purple', linestyle='--', linewidth=2, label='Target Time')
    plt.fill_between([185, 215], 0, 150, color='purple', alpha=0.1, label='Time Window')
    
    # Highlight goal achievement
    goal_achieved = df_episode['goal_achieved']
    for i, achieved in enumerate(goal_achieved):
        if achieved and i > 0 and not goal_achieved.iloc[i-1]:
            plt.axvline(x=steps.iloc[i], color='green', linewidth=3, alpha=0.7)
            plt.text(steps.iloc[i], temps.iloc[i] + 5, 'GOAL!', 
                    rotation=90, color='green', fontweight='bold')
    
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Trajectory', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(15, 110)
    
    # Plot 2: Actions and Power Usage
    ax2 = plt.subplot(6, 2, 2)
    actions = df_episode['actions'][1:]  # Skip initial None
    powers = df_episode['powers'][1:]    # Skip initial 0
    action_steps = steps[1:]
    
    colors = ['blue' if p == 0 else 'red' for p in powers]
    bars = plt.bar(action_steps, powers, color=colors, alpha=0.7, width=0.8)
    
    # Add action numbers as text
    for i, (step, action, power) in enumerate(zip(action_steps, actions, powers)):
        if power > 0:  # Only label heating actions
            plt.text(step, power + 100, f'A{action}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Time Step')
    plt.ylabel('Power (W)')
    plt.title('Power Usage (Blue=Off, Red=Heat)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3500)
    
    # Plot 3: Energy Consumption and Efficiency
    ax3 = plt.subplot(6, 2, 3)
    energy_Wh = df_episode['energy_consumed_Wh']
    efficiency = df_episode['energy_efficiency']
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(steps, energy_Wh, 'orange', linewidth=2.5, label='Energy (Wh)')
    line2 = ax3_twin.plot(steps, efficiency * 100, 'green', linewidth=2.5, label='Efficiency (%)')
    
    # Theoretical minimum line
    theoretical_min = 1.0 * 4184 * 80 / 3600  # 1kg water, 80°C rise
    ax3.axhline(y=theoretical_min, color='orange', linestyle='--', 
                label=f'Min Energy ({theoretical_min:.1f}Wh)')
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Energy Consumed (Wh)', color='orange')
    ax3_twin.set_ylabel('Energy Efficiency (%)', color='green')
    ax3.set_title('Energy Consumption & Efficiency', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rewards
    ax4 = plt.subplot(6, 2, 4)
    rewards = df_episode['rewards'][1:]  # Skip initial 0
    cum_rewards = df_episode['cumulative_rewards']
    
    ax4_twin = ax4.twinx()
    
    # Bar plot for individual rewards
    reward_colors = ['red' if r < 0 else 'green' if r > 100 else 'gray' for r in rewards]
    bars = ax4.bar(action_steps, rewards, color=reward_colors, alpha=0.7, width=0.8)
    
    # Line plot for cumulative rewards
    ax4_twin.plot(steps, cum_rewards, 'purple', linewidth=2.5, label='Cumulative')
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Step Reward', color='black')
    ax4_twin.set_ylabel('Cumulative Reward', color='purple')
    ax4.set_title('Reward Structure', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Temperature Error Over Time
    ax5 = plt.subplot(6, 2, 5)
    temp_errors = df_episode['temp_error']
    
    plt.plot(steps, temp_errors, 'red', linewidth=2.5, label='Temperature Error')
    plt.axhline(y=2.5, color='orange', linestyle='--', label='Tolerance (2.5°C)')
    plt.fill_between(steps, 0, temp_errors, where=(temp_errors <= 2.5), 
                     color='green', alpha=0.3, label='Within Target')
    plt.fill_between(steps, 0, temp_errors, where=(temp_errors > 2.5), 
                     color='red', alpha=0.3, label='Outside Target')
    
    plt.xlabel('Time Step')
    plt.ylabel('Temperature Error (°C)')
    plt.title('Temperature Error from Target', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: MCTS Statistics (if available)
    ax6 = plt.subplot(6, 2, 6)
    if not df_iteration.empty:
        iter_steps = df_iteration['step']
        goal_counts = df_iteration['goal_found_count']
        avg_rewards = df_iteration['avg_reward']
        
        ax6_twin = ax6.twinx()
        
        bars = ax6.bar(iter_steps, goal_counts, alpha=0.7, color='blue', label='Goals Found')
        line = ax6_twin.plot(iter_steps, avg_rewards, 'red', linewidth=2.5, label='Avg Reward')
        
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Goals Found in MCTS', color='blue')
        ax6_twin.set_ylabel('Average Rollout Reward', color='red')
        ax6.set_title('MCTS Search Quality', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No MCTS iteration data available', 
                ha='center', va='center', transform=ax6.transAxes)
        plt.title('MCTS Statistics', fontsize=14, fontweight='bold')
    
    # Plot 7: Strategy Phase Analysis
    ax7 = plt.subplot(6, 2, 7)
    
    # Color-code by strategy phase
    phase_colors = []
    phase_labels = []
    
    for i in range(1, len(steps)):
        temp = temps.iloc[i]
        step = steps.iloc[i]
        action = df_episode['actions'].iloc[i]
        power = df_episode['powers'].iloc[i]
        
        if temp < 95:
            phase_colors.append('red')
            phase_labels.append('Heating Phase')
        elif temp >= 97.5 and power == 0:
            phase_colors.append('green')
            phase_labels.append('Coasting Phase')
        elif abs(step - 200) <= 15:
            phase_colors.append('blue')
            phase_labels.append('Deadline Phase')
        else:
            phase_colors.append('gray')
            phase_labels.append('Transition')
    
    scatter = plt.scatter(action_steps, temps[1:], c=phase_colors, s=30, alpha=0.7)
    plt.plot(steps, temps, 'k-', linewidth=1, alpha=0.3)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Heating Phase'),
        Patch(facecolor='green', label='Coasting Phase'),
        Patch(facecolor='blue', label='Deadline Phase'),
        Patch(facecolor='gray', label='Transition')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.axhline(y=100, color='purple', linestyle='--', alpha=0.5)
    plt.axvline(x=200, color='purple', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.title('Control Strategy Phases', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Action Selection Analysis
    ax8 = plt.subplot(6, 2, 8)
    
    # Analyze action patterns
    if action_stats:
        # Get data for last few steps where we have good statistics
        recent_steps = min(10, len(action_stats))
        step_indices = list(range(len(action_stats) - recent_steps, len(action_stats)))
        
        action_0_visits = []
        action_1_visits = []
        
        for i in step_indices:
            stats = action_stats[i]
            action_0_visits.append(stats.get(0, {}).get('visits', 0))
            action_1_visits.append(stats.get(1, {}).get('visits', 0))
        
        x_pos = np.arange(len(step_indices))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, action_0_visits, width, label='Action 0 (Off)', alpha=0.7)
        bars2 = plt.bar(x_pos + width/2, action_1_visits, width, label='Action 1 (Heat)', alpha=0.7)
        
        plt.xlabel('Recent Steps (relative)')
        plt.ylabel('MCTS Visits')
        plt.title('MCTS Action Exploration (Recent Steps)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add step labels
        step_labels = [f'{len(action_stats) - recent_steps + i + 1}' for i in range(len(step_indices))]
        plt.xticks(x_pos, step_labels)
    else:
        plt.text(0.5, 0.5, 'No action statistics available', 
                ha='center', va='center', transform=ax8.transAxes)
        plt.title('Action Selection Analysis', fontsize=14, fontweight='bold')
    
    # Plot 9: Energy Efficiency Histogram
    ax9 = plt.subplot(6, 2, 9)
    
    # Calculate efficiency at different time points
    efficiency_series = df_episode['energy_efficiency'][1:]  # Skip initial
    
    plt.plot(action_steps, efficiency_series * 100, 'purple', linewidth=2.5, label='Efficiency')
    plt.axhline(y=100, color='green', linestyle='--', label='Perfect (100%)')
    plt.axhline(y=90, color='orange', linestyle=':', label='Excellent (90%)')
    plt.axhline(y=80, color='red', linestyle=':', label='Good (80%)')
    
    plt.fill_between(action_steps, 0, efficiency_series * 100, alpha=0.3, color='purple')
    
    plt.xlabel('Time Step')
    plt.ylabel('Energy Efficiency (%)')
    plt.title('Energy Efficiency Evolution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 110)
    
    # Plot 10: Strategic Timeline
    ax10 = plt.subplot(6, 2, 10)
    
    # Create strategic timeline view
    step_range = range(0, min(len(steps), 250), 10)  # Sample every 10 steps
    
    timeline_data = []
    for s in step_range:
        if s < len(steps):
            temp = temps.iloc[s]
            remaining = 200 - s
            if s > 0 and s < len(df_episode['actions']):
                heating = df_episode['powers'].iloc[s] > 0
            else:
                heating = False
            
            timeline_data.append({
                'step': s,
                'temp': temp,
                'remaining': remaining,
                'heating': heating,
                'temp_gap': 100 - temp,
                'urgency': max(0, 1 - remaining / 112)  # 112 ≈ optimal heating time
            })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    if not df_timeline.empty:
        # Bubble chart: x=step, y=temp, size=urgency, color=heating
        colors = ['red' if h else 'blue' for h in df_timeline['heating']]
        sizes = [max(20, u * 200) for u in df_timeline['urgency']]
        
        scatter = plt.scatter(df_timeline['step'], df_timeline['temp'], 
                            c=colors, s=sizes, alpha=0.6)
        
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7)
        plt.axvline(x=200, color='purple', linestyle='--', alpha=0.7)
        
        # Add annotations for key points
        for i, row in df_timeline.iterrows():
            if row['heating'] and row['step'] > 0:
                plt.annotate(f"Heat@{row['step']}", 
                           (row['step'], row['temp']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.title('Strategic Timeline (Red=Heating, Blue=Waiting, Size=Urgency)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No timeline data available', 
                ha='center', va='center', transform=ax10.transAxes)
        plt.title('Strategic Timeline', fontsize=14, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    
    # Save main plot
    plot_file = os.path.join(save_dir, "comprehensive_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary plot
    create_summary_plot(episode_data, save_dir)
    
    # Create strategy analysis plot
    create_strategy_analysis(episode_data, action_stats, save_dir)
    
    print(f"Plots saved to: {save_dir}")


def create_summary_plot(episode_data, save_dir):
    """Create a concise summary plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    df = pd.DataFrame(episode_data)
    steps = df['steps']
    
    # Temperature and target
    ax1 = axes[0, 0]
    ax1.plot(steps, df['temperatures'], 'b-', linewidth=3, label='Temperature')
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Target')
    ax1.axhline(y=97.5, color='red', linestyle=':', alpha=0.7)
    ax1.axhline(y=102.5, color='red', linestyle=':', alpha=0.7)
    ax1.axvline(x=200, color='purple', linestyle='--', linewidth=2, label='Deadline')
    ax1.fill_between([185, 215], 0, 150, color='purple', alpha=0.1)
    
    # Mark goal achievement
    for i, achieved in enumerate(df['goal_achieved']):
        if achieved and i > 0 and not df['goal_achieved'].iloc[i-1]:
            ax1.axvline(x=steps.iloc[i], color='green', linewidth=3)
            ax1.text(steps.iloc[i], df['temperatures'].iloc[i] + 5, 'GOAL!', 
                    rotation=90, color='green', fontweight='bold')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Control', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(15, 110)
    
    # Energy consumption
    ax2 = axes[0, 1]
    ax2.plot(steps, df['energy_consumed_Wh'], 'orange', linewidth=3, label='Energy Used')
    
    theoretical_min = 1.0 * 4184 * 80 / 3600
    ax2.axhline(y=theoretical_min, color='green', linestyle='--', 
                linewidth=2, label=f'Theoretical Min ({theoretical_min:.1f}Wh)')
    
    ax2.fill_between(steps, theoretical_min, df['energy_consumed_Wh'], 
                     where=(df['energy_consumed_Wh'] > theoretical_min),
                     color='red', alpha=0.3, label='Excess Energy')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Energy (Wh)')
    ax2.set_title('Energy Consumption', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Actions
    ax3 = axes[1, 0]
    actions_clean = [a for a in df['actions'] if a is not None]
    powers_clean = [df['powers'].iloc[i+1] for i in range(len(actions_clean))]
    steps_clean = list(range(1, len(actions_clean) + 1))
    
    colors = ['blue' if p == 0 else 'red' for p in powers_clean]
    bars = ax3.bar(steps_clean, powers_clean, color=colors, alpha=0.7, width=0.8)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Power (W)')
    ax3.set_title('Heating Actions (Blue=Off, Red=On)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Efficiency over time
    ax4 = axes[1, 1]
    efficiency_pct = [e * 100 for e in df['energy_efficiency']]
    ax4.plot(steps, efficiency_pct, 'purple', linewidth=3, label='Efficiency')
    ax4.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Perfect')
    ax4.axhline(y=90, color='orange', linestyle=':', linewidth=2, label='Excellent')
    
    ax4.fill_between(steps, 0, efficiency_pct, alpha=0.3, color='purple')
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Energy Efficiency (%)')
    ax4.set_title('Energy Efficiency', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 110)
    
    plt.tight_layout()
    
    summary_file = os.path.join(save_dir, "summary_plot.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_strategy_analysis(episode_data, action_stats, save_dir):
    """Create detailed strategy analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    df = pd.DataFrame(episode_data)
    
    # Heating pattern analysis
    ax1 = axes[0, 0]
    actions = [a for a in df['actions'] if a is not None]
    
    if actions:
        # Find heating periods
        heating_periods = []
        current_period = None
        
        for i, action in enumerate(actions):
            step = i + 1
            is_heating = df['powers'].iloc[step] > 0
            
            if is_heating:
                if current_period is None:
                    current_period = {'start': step, 'end': step}
                else:
                    current_period['end'] = step
            else:
                if current_period is not None:
                    heating_periods.append(current_period)
                    current_period = None
        
        if current_period is not None:
            heating_periods.append(current_period)
        
        # Plot heating periods
        for i, period in enumerate(heating_periods):
            ax1.barh(i, period['end'] - period['start'] + 1, 
                    left=period['start'], alpha=0.7, 
                    label=f"Period {i+1}: steps {period['start']}-{period['end']}")
        
        ax1.axvline(x=200, color='red', linestyle='--', label='Deadline')
        ax1.axvline(x=88, color='orange', linestyle=':', label='Optimal Start (~88)')
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Heating Period')
        ax1.set_title('Heating Periods Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Temperature vs time to deadline
    ax2 = axes[0, 1]
    time_to_deadline = [200 - s for s in df['steps']]
    scatter = ax2.scatter(time_to_deadline, df['temperatures'], 
                         c=df['steps'], cmap='viridis', alpha=0.6)
    
    # Add optimal heating curve (approximate)
    optimal_times = np.linspace(200, 0, 100)
    optimal_temps = []
    for t in optimal_times:
        if t > 112:  # Before optimal start
            optimal_temps.append(20)
        else:  # During heating
            progress = (112 - t) / 112
            optimal_temps.append(20 + 80 * progress)
    
    ax2.plot(optimal_times, optimal_temps, 'r--', linewidth=2, 
             label='Theoretical Optimal', alpha=0.7)
    
    ax2.set_xlabel('Steps to Deadline')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Temperature vs Time to Deadline', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Actual Step')
    
    # Decision making analysis
    ax3 = axes[1, 0]
    
    if action_stats:
        # Analyze decision confidence over time
        steps_analyzed = []
        action_0_confidence = []
        action_1_confidence = []
        
        for i, stats in enumerate(action_stats):
            if 0 in stats and 1 in stats:
                total_visits = stats[0]['visits'] + stats[1]['visits']
                if total_visits > 0:
                    steps_analyzed.append(i + 1)
                    action_0_confidence.append(stats[0]['visits'] / total_visits)
                    action_1_confidence.append(stats[1]['visits'] / total_visits)
        
        if steps_analyzed:
            ax3.plot(steps_analyzed, action_0_confidence, 'b-', linewidth=2, 
                    label='Wait (Action 0)', marker='o', markersize=3)
            ax3.plot(steps_analyzed, action_1_confidence, 'r-', linewidth=2, 
                    label='Heat (Action 1)', marker='s', markersize=3)
            
            ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax3.axvline(x=88, color='orange', linestyle=':', label='Optimal Start')
            ax3.axvline(x=200, color='purple', linestyle='--', label='Deadline')
            
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('MCTS Action Preference')
            ax3.set_title('MCTS Decision Confidence', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
    
    # Strategy efficiency analysis
    ax4 = axes[1, 1]
    
    # Calculate rolling efficiency and temperature error
    window_size = 10
    if len(df) > window_size:
        rolling_efficiency = df['energy_efficiency'].rolling(window=window_size).mean()
        rolling_temp_error = df['temp_error'].rolling(window=window_size).mean()
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(df['steps'], rolling_efficiency * 100, 'green', 
                        linewidth=2, label='Efficiency (10-step avg)')
        line2 = ax4_twin.plot(df['steps'], rolling_temp_error, 'red', 
                             linewidth=2, label='Temp Error (10-step avg)')
        
        ax4.axhline(y=90, color='green', linestyle='--', alpha=0.5)
        ax4_twin.axhline(y=2.5, color='red', linestyle='--', alpha=0.5)
        
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Efficiency (%)', color='green')
        ax4_twin.set_ylabel('Temperature Error (°C)', color='red')
        ax4.set_title('Strategy Quality Metrics', fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    strategy_file = os.path.join(save_dir, "strategy_analysis.png")
    plt.savefig(strategy_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_text_report(episode_data, iteration_stats, save_dir):
    """Generate a detailed text report of the results"""
    
    df = pd.DataFrame(episode_data)
    
    report_file = os.path.join(save_dir, "analysis_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("MCTS KETTLE CONTROL - DETAILED ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Episode Summary
        f.write("EPISODE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total steps: {len(df) - 1}\n")
        f.write(f"Final temperature: {df['temperatures'].iloc[-1]:.1f}°C\n")
        f.write(f"Target temperature: 100°C (±2.5°C)\n")
        f.write(f"Final energy consumption: {df['energy_consumed_Wh'].iloc[-1]:.1f} Wh\n")
        f.write(f"Final energy efficiency: {df['energy_efficiency'].iloc[-1]:.1%}\n")
        f.write(f"Goal achieved: {'YES' if df['goal_achieved'].iloc[-1] else 'NO'}\n")
        
        if df['goal_achieved'].any():
            goal_step = df[df['goal_achieved']].index[0]
            f.write(f"Goal achieved at step: {goal_step}\n")
        
        f.write(f"Total reward: {df['cumulative_rewards'].iloc[-1]:.2f}\n\n")
        
        # Strategy Analysis
        actions = [a for a in df['actions'] if a is not None]
        if actions:
            heating_actions = sum(1 for i in range(1, len(df)) if df['powers'].iloc[i] > 0)
            total_actions = len(actions)
            
            f.write("STRATEGY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total actions taken: {total_actions}\n")
            f.write(f"Heating actions: {heating_actions} ({100*heating_actions/total_actions:.1f}%)\n")
            f.write(f"Waiting actions: {total_actions - heating_actions} ({100*(total_actions-heating_actions)/total_actions:.1f}%)\n")
            
            # Find first and last heating
            first_heat = None
            last_heat = None
            for i in range(1, len(df)):
                if df['powers'].iloc[i] > 0:
                    if first_heat is None:
                        first_heat = i
                    last_heat = i
            
            if first_heat:
                f.write(f"First heating at step: {first_heat}\n")
                f.write(f"Last heating at step: {last_heat}\n")
                f.write(f"Heating duration: {last_heat - first_heat + 1} steps\n")
                f.write(f"Started heating with {200 - first_heat} steps to deadline\n")
            
            # Strategy assessment
            f.write("\nSTRATEGY ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            
            if first_heat and first_heat > 150:
                f.write("✓ DELAYED HEATING STRATEGY DETECTED\n")
                f.write("  Agent successfully learned to wait before heating\n")
            elif first_heat and first_heat > 100:
                f.write("⚠ MODERATE DELAY STRATEGY\n")
                f.write("  Agent showed some delay but could be more efficient\n")
            else:
                f.write("✗ EARLY HEATING STRATEGY\n")
                f.write("  Agent heated too early, wasting energy\n")
            
            efficiency = df['energy_efficiency'].iloc[-1]
            if efficiency > 0.9:
                f.write("✓ EXCELLENT ENERGY EFFICIENCY (>90%)\n")
            elif efficiency > 0.8:
                f.write("✓ GOOD ENERGY EFFICIENCY (>80%)\n")
            elif efficiency > 0.7:
                f.write("⚠ MODERATE ENERGY EFFICIENCY (>70%)\n")
            else:
                f.write("✗ POOR ENERGY EFFICIENCY (<70%)\n")
        
        # MCTS Performance
        if iteration_stats:
            f.write("\nMCTS PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            
            df_iter = pd.DataFrame(iteration_stats)
            avg_goals_found = df_iter['goal_found_count'].mean()
            max_goals_found = df_iter['goal_found_count'].max()
            avg_rollout_reward = df_iter['avg_reward'].mean()
            
            f.write(f"Average goals found per step: {avg_goals_found:.1f}\n")
            f.write(f"Maximum goals found in one step: {max_goals_found}\n")
            f.write(f"Average rollout reward: {avg_rollout_reward:.2f}\n")
            
            # Learning progression
            early_goals = df_iter['goal_found_count'][:len(df_iter)//3].mean()
            late_goals = df_iter['goal_found_count'][2*len(df_iter)//3:].mean()
            
            f.write(f"Early episode goal finding: {early_goals:.1f}\n")
            f.write(f"Late episode goal finding: {late_goals:.1f}\n")
            
            if late_goals > early_goals * 1.5:
                f.write("✓ MCTS IMPROVED OVER TIME\n")
            elif late_goals > early_goals:
                f.write("⚠ SLIGHT MCTS IMPROVEMENT\n")
            else:
                f.write("✗ NO CLEAR MCTS IMPROVEMENT\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        
        if df['goal_achieved'].iloc[-1]:
            f.write("✓ Goal was achieved successfully!\n")
            
            if df['energy_efficiency'].iloc[-1] < 0.9:
                f.write("• Consider increasing MCTS iterations for better energy efficiency\n")
                f.write("• The optimal strategy is to wait until ~step 88 before heating\n")
        else:
            f.write("✗ Goal was not achieved\n")
            f.write("• Increase MCTS iterations per action\n")
            f.write("• Check if enough time was allowed for heating\n")
            f.write("• Consider adjusting UCB exploration parameter\n")
        
        if first_heat and first_heat < 100:
            f.write("• Agent should learn to wait longer before starting to heat\n")
            f.write("• This would improve energy efficiency significantly\n")
        
        f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Analysis report saved to: {report_file}")


def visualize_results(episode_data, iteration_stats, action_stats, save_dir):
    """Main function to create all visualizations and reports"""
    
    # Create all plots
    create_comprehensive_plots(episode_data, iteration_stats, action_stats, save_dir)
    
    # Generate text report
    generate_text_report(episode_data, iteration_stats, save_dir)
    
    # Print summary
    df = pd.DataFrame(episode_data)
    print(f"\nVisualization Summary:")
    print(f"Goal achieved: {'YES' if df['goal_achieved'].iloc[-1] else 'NO'}")
    print(f"Final temperature: {df['temperatures'].iloc[-1]:.1f}°C")
    print(f"Energy efficiency: {df['energy_efficiency'].iloc[-1]:.1%}")
    print(f"Total energy: {df['energy_consumed_Wh'].iloc[-1]:.1f} Wh")
    
    actions = [a for a in df['actions'] if a is not None]
    if actions:
        heating_actions = sum(1 for i in range(1, len(df)) if df['powers'].iloc[i] > 0)
        print(f"Heating ratio: {heating_actions}/{len(actions)} ({100*heating_actions/len(actions):.1f}%)")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        
        # Load data
        episode_file = os.path.join(results_dir, "episode_data.csv")
        iteration_file = os.path.join(results_dir, "iteration_stats.csv")
        
        if os.path.exists(episode_file):
            episode_df = pd.read_csv(episode_file)
            episode_data = episode_df.to_dict('list')
            
            iteration_stats = []
            if os.path.exists(iteration_file):
                iteration_df = pd.read_csv(iteration_file)
                iteration_stats = iteration_df.to_dict('records')
            
            visualize_results(episode_data, iteration_stats, [], results_dir)
        else:
            print(f"Episode data file not found: {episode_file}")
    else:
        print("Usage: python visualization_module.py <results_directory>")