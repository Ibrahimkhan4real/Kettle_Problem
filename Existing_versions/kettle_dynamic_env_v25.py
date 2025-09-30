import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from math import log, sqrt
import os

# Import the energy-efficient environment
# Note: Update this import to match your file name
# from kettle_dynamic_env_v4_energy_efficient import KettleEnv
from kettle_dynamic_env_v23 import KettleEnv  # Change this when using new env

# --- Configuration ---
EPISODES_TO_RUN = 1
MAX_TIMESTEPS_PER_EPISODE = 250

# MCTS-RAVE Hyperparameters (tuned for energy efficiency)
UCB_C = 1.5  # Lower to focus on exploitation once good strategies found
RAVE_C = 8000.0  # Higher to maintain RAVE influence longer
MCTS_ITERATIONS_PER_ACTION = 10000
MAX_ROLLOUT_DEPTH = 100
DISCOUNT_FACTOR = 0.99
TREE_REUSE = True
USE_RAVE = True
OPTIMISTIC_INIT = -5.0  # Negative to account for energy penalties
PROGRESSIVE_WIDENING_ALPHA = 0.5
MIN_VISITS_FOR_EXPANSION = 10


class EnergyEfficientRAVENode:
    """
    MCTS Node optimized for energy-efficient control
    """
    def __init__(self, env_state_copy, done_flag, parent_node, observation, action_idx_leading_to_node):
        self.children = {}
        
        # Standard MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        
        # RAVE statistics
        self.rave_visits = {}
        self.rave_values = {}
        
        # Node properties
        self.env_state = env_state_copy
        self.observation = observation
        self.is_terminal = done_flag
        self.parent = parent_node
        self.action_that_led_here = action_idx_leading_to_node
        
        # Available actions
        self.untried_actions = list(range(env_state_copy.action_space.n))
        random.shuffle(self.untried_actions)
        
        # Energy tracking
        self.immediate_reward = 0.0
        self.cumulative_energy = 0.0  # Track energy used to reach this state
        if parent_node and action_idx_leading_to_node is not None:
            parent_energy = parent_node.cumulative_energy
            power_used = env_state_copy.actions[action_idx_leading_to_node]
            self.cumulative_energy = parent_energy + power_used * env_state_copy.dt
        
        # Initialize with less optimistic value for energy penalty environment
        self._initial_q = OPTIMISTIC_INIT
        
        # Track best rollout seen
        self.best_rollout_value = -float('inf')
        self.best_efficiency_seen = 0.0

    def get_rave_value(self, action):
        """Get RAVE value for an action"""
        if action not in self.rave_visits or self.rave_visits[action] == 0:
            return 0.0
        return self.rave_values[action] / self.rave_visits[action]

    def get_beta(self, child_visits, rave_constant=RAVE_C):
        """Calculate RAVE weight"""
        return sqrt(rave_constant / (3 * child_visits + rave_constant))

    def get_value_estimate(self):
        """Get current value estimate"""
        if self.visits == 0:
            return self._initial_q
        return self.value_sum / self.visits

    def get_uct_rave_value(self, action, c_param=UCB_C, use_rave=USE_RAVE):
        """
        UCT-RAVE value calculation with energy efficiency consideration
        """
        if action not in self.children:
            return float('inf')
        
        child = self.children[action]
        
        if child.visits == 0:
            return float('inf')
        
        # Standard UCT components
        exploitation = child.value_sum / child.visits
        
        # Adaptive exploration based on context
        steps_to_deadline = self.observation[1] if self.observation is not None else 0
        current_temp = self.observation[0] + 100 if self.observation is not None else 20
        
        # Reduce exploration when we're already efficient
        if abs(current_temp - 100) < 2.5 and steps_to_deadline > -15:
            exploration_multiplier = 0.5  # Less exploration when at target
        elif -20 <= steps_to_deadline <= 20:
            exploration_multiplier = 1.5  # More exploration near deadline
        else:
            exploration_multiplier = 1.0
            
        exploration = exploration_multiplier * c_param * sqrt(log(self.visits) / child.visits)
        
        if not use_rave:
            return exploitation + exploration
        
        # RAVE component
        rave_value = self.get_rave_value(action)
        beta = self.get_beta(child.visits)
        
        # Combine UCT and RAVE values
        combined_value = (1 - beta) * exploitation + beta * rave_value
        
        return combined_value + exploration

    def select_child(self, c_param=UCB_C):
        """Select best child using UCT-RAVE"""
        if not self.children:
            return None
        
        # Calculate UCT-RAVE values for all children
        action_values = []
        for action, child in self.children.items():
            value = self.get_uct_rave_value(action, c_param)
            action_values.append((action, value, child))
        
        # Select action with highest value
        best_action, best_value, best_child = max(action_values, key=lambda x: x[1])
        
        return best_child

    def should_expand(self):
        """Progressive widening"""
        if self.is_terminal:
            return False
        if not self.untried_actions:
            return False
        if not self.children:
            return True
        k = len(self.children)
        return self.visits >= MIN_VISITS_FOR_EXPANSION * (k ** PROGRESSIVE_WIDENING_ALPHA)

    def expand(self):
        """Expand one untried action"""
        if not self.untried_actions or self.is_terminal:
            return None
        
        action = self.untried_actions.pop()
        
        # Create child state
        child_env = deepcopy(self.env_state)
        obs, reward, done, truncated, info = child_env.step(action)
        child_is_terminal = done or truncated
        
        # Create child node
        child = EnergyEfficientRAVENode(child_env, child_is_terminal, self, obs, action)
        child.immediate_reward = reward
        self.children[action] = child
        
        return child

    def update(self, simulation_value, action_sequence, efficiency_achieved=0.0):
        """Update node statistics with RAVE"""
        self.visits += 1
        self.value_sum += simulation_value
        self.best_efficiency_seen = max(self.best_efficiency_seen, efficiency_achieved)
        
        # Update RAVE statistics
        if USE_RAVE:
            for action in action_sequence:
                if action not in self.rave_visits:
                    self.rave_visits[action] = 0
                    self.rave_values[action] = 0.0
                
                self.rave_visits[action] += 1
                self.rave_values[action] += simulation_value

    def energy_efficient_rollout(self, depth=0):
        """
        Energy-efficient rollout policy that explores delayed heating
        """
        if self.is_terminal or depth >= MAX_ROLLOUT_DEPTH:
            return 0.0, [], 0.0
        
        rollout_env = deepcopy(self.env_state)
        cumulative_reward = 0.0
        discount = 1.0
        action_sequence = []
        total_energy = 0.0
        
        # Calculate optimal heating time (approximately 112 steps for 20->100°C)
        temp_rise_needed = max(0, rollout_env.T_target - rollout_env.current_temp)
        # Account for heat loss during heating
        heating_steps_needed = int(temp_rise_needed * rollout_env.m * rollout_env.c / 
                                  (rollout_env.actions[1] - rollout_env.hA * temp_rise_needed/2))
        
        # Choose rollout strategy
        strategy = random.random()
        
        for step in range(MAX_ROLLOUT_DEPTH - depth):
            # Get current state
            current_temp = rollout_env.current_temp
            current_step = rollout_env.steps
            target_temp = rollout_env.T_target
            target_step = rollout_env.target_deadline_step
            
            temp_error = abs(current_temp - target_temp)
            steps_to_deadline = target_step - current_step
            
            # Multiple strategies with different probabilities
            if strategy < 0.4:  # 40% - Optimal delayed heating
                # Wait until the last moment, then heat
                if steps_to_deadline <= heating_steps_needed + 5:
                    action = 1 if current_temp < target_temp else 0
                else:
                    action = 0  # Wait
            
            elif strategy < 0.6:  # 20% - Slightly early heating
                # Start heating a bit early for safety margin
                if steps_to_deadline <= heating_steps_needed + 20:
                    action = 1 if current_temp < target_temp - 1 else 0
                else:
                    action = 0
            
            elif strategy < 0.8:  # 20% - Traditional heat and coast
                # Old strategy for comparison
                if steps_to_deadline > 30 and current_temp < target_temp - 5:
                    action = 1 if random.random() < 0.8 else 0
                elif temp_error <= 2.5:
                    action = 0  # Coast when at target
                else:
                    action = 1 if current_temp < target_temp else 0
            
            else:  # 20% - Random exploration
                # Pure random to ensure exploration
                action = rollout_env.action_space.sample()
            
            action_sequence.append(action)
            
            # Track energy
            power = rollout_env.actions[action]
            total_energy += power * rollout_env.dt
            
            _, reward, done, truncated, info = rollout_env.step(action)
            cumulative_reward += discount * reward
            discount *= DISCOUNT_FACTOR
            
            # Check if we achieved the goal
            if 'goal_achieved' in info and info['goal_achieved']:
                # Bonus for achieving goal efficiently
                efficiency = info.get('energy_efficiency', 0.0)
                cumulative_reward += discount * 500 * efficiency
                break
            
            if done or truncated:
                break
        
        # Calculate final efficiency
        if hasattr(rollout_env, 'theoretical_min_energy'):
            final_efficiency = rollout_env.theoretical_min_energy / max(total_energy, 1)
        else:
            final_efficiency = 0.0
        
        return cumulative_reward, action_sequence, final_efficiency

    def detach_from_parent(self):
        """Detach from parent for tree reuse"""
        self.parent = None

    def get_action_stats(self):
        """Get statistics for all actions"""
        stats = {}
        
        for action in range(self.env_state.action_space.n):
            stats[action] = {
                'visits': 0,
                'avg_value': 0.0,
                'rave_visits': self.rave_visits.get(action, 0),
                'rave_value': self.get_rave_value(action),
                'is_child': action in self.children,
                'energy_impact': self.env_state.actions[action] * self.env_state.dt / 3600  # Wh
            }
            
            if action in self.children:
                child = self.children[action]
                stats[action]['visits'] = child.visits
                if child.visits > 0:
                    stats[action]['avg_value'] = child.value_sum / child.visits
                stats[action]['best_efficiency'] = child.best_efficiency_seen
        
        return stats


def energy_efficient_mcts_iteration(root, found_goal_reward=False):
    """
    MCTS-RAVE iteration optimized for energy efficiency
    """
    # 1. Selection
    node = root
    action_sequence = []
    
    # Occasionally do random exploration if we haven't found efficient solution
    if not found_goal_reward and random.random() < 0.05:  # Less random exploration
        while node.children and not node.is_terminal and random.random() < 0.8:
            if node.untried_actions:
                break
            random_child = random.choice(list(node.children.values()))
            node = random_child
            if node.action_that_led_here is not None:
                action_sequence.append(node.action_that_led_here)
    else:
        # Standard UCT-RAVE selection
        while node.children and not node.is_terminal:
            node = node.select_child()
            if node is None:
                return False
            
            if node.action_that_led_here is not None:
                action_sequence.append(node.action_that_led_here)
    
    # 2. Expansion
    if not node.is_terminal and node.should_expand():
        child = node.expand()
        if child:
            node = child
            action_sequence.append(child.action_that_led_here)
    
    # 3. Rollout with energy-efficient policy
    rollout_value, rollout_actions, efficiency = node.energy_efficient_rollout()
    
    # Check if we found an efficient solution
    found_goal = rollout_value > 1000 and efficiency > 0.8  # High reward AND efficient
    
    # Combine action sequences for RAVE
    full_action_sequence = action_sequence + rollout_actions
    
    # 4. Backpropagation with efficiency tracking
    total_value = rollout_value
    current_node = node
    accumulated_value = total_value
    
    while current_node is not None:
        remaining_actions = [a for a in full_action_sequence if a != current_node.action_that_led_here]
        current_node.update(accumulated_value, remaining_actions, efficiency)
        
        current_node.best_rollout_value = max(current_node.best_rollout_value, accumulated_value)
        
        if current_node.parent is not None and current_node.immediate_reward != 0:
            accumulated_value = current_node.immediate_reward + DISCOUNT_FACTOR * accumulated_value
        
        current_node = current_node.parent
    
    return found_goal


def select_energy_efficient_action(root, num_iterations, step_num):
    """Select action considering energy efficiency"""
    
    if root.is_terminal:
        return None, root
    
    found_efficient_solution_count = 0
    
    # Run MCTS-RAVE iterations
    for i in range(num_iterations):
        found_goal = energy_efficient_mcts_iteration(root, found_efficient_solution_count > 0)
        if found_goal:
            found_efficient_solution_count += 1
        
        if num_iterations >= 1000 and (i + 1) % 1000 == 0:
            if found_efficient_solution_count > 0:
                print(f"   Progress: {i+1}/{num_iterations} iterations "
                      f"(found {found_efficient_solution_count} efficient solutions)")
            else:
                print(f"   Progress: {i+1}/{num_iterations} iterations")
    
    # Debug output
    print(f"\nStep {step_num} - Analysis")
    print(f"Temperature: {root.observation[0]+100:.1f}°C, Steps to deadline: {root.observation[1]:.0f}")
    print(f"Found efficient solutions in {found_efficient_solution_count}/{num_iterations} iterations")
    print("Action statistics:")
    
    stats = root.get_action_stats()
    
    # Sort by visits
    sorted_actions = sorted(stats.keys(), key=lambda a: stats[a]['visits'], reverse=True)
    
    for action in sorted_actions:
        s = stats[action]
        power = root.env_state.actions[action]
        
        if s['visits'] > 0 or s['rave_visits'] > 0:
            print(f"  Action {action} ({power}W, {s['energy_impact']:.3f}Wh):")
            print(f"    Direct: {s['visits']} visits, avg_value={s['avg_value']:.3f}")
            if s['visits'] > 0 and 'best_efficiency' in s:
                print(f"    Best efficiency seen: {s['best_efficiency']:.1%}")
            print(f"    RAVE: {s['rave_visits']} visits, rave_value={s['rave_value']:.3f}")
    
    print(f"Best rollout value: {root.best_rollout_value:.3f}")
    print(f"Best efficiency seen: {root.best_efficiency_seen:.1%}")
    
    # Select action
    if not root.children:
        return None, root
    
    # Action selection considering both value and energy efficiency
    current_temp = root.observation[0] + 100
    steps_to_deadline = root.observation[1]
    
    # If at target temperature, prefer no heating
    if abs(current_temp - 100) < 2.5 and steps_to_deadline > -20:
        # At target: heavily prefer action 0 (no heating)
        if 0 in root.children and root.children[0].visits > num_iterations * 0.1:
            best_action = 0
            print("\nAt target temperature - selecting no heating to save energy")
        else:
            best_action = max(root.children.keys(), 
                              key=lambda a: root.children[a].visits)
    else:
        # Not at target: use standard selection
        best_action = max(root.children.keys(), 
                          key=lambda a: root.children[a].visits)
    
    best_child = root.children[best_action]
    
    print(f"\nSelected action: {best_action} ({root.env_state.actions[best_action]}W)")
    
    # Tree reuse
    if TREE_REUSE:
        best_child.detach_from_parent()
        return best_action, best_child
    else:
        return best_action, None


def run_energy_efficient_episode():
    """Run episode with energy-efficient MCTS-RAVE"""
    
    # Initialize environment with potential-based reward shaping
    # When using the new environment with shaped rewards, use these parameters:
    env = KettleEnv(
        initial_temp=20.0,
        ambient_temp=25.0,
        max_steps=250,
        target_deadline_step=200,
        # discount_factor=0.99,    # For shaped rewards - uncomment when using new env
        # goal_reward=1000.0,      # Sparse goal reward
        # energy_cost_per_kj=0.01, # Energy penalty
        # shaping_scale=1.0        # Potential shaping strength
    )
    
    obs, info = env.reset()
    
    # Initialize root node
    root = EnergyEfficientRAVENode(
        env_state_copy=deepcopy(env),
        done_flag=False,
        parent_node=None,
        observation=obs,
        action_idx_leading_to_node=None
    )
    
    # Episode data storage
    temps = [obs[0] + 100]
    actions = []
    rewards = []
    total_reward = 0
    energy_consumed_Wh = 0
    
    print(f"\n{'='*60}")
    print(f"Energy-Efficient MCTS-RAVE Episode")
    print(f"Initial temperature: {temps[0]:.1f}°C")
    print(f"Target: 100°C at step 200 (±15 steps, ±2.5°C)")
    print(f"Objective: Achieve goal with MINIMAL energy consumption")
    if 'theoretical_min_energy_Wh' in info:
        print(f"Theoretical minimum energy: {info['theoretical_min_energy_Wh']:.1f} Wh")
    print(f"MCTS iterations: {MCTS_ITERATIONS_PER_ACTION}")
    print(f"{'='*60}")
    
    # Run episode
    done = False
    step = 0
    goal_achieved = False
    
    while not done and step < MAX_TIMESTEPS_PER_EPISODE:
        # Select action
        action, next_root = select_energy_efficient_action(
            root, MCTS_ITERATIONS_PER_ACTION, step + 1
        )
        
        if action is None:
            print("No valid action found!")
            break
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        # Record data
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        temps.append(obs[0] + 100)
        
        # Track energy
        power = env.actions[action]
        energy_consumed_Wh += power * env.dt / 3600
        
        # Check goal achievement
        if 'goal_achieved' in info and info['goal_achieved'] and not goal_achieved:
            goal_achieved = True
            print(f"\n*** GOAL ACHIEVED at step {step+1}! ***")
            print(f"    Energy used so far: {energy_consumed_Wh:.1f} Wh")
            if 'energy_efficiency' in info:
                print(f"    Current efficiency: {info['energy_efficiency']:.1%}")
        
        # Progress output
        print(f"\nStep {step+1}: Action={action} ({power}W)")
        print(f"  Temperature: {temps[-1]:.1f}°C (change: {temps[-1]-temps[-2]:+.1f}°C)")
        print(f"  Reward: {reward:.3f}, Total: {total_reward:.3f}")
        print(f"  Energy used: {energy_consumed_Wh:.1f} Wh")
        
        # Update root for next iteration
        if TREE_REUSE and next_root is not None:
            root = next_root
        else:
            root = EnergyEfficientRAVENode(
                env_state_copy=deepcopy(env),
                done_flag=done,
                parent_node=None,
                observation=obs,
                action_idx_leading_to_node=None
            )
        
        step += 1
    
    # Episode summary
    print(f"\n{'='*40}")
    print(f"Episode Summary:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final temperature: {temps[-1]:.1f}°C")
    print(f"Steps taken: {len(actions)}")
    print(f"Goal achieved: {'YES' if goal_achieved else 'NO'}")
    
    if actions:
        on_actions = sum(1 for a in actions if env.actions[a] > 0)
        print(f"Heating actions: {on_actions}/{len(actions)} ({100*on_actions/len(actions):.1f}%)")
        print(f"Total energy consumed: {energy_consumed_Wh:.1f} Wh")
        
        if hasattr(env, 'theoretical_min_energy'):
            theoretical_min = env.theoretical_min_energy / 3600
            efficiency = theoretical_min / energy_consumed_Wh if energy_consumed_Wh > 0 else 0
            print(f"Energy efficiency: {efficiency:.1%}")
            print(f"  (Theoretical minimum: {theoretical_min:.1f} Wh)")
    
    env.close()
    
    return temps, actions, rewards, total_reward, env.actions, env.target_deadline_step, energy_consumed_Wh


def plot_energy_efficient_results(temps, actions, rewards, env_actions, target_deadline, total_energy_Wh):
    """Enhanced plotting with energy efficiency focus"""
    
    plot_dir = "Experiments/Energy_Efficient_MCTS_RAVE"
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    
    steps = range(len(temps))
    
    # 1. Temperature trajectory
    ax = axes[0]
    ax.plot(steps, temps, 'b-', linewidth=2.5, label='Temperature')
    ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Target (100°C)')
    ax.axhline(y=97.5, color='r', linestyle=':', alpha=0.5)
    ax.axhline(y=102.5, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=target_deadline, color='m', linestyle='--', linewidth=2, label=f'Deadline')
    ax.fill_between([target_deadline-15, target_deadline+15], 0, 150, 
                    color='m', alpha=0.1, label='Time window')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Trajectory - Energy Efficient Control')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(temps)-1, 250))
    
    # 2. Actions with energy annotations
    if actions:
        ax = axes[1]
        action_powers = [env_actions[a] for a in actions]
        ax.step(range(len(actions)), action_powers, 'g-', where='post', linewidth=2.5)
        
        # Highlight energy-saving periods
        for i in range(len(actions)):
            if action_powers[i] == 0 and i > 0 and temps[i] > 95:
                ax.axvspan(i, i+1, alpha=0.2, color='green', label='Energy saving' if i == 0 else '')
        
        ax.set_ylabel('Power (W)')
        ax.set_title('Heating Power Applied (Green shading = energy saving while hot)')
        ax.set_ylim(-100, 3500)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(len(actions), 250))
    
    # 3. Instantaneous reward
    if rewards:
        ax = axes[2]
        colors = ['red' if r < 0 else 'green' for r in rewards]
        bars = ax.bar(range(len(rewards)), rewards, color=colors, alpha=0.7)
        
        for i, r in enumerate(rewards):
            if r > 100:
                bars[i].set_alpha(1.0)
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
        
        ax.set_ylabel('Reward')
        ax.set_title('Reward per Step (includes energy penalties)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlim(-0.5, max(len(rewards), 250) - 0.5)
    
    # 4. Cumulative energy consumption
    if actions:
        ax = axes[3]
        energy_watts = [env_actions[a] for a in actions]
        cumulative_energy_Wh = np.cumsum(energy_watts) / 3600
        
        ax.plot(range(len(actions)), cumulative_energy_Wh, 'orange', linewidth=2.5, label='Actual')
        
        # Theoretical minimum line
        theoretical_min = 1.0 * 4184 * 80 / 3600  # 1kg water, 80°C rise
        ax.axhline(y=theoretical_min, color='green', linestyle='--', 
                   label=f'Theoretical min ({theoretical_min:.1f} Wh)')
        
        ax.fill_between(range(len(actions)), theoretical_min, cumulative_energy_Wh, 
                        where=(cumulative_energy_Wh > theoretical_min),
                        color='red', alpha=0.2, label='Excess energy')
        
        ax.set_ylabel('Energy (Wh)')
        ax.set_xlabel('Time Step')
        ax.set_title('Cumulative Energy Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(len(actions), 250))
    
    # 5. Energy efficiency over time
    if actions and len(actions) > 0:
        ax = axes[4]
        theoretical_min = 1.0 * 4184 * 80 / 3600
        cumulative_energy_Wh = np.cumsum([env_actions[a] for a in actions]) / 3600
        efficiency = theoretical_min / np.maximum(cumulative_energy_Wh, 1)
        
        ax.plot(range(len(actions)), efficiency * 100, 'purple', linewidth=2.5)
        ax.axhline(y=100, color='green', linestyle='--', label='Perfect efficiency')
        ax.axhline(y=90, color='orange', linestyle=':', label='90% efficiency')
        ax.fill_between(range(len(actions)), 0, efficiency * 100, alpha=0.3, color='purple')
        
        ax.set_ylabel('Efficiency (%)')
        ax.set_xlabel('Time Step')
        ax.set_title('Energy Efficiency Over Time')
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(len(actions), 250))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'energy_efficient_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Strategy visualization
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    if len(temps) > 1 and len(actions) > 0:
        # Color code by strategy phase
        colors = []
        for i in range(len(actions)):
            if temps[i] < 95:
                colors.append('red')  # Heating phase
            elif temps[i] > 98 and actions[i] == 0:
                colors.append('green')  # Coasting phase
            elif abs(i - target_deadline) < 15:
                colors.append('blue')  # Deadline phase
            else:
                colors.append('gray')  # Other
        
        scatter = ax.scatter(range(len(actions)), temps[1:], c=colors, s=20, alpha=0.6)
        ax.plot(steps, temps, 'k-', linewidth=0.5, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Heating phase'),
            Patch(facecolor='green', label='Coasting (energy saving)'),
            Patch(facecolor='blue', label='Deadline maintenance'),
            Patch(facecolor='gray', label='Transition')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=target_deadline, color='m', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Control Strategy Phases')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'strategy_phases.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {plot_dir}/")


if __name__ == "__main__":
    print("=== Energy-Efficient MCTS-RAVE for Kettle Control ===")
    
    # Run episode
    results = run_energy_efficient_episode()
    temps, actions, rewards, total_reward, env_actions, target_deadline, total_energy = results
    
    # Plot results
    if temps:
        plot_energy_efficient_results(temps, actions, rewards, env_actions, target_deadline, total_energy)
        
        # Final analysis
        print(f"\n{'='*50}")
        print("Energy Efficiency Analysis:")
        print(f"{'='*50}")
        
        if len(temps) > 1:
            final_temp = temps[-1]
            temp_error = abs(final_temp - 100)
            print(f"Final temperature: {final_temp:.1f}°C (error: {temp_error:.1f}°C)")
            
            # Check goal achievement
            final_step = len(actions)
            time_error = abs(final_step - target_deadline)
            
            goal_achieved = temp_error <= 2.5 and time_error <= 15
            
            if goal_achieved:
                print("\n✓ GOAL ACHIEVED!")
                print(f"  Temperature: {final_temp:.1f}°C (within 97.5-102.5°C)")
                print(f"  Time: step {final_step} (within {target_deadline}±15)")
            else:
                print("\n✗ Goal not achieved")
                if temp_error > 2.5:
                    print(f"  Temperature error: {temp_error:.1f}°C > 2.5°C")
                if time_error > 15:
                    print(f"  Time error: {time_error} steps > 15 steps")
        
        if actions and total_energy > 0:
            print(f"\nEnergy Performance:")
            print(f"  Total energy consumed: {total_energy:.1f} Wh")
            
            theoretical_min = 1.0 * 4184 * 80 / 3600  # 1kg water, 80°C rise
            efficiency = (theoretical_min / total_energy) * 100
            
            print(f"  Theoretical minimum: {theoretical_min:.1f} Wh")
            print(f"  Energy efficiency: {efficiency:.1f}%")
            
            # Analyze strategy
            on_actions = sum(1 for a in actions if env_actions[a] > 0)
            heating_percentage = 100 * on_actions / len(actions)
            
            print(f"\nControl Strategy:")
            print(f"  Heating actions: {on_actions}/{len(actions)} ({heating_percentage:.1f}%)")
            print(f"  Coasting actions: {len(actions)-on_actions}/{len(actions)} ({100-heating_percentage:.1f}%)")
            
            # Find when heating starts and stops
            first_heat = None
            last_heat = None
            for i, a in enumerate(actions):
                if env_actions[a] > 0:
                    if first_heat is None:
                        first_heat = i + 1
                    last_heat = i + 1
            
            if first_heat:
                print(f"  First heating at step: {first_heat}")
                print(f"  Last heating at step: {last_heat}")
                print(f"  Heating duration: {last_heat - first_heat + 1} steps")
            
            # Find longest coasting period
            max_coast = 0
            current_coast = 0
            coast_start = 0
            best_coast_period = (0, 0)
            
            for i, a in enumerate(actions):
                if env_actions[a] == 0:
                    if current_coast == 0:
                        coast_start = i
                    current_coast += 1
                    if current_coast > max_coast:
                        max_coast = current_coast
                        best_coast_period = (coast_start + 1, i + 1)
                else:
                    current_coast = 0
            
            print(f"  Longest coasting period: {max_coast} steps (steps {best_coast_period[0]}-{best_coast_period[1]})")
            
            # Check if delayed heating strategy was used
            if first_heat and first_heat > 50:
                print(f"\n✓ DELAYED HEATING STRATEGY DETECTED!")
                print(f"  Waited until step {first_heat} before heating")
                print(f"  Theoretical optimal start: ~step 88")
            
            if efficiency > 90:
                print("\n⭐ EXCELLENT: Achieved >90% energy efficiency!")
            elif efficiency > 80:
                print("\n✓ GOOD: Achieved >80% energy efficiency")
            else:
                print("\n⚠ IMPROVEMENT NEEDED: Energy efficiency below 80%")