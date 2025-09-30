import numpy as np
import random
from copy import deepcopy
from math import log, sqrt
import csv
import os
from datetime import datetime

class MCTSNode:
    """MCTS Node with RAVE for kettle control"""
    
    def __init__(self, env_state, done, parent, observation, action_taken):
        self.env_state = env_state
        self.observation = observation
        self.is_terminal = done
        self.parent = parent
        self.action_taken = action_taken
        
        # MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        
        # RAVE statistics
        self.rave_visits = {}
        self.rave_values = {}
        
        # Available actions
        self.untried_actions = list(range(env_state.action_space.n))
        random.shuffle(self.untried_actions)
        
        # Optimistic initialization
        self.initial_value = 0.0
    
    def get_rave_value(self, action):
        """Get RAVE value for action"""
        if action not in self.rave_visits or self.rave_visits[action] == 0:
            return 0.0
        return self.rave_values[action] / self.rave_visits[action]
    
    def get_beta(self, child_visits, rave_constant=1000):
        """Calculate RAVE weight using beta function"""
        return sqrt(rave_constant / (3 * child_visits + rave_constant))
    
    def get_ucb_rave_value(self, action, c_param=1.4):
        """Calculate UCB-RAVE value for action selection"""
        if action not in self.children:
            return float('inf')  # Unvisited children have infinite value
        
        child = self.children[action]
        if child.visits == 0:
            return float('inf')
        
        # UCB components
        exploitation = child.value_sum / child.visits
        exploration = c_param * sqrt(log(self.visits) / child.visits)
        
        # RAVE component
        rave_value = self.get_rave_value(action)
        beta = self.get_beta(child.visits)
        
        # Combine UCB and RAVE
        combined_value = (1 - beta) * exploitation + beta * rave_value
        
        return combined_value + exploration
    
    def select_child(self):
        """Select best child using UCB-RAVE"""
        if not self.children:
            return None
        
        best_action = max(self.children.keys(), 
                         key=lambda a: self.get_ucb_rave_value(a))
        return self.children[best_action]
    
    def expand(self):
        """Expand by trying an untried action"""
        if not self.untried_actions or self.is_terminal:
            return None
        
        action = self.untried_actions.pop()
        
        # Create child environment
        child_env = deepcopy(self.env_state)
        obs, reward, done, truncated, info = child_env.step(action)
        
        # Create child node
        child = MCTSNode(child_env, done or truncated, self, obs, action)
        self.children[action] = child
        
        return child
    
    def rollout(self, max_depth=100):
        """Perform random rollout from this node"""
        if self.is_terminal:
            return 0.0, []
        
        rollout_env = deepcopy(self.env_state)
        total_reward = 0.0
        action_sequence = []
        
        for _ in range(max_depth):
            # Simple random policy for rollout
            action = rollout_env.action_space.sample()
            action_sequence.append(action)
            
            obs, reward, done, truncated, info = rollout_env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        return total_reward, action_sequence
    
    def update(self, value, action_sequence):
        """Update node statistics including RAVE"""
        self.visits += 1
        self.value_sum += value
        
        # Update RAVE statistics for all actions in sequence
        for action in action_sequence:
            if action not in self.rave_visits:
                self.rave_visits[action] = 0
                self.rave_values[action] = 0.0
            
            self.rave_visits[action] += 1
            self.rave_values[action] += value
    
    def best_action(self):
        """Return most visited action (final action selection)"""
        if not self.children:
            return None
        return max(self.children.keys(), 
                  key=lambda a: self.children[a].visits)


class KettleMCTS:
    """MCTS solver for kettle control problem"""
    
    def __init__(self, iterations_per_action=5000, c_param=1.4, use_tree_reuse=True):
        self.iterations_per_action = iterations_per_action
        self.c_param = c_param
        self.use_tree_reuse = use_tree_reuse
        self.root = None
        
        # Statistics tracking
        self.iteration_stats = []
        self.action_stats = []
        
    def mcts_iteration(self, root):
        """Single MCTS iteration"""
        # 1. Selection
        node = root
        action_sequence = []
        
        while node.children and not node.is_terminal:
            node = node.select_child()
            if node is None:
                break
            if node.action_taken is not None:
                action_sequence.append(node.action_taken)
        
        # 2. Expansion
        if not node.is_terminal and node.untried_actions:
            child = node.expand()
            if child:
                node = child
                action_sequence.append(child.action_taken)
        
        # 3. Rollout
        rollout_reward, rollout_actions = node.rollout()
        full_action_sequence = action_sequence + rollout_actions
        
        # 4. Backpropagation
        current = node
        while current is not None:
            # Filter actions for RAVE (only include actions available at this node)
            relevant_actions = [a for a in full_action_sequence 
                              if a in range(current.env_state.action_space.n)]
            current.update(rollout_reward, relevant_actions)
            current = current.parent
        
        return rollout_reward
    
    def select_action(self, env, step_num):
        """Select best action using MCTS"""
        # Create root node if needed
        if self.root is None or not self.use_tree_reuse:
            obs = env._get_observation()
            self.root = MCTSNode(deepcopy(env), False, None, obs, None)
        
        # Run MCTS iterations
        rewards = []
        goal_found_count = 0
        
        print(f"\nStep {step_num}: Running {self.iterations_per_action} MCTS iterations...")
        
        for i in range(self.iterations_per_action):
            reward = self.mcts_iteration(self.root)
            rewards.append(reward)
            
            if reward > 500:  # High reward indicates goal achievement
                goal_found_count += 1
            
            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"  Iteration {i+1}/{self.iterations_per_action}, "
                      f"Goals found: {goal_found_count}, "
                      f"Avg reward: {np.mean(rewards[-1000:]):.2f}")
        
        # Analyze results
        temp = env.current_temp
        steps_to_deadline = env.target_deadline_step - env.steps
        
        print(f"\nStep {step_num} Analysis:")
        print(f"  Current: {temp:.1f}°C, {steps_to_deadline} steps to deadline")
        print(f"  MCTS found {goal_found_count} goal-achieving paths")
        print(f"  Average rollout reward: {np.mean(rewards):.2f}")
        
        # Action statistics
        action_info = {}
        for action, child in self.root.children.items():
            power = env.actions[action]
            action_info[action] = {
                'visits': child.visits,
                'avg_value': child.value_sum / max(1, child.visits),
                'rave_visits': self.root.rave_visits.get(action, 0),
                'rave_value': self.root.get_rave_value(action),
                'power': power
            }
        
        # Print action statistics
        print("  Action statistics:")
        for action in sorted(action_info.keys()):
            info = action_info[action]
            print(f"    Action {action} ({info['power']}W): "
                  f"{info['visits']} visits, "
                  f"value={info['avg_value']:.3f}, "
                  f"RAVE={info['rave_value']:.3f}")
        
        # Select best action
        best_action = self.root.best_action()
        
        # Store statistics
        self.iteration_stats.append({
            'step': step_num,
            'goal_found_count': goal_found_count,
            'avg_reward': np.mean(rewards),
            'best_action': best_action,
            'temperature': temp,
            'steps_to_deadline': steps_to_deadline
        })
        
        self.action_stats.append(action_info)
        
        # Tree reuse: move to selected child
        if self.use_tree_reuse and best_action is not None:
            if best_action in self.root.children:
                new_root = self.root.children[best_action]
                new_root.parent = None  # Detach from parent
                self.root = new_root
            else:
                self.root = None
        else:
            self.root = None
        
        print(f"  Selected action: {best_action} ({env.actions[best_action]}W)")
        
        return best_action


def run_mcts_episode(iterations_per_action=5000, save_results=True):
    """Run single MCTS episode on kettle environment"""
    from Kettle_env_v1 import KettleEnv
    
    # Initialize environment
    env = KettleEnv()
    mcts = KettleMCTS(iterations_per_action=iterations_per_action)
    
    # Episode data
    episode_data = {
        'steps': [],
        'temperatures': [],
        'actions': [],
        'powers': [],
        'rewards': [],
        'cumulative_rewards': [],
        'energy_consumed_J': [],
        'energy_consumed_Wh': [],
        'energy_efficiency': [],
        'goal_achieved': [],
        'temp_error': [],
        'steps_to_deadline': [],
        'base_rewards': [],
        'shaping_rewards': [],
        'potential_values': []
    }
    
    # Initialize
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print("="*60)
    print("MCTS Kettle Control Episode")
    print(f"Target: {env.target_temp}°C at step {env.target_deadline_step}")
    print(f"MCTS iterations per action: {iterations_per_action}")
    print("="*60)
    
    # Initial state
    episode_data['steps'].append(step)
    episode_data['temperatures'].append(env.current_temp)
    episode_data['actions'].append(None)
    episode_data['powers'].append(0)
    episode_data['rewards'].append(0)
    episode_data['cumulative_rewards'].append(0)
    episode_data['energy_consumed_J'].append(0)
    episode_data['energy_consumed_Wh'].append(0)
    episode_data['energy_efficiency'].append(1.0)
    episode_data['goal_achieved'].append(False)
    episode_data['temp_error'].append(abs(env.current_temp - env.target_temp))
    episode_data['steps_to_deadline'].append(env.target_deadline_step - step)
    episode_data['base_rewards'].append(0)
    episode_data['shaping_rewards'].append(0)
    episode_data['potential_values'].append(info.get('initial_potential', 0.0))
    
    # Run episode
    while not done and step < env.max_steps:
        # Select action using MCTS
        action = mcts.select_action(env, step + 1)
        
        if action is None:
            print("No action selected, stopping episode")
            break
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Store data
        episode_data['steps'].append(step)
        episode_data['temperatures'].append(info['temperature'])
        episode_data['actions'].append(action)
        episode_data['powers'].append(info['power'])
        episode_data['rewards'].append(reward)
        episode_data['cumulative_rewards'].append(total_reward)
        episode_data['energy_consumed_J'].append(info['total_energy_J'])
        episode_data['energy_consumed_Wh'].append(info['total_energy_Wh'])
        episode_data['energy_efficiency'].append(info['energy_efficiency'])
        episode_data['goal_achieved'].append(info['goal_achieved'])
        episode_data['temp_error'].append(info['temp_error'])
        episode_data['steps_to_deadline'].append(info['steps_to_deadline'])
        
        # Store reward components if available
        if 'base_reward' in info:
            episode_data['base_rewards'].append(info['base_reward'])
            episode_data['shaping_rewards'].append(info['shaping_reward'])
            episode_data['potential_values'].append(info.get('potential_after', 0.0))
        else:
            episode_data['base_rewards'].append(reward)
            episode_data['shaping_rewards'].append(0.0)
            episode_data['potential_values'].append(0.0)
        
        # Print progress with reward shaping info
        print(f"\nStep {step}: Action={action} ({info['power']}W)")
        print(f"  Temp: {info['temperature']:.1f}°C (error: {info['temp_error']:.1f}°C)")
        print(f"  Reward: {reward:.2f} (Base: {info.get('base_reward', reward):.2f}, "
              f"  Shaping: {info.get('shaping_reward', 0.0):.2f})")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Energy: {info['total_energy_Wh']:.1f}Wh (eff: {info['energy_efficiency']:.1%})")
        
        if info['goal_achieved']:
            print("  *** GOAL ACHIEVED! ***")
        
        done = done or truncated
    
    # Episode summary
    final_temp = episode_data['temperatures'][-1]
    final_energy = episode_data['energy_consumed_Wh'][-1]
    final_efficiency = episode_data['energy_efficiency'][-1]
    goal_achieved = episode_data['goal_achieved'][-1]
    
    print(f"\n" + "="*40)
    print("Episode Summary:")
    print(f"  Final temperature: {final_temp:.1f}°C")
    print(f"  Total energy: {final_energy:.1f}Wh")
    print(f"  Energy efficiency: {final_efficiency:.1%}")
    print(f"  Goal achieved: {'YES' if goal_achieved else 'NO'}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps taken: {step}")
    
    # Calculate strategy metrics
    actions = [a for a in episode_data['actions'] if a is not None]
    if actions:
        heating_actions = sum(1 for a in actions if env.actions[a] > 0)
        heating_percentage = 100 * heating_actions / len(actions)
        
        # Find first and last heating
        first_heat = next((i for i, a in enumerate(actions) if env.actions[a] > 0), None)
        last_heat = max((i for i, a in enumerate(actions) if env.actions[a] > 0), default=None)
        
        print(f"  Heating actions: {heating_actions}/{len(actions)} ({heating_percentage:.1f}%)")
        if first_heat is not None:
            print(f"  First heating: step {first_heat + 1}")
        if last_heat is not None:
            print(f"  Last heating: step {last_heat + 1}")
    
    env.close()
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_episode_results(episode_data, mcts.iteration_stats, mcts.action_stats, timestamp)
    
    return episode_data, mcts.iteration_stats, mcts.action_stats


def save_episode_results(episode_data, iteration_stats, action_stats, timestamp):
    """Save episode results to CSV files"""
    
    # Create results directory
    results_dir = f"results/mcts_kettle_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save episode data
    episode_file = os.path.join(results_dir, "episode_data.csv")
    with open(episode_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(episode_data.keys())
        
        # Data rows
        for i in range(len(episode_data['steps'])):
            row = [episode_data[key][i] for key in episode_data.keys()]
            writer.writerow(row)
    
    # Save iteration statistics
    iter_file = os.path.join(results_dir, "iteration_stats.csv")
    with open(iter_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if iteration_stats:
            writer.writerow(iteration_stats[0].keys())
            for stat in iteration_stats:
                writer.writerow(stat.values())
    
    # Save action statistics
    action_file = os.path.join(results_dir, "action_stats.csv")
    with open(action_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['step', 'action', 'visits', 'avg_value', 'rave_visits', 'rave_value', 'power'])
        
        # Data
        for step, stats in enumerate(action_stats):
            for action, info in stats.items():
                writer.writerow([step + 1, action, info['visits'], info['avg_value'], 
                               info['rave_visits'], info['rave_value'], info['power']])
    
    print(f"\nResults saved to: {results_dir}")
    return results_dir


def run_mcts_episode_with_shaping(iterations_per_action=5000, enable_reward_shaping=True, save_results=True):
    """Run MCTS episode with configurable reward shaping"""
    from Kettle_env_v1 import KettleEnv
    
    # Initialize environment with shaping configuration
    env = KettleEnv(enable_reward_shaping=enable_reward_shaping)
    mcts = KettleMCTS(iterations_per_action=iterations_per_action)
    
    # Episode data (same structure as run_mcts_episode)
    episode_data = {
        'steps': [],
        'temperatures': [],
        'actions': [],
        'powers': [],
        'rewards': [],
        'cumulative_rewards': [],
        'energy_consumed_J': [],
        'energy_consumed_Wh': [],
        'energy_efficiency': [],
        'goal_achieved': [],
        'temp_error': [],
        'steps_to_deadline': [],
        'base_rewards': [],
        'shaping_rewards': [],
        'potential_values': []
    }
    
    # Initialize
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print("="*60)
    print(f"MCTS Kettle Control Episode (Shaping: {'ON' if enable_reward_shaping else 'OFF'})")
    print(f"Target: {env.target_temp}°C at step {env.target_deadline_step}")
    print(f"MCTS iterations per action: {iterations_per_action}")
    if enable_reward_shaping:
        print(f"Reward shaping: γ={env.gamma}, scale={env.potential_scale}")
    print("="*60)
    
    # Initial state
    episode_data['steps'].append(step)
    episode_data['temperatures'].append(env.current_temp)
    episode_data['actions'].append(None)
    episode_data['powers'].append(0)
    episode_data['rewards'].append(0)
    episode_data['cumulative_rewards'].append(0)
    episode_data['energy_consumed_J'].append(0)
    episode_data['energy_consumed_Wh'].append(0)
    episode_data['energy_efficiency'].append(1.0)
    episode_data['goal_achieved'].append(False)
    episode_data['temp_error'].append(abs(env.current_temp - env.target_temp))
    episode_data['steps_to_deadline'].append(env.target_deadline_step - step)
    episode_data['base_rewards'].append(0)
    episode_data['shaping_rewards'].append(0)
    episode_data['potential_values'].append(info.get('initial_potential', 0.0))
    
    # Run episode
    while not done and step < env.max_steps:
        # Select action using MCTS
        action = mcts.select_action(env, step + 1)
        
        if action is None:
            print("No action selected, stopping episode")
            break
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Store data
        episode_data['steps'].append(step)
        episode_data['temperatures'].append(info['temperature'])
        episode_data['actions'].append(action)
        episode_data['powers'].append(info['power'])
        episode_data['rewards'].append(reward)
        episode_data['cumulative_rewards'].append(total_reward)
        episode_data['energy_consumed_J'].append(info['total_energy_J'])
        episode_data['energy_consumed_Wh'].append(info['total_energy_Wh'])
        episode_data['energy_efficiency'].append(info['energy_efficiency'])
        episode_data['goal_achieved'].append(info['goal_achieved'])
        episode_data['temp_error'].append(info['temp_error'])
        episode_data['steps_to_deadline'].append(info['steps_to_deadline'])
        
        # Store reward components
        episode_data['base_rewards'].append(info.get('base_reward', reward))
        episode_data['shaping_rewards'].append(info.get('shaping_reward', 0.0))
        episode_data['potential_values'].append(info.get('potential_after', 0.0))
        
        # Print progress with reward shaping info
        print(f"\nStep {step}: Action={action} ({info['power']}W)")
        print(f"  Temp: {info['temperature']:.1f}°C (error: {info['temp_error']:.1f}°C)")
        
        if enable_reward_shaping and 'base_reward' in info:
            print(f"  Reward: {reward:.2f} (Base: {info['base_reward']:.2f}, "
                  f"Shaping: {info['shaping_reward']:.2f})")
            print(f"  Potential: {info.get('potential_after', 0.0):.2f}")
        else:
            print(f"  Reward: {reward:.2f}")
        
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Energy: {info['total_energy_Wh']:.1f}Wh (eff: {info['energy_efficiency']:.1%})")
        
        if info.get('should_be_heating', False):
            print(f"  Strategy: Should be heating (past step {info.get('optimal_start_step', 0)})")
        else:
            time_to_start = info.get('time_to_optimal_start', 0)
            if time_to_start > 0:
                print(f"  Strategy: Wait {time_to_start} more steps before heating")
        
        if info['goal_achieved']:
            print("  *** GOAL ACHIEVED! ***")
        
        done = done or truncated
    
    # Episode summary
    final_temp = episode_data['temperatures'][-1]
    final_energy = episode_data['energy_consumed_Wh'][-1]
    final_efficiency = episode_data['energy_efficiency'][-1]
    goal_achieved = episode_data['goal_achieved'][-1]
    
    print(f"\n" + "="*40)
    print("Episode Summary:")
    print(f"  Final temperature: {final_temp:.1f}°C")
    print(f"  Total energy: {final_energy:.1f}Wh")
    print(f"  Energy efficiency: {final_efficiency:.1%}")
    print(f"  Goal achieved: {'YES' if goal_achieved else 'NO'}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps taken: {step}")
    
    # Reward shaping analysis
    if enable_reward_shaping and len(episode_data['shaping_rewards']) > 1:
        total_shaping = sum(episode_data['shaping_rewards'][1:])  # Skip initial 0
        total_base = sum(episode_data['base_rewards'][1:])
        print(f"  Base reward total: {total_base:.2f}")
        print(f"  Shaping reward total: {total_shaping:.2f}")
        print(f"  Shaping contribution: {100*total_shaping/total_reward:.1f}%")
    
    # Calculate strategy metrics
    actions = [a for a in episode_data['actions'] if a is not None]
    if actions:
        heating_actions = sum(1 for i in range(1, len(episode_data['steps'])) 
                             if episode_data['powers'][i] > 0)
        heating_percentage = 100 * heating_actions / len(actions)
        
        # Find first and last heating
        first_heat = next((i for i, p in enumerate(episode_data['powers'][1:], 1) if p > 0), None)
        last_heat = max((i for i, p in enumerate(episode_data['powers'][1:], 1) if p > 0), default=None)
        
        print(f"  Heating actions: {heating_actions}/{len(actions)} ({heating_percentage:.1f}%)")
        if first_heat is not None:
            print(f"  First heating: step {first_heat}")
        if last_heat is not None:
            print(f"  Last heating: step {last_heat}")
    
    env.close()
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_shaping_{enable_reward_shaping}"
        save_episode_results(episode_data, mcts.iteration_stats, mcts.action_stats, timestamp + suffix)
    
    return episode_data, mcts.iteration_stats, mcts.action_stats


# Keep the original function for backward compatibility
def run_mcts_episode(iterations_per_action=5000, save_results=True):
    """Run MCTS episode with default settings (shaping enabled)"""
    return run_mcts_episode_with_shaping(iterations_per_action, True, save_results)