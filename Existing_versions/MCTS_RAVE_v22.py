"""
MCTS with RAVE (Rapid Action Value Estimation) implementation for the kettle control problem.
This should outperform plain MCTS by sharing action values across the tree.
"""

import numpy as np
from copy import deepcopy
from math import log, sqrt
import random

# Configuration
MCTS_ITERATIONS_PER_ACTION = 1000
UCB_C = 1.5
RAVE_C = 1000.0
DISCOUNT_FACTOR = 0.99
MAX_ROLLOUT_DEPTH = 200
TREE_REUSE = True


class MCTSRAVENode:
    """MCTS Node with RAVE support"""
    
    def __init__(self, env_state, done, parent, observation, action_leading_here):
        self.env_state = env_state
        self.observation = observation
        self.is_terminal = done
        self.parent = parent
        self.action_that_led_here = action_leading_here
        
        # Standard MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        
        # RAVE statistics
        self.rave_visits = {}  # action -> count
        self.rave_values = {}  # action -> value sum
        
        # Children
        self.children = {}
        self.untried_actions = list(range(env_state.action_space.n))
        random.shuffle(self.untried_actions)
        
        # Track immediate reward
        self.immediate_reward = 0.0
        
    def get_rave_value(self, action):
        """Get RAVE value for an action"""
        if action not in self.rave_visits or self.rave_visits[action] == 0:
            return 0.0
        return self.rave_values[action] / self.rave_visits[action]
    
    def get_beta(self, child_visits, rave_constant=RAVE_C):
        """Calculate RAVE weight (beta)"""
        return sqrt(rave_constant / (3 * child_visits + rave_constant))
    
    def uct_rave_value(self, action, c_param=UCB_C):
        """Calculate UCT value with RAVE"""
        if action not in self.children:
            return float('inf')
        
        child = self.children[action]
        if child.visits == 0:
            return float('inf')
        
        # Standard UCT
        exploitation = child.value_sum / child.visits
        exploration = c_param * sqrt(log(self.visits) / child.visits)
        
        # RAVE component
        rave_value = self.get_rave_value(action)
        beta = self.get_beta(child.visits)
        
        # Combine UCT and RAVE
        combined_value = (1 - beta) * exploitation + beta * rave_value
        
        return combined_value + exploration
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=UCB_C):
        """Select best child using UCT-RAVE"""
        if not self.children:
            return None
        
        best_action = max(self.children.keys(), 
                         key=lambda a: self.uct_rave_value(a, c_param))
        return self.children[best_action]
    
    def expand(self):
        """Expand an untried action"""
        action = self.untried_actions.pop()
        
        # Create child state
        child_env = deepcopy(self.env_state)
        obs, reward, done, truncated, info = child_env.step(action)
        
        child = MCTSRAVENode(child_env, done or truncated, self, obs, action)
        child.immediate_reward = reward
        self.children[action] = child
        
        return child
    
    def rollout(self):
        """Perform biased rollout"""
        if self.is_terminal:
            return 0.0, []
        
        rollout_env = deepcopy(self.env_state)
        cumulative_reward = 0.0
        discount = 1.0
        action_sequence = []
        
        for _ in range(MAX_ROLLOUT_DEPTH):
            # Intelligent rollout policy
            current_temp = rollout_env.current_temp
            current_step = rollout_env.steps
            target_step = rollout_env.target_deadline_step
            
            # Calculate approximate heating time needed
            temp_diff = max(0, 100 - current_temp)
            heating_time = int(temp_diff * 1.3)  # Approximate
            
            if current_step < target_step - heating_time - 10:
                # Too early - don't heat
                action = 0
            elif current_step >= target_step - heating_time and current_temp < 98:
                # Time to heat
                action = 1
            elif current_temp >= 98 and current_temp <= 102:
                # At target - maintain with minimal heating
                action = 0 if random.random() < 0.8 else 1
            else:
                # Default random
                action = rollout_env.action_space.sample()
            
            action_sequence.append(action)
            _, reward, done, truncated, _ = rollout_env.step(action)
            cumulative_reward += discount * reward
            discount *= DISCOUNT_FACTOR
            
            if done or truncated:
                break
        
        return cumulative_reward, action_sequence
    
    def update(self, value, action_sequence):
        """Update node statistics with RAVE"""
        self.visits += 1
        self.value_sum += value
        
        # Update RAVE statistics for all actions in sequence
        for action in action_sequence:
            if action not in self.rave_visits:
                self.rave_visits[action] = 0
                self.rave_values[action] = 0.0
            
            self.rave_visits[action] += 1
            self.rave_values[action] += value
    
    def tree_policy(self):
        """Select or expand nodes until a leaf is reached"""
        current_node = self
        
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
                if current_node is None:
                    break
        
        return current_node
    
    def best_action(self):
        """Select action with highest visit count"""
        if not self.children:
            return None
            
        return max(self.children.items(), key=lambda x: x[1].visits)[0]
    
    def get_action_stats(self):
        """Get statistics for all actions"""
        stats = {}
        for action in range(self.env_state.action_space.n):
            stats[action] = {
                'visits': 0,
                'avg_value': 0.0,
                'rave_visits': self.rave_visits.get(action, 0),
                'rave_value': self.get_rave_value(action)
            }
            
            if action in self.children:
                child = self.children[action]
                stats[action]['visits'] = child.visits
                if child.visits > 0:
                    stats[action]['avg_value'] = child.value_sum / child.visits
        
        return stats


def mcts_rave_planning(root, n_iterations):
    """Run MCTS with RAVE planning"""
    for i in range(n_iterations):
        # Selection & Expansion
        current = root
        path = []
        
        # Traverse down to leaf
        while not current.is_terminal and current.is_fully_expanded():
            current = current.best_child()
            if current is None:
                break
            if current.action_that_led_here is not None:
                path.append(current.action_that_led_here)
        
        # Expand if possible
        if current and not current.is_terminal and not current.is_fully_expanded():
            current = current.expand()
            if current.action_that_led_here is not None:
                path.append(current.action_that_led_here)
        
        # Rollout
        if current:
            rollout_value, rollout_actions = current.rollout()
            
            # Combine paths for RAVE
            full_sequence = path + rollout_actions
            
            # Backpropagation with RAVE
            value = rollout_value
            node = current
            
            while node is not None:
                # Get actions that appear after this node in the sequence
                if node.action_that_led_here is not None:
                    remaining_actions = [a for a in full_sequence if a != node.action_that_led_here]
                else:
                    remaining_actions = full_sequence
                
                node.update(value, remaining_actions)
                
                # Apply discount
                if node.parent is not None and node.immediate_reward != 0:
                    value = node.immediate_reward + DISCOUNT_FACTOR * value
                
                node = node.parent
        
        if (i + 1) % 1000 == 0 and n_iterations >= 1000:
            print(f"   MCTS-RAVE Progress: {i+1}/{n_iterations} iterations", end='\r')
    
    print()  # New line after progress
    return root.best_action()


class MCTSRAVEAgent:
    """MCTS with RAVE agent for kettle control"""
    
    def __init__(self, iterations_per_action=MCTS_ITERATIONS_PER_ACTION,
                 ucb_c=UCB_C, rave_c=RAVE_C, tree_reuse=TREE_REUSE):
        self.iterations_per_action = iterations_per_action
        self.ucb_c = ucb_c
        self.rave_c = rave_c
        self.tree_reuse = tree_reuse
        self.root = None
    
    def reset(self):
        """Reset the agent for a new episode"""
        self.root = None
    
    def get_action(self, env, obs, step_num=None):
        """Get action using MCTS-RAVE planning"""
        # Create or update root
        if self.root is None or not self.tree_reuse:
            self.root = MCTSRAVENode(
                env_state=deepcopy(env),
                done=False,
                parent=None,
                observation=obs,
                action_leading_here=None
            )
        
        # Run MCTS-RAVE
        if step_num is not None:
            print(f"\nStep {step_num} - MCTS-RAVE Analysis")
            print(f"Temperature: {obs[0]+100:.1f}°C, Steps to deadline: {obs[1]:.0f}")
        
        action = mcts_rave_planning(self.root, self.iterations_per_action)
        
        if action is None:
            print("Warning: No action selected by MCTS-RAVE")
            return 0
        
        # Print statistics
        stats = self.root.get_action_stats()
        print("Action statistics:")
        for a in sorted(stats.keys()):
            s = stats[a]
            print(f"  Action {a}: Direct: {s['visits']} visits, avg={s['avg_value']:.3f}")
            print(f"           RAVE: {s['rave_visits']} visits, value={s['rave_value']:.3f}")
        print(f"Selected action: {action}")
        
        # Update root for tree reuse
        if self.tree_reuse and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None
        
        return action
    
    def run_episode(self, env, render=False, verbose=True):
        """Run a complete episode"""
        obs, info = env.reset()
        self.reset()
        
        # Episode data
        trajectory = {
            'observations': [obs],
            'actions': [],
            'rewards': [],
            'temperatures': [info['current_temperature']],
            'energy_Wh': [0],
            'steps': [0]
        }
        
        done = False
        total_reward = 0
        step = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print("Starting MCTS-RAVE Episode")
            print(f"Initial temperature: {trajectory['temperatures'][0]:.1f}°C")
            print(f"Target: 100°C at step 200 (±15 steps, ±2.5°C)")
            print(f"MCTS-RAVE iterations per action: {self.iterations_per_action}")
            print(f"UCB_C: {self.ucb_c}, RAVE_C: {self.rave_c}")
            print(f"{'='*60}")
        
        while not done and step < env.max_steps:
            # Get action from MCTS-RAVE
            action = self.get_action(env, obs, step + 1 if verbose else None)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Record data
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['temperatures'].append(info['temperature'])
            trajectory['energy_Wh'].append(info['total_energy_Wh'])
            trajectory['steps'].append(info['current_step'])
            
            total_reward += reward
            
            if verbose:
                print(f"\nStep {step+1}: Action={action} ({env.actions[action]}W)")
                print(f"  Temperature: {info['temperature']:.1f}°C")
                print(f"  Reward: {reward:.3f}, Total: {total_reward:.3f}")
                print(f"  Energy: {info['total_energy_Wh']:.1f} Wh")
            
            if render:
                env.render()
            
            step += 1
            done = done or truncated
        
        # Summary
        trajectory['total_reward'] = total_reward
        trajectory['goal_achieved'] = info.get('goal_achieved', False)
        trajectory['final_temperature'] = trajectory['temperatures'][-1]
        trajectory['final_energy_Wh'] = trajectory['energy_Wh'][-1]
        trajectory['episode_length'] = len(trajectory['actions'])
        
        if verbose:
            print(f"\n{'='*40}")
            print("Episode Summary:")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final temperature: {trajectory['final_temperature']:.1f}°C")
            print(f"Total energy: {trajectory['final_energy_Wh']:.1f} Wh")
            print(f"Goal achieved: {'YES' if trajectory['goal_achieved'] else 'NO'}")
            
            if trajectory['final_energy_Wh'] > 0:
                theoretical_min = 93.0
                efficiency = theoretical_min / trajectory['final_energy_Wh'] * 100
                print(f"Energy efficiency: {efficiency:.1f}%")
        
        return trajectory


def test_mcts_rave():
    """Test the MCTS-RAVE implementation"""
    from kettle_dynamic_env_v24 import KettleEnv
    
    env = KettleEnv(
        initial_temp=20.0,
        ambient_temp=25.0,
        max_steps=250,
        target_deadline_step=200
    )
    
    agent = MCTSRAVEAgent(iterations_per_action=2000)
    trajectory = agent.run_episode(env, render=False, verbose=True)
    
    return trajectory


if __name__ == "__main__":
    print("Testing MCTS with RAVE Implementation...")
    test_mcts_rave()