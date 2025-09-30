"""
Plain MCTS implementation (without RAVE) for the kettle control problem.
This serves as a baseline to compare against MCTS with RAVE.
"""

import numpy as np
from copy import deepcopy
from math import log, sqrt
import random

# Configuration
MCTS_ITERATIONS_PER_ACTION = 1000
UCB_C = 1.5
DISCOUNT_FACTOR = 0.99
MAX_ROLLOUT_DEPTH = 200
TREE_REUSE = True


class MCTSNode:
    """Standard MCTS Node without RAVE"""
    
    def __init__(self, env_state, done, parent, observation, action_leading_here):
        self.env_state = env_state
        self.observation = observation
        self.is_terminal = done
        self.parent = parent
        self.action_that_led_here = action_leading_here
        
        # Standard MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        
        # Children
        self.children = {}
        self.untried_actions = list(range(env_state.action_space.n))
        random.shuffle(self.untried_actions)
        
        # Track immediate reward for backpropagation
        self.immediate_reward = 0.0
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=UCB_C):
        """Select best child using UCB1"""
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                weight = float('inf')
            else:
                exploitation = child.value_sum / child.visits
                exploration = c_param * sqrt(log(self.visits) / child.visits)
                weight = exploitation + exploration
            choices_weights.append((child, weight))
        
        return max(choices_weights, key=lambda x: x[1])[0]
    
    def expand(self):
        """Expand an untried action"""
        action = self.untried_actions.pop()
        
        # Create child state
        child_env = deepcopy(self.env_state)
        obs, reward, done, truncated, info = child_env.step(action)
        
        child = MCTSNode(child_env, done or truncated, self, obs, action)
        child.immediate_reward = reward
        self.children[action] = child
        
        return child
    
    def rollout(self):
        """Perform random rollout"""
        if self.is_terminal:
            return 0.0
        
        rollout_env = deepcopy(self.env_state)
        cumulative_reward = 0.0
        discount = 1.0
        
        for _ in range(MAX_ROLLOUT_DEPTH):
            # Biased random policy for rollout
            if rollout_env.current_temp < 95 and rollout_env.steps > 80:
                # Prefer heating when cold and not too early
                action = 1 if random.random() < 0.8 else 0
            else:
                action = rollout_env.action_space.sample()
            
            _, reward, done, truncated, _ = rollout_env.step(action)
            cumulative_reward += discount * reward
            discount *= DISCOUNT_FACTOR
            
            if done or truncated:
                break
        
        return cumulative_reward
    
    def backpropagate(self, value):
        """Backpropagate value up the tree"""
        self.visits += 1
        self.value_sum += value
        
        if self.parent:
            parent_value = self.immediate_reward + DISCOUNT_FACTOR * value
            self.parent.backpropagate(parent_value)
    
    def tree_policy(self):
        """Select or expand nodes until a leaf is reached"""
        current_node = self
        
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        
        return current_node
    
    def best_action(self):
        """Select action with highest visit count"""
        if not self.children:
            return None
            
        return max(self.children.items(), key=lambda x: x[1].visits)[0]
    
    def get_action_stats(self):
        """Get statistics for all actions"""
        stats = {}
        for action, child in self.children.items():
            if child.visits > 0:
                stats[action] = {
                    'visits': child.visits,
                    'avg_value': child.value_sum / child.visits
                }
            else:
                stats[action] = {'visits': 0, 'avg_value': 0.0}
        return stats


def mcts_planning(root, n_iterations):
    """Run MCTS planning from root node"""
    for i in range(n_iterations):
        # Selection & Expansion
        leaf = root.tree_policy()
        
        # Simulation
        value = leaf.rollout()
        
        # Backpropagation
        leaf.backpropagate(value)
        
        if (i + 1) % 1000 == 0 and n_iterations >= 1000:
            print(f"   MCTS Progress: {i+1}/{n_iterations} iterations", end='\r')
    
    print()  # New line after progress
    return root.best_action()


class PlainMCTSAgent:
    """Plain MCTS agent for kettle control"""
    
    def __init__(self, iterations_per_action=MCTS_ITERATIONS_PER_ACTION, 
                 ucb_c=UCB_C, tree_reuse=TREE_REUSE):
        self.iterations_per_action = iterations_per_action
        self.ucb_c = ucb_c
        self.tree_reuse = tree_reuse
        self.root = None
        
    def reset(self):
        """Reset the agent for a new episode"""
        self.root = None
    
    def get_action(self, env, obs, step_num=None):
        """Get action using MCTS planning"""
        # Create or update root
        if self.root is None or not self.tree_reuse:
            self.root = MCTSNode(
                env_state=deepcopy(env),
                done=False,
                parent=None,
                observation=obs,
                action_leading_here=None
            )
        
        # Run MCTS
        if step_num is not None:
            print(f"\nStep {step_num} - Plain MCTS Analysis")
            print(f"Temperature: {obs[0]+100:.1f}°C, Steps to deadline: {obs[1]:.0f}")
        
        action = mcts_planning(self.root, self.iterations_per_action)
        
        if action is None:
            print("Warning: No action selected by MCTS")
            return 0  # Default to no heating
        
        # Print statistics
        stats = self.root.get_action_stats()
        print("Action statistics:")
        for a in sorted(stats.keys()):
            s = stats[a]
            print(f"  Action {a}: {s['visits']} visits, avg_value={s['avg_value']:.3f}")
        print(f"Selected action: {action}")
        
        # Update root for tree reuse
        if self.tree_reuse and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None  # Detach from parent
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
            print("Starting Plain MCTS Episode")
            print(f"Initial temperature: {trajectory['temperatures'][0]:.1f}°C")
            print(f"Target: 100°C at step 200 (±15 steps, ±2.5°C)")
            print(f"MCTS iterations per action: {self.iterations_per_action}")
            print(f"{'='*60}")
        
        while not done and step < env.max_steps:
            # Get action from MCTS
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
                theoretical_min = 93.0  # Wh
                efficiency = theoretical_min / trajectory['final_energy_Wh'] * 100
                print(f"Energy efficiency: {efficiency:.1f}%")
        
        return trajectory


def test_plain_mcts():
    """Test the plain MCTS implementation"""
    from kettle_dynamic_env_v24 import KettleEnv
    
    env = KettleEnv(
        initial_temp=20.0,
        ambient_temp=25.0,
        max_steps=250,
        target_deadline_step=200  # Can toggle this
    )
    
    agent = PlainMCTSAgent(iterations_per_action=2000)  # Fewer iterations for testing
    trajectory = agent.run_episode(env, render=False, verbose=True)
    
    return trajectory


if __name__ == "__main__":
    print("Testing Plain MCTS Implementation...")
    test_plain_mcts()