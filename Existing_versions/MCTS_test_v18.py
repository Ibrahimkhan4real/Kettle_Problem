import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from math import log, sqrt
import os

from kettle_dynamic_env_v23 import KettleEnv

# --- Configuration ---
EPISODES_TO_RUN = 1
MAX_TIMESTEPS_PER_EPISODE = 250

# MCTS Hyperparameters
UCB_C = 1.5  # Reduced to better match reward scale
MCTS_ITERATIONS_PER_ACTION = 4000
MAX_ROLLOUT_DEPTH = 200  # Reduced for faster rollouts
DISCOUNT_FACTOR = 0.99  # Discount future rewards
TREE_REUSE = False  # Reuse tree between timesteps


class Node:
    """
    Represents a node in the Monte Carlo Search Tree
    """
    def __init__(self, env_state_copy, done_flag, parent_node, observation, action_idx_leading_to_node):
        self.children = {}
        self.total_simulation_reward = 0.0
        self.num_visits = 0
        
        self.env_state = env_state_copy
        self.observation = observation
        self.is_terminal = done_flag
        
        self.parent = parent_node
        self.action_that_led_here = action_idx_leading_to_node
        
        # Store value estimates separately for better debugging
        self.immediate_reward = 0.0
        self.value_estimate = 0.0

    def get_ucb1_score(self) -> float:
        """Calculates the UCB1 score for this node."""
        if self.num_visits == 0:
            return float("inf")
        
        parent_visits = self.parent.num_visits if self.parent else 1
        
        exploitation = self.value_estimate
        exploration = UCB_C * sqrt(log(parent_visits) / self.num_visits)
        
        return exploitation + exploration

    def detach_from_parent(self):
        """Detaches this node from its parent"""
        if self.parent and self.action_that_led_here is not None:
            if self.action_that_led_here in self.parent.children:
                del self.parent.children[self.action_that_led_here]
        self.parent = None

    def is_fully_expanded(self):
        """Check if all possible actions have been tried from this node"""
        if self.is_terminal:
            return True
        return len(self.children) == self.env_state.action_space.n

    def expand(self):
        """
        Expands this node by creating ONE new child node for an untried action.
        Returns the new child node.
        """
        if self.is_terminal or self.is_fully_expanded():
            return None
        
        # Find untried actions
        tried_actions = set(self.children.keys())
        possible_actions = set(range(self.env_state.action_space.n))
        untried_actions = possible_actions - tried_actions
        
        if not untried_actions:
            return None
        
        # Select a random untried action
        action_idx = random.choice(list(untried_actions))
        
        # Create child node
        child_env_state = deepcopy(self.env_state)
        obs, reward, done, truncated, _ = child_env_state.step(action_idx)
        child_is_terminal = done or truncated
        
        child_node = Node(child_env_state, child_is_terminal, self, obs, action_idx)
        child_node.immediate_reward = reward
        self.children[action_idx] = child_node
        
        return child_node

    def rollout(self, depth=0) -> float:
        """
        Performs a random simulation from this node's state.
        Returns the discounted cumulative reward.
        """
        if self.is_terminal or depth >= MAX_ROLLOUT_DEPTH:
            return 0.0
        
        rollout_env = deepcopy(self.env_state)
        cumulative_reward = 0.0
        discount = 1.0
        current_depth = 0
        done = False
        
        while not done and current_depth < MAX_ROLLOUT_DEPTH:
            action = rollout_env.action_space.sample()
            _, reward, term, trunc, _ = rollout_env.step(action)
            done = term or trunc
            
            cumulative_reward += discount * reward
            discount *= DISCOUNT_FACTOR
            current_depth += 1
        
        return cumulative_reward

    def backpropagate(self, value):
        """
        Propagates the value estimate up the tree
        """
        self.num_visits += 1
        self.total_simulation_reward += value
        self.value_estimate = self.total_simulation_reward / self.num_visits
        
        if self.parent:
            # Pass the discounted value to parent
            parent_value = self.immediate_reward + DISCOUNT_FACTOR * value
            self.parent.backpropagate(parent_value)

    def best_child(self, c_param=0.0):
        """
        Select best child based on UCB score (exploration) or value (exploitation)
        c_param=0 means pure exploitation (for action selection)
        c_param>0 means exploration (for tree traversal)
        """
        if not self.children:
            return None
        
        if c_param == 0:
            # Pure exploitation: choose child with highest average value
            return max(self.children.values(), 
                      key=lambda n: n.value_estimate if n.num_visits > 0 else -float('inf'))
        else:
            # UCB selection
            return max(self.children.values(), 
                      key=lambda n: n.get_ucb1_score())

    def get_action_values(self):
        """Returns a dictionary of action -> (visits, avg_value) for debugging"""
        action_values = {}
        for action, child in self.children.items():
            if child.num_visits > 0:
                action_values[action] = (child.num_visits, child.value_estimate)
            else:
                action_values[action] = (0, 0.0)
        return action_values


def mcts_iteration(root: Node):
    """Performs one MCTS iteration: Selection, Expansion, Simulation, Backpropagation"""
    
    # 1. Selection - traverse tree using UCB
    node = root
    while not node.is_terminal and node.is_fully_expanded():
        node = node.best_child(c_param=UCB_C)
        if node is None:  # Safety check
            break
    
    # 2. Expansion - add new child if not terminal
    if not node.is_terminal and not node.is_fully_expanded():
        node = node.expand() or node
    
    # 3. Simulation - random rollout
    rollout_value = node.rollout()
    
    # 4. Backpropagation
    node.backpropagate(rollout_value)


def select_action_mcts(root: Node, num_iterations: int):
    """
    Runs MCTS iterations and selects the best action.
    Returns the action index and the root node for the next state.
    """
    if root.is_terminal:
        return None, root
    
    # Run MCTS iterations
    for _ in range(num_iterations):
        mcts_iteration(root)
    
    # Debug output
    action_values = root.get_action_values()
    print(f"\nMCTS Analysis (Temp: {root.observation[0]+100:.1f}°C, Steps to deadline: {root.observation[1]:.0f})")
    print("Action stats:")
    for action in sorted(action_values.keys()):
        visits, avg_value = action_values[action]
        power = root.env_state.actions[action]
        print(f"  Action {action} ({power}W): {visits} visits, avg_value={avg_value:.3f}")
    
    # Select action with most visits
    if not root.children:
        return None, root
    
    best_action = max(root.children.keys(), 
                     key=lambda a: root.children[a].num_visits)
    
    # Get the next root (reuse tree)
    next_root = root.children[best_action]
    if TREE_REUSE:
        next_root.detach_from_parent()
    
    return best_action, next_root


def run_mcts_episode():
    """Runs a single episode with MCTS control"""
    
    # Initialize environment
    env = KettleEnv(
        initial_temp=20.0,
        ambient_temp=25.0,
        max_steps=250,
        target_deadline_step=200
    )
    
    obs, _ = env.reset()
    root = Node(
        env_state_copy=deepcopy(env),
        done_flag=False,
        parent_node=None,
        observation=obs,
        action_idx_leading_to_node=None
    )
    
    # Episode data storage
    temps = [obs[0] + 100]  # Convert to actual temperature
    actions = []
    rewards = []
    total_reward = 0
    
    print(f"\nStarting MCTS Episode")
    print(f"Initial temp: {temps[0]:.1f}°C, Target: 100°C by step 200")
    print(f"MCTS iterations per action: {MCTS_ITERATIONS_PER_ACTION}")
    
    # Run episode
    done = False
    step = 0
    
    while not done and step < MAX_TIMESTEPS_PER_EPISODE:
        # Select action using MCTS
        action, next_root = select_action_mcts(root, MCTS_ITERATIONS_PER_ACTION)
        
        if action is None:
            print("No valid action found")
            break
        
        # Execute action in real environment
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        # Record data
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        temps.append(obs[0] + 100)  # Convert to actual temperature
        
        # Update root for next iteration
        if TREE_REUSE and next_root is not None:
            root = next_root
        else:
            # Create new root if not reusing tree
            root = Node(
                env_state_copy=deepcopy(env),
                done_flag=done,
                parent_node=None,
                observation=obs,
                action_idx_leading_to_node=None
            )
        
        # Progress output
        power = env.actions[action]
        print(f"Step {step+1}: Action={action} ({power}W), "
              f"Temp={temps[-1]:.1f}°C, Reward={reward:.3f}")
        
        step += 1
    
    # Episode summary
    print(f"\nEpisode Summary:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final temperature: {temps[-1]:.1f}°C")
    print(f"Steps taken: {len(actions)}")
    
    env.close()
    
    return temps, actions, rewards, total_reward


def plot_results(temps, actions, rewards, env_actions, target_deadline):
    """Plot episode results"""
    
    plot_dir = "Experiments/MCTS_Improved_Plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    steps = range(len(temps))
    
    # Plot 1: Temperature trajectory
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Temperature
    ax1.plot(steps, temps, 'b-', linewidth=2, label='Temperature')
    ax1.axhline(y=100, color='r', linestyle='--', label='Target (100°C)')
    ax1.axvline(x=target_deadline, color='m', linestyle='--', label=f'Deadline (step {target_deadline})')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Actions
    if actions:
        action_powers = [env_actions[a] for a in actions]
        ax2.step(range(len(actions)), action_powers, 'g-', where='post', linewidth=2)
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Heating Power Applied')
        ax2.set_ylim(-100, 3500)
        ax2.grid(True, alpha=0.3)
    
    # Cumulative reward
    if rewards:
        cum_rewards = np.cumsum(rewards)
        ax3.plot(range(len(rewards)), cum_rewards, 'r-', linewidth=2)
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_xlabel('Time Step')
        ax3.set_title('Cumulative Reward')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mcts_episode_results.png'), dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {plot_dir}")


if __name__ == "__main__":
    print("=== Improved MCTS for Kettle Control ===")
    
    # Run episode
    temps, actions, rewards, total_reward = run_mcts_episode()
    
    # Get environment parameters for plotting
    env = KettleEnv()
    env_actions = env.actions
    target_deadline = env.target_deadline_step
    env.close()
    
    # Plot results
    if temps:
        plot_results(temps, actions, rewards, env_actions, target_deadline)