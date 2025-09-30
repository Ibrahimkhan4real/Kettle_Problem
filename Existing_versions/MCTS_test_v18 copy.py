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
UCB_C = 2.0  # Exploration constant
MCTS_ITERATIONS_PER_ACTION = 5000  # Increased for better exploration
MAX_ROLLOUT_DEPTH = 100  # Increased to see further ahead
DISCOUNT_FACTOR = 0.995  # Less aggressive discounting
TREE_REUSE = True
INITIAL_Q_VALUE = 0.0  # Optimistic initialization
MIN_VISITS_FOR_EXPANSION = 10  # Progressive widening threshold


class Node:
    """
    Enhanced MCTS Node with better initialization and statistics
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
        
        # Enhanced statistics
        self.immediate_reward = 0.0
        self.max_reward_seen = -float('inf')
        self.untried_actions = list(range(env_state_copy.action_space.n))
        random.shuffle(self.untried_actions)  # Randomize order
        
        # Initialize with optimistic value
        self._initial_q = INITIAL_Q_VALUE

    def get_value_estimate(self):
        """Get current value estimate with optimistic initialization"""
        if self.num_visits == 0:
            return self._initial_q
        return self.total_simulation_reward / self.num_visits

    def get_ucb_score(self, c_param=UCB_C, parent_visits=None):
        """Enhanced UCB score calculation"""
        if self.num_visits == 0:
            return float('inf')
        
        if parent_visits is None:
            parent_visits = self.parent.num_visits if self.parent else 1
        
        exploitation = self.get_value_estimate()
        exploration = c_param * sqrt(2 * log(parent_visits) / self.num_visits)
        
        return exploitation + exploration

    def detach_from_parent(self):
        """Detaches this node from its parent"""
        if self.parent and self.action_that_led_here is not None:
            if self.action_that_led_here in self.parent.children:
                del self.parent.children[self.action_that_led_here]
        self.parent = None

    def select_untried_action(self):
        """Select an untried action for progressive widening"""
        if not self.untried_actions:
            return None
        return self.untried_actions.pop()

    def is_fully_expanded(self):
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0

    def should_expand(self):
        """Progressive widening: only expand if visited enough times"""
        if self.is_terminal:
            return False
        if not self.untried_actions:
            return False
        # Use progressive widening formula
        k = len(self.children)
        return self.num_visits >= MIN_VISITS_FOR_EXPANSION * (k + 1)

    def expand(self):
        """Expand by adding one new child"""
        if self.is_terminal or not self.untried_actions:
            return None
        
        action = self.select_untried_action()
        if action is None:
            return None
        
        # Create child node
        child_env = deepcopy(self.env_state)
        obs, reward, done, truncated, _ = child_env.step(action)
        child_is_terminal = done or truncated
        
        child = Node(child_env, child_is_terminal, self, obs, action)
        child.immediate_reward = reward
        self.children[action] = child
        
        return child

    def best_child(self, c_param=UCB_C):
        """Select best child using UCB"""
        if not self.children:
            return None
        
        parent_visits = self.num_visits
        return max(self.children.values(), 
                   key=lambda n: n.get_ucb_score(c_param, parent_visits))

    def robust_child(self):
        """Select most visited child (for final action selection)"""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda n: n.num_visits)

    def rollout(self, depth=0):
        """Enhanced rollout with epsilon-greedy strategy"""
        if self.is_terminal or depth >= MAX_ROLLOUT_DEPTH:
            return 0.0
        
        rollout_env = deepcopy(self.env_state)
        cumulative_reward = 0.0
        discount = 1.0
        
        for _ in range(MAX_ROLLOUT_DEPTH - depth):
            # Epsilon-greedy rollout: mostly random, occasionally try specific actions
            if random.random() < 0.9:
                action = rollout_env.action_space.sample()
            else:
                # Try alternating actions to explore different strategies
                action = random.choice([0, 1])
            
            _, reward, done, truncated, _ = rollout_env.step(action)
            cumulative_reward += discount * reward
            discount *= DISCOUNT_FACTOR
            
            if done or truncated:
                break
        
        return cumulative_reward

    def backpropagate(self, value):
        """Backpropagate value through the tree"""
        self.num_visits += 1
        self.total_simulation_reward += value
        self.max_reward_seen = max(self.max_reward_seen, value)
        
        if self.parent:
            parent_value = self.immediate_reward + DISCOUNT_FACTOR * value
            self.parent.backpropagate(parent_value)


def mcts_iteration(root: Node):
    """Single MCTS iteration with progressive widening"""
    
    # 1. Selection - traverse down the tree
    node = root
    path = [node]
    
    while not node.is_terminal:
        if node.should_expand():
            # Expansion possible
            child = node.expand()
            if child:
                path.append(child)
                node = child
                break
        elif node.children:
            # Continue selection
            node = node.best_child()
            if node:
                path.append(node)
            else:
                break
        else:
            # Leaf node that can't expand yet
            break
    
    # 2. Rollout from leaf
    value = node.rollout()
    
    # 3. Backpropagation
    node.backpropagate(value)


def run_mcts_with_restarts(root: Node, num_iterations: int):
    """Run MCTS with periodic random exploration injections"""
    
    # Run most iterations normally
    main_iterations = int(num_iterations * 0.9)
    exploration_iterations = num_iterations - main_iterations
    
    # Main MCTS iterations
    for i in range(main_iterations):
        mcts_iteration(root)
    
    # Exploration boost: force exploration of less-visited branches
    for _ in range(exploration_iterations):
        # Start from root and take random actions until we hit an unexplored node
        node = root
        while node.children and random.random() < 0.7:
            # Sometimes pick randomly instead of UCB
            if random.random() < 0.3:
                node = random.choice(list(node.children.values()))
            else:
                node = node.best_child(c_param=UCB_C * 2)  # Higher exploration
        
        if not node.is_terminal:
            mcts_iteration(node)


def select_action_mcts(root: Node, num_iterations: int, step_num: int):
    """
    Run MCTS and select best action
    """
    if root.is_terminal:
        return None, root
    
    # Run MCTS
    run_mcts_with_restarts(root, num_iterations)
    
    # Debug output
    print(f"\nStep {step_num} - MCTS Analysis (Temp: {root.observation[0]+100:.1f}°C, Steps to deadline: {root.observation[1]:.0f})")
    print("Action statistics:")
    
    action_data = []
    for action, child in root.children.items():
        if child.num_visits > 0:
            avg_value = child.get_value_estimate()
            max_value = child.max_reward_seen
            power = root.env_state.actions[action]
            action_data.append((action, child.num_visits, avg_value, max_value, power))
    
    # Sort by visits
    action_data.sort(key=lambda x: x[1], reverse=True)
    
    for action, visits, avg_val, max_val, power in action_data:
        print(f"  Action {action} ({power}W): {visits} visits, "
              f"avg_value={avg_val:.3f}, max_value={max_val:.3f}")
    
    # Select action - use robust child (most visits)
    best_child = root.robust_child()
    if not best_child:
        return None, root
    
    best_action = best_child.action_that_led_here
    
    # Tree reuse
    if TREE_REUSE:
        best_child.detach_from_parent()
        next_root = best_child
    else:
        next_root = None
    
    return best_action, next_root


def run_mcts_episode():
    """Run a single episode with enhanced MCTS"""
    
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
    
    # Episode data
    temps = [obs[0] + 100]
    actions = []
    rewards = []
    total_reward = 0
    
    print(f"\n{'='*60}")
    print(f"Starting Enhanced MCTS Episode")
    print(f"Initial temp: {temps[0]:.1f}°C, Target: 100°C by step 200")
    print(f"MCTS iterations: {MCTS_ITERATIONS_PER_ACTION}, UCB_C: {UCB_C}")
    print(f"Progressive widening threshold: {MIN_VISITS_FOR_EXPANSION}")
    print(f"{'='*60}")
    
    # Run episode
    done = False
    step = 0
    
    while not done and step < MAX_TIMESTEPS_PER_EPISODE:
        # Select action
        action, next_root = select_action_mcts(root, MCTS_ITERATIONS_PER_ACTION, step + 1)
        
        if action is None:
            print("No valid action found")
            break
        
        # Execute in environment
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        # Record data
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        temps.append(obs[0] + 100)
        
        # Update root
        if TREE_REUSE and next_root is not None:
            root = next_root
        else:
            root = Node(
                env_state_copy=deepcopy(env),
                done_flag=done,
                parent_node=None,
                observation=obs,
                action_idx_leading_to_node=None
            )
        
        # Progress
        power = env.actions[action]
        print(f"\nStep {step+1}: Action={action} ({power}W), "
              f"Temp={temps[-1]:.1f}°C, Reward={reward:.3f}, "
              f"Total={total_reward:.3f}")
        
        step += 1
    
    # Summary
    print(f"\n{'='*40}")
    print(f"Episode Summary:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final temperature: {temps[-1]:.1f}°C")
    print(f"Steps taken: {len(actions)}")
    
    # Action statistics
    if actions:
        on_actions = sum(1 for a in actions if env.actions[a] > 0)
        print(f"Heating actions: {on_actions}/{len(actions)} ({100*on_actions/len(actions):.1f}%)")
    
    env.close()
    
    return temps, actions, rewards, total_reward, env.actions, env.target_deadline_step


def plot_results(temps, actions, rewards, env_actions, target_deadline):
    """Enhanced plotting"""
    
    plot_dir = "Experiments/MCTS_Enhanced_Plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    steps = range(len(temps))
    
    # Temperature
    ax = axes[0]
    ax.plot(steps, temps, 'b-', linewidth=2.5, label='Temperature')
    ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Target (100°C)')
    ax.axhline(y=97.5, color='r', linestyle=':', alpha=0.5, label='Acceptable range')
    ax.axhline(y=102.5, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=target_deadline, color='m', linestyle='--', linewidth=2, label=f'Deadline')
    ax.fill_between([target_deadline-15, target_deadline+15], 0, 150, 
                    color='m', alpha=0.1, label='Time window')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Trajectory')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(temps)-1)
    
    # Actions
    if actions:
        ax = axes[1]
        action_powers = [env_actions[a] for a in actions]
        ax.step(range(len(actions)), action_powers, 'g-', where='post', linewidth=2.5)
        ax.fill_between(range(len(actions)), 0, action_powers, step='post', alpha=0.3, color='g')
        ax.set_ylabel('Power (W)')
        ax.set_title('Heating Power Applied')
        ax.set_ylim(-100, 3500)
        ax.grid(True, alpha=0.3)
    
    # Reward per step
    if rewards:
        ax = axes[2]
        ax.bar(range(len(rewards)), rewards, color=['red' if r < 0 else 'green' for r in rewards])
        ax.set_ylabel('Reward')
        ax.set_title('Reward per Step')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Cumulative reward
    if rewards:
        ax = axes[3]
        cum_rewards = np.cumsum(rewards)
        ax.plot(range(len(rewards)), cum_rewards, 'purple', linewidth=2.5)
        ax.fill_between(range(len(rewards)), 0, cum_rewards, alpha=0.3, color='purple')
        ax.set_ylabel('Cumulative Reward')
        ax.set_xlabel('Time Step')
        ax.set_title('Cumulative Reward')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mcts_enhanced_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {plot_dir}/")


if __name__ == "__main__":
    print("=== Enhanced MCTS for Kettle Control ===")
    
    # Run episode
    temps, actions, rewards, total_reward, env_actions, target_deadline = run_mcts_episode()
    
    # Plot results
    if temps:
        plot_results(temps, actions, rewards, env_actions, target_deadline)