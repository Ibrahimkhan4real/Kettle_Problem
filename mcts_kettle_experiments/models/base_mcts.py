# models/base_mcts.py
"""
Abstract base class for MCTS implementations in the kettle environment.
Provides consistent interface and common functionality.
"""

from abc import ABC, abstractmethod
import numpy as np
import json
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class MCTSStats:
    """Statistics collected during MCTS search"""
    total_iterations: int = 0
    total_time: float = 0.0
    tree_size: int = 0
    max_depth: int = 0
    avg_rollout_length: float = 0.0
    goal_found_iterations: int = 0
    best_value_found: float = -float('inf')
    nodes_expanded: int = 0
    rollouts_performed: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EpisodeResult:
    """Results from a single episode"""
    episode_id: int
    algorithm_name: str
    config: Dict[str, Any]
    
    # Environment outcomes
    final_temperature: float
    final_step: int
    goal_achieved: bool
    total_reward: float
    energy_consumed_wh: float
    energy_efficiency: float
    
    # Trajectory data
    temperatures: List[float]
    actions: List[int]
    rewards: List[float]
    mcts_iterations_per_step: List[int]
    
    # MCTS statistics
    mcts_stats: MCTSStats
    
    # Timing
    total_time: float
    avg_time_per_step: float
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['mcts_stats'] = self.mcts_stats.to_dict()
        return result


class MCTSNode:
    """Base MCTS Node class"""
    
    def __init__(self, env_state, parent=None, action=None, observation=None):
        self.env_state = env_state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.observation = observation
        
        # MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.untried_actions = list(range(env_state.action_space.n))
        np.random.shuffle(self.untried_actions)
        
        # Terminal state
        self.is_terminal = False
        
        # Additional tracking
        self.depth = 0 if parent is None else parent.depth + 1
        self.creation_time = time.time()
    
    def get_value(self) -> float:
        """Get average value of this node"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def get_uct_value(self, c_param: float = 1.414) -> float:
        """Calculate UCT value for selection"""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.get_value()
        
        exploitation = self.get_value()
        exploration = c_param * np.sqrt(np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    def update(self, value: float):
        """Update node statistics"""
        self.visits += 1
        self.value_sum += value


class BaseMCTS(ABC):
    """Abstract base class for MCTS algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm_name = self.__class__.__name__
        
        # Extract common parameters
        self.iterations_per_action = config.get('iterations_per_action', 1000)
        self.c_param = config.get('c_param', 1.414)
        self.max_rollout_depth = config.get('max_rollout_depth', 100)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.tree_reuse = config.get('tree_reuse', True)
        
        # Statistics tracking
        self.stats = MCTSStats()
        self.episode_data = []
        
        # Tree root
        self.root = None
        
    @abstractmethod
    def create_node(self, env_state, parent=None, action=None, observation=None) -> MCTSNode:
        """Create a new MCTS node"""
        pass
    
    @abstractmethod
    def selection(self, node: MCTSNode) -> MCTSNode:
        """Selection phase of MCTS"""
        pass
    
    @abstractmethod
    def expansion(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expansion phase of MCTS"""
        pass
    
    @abstractmethod
    def rollout(self, node: MCTSNode) -> float:
        """Rollout/simulation phase of MCTS"""
        pass
    
    @abstractmethod
    def backpropagation(self, node: MCTSNode, value: float):
        """Backpropagation phase of MCTS"""
        pass
    
    def mcts_iteration(self) -> bool:
        """Single MCTS iteration"""
        if self.root is None or self.root.is_terminal:
            return False
        
        # 1. Selection
        selected_node = self.selection(self.root)
        
        # 2. Expansion
        if not selected_node.is_terminal:
            expanded_node = self.expansion(selected_node)
            if expanded_node is not None:
                selected_node = expanded_node
        
        # 3. Rollout
        rollout_value = self.rollout(selected_node)
        
        # 4. Backpropagation
        self.backpropagation(selected_node, rollout_value)
        
        # Update statistics
        self.stats.total_iterations += 1
        self.stats.rollouts_performed += 1
        self.stats.best_value_found = max(self.stats.best_value_found, rollout_value)
        
        return True
    
    def search(self, root_state, observation=None, iterations: Optional[int] = None) -> int:
        """Run MCTS search and return best action"""
        start_time = time.time()
        
        # Initialize root if needed
        if not self.tree_reuse or self.root is None:
            self.root = self.create_node(root_state, observation=observation)
        
        iterations = iterations or self.iterations_per_action
        
        # Run MCTS iterations
        for i in range(iterations):
            success = self.mcts_iteration()
            if not success:
                break
        
        # Select best action
        best_action = self.select_best_action()
        
        # Update statistics
        search_time = time.time() - start_time
        self.stats.total_time += search_time
        self.stats.tree_size = self.get_tree_size()
        self.stats.max_depth = self.get_max_depth()
        
        return best_action
    
    def select_best_action(self) -> int:
        """Select the best action based on visit counts"""
        if not self.root.children:
            return np.random.choice(self.root.env_state.action_space.n)
        
        # Select action with most visits
        best_action = max(self.root.children.keys(), 
                         key=lambda a: self.root.children[a].visits)
        return best_action
    
    def update_root(self, action: int, new_state, new_observation=None):
        """Update root after action is taken"""
        if self.tree_reuse and action in self.root.children:
            # Reuse subtree
            new_root = self.root.children[action]
            new_root.parent = None
            self.root = new_root
        else:
            # Create new root
            self.root = self.create_node(new_state, observation=new_observation)
    
    def get_tree_size(self) -> int:
        """Count total nodes in tree"""
        if self.root is None:
            return 0
        
        count = 0
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        
        return count
    
    def get_max_depth(self) -> int:
        """Get maximum depth of tree"""
        if self.root is None:
            return 0
        
        max_depth = 0
        stack = [(self.root, 0)]
        
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in node.children.values():
                stack.append((child, depth + 1))
        
        return max_depth
    
    def get_action_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for each action"""
        if self.root is None or not self.root.children:
            return {}
        
        stats = {}
        for action, child in self.root.children.items():
            stats[action] = {
                'visits': child.visits,
                'average_value': child.get_value(),
                'uct_value': child.get_uct_value(self.c_param),
                'visit_percentage': child.visits / self.root.visits if self.root.visits > 0 else 0
            }
        
        return stats
    
    def run_episode(self, env, max_steps: int = 250) -> EpisodeResult:
        """Run a complete episode"""
        start_time = time.time()
        
        # Reset environment
        obs, info = env.reset()
        
        # Episode data
        temperatures = [obs[0] + 100]  # Convert to actual temperature
        actions = []
        rewards = []
        step_times = []
        iterations_per_step = []
        
        # Initialize root
        self.root = self.create_node(deepcopy(env), observation=obs)
        
        # Run episode
        step = 0
        total_reward = 0.0
        done = False
        
        while not done and step < max_steps:
            step_start = time.time()
            
            # MCTS search
            action = self.search(deepcopy(env), obs, self.iterations_per_action)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Record data
            actions.append(action)
            rewards.append(reward)
            temperatures.append(obs[0] + 100)
            total_reward += reward
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            iterations_per_step.append(self.iterations_per_action)
            
            # Update root for next iteration
            if not done:
                self.update_root(action, deepcopy(env), obs)
            
            step += 1
        
        # Calculate final metrics
        total_time = time.time() - start_time
        energy_consumed = sum(env.actions[a] * env.dt for a in actions) / 3600  # Wh
        
        # Energy efficiency
        if hasattr(env, 'theoretical_min_energy'):
            theoretical_min = env.theoretical_min_energy / 3600
            energy_efficiency = theoretical_min / energy_consumed if energy_consumed > 0 else 0
        else:
            energy_efficiency = 0.0
        
        # Goal achievement
        final_temp = temperatures[-1]
        temp_error = abs(final_temp - 100)
        time_error = abs(len(actions) - getattr(env, 'target_deadline_step', 200))
        goal_achieved = temp_error <= 2.5 and time_error <= 15
        
        # Create result
        result = EpisodeResult(
            episode_id=len(self.episode_data),
            algorithm_name=self.algorithm_name,
            config=self.config.copy(),
            final_temperature=final_temp,
            final_step=len(actions),
            goal_achieved=goal_achieved,
            total_reward=total_reward,
            energy_consumed_wh=energy_consumed,
            energy_efficiency=energy_efficiency,
            temperatures=temperatures,
            actions=actions,
            rewards=rewards,
            mcts_iterations_per_step=iterations_per_step,
            mcts_stats=deepcopy(self.stats),
            total_time=total_time,
            avg_time_per_step=np.mean(step_times) if step_times else 0.0
        )
        
        self.episode_data.append(result)
        return result
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = MCTSStats()
        self.root = None
    
    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration"""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Update algorithm configuration"""
        self.config.update(config)
        self.iterations_per_action = config.get('iterations_per_action', self.iterations_per_action)
        self.c_param = config.get('c_param', self.c_param)
        self.max_rollout_depth = config.get('max_rollout_depth', self.max_rollout_depth)
        self.discount_factor = config.get('discount_factor', self.discount_factor)
        self.tree_reuse = config.get('tree_reuse', self.tree_reuse)