# models/mcts_uct.py
"""
Standard UCT (Upper Confidence bounds applied to Trees) implementation
for the energy-efficient kettle control problem.
"""

import numpy as np
import random
from copy import deepcopy
from math import sqrt, log
from typing import Dict, List, Optional, Any

from .base_mcts import BaseMCTS, MCTSNode


class UCTNode(MCTSNode):
    """Standard UCT Node"""
    
    def __init__(self, env_state, parent=None, action=None, observation=None):
        super().__init__(env_state, parent, action, observation)
        
        # Energy tracking for efficient rollouts
        self.cumulative_energy = 0.0
        if parent and action is not None:
            parent_energy = parent.cumulative_energy
            power_used = env_state.actions[action]
            self.cumulative_energy = parent_energy + power_used * env_state.dt
        
        # Performance tracking
        self.best_rollout_value = -float('inf')
        self.best_efficiency_seen = 0.0
    
    def get_uct_value(self, c_param: float = 1.414) -> float:
        """Calculate UCT value for action selection"""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.get_value()
        
        # Standard UCT formula
        exploitation = self.get_value()
        
        # Adaptive exploration based on context
        if self.observation is not None:
            steps_to_deadline = self.observation[1]
            current_temp = self.observation[0] + 100
            
            # Reduce exploration when already efficient
            if abs(current_temp - 100) < 2.5 and steps_to_deadline > -15:
                exploration_multiplier = 0.5
            elif -20 <= steps_to_deadline <= 20:
                exploration_multiplier = 1.5
            else:
                exploration_multiplier = 1.0
        else:
            exploration_multiplier = 1.0
        
        exploration = exploration_multiplier * c_param * sqrt(log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def should_expand(self, min_visits: int = 1) -> bool:
        """Check if node should be expanded"""
        return (not self.is_terminal and 
                self.untried_actions and 
                self.visits >= min_visits)


class UCT(BaseMCTS):
    """Standard UCT implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # UCT-specific parameters
        self.min_visits_for_expansion = config.get('min_visits_for_expansion', 1)
        self.optimistic_init = config.get('optimistic_init', 0.0)
        
        # Algorithm name
        self.algorithm_name = "UCT"
    
    def create_node(self, env_state, parent=None, action=None, observation=None) -> UCTNode:
        """Create a new UCT node"""
        node = UCTNode(env_state, parent, action, observation)
        node.is_terminal = getattr(env_state, 'steps', 0) >= getattr(env_state, 'max_steps', 250)
        return node
    
    def selection(self, node: UCTNode) -> UCTNode:
        """Selection phase using UCT"""
        current = node
        
        while current.children and not current.is_terminal:
            if not current.is_fully_expanded() and current.should_expand(self.min_visits_for_expansion):
                break
            
            # Select child with highest UCT value
            if not current.children:
                break
            
            best_action = max(current.children.keys(),
                            key=lambda a: current.children[a].get_uct_value(self.c_param))
            
            current = current.children[best_action]
        
        return current
    
    def expansion(self, node: UCTNode) -> Optional[UCTNode]:
        """Expansion phase"""
        if node.is_terminal or not node.untried_actions:
            return None
        
        if not node.should_expand(self.min_visits_for_expansion):
            return None
        
        # Select untried action
        action = node.untried_actions.pop()
        
        # Create child state
        child_env = deepcopy(node.env_state)
        obs, reward, done, truncated, info = child_env.step(action)
        child_is_terminal = done or truncated
        
        # Create child node
        child = UCTNode(child_env, node, action, obs)
        child.is_terminal = child_is_terminal
        node.children[action] = child
        
        self.stats.nodes_expanded += 1
        
        return child
    
    def rollout(self, node: UCTNode) -> float:
        """Energy-efficient rollout policy"""
        if node.is_terminal:
            return 0.0
        
        rollout_env = deepcopy(node.env_state)
        cumulative_reward = 0.0
        discount = 1.0
        total_energy = 0.0
        depth = 0
        
        # Calculate optimal heating strategy
        temp_rise_needed = max(0, rollout_env.T_target - rollout_env.current_temp)
        heating_power = rollout_env.actions[1] if len(rollout_env.actions) > 1 else 3000
        
        if heating_power > 0:
            heating_steps_needed = int(temp_rise_needed * rollout_env.m * rollout_env.c / 
                                     (heating_power - rollout_env.hA * temp_rise_needed/2))
        else:
            heating_steps_needed = 50
        
        # Choose rollout strategy
        strategy = random.random()
        
        while depth < self.max_rollout_depth:
            current_temp = rollout_env.current_temp
            current_step = rollout_env.steps
            target_temp = rollout_env.T_target
            target_step = getattr(rollout_env, 'target_deadline_step', 200)
            
            temp_error = abs(current_temp - target_temp)
            steps_to_deadline = target_step - current_step
            
            # Multiple rollout strategies
            if strategy < 0.3:  # Optimal delayed heating
                if steps_to_deadline <= heating_steps_needed + 5:
                    action = 1 if current_temp < target_temp else 0
                else:
                    action = 0
            elif strategy < 0.5:  # Early heating with coasting
                if steps_to_deadline <= heating_steps_needed + 20:
                    action = 1 if current_temp < target_temp - 1 else 0
                else:
                    action = 0
            elif strategy < 0.7:  # Heat and coast strategy
                if steps_to_deadline > 30 and current_temp < target_temp - 5:
                    action = 1 if random.random() < 0.7 else 0
                elif temp_error <= 2.5:
                    action = 0
                else:
                    action = 1 if current_temp < target_temp else 0
            elif strategy < 0.9:  # Conservative heating
                if current_temp < target_temp - 10:
                    action = 1
                elif current_temp < target_temp and steps_to_deadline < 50:
                    action = 1
                else:
                    action = 0
            else:  # Random exploration
                action = rollout_env.action_space.sample()
            
            # Track energy
            power = rollout_env.actions[action]
            total_energy += power * rollout_env.dt
            
            # Execute action
            _, reward, done, truncated, info = rollout_env.step(action)
            cumulative_reward += discount * reward
            discount *= self.discount_factor
            
            # Check goal achievement
            if info.get('goal_achieved', False):
                efficiency = info.get('energy_efficiency', 0.0)
                cumulative_reward += discount * 500 * efficiency
                break
            
            if done or truncated:
                break
            
            depth += 1
        
        # Calculate efficiency bonus
        if hasattr(rollout_env, 'theoretical_min_energy') and total_energy > 0:
            efficiency = rollout_env.theoretical_min_energy / total_energy
            node.best_efficiency_seen = max(node.best_efficiency_seen, efficiency)
        
        return cumulative_reward
    
    def backpropagation(self, node: UCTNode, value: float):
        """Backpropagation phase"""
        current = node
        accumulated_value = value
        
        while current is not None:
            # Update statistics
            current.update(accumulated_value)
            
            # Track best values
            current.best_rollout_value = max(current.best_rollout_value, accumulated_value)
            
            # Move up the tree with discounting
            if current.parent is not None:
                accumulated_value = accumulated_value * self.discount_factor
            
            current = current.parent
    
    def select_best_action(self) -> int:
        """Select best action considering energy efficiency"""
        if not self.root.children:
            return np.random.choice(self.root.env_state.action_space.n)
        
        # Get current state information
        if self.root.observation is not None:
            current_temp = self.root.observation[0] + 100
            steps_to_deadline = self.root.observation[1]
            
            # If at target temperature, prefer no heating
            if (abs(current_temp - 100) < 2.5 and steps_to_deadline > -20 and 
                0 in self.root.children and 
                self.root.children[0].visits > self.iterations_per_action * 0.1):
                return 0
        
        # Standard selection by visit count
        best_action = max(self.root.children.keys(), 
                         key=lambda a: self.root.children[a].visits)
        return best_action
    
    def get_action_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed action statistics"""
        if self.root is None or not self.root.children:
            return {}
        
        stats = {}
        for action in range(self.root.env_state.action_space.n):
            power = self.root.env_state.actions[action]
            energy_impact = power * self.root.env_state.dt / 3600  # Wh
            
            stats[action] = {
                'visits': 0,
                'average_value': 0.0,
                'uct_value': 0.0,
                'is_child': action in self.root.children,
                'energy_impact': energy_impact,
                'power_w': power
            }
            
            if action in self.root.children:
                child = self.root.children[action]
                stats[action]['visits'] = child.visits
                stats[action]['average_value'] = child.get_value()
                stats[action]['uct_value'] = child.get_uct_value(self.c_param)
                stats[action]['best_efficiency'] = child.best_efficiency_seen
                stats[action]['best_rollout_value'] = child.best_rollout_value
            
            if self.root.visits > 0:
                stats[action]['visit_percentage'] = stats[action]['visits'] / self.root.visits
            else:
                stats[action]['visit_percentage'] = 0.0
        
        return stats
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm-specific information"""
        info = {
            'algorithm': self.algorithm_name,
            'c_param': self.c_param,
            'min_visits_for_expansion': self.min_visits_for_expansion,
            'optimistic_init': self.optimistic_init,
            'iterations_per_action': self.iterations_per_action,
            'max_rollout_depth': self.max_rollout_depth,
            'discount_factor': self.discount_factor,
            'tree_reuse': self.tree_reuse
        }
        
        if self.root is not None:
            info.update({
                'tree_size': self.get_tree_size(),
                'max_depth': self.get_max_depth(),
                'root_visits': self.root.visits,
                'children_count': len(self.root.children)
            })
        
        return info