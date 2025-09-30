# utils/config_manager.py
"""
Configuration management for MCTS experiments.
Handles loading, validation, and generation of experiment configurations.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools
import copy


@dataclass
class EnvironmentConfig:
    """Configuration for the kettle environment"""
    water_volume: float = 1.0
    initial_temp: float = 20.0
    ambient_temp: float = 20.0
    max_steps: int = 250
    target_deadline_step: int = 200
    
    # Reward shaping parameters
    gamma: float = 0.99
    goal_reward: float = 1000.0
    energy_cost_per_kJ: float = 0.01
    efficiency_bonus_scale: float = 200.0
    
    # Potential-based shaping
    enable_reward_shaping: bool = True
    potential_scale: float = 100.0
    temp_potential_weight: float = 1.0
    time_potential_weight: float = 0.5
    energy_potential_weight: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MCTSConfig:
    """Base configuration for MCTS algorithms"""
    algorithm_name: str = "MCTS-RAVE"
    iterations_per_action: int = 1000
    c_param: float = 1.414
    max_rollout_depth: int = 200
    discount_factor: float = 0.99
    tree_reuse: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAVEConfig(MCTSConfig):
    """Configuration for MCTS-RAVE"""
    algorithm_name: str = "MCTS-RAVE"
    rave_constant: float = 8000.0
    use_rave: bool = True
    progressive_widening_alpha: float = 0.5
    min_visits_for_expansion: int = 10
    optimistic_init: float = -5.0


@dataclass
class UCTConfig(MCTSConfig):
    """Configuration for standard UCT"""
    algorithm_name: str = "UCT"
    # UCT-specific parameters can be added here


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str
    description: str
    environment: EnvironmentConfig
    algorithm: Union[MCTSConfig, RAVEConfig, UCTConfig]
    
    # Experiment parameters
    num_episodes: int = 10
    max_steps_per_episode: int = 250
    random_seed: Optional[int] = None
    
    # Output settings
    save_trajectories: bool = True
    save_mcts_stats: bool = True
    generate_plots: bool = True
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'environment': self.environment.to_dict(),
            'algorithm': self.algorithm.to_dict(),
            'num_episodes': self.num_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'random_seed': self.random_seed,
            'save_trajectories': self.save_trajectories,
            'save_mcts_stats': self.save_mcts_stats,
            'generate_plots': self.generate_plots,
            'verbose': self.verbose
        }


class ConfigManager:
    """Manages experiment configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Create default config files if they don't exist
        self._create_default_configs()
    
    def _create_default_configs(self):
        """Create default configuration files"""
        
        # Base configuration
        base_config = {
            "environment_defaults": {
                "water_volume": 1.0,
                "initial_temp": 20.0,
                "ambient_temp": 20.0,
                "max_steps": 250,
                "target_deadline_step": 200,
                "gamma": 0.99,
                "goal_reward": 1000.0,
                "energy_cost_per_kJ": 0.01,
                "efficiency_bonus_scale": 200.0,
                "enable_reward_shaping": True,
                "potential_scale": 100.0,
                "temp_potential_weight": 1.0,
                "time_potential_weight": 0.5,
                "energy_potential_weight": 0.2
            },
            "experiment_defaults": {
                "num_episodes": 10,
                "max_steps_per_episode": 250,
                "save_trajectories": True,
                "save_mcts_stats": True,
                "generate_plots": True,
                "verbose": True
            }
        }
        
        # MCTS algorithm variants
        mcts_variants = {
            "MCTS-RAVE": {
                "algorithm_name": "MCTS-RAVE",
                "iterations_per_action": 1000,
                "c_param": 1.5,
                "max_rollout_depth": 200,
                "discount_factor": 0.99,
                "tree_reuse": True,
                "rave_constant": 8000.0,
                "use_rave": True,
                "progressive_widening_alpha": 0.5,
                "min_visits_for_expansion": 10,
                "optimistic_init": -5.0
            },
            "UCT": {
                "algorithm_name": "UCT",
                "iterations_per_action": 1000,
                "c_param": 1.414,
                "max_rollout_depth": 200,
                "discount_factor": 0.99,
                "tree_reuse": True
            },
            "MCTS-RAVE-HighIterations": {
                "algorithm_name": "MCTS-RAVE",
                "iterations_per_action": 5000,
                "c_param": 1.5,
                "max_rollout_depth": 200,
                "discount_factor": 0.99,
                "tree_reuse": True,
                "rave_constant": 8000.0,
                "use_rave": True,
                "progressive_widening_alpha": 0.5,
                "min_visits_for_expansion": 10,
                "optimistic_init": -5.0
            },
            "MCTS-NoRAVE": {
                "algorithm_name": "MCTS-RAVE",
                "iterations_per_action": 1000,
                "c_param": 1.414,
                "max_rollout_depth": 200,
                "discount_factor": 0.99,
                "tree_reuse": True,
                "rave_constant": 8000.0,
                "use_rave": False,
                "progressive_widening_alpha": 0.5,
                "min_visits_for_expansion": 10,
                "optimistic_init": -5.0
            }
        }
        
        # Environment variations
        environment_configs = {
            "standard": {
                "description": "Standard kettle environment",
                "initial_temp": 20.0,
                "ambient_temp": 20.0,
                "enable_reward_shaping": True
            },
            "hot_ambient": {
                "description": "Hot ambient temperature environment",
                "initial_temp": 20.0,
                "ambient_temp": 35.0,
                "enable_reward_shaping": True
            },
            "cold_start": {
                "description": "Cold start temperature",
                "initial_temp": 5.0,
                "ambient_temp": 20.0,
                "enable_reward_shaping": True
            },
            "no_shaping": {
                "description": "No reward shaping - sparse rewards only",
                "initial_temp": 20.0,
                "ambient_temp": 20.0,
                "enable_reward_shaping": False
            },
            "tight_deadline": {
                "description": "Tighter deadline constraint",
                "initial_temp": 20.0,
                "ambient_temp": 20.0,
                "target_deadline_step": 180,
                "enable_reward_shaping": True
            }
        }
        
        # Save configuration files
        self._save_json(base_config, "base_config.json")
        self._save_json(mcts_variants, "mcts_variants.json")
        self._save_json(environment_configs, "environment_configs.json")
    
    def _save_json(self, data: Dict[str, Any], filename: str):
        """Save data to JSON file"""
        filepath = self.config_dir / filename
        if not filepath.exists():  # Only create if doesn't exist
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    def load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        with open(self.config_dir / "base_config.json", 'r') as f:
            return json.load(f)
    
    def load_mcts_variants(self) -> Dict[str, Any]:
        """Load MCTS algorithm variants"""
        with open(self.config_dir / "mcts_variants.json", 'r') as f:
            return json.load(f)
    
    def load_environment_configs(self) -> Dict[str, Any]:
        """Load environment configurations"""
        with open(self.config_dir / "environment_configs.json", 'r') as f:
            return json.load(f)
    
    def create_experiment_config(self, 
                           experiment_name: str,
                           algorithm_name: str = "MCTS-RAVE",
                           environment_name: str = "standard",
                           description: str = "",
                           **overrides) -> ExperimentConfig:
        """Create a complete experiment configuration"""
        
        # Load base configurations
        base_config = self.load_base_config()
        mcts_variants = self.load_mcts_variants()
        env_configs = self.load_environment_configs()
        
        # Get algorithm configuration
        if algorithm_name not in mcts_variants:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        algorithm_config = mcts_variants[algorithm_name].copy()
        
        # Get environment configuration
        if environment_name not in env_configs:
            raise ValueError(f"Unknown environment: {environment_name}")
        env_config = env_configs[environment_name].copy()
        
        # Merge with defaults
        env_defaults = base_config["environment_defaults"].copy()
        env_defaults.update(env_config)
        
        # Remove description field before creating EnvironmentConfig
        env_defaults.pop('description', None)
        
        exp_defaults = base_config["experiment_defaults"].copy()
        
        # Apply overrides
        for key, value in overrides.items():
            if key in env_defaults:
                env_defaults[key] = value
            elif key in algorithm_config:
                algorithm_config[key] = value
            elif key in exp_defaults:
                exp_defaults[key] = value
        
        # Create configuration objects
        environment = EnvironmentConfig(**env_defaults)
        
        if algorithm_name == "MCTS-RAVE":
            algorithm = RAVEConfig(**algorithm_config)
        else:
            algorithm = MCTSConfig(**algorithm_config)
        
        # Create experiment configuration
        experiment = ExperimentConfig(
            experiment_name=experiment_name,
            description=description or f"{algorithm_name} on {environment_name} environment",
            environment=environment,
            algorithm=algorithm,
            **exp_defaults
        )
        
        return experiment
    
    def generate_parameter_sweep(self, 
                               base_config: ExperimentConfig,
                               parameter_ranges: Dict[str, List[Any]]) -> List[ExperimentConfig]:
        """Generate configurations for parameter sweep"""
        
        configs = []
        
        # Get all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        for i, combination in enumerate(itertools.product(*param_values)):
            # Create new config
            config = copy.deepcopy(base_config)
            
            # Apply parameter values
            config_name_parts = [base_config.experiment_name]
            
            for param_name, param_value in zip(param_names, combination):
                # Update configuration
                if hasattr(config.algorithm, param_name):
                    setattr(config.algorithm, param_name, param_value)
                elif hasattr(config.environment, param_name):
                    setattr(config.environment, param_name, param_value)
                else:
                    setattr(config, param_name, param_value)
                
                # Add to name
                config_name_parts.append(f"{param_name}_{param_value}")
            
            # Update experiment name
            config.experiment_name = "_".join(str(p) for p in config_name_parts)
            config.description = f"Parameter sweep: {dict(zip(param_names, combination))}"
            
            configs.append(config)
        
        return configs
    
    def save_experiment_config(self, config: ExperimentConfig, filepath: Optional[str] = None):
        """Save experiment configuration to file"""
        if filepath is None:
            filepath = self.config_dir / f"{config.experiment_name}_config.json"
        else:
            filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def load_experiment_config(self, filepath: str) -> ExperimentConfig:
        """Load experiment configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct objects
        env_config = EnvironmentConfig(**data['environment'])
        
        algorithm_name = data['algorithm']['algorithm_name']
        if algorithm_name == "MCTS-RAVE":
            alg_config = RAVEConfig(**data['algorithm'])
        else:
            alg_config = MCTSConfig(**data['algorithm'])
        
        # Create experiment config
        experiment_data = {k: v for k, v in data.items() 
                          if k not in ['environment', 'algorithm']}
        
        return ExperimentConfig(
            environment=env_config,
            algorithm=alg_config,
            **experiment_data
        )
    
    def get_predefined_experiments(self) -> List[Dict[str, str]]:
        """Get list of predefined experiment configurations"""
        mcts_variants = self.load_mcts_variants()
        env_configs = self.load_environment_configs()
        
        experiments = []
        
        for alg_name in mcts_variants.keys():
            for env_name in env_configs.keys():
                experiments.append({
                    'name': f"{alg_name}_{env_name}",
                    'algorithm': alg_name,
                    'environment': env_name,
                    'description': f"{alg_name} algorithm on {env_name} environment"
                })
        
        return experiments
    
    def create_comparison_suite(self) -> List[ExperimentConfig]:
        """Create a suite of experiments for algorithm comparison"""
        
        # Standard comparison experiments
        experiments = []
        
        algorithms = ["MCTS-RAVE", "UCT", "MCTS-NoRAVE"]
        environments = ["standard", "no_shaping", "hot_ambient"]
        
        for alg in algorithms:
            for env in environments:
                config = self.create_experiment_config(
                    experiment_name=f"comparison_{alg}_{env}",
                    algorithm_name=alg,
                    environment_name=env,
                    num_episodes=20,  # More episodes for comparison
                    description=f"Comparison experiment: {alg} on {env}"
                )
                experiments.append(config)
        
        return experiments
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """Validate experiment configuration and return any issues"""
        issues = []
        
        # Validate environment parameters
        if config.environment.water_volume <= 0:
            issues.append("Water volume must be positive")
        
        if config.environment.max_steps <= 0:
            issues.append("Max steps must be positive")
        
        if config.environment.target_deadline_step > config.environment.max_steps:
            issues.append("Target deadline step cannot exceed max steps")
        
        if not (0 <= config.environment.gamma < 1):
            issues.append("Gamma must be in [0, 1)")
        
        # Validate algorithm parameters
        if config.algorithm.iterations_per_action <= 0:
            issues.append("Iterations per action must be positive")
        
        if config.algorithm.c_param < 0:
            issues.append("UCB constant must be non-negative")
        
        if config.algorithm.max_rollout_depth <= 0:
            issues.append("Max rollout depth must be positive")
        
        # RAVE-specific validation
        if isinstance(config.algorithm, RAVEConfig):
            if config.algorithm.rave_constant <= 0:
                issues.append("RAVE constant must be positive")
            
            if not (0 < config.algorithm.progressive_widening_alpha <= 1):
                issues.append("Progressive widening alpha must be in (0, 1]")
        
        # Experiment parameters
        if config.num_episodes <= 0:
            issues.append("Number of episodes must be positive")
        
        return issues