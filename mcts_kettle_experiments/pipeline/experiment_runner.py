# pipeline/experiment_runner.py
"""
Main experiment runner for MCTS kettle control experiments.
Orchestrates the entire testing pipeline from configuration to results.
"""

import os
import json
import time
import datetime
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import logging

# Import our components
from utils.config_manager import ConfigManager, ExperimentConfig
from utils.data_collector import DataCollector
from utils.visualization_utils import VisualizationUtils
from utils.metrics_calculator import MetricsCalculator
from models.base_mcts import BaseMCTS, EpisodeResult
from models.mcts_rave import MCTSRAVE
from models.kettle_dynamic_env_v26 import KettleEnv


class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, base_dir: str = "mcts_kettle_experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.config_manager = ConfigManager(self.base_dir / "configs")
        self.data_collector = DataCollector(self.base_dir)
        self.visualization_utils = VisualizationUtils(self.base_dir)
        self.metrics_calculator = MetricsCalculator()
        
        # Setup logging
        self._setup_logging()
        
        # Algorithm registry
        self.algorithm_registry = {
            'MCTS-RAVE': MCTSRAVE,
            'UCT': MCTSRAVE,  # Will use RAVE class with RAVE disabled
            'MCTS-NoRAVE': MCTSRAVE  # Will use RAVE class with RAVE disabled
        }
        
        self.logger.info("ExperimentRunner initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"experiment_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def register_algorithm(self, name: str, algorithm_class: Type[BaseMCTS]):
        """Register a new algorithm class"""
        self.algorithm_registry[name] = algorithm_class
        self.logger.info(f"Registered algorithm: {name}")
    
    def create_environment(self, env_config: Dict[str, Any]) -> KettleEnv:
        """Create kettle environment from configuration"""
        return KettleEnv(**env_config)
    
    def create_algorithm(self, alg_config: Dict[str, Any]) -> BaseMCTS:
        """Create MCTS algorithm from configuration"""
        algorithm_name = alg_config['algorithm_name']
        
        if algorithm_name not in self.algorithm_registry:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algorithm_class = self.algorithm_registry[algorithm_name]
        return algorithm_class(alg_config)
    
    def run_single_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment with the given configuration"""
        
        self.logger.info(f"Starting experiment: {config.experiment_name}")
        start_time = time.time()
        
        # Validate configuration
        issues = self.config_manager.validate_config(config)
        if issues:
            error_msg = f"Configuration validation failed: {issues}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create experiment directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.base_dir / "experiments" / f"{timestamp}_{config.experiment_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        self.config_manager.save_experiment_config(config, config_path)
        
        try:
            # Setup random seed
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
            
            # Create environment and algorithm
            env = self.create_environment(config.environment.to_dict())
            algorithm = self.create_algorithm(config.algorithm.to_dict())
            
            # Run episodes
            episode_results = []
            episode_data_dir = exp_dir / "episode_data"
            episode_data_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Running {config.num_episodes} episodes...")
            
            for episode_id in range(config.num_episodes):
                self.logger.info(f"Episode {episode_id + 1}/{config.num_episodes}")
                
                # Reset algorithm statistics
                algorithm.reset_statistics()
                
                # Run episode
                episode_result = algorithm.run_episode(env, config.max_steps_per_episode)
                episode_result.episode_id = episode_id
                
                episode_results.append(episode_result)
                
                # Save episode data
                if config.save_trajectories:
                    episode_file = episode_data_dir / f"episode_{episode_id:03d}.json"
                    self.data_collector.save_episode_data(episode_result, episode_file)
                
                # Log progress
                if config.verbose:
                    self._log_episode_summary(episode_result)
            
            # Calculate metrics
            self.logger.info("Calculating metrics...")
            metrics = self.metrics_calculator.calculate_experiment_metrics(episode_results)
            
            # Save metrics
            metrics_dir = exp_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            self.data_collector.save_metrics(metrics, metrics_dir / "performance_metrics.json")
            
            # Save aggregated data
            if config.save_trajectories:
                self._save_aggregated_data(episode_results, exp_dir)
            
            # Generate visualizations
            if config.generate_plots:
                self.logger.info("Generating visualizations...")
                plots_dir = exp_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                self.visualization_utils.generate_experiment_plots(
                    episode_results, plots_dir, config.experiment_name
                )
            
            # Create experiment summary
            total_time = time.time() - start_time
            summary = self._create_experiment_summary(
                config, episode_results, metrics, total_time
            )
            
            # Save summary
            summary_path = exp_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Experiment completed in {total_time:.2f}s")
            self.logger.info(f"Results saved to: {exp_dir}")
            
            return {
                'config': config,
                'results': episode_results,
                'metrics': metrics,
                'summary': summary,
                'experiment_dir': str(exp_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Save error information
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(exp_dir / "error.json", 'w') as f:
                json.dump(error_info, f, indent=2)
            
            raise
    
    def run_experiment_batch(self, configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
        """Run multiple experiments"""
        
        self.logger.info(f"Starting batch of {len(configs)} experiments")
        
        results = []
        failed_experiments = []
        
        for i, config in enumerate(configs):
            self.logger.info(f"Batch progress: {i + 1}/{len(configs)}")
            
            try:
                result = self.run_single_experiment(config)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Experiment {config.experiment_name} failed: {str(e)}")
                failed_experiments.append({
                    'config': config,
                    'error': str(e)
                })
        
        # Log batch summary
        self.logger.info(f"Batch completed: {len(results)} successful, {len(failed_experiments)} failed")
        
        if failed_experiments:
            self.logger.warning("Failed experiments:")
            for failed in failed_experiments:
                self.logger.warning(f"  - {failed['config'].experiment_name}: {failed['error']}")
        
        return results
    
    def run_comparison_study(self, 
                           algorithms: List[str],
                           environments: List[str] = None,
                           num_episodes: int = 20) -> Dict[str, Any]:
        """Run a comprehensive comparison study"""
        
        if environments is None:
            environments = ["standard", "no_shaping", "hot_ambient"]
        
        self.logger.info(f"Starting comparison study: {algorithms} x {environments}")
        
        # Generate configurations
        configs = []
        for alg in algorithms:
            for env in environments:
                config = self.config_manager.create_experiment_config(
                    experiment_name=f"comparison_{alg}_{env}",
                    algorithm_name=alg,
                    environment_name=env,
                    num_episodes=num_episodes,
                    description=f"Comparison: {alg} on {env}"
                )
                configs.append(config)
        
        # Run experiments
        results = self.run_experiment_batch(configs)
        
        # Analyze results
        self.logger.info("Performing comparative analysis...")
        comparison_results = self._analyze_comparison_results(results, algorithms, environments)
        
        # Save comparison results
        comparison_dir = self.base_dir / "results" / "comparative_analysis"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = comparison_dir / f"comparison_{timestamp}.json"
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Generate comparison plots
        self.visualization_utils.generate_comparison_plots(
            results, comparison_dir, algorithms, environments
        )
        
        self.logger.info(f"Comparison study completed. Results saved to: {comparison_dir}")
        
        return comparison_results
    
    def run_parameter_sweep(self, 
                          base_config: ExperimentConfig,
                          parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Run parameter sweep experiment"""
        
        self.logger.info(f"Starting parameter sweep: {list(parameter_ranges.keys())}")
        
        # Generate configurations
        configs = self.config_manager.generate_parameter_sweep(base_config, parameter_ranges)
        
        self.logger.info(f"Generated {len(configs)} configurations")
        
        # Run experiments
        results = self.run_experiment_batch(configs)
        
        # Analyze parameter sensitivity
        self.logger.info("Analyzing parameter sensitivity...")
        sensitivity_results = self._analyze_parameter_sensitivity(
            results, parameter_ranges
        )
        
        # Save results
        sweep_dir = self.base_dir / "results" / "parameter_sweeps"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_file = sweep_dir / f"parameter_sweep_{timestamp}.json"
        
        with open(sweep_file, 'w') as f:
            json.dump(sensitivity_results, f, indent=2)
        
        # Generate parameter sensitivity plots
        self.visualization_utils.generate_parameter_plots(
            results, sweep_dir, parameter_ranges
        )
        
        self.logger.info(f"Parameter sweep completed. Results saved to: {sweep_dir}")
        
        return sensitivity_results
    
    def _log_episode_summary(self, result: EpisodeResult):
        """Log summary of episode results"""
        goal_status = "✓" if result.goal_achieved else "✗"
        self.logger.info(
            f"  Episode {result.episode_id}: {goal_status} "
            f"Temp: {result.final_temperature:.1f}°C, "
            f"Energy: {result.energy_consumed_wh:.1f}Wh, "
            f"Efficiency: {result.energy_efficiency:.1%}, "
            f"Reward: {result.total_reward:.1f}"
        )
    
    def _save_aggregated_data(self, episode_results: List[EpisodeResult], exp_dir: Path):
        """Save aggregated episode data as CSV"""
        
        # Create DataFrame with key metrics
        data = []
        for result in episode_results:
            data.append({
                'episode_id': result.episode_id,
                'algorithm': result.algorithm_name,
                'goal_achieved': result.goal_achieved,
                'final_temperature': result.final_temperature,
                'final_step': result.final_step,
                'total_reward': result.total_reward,
                'energy_consumed_wh': result.energy_consumed_wh,
                'energy_efficiency': result.energy_efficiency,
                'total_time': result.total_time,
                'avg_time_per_step': result.avg_time_per_step,
                'mcts_total_iterations': result.mcts_stats.total_iterations,
                'mcts_tree_size': result.mcts_stats.tree_size,
                'mcts_max_depth': result.mcts_stats.max_depth
            })
        
        df = pd.DataFrame(data)
        df.to_csv(exp_dir / "episode_summary.csv", index=False)
        
        # Save detailed trajectory data
        trajectory_data = []
        for result in episode_results:
            for step, (temp, action, reward) in enumerate(
                zip(result.temperatures[1:], result.actions, result.rewards)
            ):
                trajectory_data.append({
                    'episode_id': result.episode_id,
                    'step': step,
                    'temperature': temp,
                    'action': action,
                    'reward': reward
                })
        
        if trajectory_data:
            trajectory_df = pd.DataFrame(trajectory_data)
            trajectory_df.to_csv(exp_dir / "trajectory_data.csv", index=False)
    
    def _create_experiment_summary(self, 
                                 config: ExperimentConfig,
                                 episode_results: List[EpisodeResult],
                                 metrics: Dict[str, Any],
                                 total_time: float) -> Dict[str, Any]:
        """Create experiment summary"""
        
        return {
            'experiment_info': {
                'name': config.experiment_name,
                'description': config.description,
                'algorithm': config.algorithm.algorithm_name,
                'num_episodes': len(episode_results),
                'total_time': total_time,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'performance_summary': {
                'success_rate': metrics['goal_achievement_rate'],
                'avg_energy_efficiency': metrics['avg_energy_efficiency'],
                'avg_final_temperature': metrics['avg_final_temperature'],
                'avg_total_reward': metrics['avg_total_reward'],
                'avg_energy_consumed': metrics['avg_energy_consumed']
            },
            'mcts_summary': {
                'avg_iterations_per_episode': metrics.get('avg_mcts_iterations', 0),
                'avg_tree_size': metrics.get('avg_tree_size', 0),
                'avg_max_depth': metrics.get('avg_max_depth', 0),
                'avg_time_per_step': metrics.get('avg_time_per_step', 0)
            },
            'configuration': config.to_dict()
        }
    
    def _analyze_comparison_results(self, 
                                  results: List[Dict[str, Any]],
                                  algorithms: List[str],
                                  environments: List[str]) -> Dict[str, Any]:
        """Analyze results from comparison study"""
        
        analysis = {
            'algorithms': algorithms,
            'environments': environments,
            'summary': {},
            'detailed_comparison': {},
            'statistical_tests': {},
            'rankings': {}
        }
        
        # Group results by algorithm and environment
        grouped_results = {}
        for result in results:
            alg = result['config'].algorithm.algorithm_name
            env_name = result['config'].experiment_name.split('_')[-1]  # Extract env name
            
            key = f"{alg}_{env_name}"
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].extend(result['results'])
        
        # Calculate statistics for each group
        for key, episode_results in grouped_results.items():
            alg, env = key.split('_', 1)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_experiment_metrics(episode_results)
            
            analysis['detailed_comparison'][key] = {
                'algorithm': alg,
                'environment': env,
                'metrics': metrics,
                'num_episodes': len(episode_results)
            }
        
        # Overall rankings
        for metric in ['goal_achievement_rate', 'avg_energy_efficiency', 'avg_total_reward']:
            rankings = sorted(
                analysis['detailed_comparison'].items(),
                key=lambda x: x[1]['metrics'].get(metric, 0),
                reverse=True
            )
            
            analysis['rankings'][metric] = [
                {'algorithm_env': key, 'value': data['metrics'].get(metric, 0)}
                for key, data in rankings
            ]
        
        return analysis
    
    def _analyze_parameter_sensitivity(self, 
                                     results: List[Dict[str, Any]],
                                     parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze parameter sensitivity"""
        
        analysis = {
            'parameters': list(parameter_ranges.keys()),
            'parameter_ranges': parameter_ranges,
            'sensitivity_analysis': {},
            'best_configurations': {},
            'parameter_effects': {}
        }
        
        # Extract parameter values and performance metrics
        data_points = []
        for result in results:
            config = result['config']
            metrics = result['metrics']
            
            point = {}
            
            # Extract parameter values
            for param_name in parameter_ranges.keys():
                if hasattr(config.algorithm, param_name):
                    point[param_name] = getattr(config.algorithm, param_name)
                elif hasattr(config.environment, param_name):
                    point[param_name] = getattr(config.environment, param_name)
                elif hasattr(config, param_name):
                    point[param_name] = getattr(config, param_name)
            
            # Add performance metrics
            point.update({
                'goal_achievement_rate': metrics['goal_achievement_rate'],
                'avg_energy_efficiency': metrics['avg_energy_efficiency'],
                'avg_total_reward': metrics['avg_total_reward']
            })
            
            data_points.append(point)
        
        # Find best configurations
        for metric in ['goal_achievement_rate', 'avg_energy_efficiency', 'avg_total_reward']:
            best_point = max(data_points, key=lambda x: x.get(metric, 0))
            
            analysis['best_configurations'][metric] = {
                'parameters': {k: v for k, v in best_point.items() 
                             if k in parameter_ranges},
                'performance': best_point.get(metric, 0)
            }
        
        return analysis
    
    def load_experiment_results(self, experiment_dir: str) -> Dict[str, Any]:
        """Load results from a completed experiment"""
        
        exp_path = Path(experiment_dir)
        
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Load configuration
        config_file = exp_path / "config.json"
        if config_file.exists():
            config = self.config_manager.load_experiment_config(config_file)
        else:
            config = None
        
        # Load summary
        summary_file = exp_path / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = None
        
        # Load episode data
        episode_results = []
        episode_dir = exp_path / "episode_data"
        
        if episode_dir.exists():
            for episode_file in sorted(episode_dir.glob("episode_*.json")):
                episode_results.append(
                    self.data_collector.load_episode_data(episode_file)
                )
        
        # Load metrics
        metrics_file = exp_path / "metrics" / "performance_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = None
        
        return {
            'config': config,
            'summary': summary,
            'episode_results': episode_results,
            'metrics': metrics,
            'experiment_dir': str(exp_path)
        }