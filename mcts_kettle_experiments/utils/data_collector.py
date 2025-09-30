# utils/data_collector.py
"""
Data collection and management utilities for MCTS experiments.
Handles saving, loading, and organizing experimental data.
"""

import json
import csv
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import datetime
import h5py

from models.base_mcts import EpisodeResult, MCTSStats


class DataCollector:
    """Handles data collection and storage for experiments"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "csv_exports").mkdir(exist_ok=True)
        (self.data_dir / "json_exports").mkdir(exist_ok=True)
        (self.data_dir / "processed_data").mkdir(exist_ok=True)
    
    def save_episode_data(self, episode_result: EpisodeResult, filepath: Union[str, Path]):
        """Save episode result to JSON file"""
        filepath = Path(filepath)
        
        # Convert to dictionary
        data = episode_result.to_dict()
        
        # Handle numpy arrays and special types
        data = self._serialize_data(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_episode_data(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load episode result from JSON file"""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert back to proper types
        data = self._deserialize_data(data)
        
        return data
    
    def save_metrics(self, metrics: Dict[str, Any], filepath: Union[str, Path]):
        """Save metrics to JSON file"""
        filepath = Path(filepath)
        
        # Serialize data
        serialized_metrics = self._serialize_data(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(serialized_metrics, f, indent=2)
    
    def load_metrics(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load metrics from JSON file"""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        return self._deserialize_data(metrics)
    
    def export_episodes_to_csv(self, 
                              episode_results: List[EpisodeResult], 
                              filepath: Union[str, Path]):
        """Export episode results to CSV format"""
        filepath = Path(filepath)
        
        # Create summary data
        data = []
        for result in episode_results:
            row = {
                'episode_id': result.episode_id,
                'algorithm_name': result.algorithm_name,
                'goal_achieved': result.goal_achieved,
                'final_temperature': result.final_temperature,
                'final_step': result.final_step,
                'total_reward': result.total_reward,
                'energy_consumed_wh': result.energy_consumed_wh,
                'energy_efficiency': result.energy_efficiency,
                'total_time': result.total_time,
                'avg_time_per_step': result.avg_time_per_step,
                
                # MCTS stats
                'mcts_total_iterations': result.mcts_stats.total_iterations,
                'mcts_total_time': result.mcts_stats.total_time,
                'mcts_tree_size': result.mcts_stats.tree_size,
                'mcts_max_depth': result.mcts_stats.max_depth,
                'mcts_avg_rollout_length': result.mcts_stats.avg_rollout_length,
                'mcts_goal_found_iterations': result.mcts_stats.goal_found_iterations,
                'mcts_best_value_found': result.mcts_stats.best_value_found,
                'mcts_nodes_expanded': result.mcts_stats.nodes_expanded,
                'mcts_rollouts_performed': result.mcts_stats.rollouts_performed
            }
            
            # Add configuration parameters
            for key, value in result.config.items():
                if isinstance(value, (int, float, str, bool)):
                    row[f'config_{key}'] = value
            
            data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def export_trajectories_to_csv(self, 
                                  episode_results: List[EpisodeResult], 
                                  filepath: Union[str, Path]):
        """Export detailed trajectory data to CSV"""
        filepath = Path(filepath)
        
        data = []
        for result in episode_results:
            episode_id = result.episode_id
            algorithm = result.algorithm_name
            
            # Add trajectory data
            max_len = max(len(result.temperatures), len(result.actions) + 1, len(result.rewards) + 1)
            
            for step in range(max_len):
                row = {
                    'episode_id': episode_id,
                    'algorithm': algorithm,
                    'step': step
                }
                
                # Temperature (includes initial state)
                if step < len(result.temperatures):
                    row['temperature'] = result.temperatures[step]
                else:
                    row['temperature'] = None
                
                # Actions and rewards (start from step 1)
                if step > 0 and (step - 1) < len(result.actions):
                    row['action'] = result.actions[step - 1]
                    row['reward'] = result.rewards[step - 1] if (step - 1) < len(result.rewards) else None
                else:
                    row['action'] = None
                    row['reward'] = None
                
                # MCTS iterations for this step
                if step > 0 and (step - 1) < len(result.mcts_iterations_per_step):
                    row['mcts_iterations'] = result.mcts_iterations_per_step[step - 1]
                else:
                    row['mcts_iterations'] = None
                
                data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def save_batch_results(self, 
                          results: List[Dict[str, Any]], 
                          batch_name: str,
                          formats: List[str] = ['json', 'csv']):
        """Save results from a batch of experiments"""
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = self.data_dir / "processed_data" / f"{batch_name}_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract all episode results
        all_episodes = []
        experiment_summaries = []
        
        for result in results:
            config = result['config']
            episode_results = result['results']
            metrics = result['metrics']
            
            all_episodes.extend(episode_results)
            
            # Create experiment summary
            experiment_summaries.append({
                'experiment_name': config.experiment_name,
                'algorithm': config.algorithm.algorithm_name,
                'num_episodes': len(episode_results),
                'goal_achievement_rate': metrics.get('goal_achievement_rate', 0),
                'avg_energy_efficiency': metrics.get('avg_energy_efficiency', 0),
                'avg_total_reward': metrics.get('avg_total_reward', 0),
                'avg_energy_consumed': metrics.get('avg_energy_consumed', 0),
                'configuration': config.to_dict()
            })
        
        # Save in requested formats
        if 'json' in formats:
            # Save detailed data
            batch_data = {
                'batch_info': {
                    'name': batch_name,
                    'timestamp': timestamp,
                    'num_experiments': len(results),
                    'total_episodes': len(all_episodes)
                },
                'experiment_summaries': experiment_summaries,
                'all_results': [r for r in results]
            }
            
            # Serialize and save
            serialized_data = self._serialize_data(batch_data)
            with open(batch_dir / "batch_results.json", 'w') as f:
                json.dump(serialized_data, f, indent=2)
        
        if 'csv' in formats:
            # Export episode summaries
            self.export_episodes_to_csv(all_episodes, batch_dir / "all_episodes.csv")
            
            # Export experiment summaries
            exp_df = pd.DataFrame(experiment_summaries)
            exp_df.to_csv(batch_dir / "experiment_summaries.csv", index=False)
            
            # Export trajectories
            self.export_trajectories_to_csv(all_episodes, batch_dir / "all_trajectories.csv")
        
        if 'hdf5' in formats:
            self._save_to_hdf5(all_episodes, batch_dir / "batch_data.h5")
        
        return str(batch_dir)
    
    def load_batch_results(self, batch_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load batch results from directory"""
        batch_path = Path(batch_dir)
        
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
        
        # Load JSON data if available
        json_file = batch_path / "batch_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            return self._deserialize_data(data)
        
        # Otherwise load from CSV files
        csv_data = {}
        
        episodes_file = batch_path / "all_episodes.csv"
        if episodes_file.exists():
            csv_data['episodes'] = pd.read_csv(episodes_file)
        
        summaries_file = batch_path / "experiment_summaries.csv"
        if summaries_file.exists():
            csv_data['summaries'] = pd.read_csv(summaries_file)
        
        trajectories_file = batch_path / "all_trajectories.csv"
        if trajectories_file.exists():
            csv_data['trajectories'] = pd.read_csv(trajectories_file)
        
        return csv_data
    
    def create_aggregated_dataset(self, 
                                experiment_dirs: List[Union[str, Path]],
                                output_name: str = "aggregated_dataset") -> str:
        """Create aggregated dataset from multiple experiments"""
        
        all_episodes = []
        all_summaries = []
        
        for exp_dir in experiment_dirs:
            exp_path = Path(exp_dir)
            
            # Load episode data
            episode_data_dir = exp_path / "episode_data"
            if episode_data_dir.exists():
                for episode_file in sorted(episode_data_dir.glob("episode_*.json")):
                    episode_data = self.load_episode_data(episode_file)
                    all_episodes.append(episode_data)
            
            # Load summary
            summary_file = exp_path / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                all_summaries.append(summary)
        
        # Save aggregated data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.data_dir / "processed_data" / f"{output_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as different formats
        aggregated_data = {
            'metadata': {
                'created': timestamp,
                'num_experiments': len(all_summaries),
                'total_episodes': len(all_episodes),
                'source_directories': [str(d) for d in experiment_dirs]
            },
            'episodes': all_episodes,
            'experiment_summaries': all_summaries
        }
        
        # JSON format
        serialized_data = self._serialize_data(aggregated_data)
        with open(output_dir / "aggregated_data.json", 'w') as f:
            json.dump(serialized_data, f, indent=2)
        
        # CSV formats
        if all_episodes:
            # Convert episode dictionaries to EpisodeResult objects for CSV export
            episode_objects = []
            for ep_data in all_episodes:
                # Reconstruct EpisodeResult object (simplified)
                episode_objects.append(self._dict_to_episode_result(ep_data))
            
            self.export_episodes_to_csv(episode_objects, output_dir / "episodes.csv")
            self.export_trajectories_to_csv(episode_objects, output_dir / "trajectories.csv")
        
        # Experiment summaries CSV
        if all_summaries:
            summary_rows = []
            for summary in all_summaries:
                row = {
                    'experiment_name': summary.get('experiment_info', {}).get('name', ''),
                    'algorithm': summary.get('experiment_info', {}).get('algorithm', ''),
                    'num_episodes': summary.get('experiment_info', {}).get('num_episodes', 0),
                    'total_time': summary.get('experiment_info', {}).get('total_time', 0),
                    'success_rate': summary.get('performance_summary', {}).get('success_rate', 0),
                    'avg_energy_efficiency': summary.get('performance_summary', {}).get('avg_energy_efficiency', 0),
                    'avg_total_reward': summary.get('performance_summary', {}).get('avg_total_reward', 0),
                    'avg_energy_consumed': summary.get('performance_summary', {}).get('avg_energy_consumed', 0)
                }
                summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(output_dir / "experiment_summaries.csv", index=False)
        
        return str(output_dir)
    
    def _save_to_hdf5(self, episode_results: List[EpisodeResult], filepath: Union[str, Path]):
        """Save episode results to HDF5 format for efficient storage"""
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            episodes_group = f.create_group('episodes')
            trajectories_group = f.create_group('trajectories')
            metadata_group = f.create_group('metadata')
            
            # Save metadata
            metadata_group.attrs['num_episodes'] = len(episode_results)
            metadata_group.attrs['created'] = datetime.datetime.now().isoformat()
            
            for i, result in enumerate(episode_results):
                ep_group = episodes_group.create_group(f'episode_{i:03d}')
                
                # Scalar values
                ep_group.attrs['episode_id'] = result.episode_id
                ep_group.attrs['algorithm_name'] = result.algorithm_name
                ep_group.attrs['goal_achieved'] = result.goal_achieved
                ep_group.attrs['final_temperature'] = result.final_temperature
                ep_group.attrs['final_step'] = result.final_step
                ep_group.attrs['total_reward'] = result.total_reward
                ep_group.attrs['energy_consumed_wh'] = result.energy_consumed_wh
                ep_group.attrs['energy_efficiency'] = result.energy_efficiency
                ep_group.attrs['total_time'] = result.total_time
                ep_group.attrs['avg_time_per_step'] = result.avg_time_per_step
                
                # Arrays
                ep_group.create_dataset('temperatures', data=result.temperatures)
                ep_group.create_dataset('actions', data=result.actions)
                ep_group.create_dataset('rewards', data=result.rewards)
                ep_group.create_dataset('mcts_iterations_per_step', data=result.mcts_iterations_per_step)
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage"""
        if isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
    
    def _deserialize_data(self, data: Any) -> Any:
        """Deserialize data from JSON storage"""
        if isinstance(data, dict):
            return {key: self._deserialize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._deserialize_data(item) for item in data]
        else:
            return data
    
    def _dict_to_episode_result(self, data: Dict[str, Any]) -> EpisodeResult:
        """Convert dictionary back to EpisodeResult object"""
        # Create MCTSStats object
        mcts_stats_data = data.get('mcts_stats', {})
        mcts_stats = MCTSStats(
            total_iterations=mcts_stats_data.get('total_iterations', 0),
            total_time=mcts_stats_data.get('total_time', 0.0),
            tree_size=mcts_stats_data.get('tree_size', 0),
            max_depth=mcts_stats_data.get('max_depth', 0),
            avg_rollout_length=mcts_stats_data.get('avg_rollout_length', 0.0),
            goal_found_iterations=mcts_stats_data.get('goal_found_iterations', 0),
            best_value_found=mcts_stats_data.get('best_value_found', -float('inf')),
            nodes_expanded=mcts_stats_data.get('nodes_expanded', 0),
            rollouts_performed=mcts_stats_data.get('rollouts_performed', 0)
        )
        
        # Create EpisodeResult object
        return EpisodeResult(
            episode_id=data.get('episode_id', 0),
            algorithm_name=data.get('algorithm_name', ''),
            config=data.get('config', {}),
            final_temperature=data.get('final_temperature', 0.0),
            final_step=data.get('final_step', 0),
            goal_achieved=data.get('goal_achieved', False),
            total_reward=data.get('total_reward', 0.0),
            energy_consumed_wh=data.get('energy_consumed_wh', 0.0),
            energy_efficiency=data.get('energy_efficiency', 0.0),
            temperatures=data.get('temperatures', []),
            actions=data.get('actions', []),
            rewards=data.get('rewards', []),
            mcts_iterations_per_step=data.get('mcts_iterations_per_step', []),
            mcts_stats=mcts_stats,
            total_time=data.get('total_time', 0.0),
            avg_time_per_step=data.get('avg_time_per_step', 0.0)
        )
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored data"""
        info = {
            'base_directory': str(self.base_dir),
            'data_directory': str(self.data_dir),
            'subdirectories': {},
            'total_size_mb': 0
        }
        
        # Check each subdirectory
        for subdir_name in ['csv_exports', 'json_exports', 'processed_data']:
            subdir = self.data_dir / subdir_name
            if subdir.exists():
                files = list(subdir.rglob('*'))
                file_count = sum(1 for f in files if f.is_file())
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                info['subdirectories'][subdir_name] = {
                    'file_count': file_count,
                    'size_mb': total_size / (1024 * 1024),
                    'latest_file': max(files, key=lambda x: x.stat().st_mtime).name if files else None
                }
                
                info['total_size_mb'] += total_size / (1024 * 1024)
        
        return info