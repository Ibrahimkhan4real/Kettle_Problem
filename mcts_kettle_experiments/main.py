# main.py
"""
Main entry point for the MCTS Kettle Environment Testing Pipeline.
Provides command-line interface for running experiments and analysis.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from pipeline.experiment_runner import ExperimentRunner
from utils.config_manager import ConfigManager
from models.mcts_rave import MCTSRAVE  
from models.mcts_uct import UCT


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MCTS Kettle Environment Testing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single experiment
  python main.py single --algorithm MCTS-RAVE --environment standard --episodes 10
  
  # Compare algorithms
  python main.py compare --algorithms MCTS-RAVE UCT --environments standard no_shaping
  
  # Parameter sweep
  python main.py sweep --base-config comparison_MCTS-RAVE_standard --parameter c_param --values 0.5 1.0 1.5 2.0
  
  # Run predefined experiments
  python main.py batch --config-file experiments.json
  
  # Load and analyze existing results
  python main.py analyze --experiment-dir experiments/20240101_120000_test_experiment
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single experiment command
    single_parser = subparsers.add_parser('single', help='Run a single experiment')
    single_parser.add_argument('--algorithm', '-a', choices=['MCTS-RAVE', 'UCT', 'MCTS-NoRAVE'], 
                              default='MCTS-RAVE', help='Algorithm to use')
    single_parser.add_argument('--environment', '-e', choices=['standard', 'hot_ambient', 'cold_start', 'no_shaping', 'tight_deadline'],
                              default='standard', help='Environment configuration')
    single_parser.add_argument('--episodes', '-n', type=int, default=10, help='Number of episodes to run')
    single_parser.add_argument('--iterations', '-i', type=int, default=1000, help='MCTS iterations per action')
    single_parser.add_argument('--name', help='Experiment name (auto-generated if not provided)')
    single_parser.add_argument('--seed', type=int, help='Random seed')
    single_parser.add_argument('--output-dir', help='Output directory (default: mcts_kettle_experiments)')
    
    # Comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple algorithms')
    compare_parser.add_argument('--algorithms', '-a', nargs='+', 
                               choices=['MCTS-RAVE', 'UCT', 'MCTS-NoRAVE'],
                               default=['MCTS-RAVE', 'UCT'], help='Algorithms to compare')
    compare_parser.add_argument('--environments', '-e', nargs='+',
                               choices=['standard', 'hot_ambient', 'cold_start', 'no_shaping', 'tight_deadline'],
                               default=['standard', 'no_shaping'], help='Environments to test')
    compare_parser.add_argument('--episodes', '-n', type=int, default=20, help='Episodes per configuration')
    compare_parser.add_argument('--output-dir', help='Output directory')
    
    # Parameter sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sensitivity analysis')
    sweep_parser.add_argument('--base-config', required=True, help='Base configuration name')
    sweep_parser.add_argument('--parameter', '-p', required=True, help='Parameter to sweep')
    sweep_parser.add_argument('--values', '-v', nargs='+', required=True, help='Parameter values to test')
    sweep_parser.add_argument('--episodes', '-n', type=int, default=10, help='Episodes per configuration')
    sweep_parser.add_argument('--output-dir', help='Output directory')
    
    # Batch experiment command
    batch_parser = subparsers.add_parser('batch', help='Run batch of predefined experiments')
    batch_parser.add_argument('--config-file', help='JSON file with experiment configurations')
    batch_parser.add_argument('--predefined', choices=['comparison', 'all_algorithms', 'all_environments'],
                             help='Run predefined experiment suite')
    batch_parser.add_argument('--output-dir', help='Output directory')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing experiment results')
    analyze_parser.add_argument('--experiment-dir', required=True, help='Experiment directory to analyze')
    analyze_parser.add_argument('--generate-plots', action='store_true', help='Regenerate plots')
    analyze_parser.add_argument('--export-csv', action='store_true', help='Export data to CSV')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available configurations and experiments')
    list_parser.add_argument('--type', choices=['algorithms', 'environments', 'experiments'], 
                            default='algorithms', help='What to list')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize experiment runner
    base_dir = args.output_dir or "mcts_kettle_experiments"
    runner = ExperimentRunner(base_dir)
    
    # Register additional algorithms
    runner.register_algorithm('UCT', UCT)
    
    try:
        if args.command == 'single':
            run_single_experiment(runner, args)
        elif args.command == 'compare':
            run_comparison(runner, args)
        elif args.command == 'sweep':
            run_parameter_sweep(runner, args)
        elif args.command == 'batch':
            run_batch_experiments(runner, args)
        elif args.command == 'analyze':
            analyze_experiment(runner, args)
        elif args.command == 'list':
            list_configurations(runner, args)
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_single_experiment(runner: ExperimentRunner, args):
    """Run a single experiment"""
    
    # Generate experiment name if not provided
    if args.name:
        experiment_name = args.name
    else:
        experiment_name = f"{args.algorithm}_{args.environment}_{args.episodes}ep"
    
    print(f"Running single experiment: {experiment_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Environment: {args.environment}")
    print(f"Episodes: {args.episodes}")
    
    # Create configuration
    config = runner.config_manager.create_experiment_config(
        experiment_name=experiment_name,
        algorithm_name=args.algorithm,
        environment_name=args.environment,
        num_episodes=args.episodes,
        iterations_per_action=args.iterations,
        random_seed=args.seed,
        description=f"Single experiment: {args.algorithm} on {args.environment}"
    )
    
    # Run experiment
    result = runner.run_single_experiment(config)
    
    # Print summary
    print_experiment_summary(result)


def run_comparison(runner: ExperimentRunner, args):
    """Run algorithm comparison"""
    
    print(f"Running comparison study:")
    print(f"Algorithms: {args.algorithms}")
    print(f"Environments: {args.environments}")
    print(f"Episodes per config: {args.episodes}")
    
    # Run comparison
    results = runner.run_comparison_study(
        algorithms=args.algorithms,
        environments=args.environments,
        num_episodes=args.episodes
    )
    
    # Print comparison summary
    print_comparison_summary(results)


def run_parameter_sweep(runner: ExperimentRunner, args):
    """Run parameter sensitivity analysis"""
    
    print(f"Running parameter sweep:")
    print(f"Base config: {args.base_config}")
    print(f"Parameter: {args.parameter}")
    print(f"Values: {args.values}")
    
    # Create base configuration
    try:
        # Try to parse as algorithm and environment
        parts = args.base_config.split('_')
        if len(parts) >= 2:
            algorithm = parts[0]
            environment = parts[1]
        else:
            algorithm = 'MCTS-RAVE'
            environment = 'standard'
        
        base_config = runner.config_manager.create_experiment_config(
            experiment_name=f"sweep_{args.parameter}",
            algorithm_name=algorithm,
            environment_name=environment,
            num_episodes=args.episodes
        )
    except Exception as e:
        print(f"Error creating base configuration: {e}")
        print("Using default MCTS-RAVE on standard environment")
        base_config = runner.config_manager.create_experiment_config(
            experiment_name=f"sweep_{args.parameter}",
            algorithm_name='MCTS-RAVE',
            environment_name='standard',
            num_episodes=args.episodes
        )
    
    # Convert parameter values to appropriate types
    param_values = []
    for value in args.values:
        try:
            # Try int first
            param_values.append(int(value))
        except ValueError:
            try:
                # Try float
                param_values.append(float(value))
            except ValueError:
                # Keep as string
                param_values.append(value)
    
    parameter_ranges = {args.parameter: param_values}
    
    # Run parameter sweep
    results = runner.run_parameter_sweep(base_config, parameter_ranges)
    
    print(f"Parameter sweep completed.")
    print(f"Optimal {args.parameter}: {results.get('optimal_value', 'N/A')}")


def run_batch_experiments(runner: ExperimentRunner, args):
    """Run batch of experiments"""
    
    if args.config_file:
        # Load from file
        print(f"Loading experiments from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            experiment_configs = json.load(f)
        
        # Convert to experiment config objects
        configs = []
        for exp_config in experiment_configs:
            config = runner.config_manager.create_experiment_config(**exp_config)
            configs.append(config)
    
    elif args.predefined:
        print(f"Running predefined experiment suite: {args.predefined}")
        
        if args.predefined == 'comparison':
            configs = runner.config_manager.create_comparison_suite()
        elif args.predefined == 'all_algorithms':
            algorithms = ['MCTS-RAVE', 'UCT', 'MCTS-NoRAVE']
            configs = []
            for alg in algorithms:
                config = runner.config_manager.create_experiment_config(
                    experiment_name=f"test_{alg}",
                    algorithm_name=alg,
                    environment_name='standard',
                    num_episodes=10
                )
                configs.append(config)
        elif args.predefined == 'all_environments':
            environments = ['standard', 'hot_ambient', 'cold_start', 'no_shaping', 'tight_deadline']
            configs = []
            for env in environments:
                config = runner.config_manager.create_experiment_config(
                    experiment_name=f"test_{env}",
                    algorithm_name='MCTS-RAVE',
                    environment_name=env,
                    num_episodes=10
                )
                configs.append(config)
    else:
        print("Error: Must provide either --config-file or --predefined")
        return
    
    print(f"Running {len(configs)} experiments...")
    
    # Run batch
    results = runner.run_experiment_batch(configs)
    
    print(f"Batch completed: {len(results)} successful experiments")


def analyze_experiment(runner: ExperimentRunner, args):
    """Analyze existing experiment results"""
    
    print(f"Analyzing experiment: {args.experiment_dir}")
    
    try:
        # Load experiment results
        result = runner.load_experiment_results(args.experiment_dir)
        
        print(f"Loaded experiment with {len(result.get('episode_results', []))} episodes")
        
        if result.get('summary'):
            print_experiment_summary({'summary': result['summary']})
        
        if args.generate_plots and result.get('episode_results'):
            print("Regenerating plots...")
            plots_dir = Path(args.experiment_dir) / "plots_regenerated"
            plots_dir.mkdir(exist_ok=True)
            
            runner.visualization_utils.generate_experiment_plots(
                result['episode_results'], plots_dir, "Regenerated Analysis"
            )
            print(f"Plots saved to: {plots_dir}")
        
        if args.export_csv and result.get('episode_results'):
            print("Exporting to CSV...")
            csv_path = Path(args.experiment_dir) / "exported_data.csv"
            runner.data_collector.export_episodes_to_csv(
                result['episode_results'], csv_path
            )
            print(f"Data exported to: {csv_path}")
    
    except Exception as e:
        print(f"Error analyzing experiment: {e}")


def list_configurations(runner: ExperimentRunner, args):
    """List available configurations"""
    
    if args.type == 'algorithms':
        print("Available algorithms:")
        for alg_name in runner.algorithm_registry.keys():
            print(f"  - {alg_name}")
    
    elif args.type == 'environments':
        env_configs = runner.config_manager.load_environment_configs()
        print("Available environments:")
        for env_name, env_config in env_configs.items():
            print(f"  - {env_name}: {env_config.get('description', 'No description')}")
    
    elif args.type == 'experiments':
        predefined = runner.config_manager.get_predefined_experiments()
        print("Available predefined experiments:")
        for exp in predefined[:10]:  # Show first 10
            print(f"  - {exp['name']}: {exp['description']}")
        
        if len(predefined) > 10:
            print(f"  ... and {len(predefined) - 10} more")


def print_experiment_summary(result: Dict[str, Any]):
    """Print experiment summary"""
    
    summary = result.get('summary', {})
    exp_info = summary.get('experiment_info', {})
    perf_summary = summary.get('performance_summary', {})
    mcts_summary = summary.get('mcts_summary', {})
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY: {exp_info.get('name', 'Unknown')}")
    print(f"{'='*60}")
    print(f"Algorithm: {exp_info.get('algorithm', 'Unknown')}")
    print(f"Episodes: {exp_info.get('num_episodes', 0)}")
    print(f"Total time: {exp_info.get('total_time', 0):.1f}s")
    
    print(f"\nPERFORMANCE:")
    print(f"  Success rate: {perf_summary.get('success_rate', 0):.1%}")
    print(f"  Avg energy efficiency: {perf_summary.get('avg_energy_efficiency', 0):.1%}")
    print(f"  Avg final temperature: {perf_summary.get('avg_final_temperature', 0):.1f}°C")
    print(f"  Avg total reward: {perf_summary.get('avg_total_reward', 0):.1f}")
    print(f"  Avg energy consumed: {perf_summary.get('avg_energy_consumed', 0):.1f} Wh")
    
    print(f"\nMCTS STATISTICS:")
    print(f"  Avg iterations per episode: {mcts_summary.get('avg_iterations_per_episode', 0):.0f}")
    print(f"  Avg tree size: {mcts_summary.get('avg_tree_size', 0):.0f}")
    print(f"  Avg max depth: {mcts_summary.get('avg_max_depth', 0):.1f}")
    print(f"  Avg time per step: {mcts_summary.get('avg_time_per_step', 0):.3f}s")
    
    print(f"\nResults saved to: {result.get('experiment_dir', 'Unknown')}")


def print_comparison_summary(results: Dict[str, Any]):
    """Print comparison summary"""
    
    print(f"\n{'='*60}")
    print(f"COMPARISON STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"Algorithms: {', '.join(results.get('algorithms', []))}")
    print(f"Environments: {', '.join(results.get('environments', []))}")
    
    # Show top performers
    rankings = results.get('rankings', {})
    
    if 'goal_achievement_rate' in rankings:
        print(f"\nTOP PERFORMERS (Goal Achievement):")
        for i, entry in enumerate(rankings['goal_achievement_rate'][:3]):
            print(f"  {i+1}. {entry['algorithm_env']}: {entry['value']:.1%}")
    
    if 'avg_energy_efficiency' in rankings:
        print(f"\nTOP PERFORMERS (Energy Efficiency):")
        for i, entry in enumerate(rankings['avg_energy_efficiency'][:3]):
            print(f"  {i+1}. {entry['algorithm_env']}: {entry['value']:.1%}")
    
    if 'avg_total_reward' in rankings:
        print(f"\nTOP PERFORMERS (Total Reward):")
        for i, entry in enumerate(rankings['avg_total_reward'][:3]):
            print(f"  {i+1}. {entry['algorithm_env']}: {entry['value']:.1f}")


if __name__ == "__main__":
    main()