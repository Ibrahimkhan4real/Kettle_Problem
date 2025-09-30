# run_experiments.py
"""
Convenient script for running common MCTS experiment scenarios.
Provides pre-configured experiment suites for quick testing and evaluation.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from pipeline.experiment_runner import ExperimentRunner
from utils.config_manager import ConfigManager
from models.mcts_rave import MCTSRAVE
from models.mcts_uct import UCT


def run_quick_test():
    """Run a quick test with minimal episodes to verify the pipeline works"""
    
    print("="*60)
    print("QUICK PIPELINE TEST")
    print("="*60)
    print("Running minimal experiments to verify pipeline functionality...")
    
    runner = ExperimentRunner("test_results")
    runner.register_algorithm('UCT', UCT)
    
    # Test each algorithm briefly
    algorithms = ['MCTS-RAVE', 'UCT']
    results = []
    
    for alg in algorithms:
        print(f"\nTesting {alg}...")
        
        config = runner.config_manager.create_experiment_config(
            experiment_name=f"quick_test_{alg}",
            algorithm_name=alg,
            environment_name='standard',
            num_episodes=3,  # Very few episodes for quick test
            iterations_per_action=100,  # Low iterations for speed
            description=f"Quick test of {alg}"
        )
        
        try:
            result = runner.run_single_experiment(config)
            results.append(result)
            
            # Quick summary
            metrics = result['metrics']
            print(f"  ✓ Success rate: {metrics.get('goal_achievement_rate', 0):.1%}")
            print(f"  ✓ Avg efficiency: {metrics.get('avg_energy_efficiency', 0):.1%}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\nQuick test completed! Results saved to: test_results/")
    print("Pipeline appears to be working correctly.")
    
    return results


def run_comprehensive_comparison():
    """Run comprehensive algorithm comparison study"""
    
    print("="*60)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*60)
    print("This will take 15-30 minutes depending on your hardware...")
    
    runner = ExperimentRunner("comprehensive_results")
    runner.register_algorithm('UCT', UCT)
    
    # Comprehensive comparison settings
    algorithms = ['MCTS-RAVE', 'UCT', 'MCTS-NoRAVE']
    environments = ['standard', 'no_shaping', 'hot_ambient', 'cold_start']
    num_episodes = 25  # Good statistical power
    
    print(f"Algorithms: {algorithms}")
    print(f"Environments: {environments}")
    print(f"Episodes per configuration: {num_episodes}")
    print(f"Total experiments: {len(algorithms) * len(environments)}")
    print(f"Total episodes: {len(algorithms) * len(environments) * num_episodes}")
    
    # Run comparison
    results = runner.run_comparison_study(
        algorithms=algorithms,
        environments=environments,
        num_episodes=num_episodes
    )
    
    # Print detailed summary
    print_detailed_comparison_results(results)
    
    return results


def run_parameter_optimization():
    """Run parameter optimization for MCTS-RAVE"""
    
    print("="*60)
    print("PARAMETER OPTIMIZATION STUDY")
    print("="*60)
    print("Optimizing key parameters for MCTS-RAVE...")
    
    runner = ExperimentRunner("parameter_optimization")
    runner.register_algorithm('UCT', UCT)
    
    # Base configuration
    base_config = runner.config_manager.create_experiment_config(
        experiment_name="param_optimization_base",
        algorithm_name='MCTS-RAVE',
        environment_name='standard',
        num_episodes=15
    )
    
    # Parameter ranges to test
    parameter_studies = {
        'c_param': [0.5, 1.0, 1.414, 2.0, 3.0],
        'rave_constant': [1000, 5000, 8000, 15000, 25000],
        'iterations_per_action': [500, 1000, 2000, 5000],
        'progressive_widening_alpha': [0.3, 0.5, 0.7, 1.0]
    }
    
    optimization_results = {}
    
    for param_name, param_values in parameter_studies.items():
        print(f"\nOptimizing {param_name}...")
        print(f"Testing values: {param_values}")
        
        try:
            results = runner.run_parameter_sweep(
                base_config, {param_name: param_values}
            )
            
            optimization_results[param_name] = results
            
            # Print best value
            optimal_value = results.get('optimal_value')
            print(f"  Optimal {param_name}: {optimal_value}")
            
            # Print sensitivity score
            sensitivity = results.get('sensitivity_score', 0)
            print(f"  Sensitivity score: {sensitivity:.3f}")
            
        except Exception as e:
            print(f"  Failed to optimize {param_name}: {e}")
    
    # Save optimization results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("parameter_optimization") / f"optimization_summary_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print(f"\nParameter optimization completed!")
    print(f"Results saved to: {results_file}")
    
    return optimization_results


def run_scalability_test():
    """Test MCTS performance with different computational budgets"""
    
    print("="*60)
    print("SCALABILITY TEST")
    print("="*60)
    print("Testing performance vs computational budget...")
    
    runner = ExperimentRunner("scalability_results")
    runner.register_algorithm('UCT', UCT)
    
    # Test different iteration counts
    iteration_budgets = [100, 500, 1000, 2000, 5000, 10000]
    algorithms = ['MCTS-RAVE', 'UCT']
    
    scalability_results = []
    
    for alg in algorithms:
        print(f"\nTesting {alg} scalability...")
        
        for iterations in iteration_budgets:
            print(f"  Testing {iterations} iterations...")
            
            config = runner.config_manager.create_experiment_config(
                experiment_name=f"scalability_{alg}_{iterations}",
                algorithm_name=alg,
                environment_name='standard',
                num_episodes=10,
                iterations_per_action=iterations,
                description=f"Scalability test: {alg} with {iterations} iterations"
            )
            
            try:
                result = runner.run_single_experiment(config)
                
                # Extract key metrics
                metrics = result['metrics']
                scalability_results.append({
                    'algorithm': alg,
                    'iterations': iterations,
                    'success_rate': metrics.get('goal_achievement_rate', 0),
                    'avg_efficiency': metrics.get('avg_energy_efficiency', 0),
                    'avg_time_per_step': metrics.get('avg_time_per_step', 0),
                    'avg_total_time': metrics.get('avg_total_time', 0)
                })
                
                print(f"    Success: {metrics.get('goal_achievement_rate', 0):.1%}, "
                      f"Efficiency: {metrics.get('avg_energy_efficiency', 0):.1%}, "
                      f"Time/step: {metrics.get('avg_time_per_step', 0):.3f}s")
                
            except Exception as e:
                print(f"    Failed: {e}")
    
    # Analyze scalability
    print(f"\nScalability Analysis:")
    print("-" * 40)
    
    for alg in algorithms:
        alg_results = [r for r in scalability_results if r['algorithm'] == alg]
        if alg_results:
            print(f"\n{alg}:")
            
            # Find best performance vs iteration trade-off
            best_efficiency = max(alg_results, key=lambda x: x['avg_efficiency'])
            fastest = min(alg_results, key=lambda x: x['avg_time_per_step'])
            
            print(f"  Best efficiency: {best_efficiency['avg_efficiency']:.1%} "
                  f"({best_efficiency['iterations']} iterations)")
            print(f"  Fastest: {fastest['avg_time_per_step']:.3f}s/step "
                  f"({fastest['iterations']} iterations)")
            
            # Diminishing returns analysis
            sorted_results = sorted(alg_results, key=lambda x: x['iterations'])
            if len(sorted_results) > 1:
                efficiency_gains = []
                for i in range(1, len(sorted_results)):
                    prev_eff = sorted_results[i-1]['avg_efficiency']
                    curr_eff = sorted_results[i]['avg_efficiency']
                    gain = curr_eff - prev_eff
                    efficiency_gains.append(gain)
                
                # Find knee point (where gains start diminishing)
                if efficiency_gains:
                    knee_idx = 0
                    for i in range(1, len(efficiency_gains)):
                        if efficiency_gains[i] < efficiency_gains[i-1] * 0.5:
                            knee_idx = i
                            break
                    
                    knee_point = sorted_results[knee_idx + 1]
                    print(f"  Recommended budget: {knee_point['iterations']} iterations "
                          f"(efficiency: {knee_point['avg_efficiency']:.1%}, "
                          f"time: {knee_point['avg_time_per_step']:.3f}s/step)")
    
    return scalability_results


def run_robustness_test():
    """Test algorithm robustness across different environments"""
    
    print("="*60)
    print("ROBUSTNESS TEST")
    print("="*60)
    print("Testing algorithm robustness across challenging environments...")
    
    runner = ExperimentRunner("robustness_results")
    runner.register_algorithm('UCT', UCT)
    
    # Test environments with varying difficulty
    test_environments = ['standard', 'hot_ambient', 'cold_start', 'no_shaping', 'tight_deadline']
    algorithms = ['MCTS-RAVE', 'UCT']
    
    robustness_results = {}
    
    for alg in algorithms:
        print(f"\nTesting {alg} robustness...")
        alg_results = {}
        
        for env in test_environments:
            print(f"  Environment: {env}")
            
            config = runner.config_manager.create_experiment_config(
                experiment_name=f"robustness_{alg}_{env}",
                algorithm_name=alg,
                environment_name=env,
                num_episodes=15,
                description=f"Robustness test: {alg} on {env}"
            )
            
            try:
                result = runner.run_single_experiment(config)
                metrics = result['metrics']
                
                alg_results[env] = {
                    'success_rate': metrics.get('goal_achievement_rate', 0),
                    'avg_efficiency': metrics.get('avg_energy_efficiency', 0),
                    'consistency': metrics.get('success_consistency', 0)
                }
                
                print(f"    Success: {metrics.get('goal_achievement_rate', 0):.1%}, "
                      f"Efficiency: {metrics.get('avg_energy_efficiency', 0):.1%}")
                
            except Exception as e:
                print(f"    Failed: {e}")
                alg_results[env] = {'success_rate': 0, 'avg_efficiency': 0, 'consistency': 0}
        
        robustness_results[alg] = alg_results
    
    # Calculate robustness metrics
    print(f"\nRobustness Analysis:")
    print("-" * 40)
    
    for alg, env_results in robustness_results.items():
        success_rates = [r['success_rate'] for r in env_results.values()]
        efficiencies = [r['avg_efficiency'] for r in env_results.values()]
        
        # Calculate variance (lower = more robust)
        success_variance = sum((x - sum(success_rates)/len(success_rates))**2 for x in success_rates) / len(success_rates)
        efficiency_variance = sum((x - sum(efficiencies)/len(efficiencies))**2 for x in efficiencies) / len(efficiencies)
        
        # Robustness score (higher = more robust)
        robustness_score = (sum(success_rates)/len(success_rates)) * 100 - success_variance * 1000
        
        print(f"\n{alg}:")
        print(f"  Average success rate: {sum(success_rates)/len(success_rates):.1%}")
        print(f"  Success variance: {success_variance:.4f}")
        print(f"  Average efficiency: {sum(efficiencies)/len(efficiencies):.1%}")
        print(f"  Efficiency variance: {efficiency_variance:.4f}")
        print(f"  Robustness score: {robustness_score:.1f}")
        
        # Find most challenging environment
        worst_env = min(env_results.items(), key=lambda x: x[1]['success_rate'])
        print(f"  Most challenging env: {worst_env[0]} "
              f"(success: {worst_env[1]['success_rate']:.1%})")
    
    return robustness_results


def run_energy_efficiency_study():
    """Focus specifically on energy efficiency performance"""
    
    print("="*60)
    print("ENERGY EFFICIENCY STUDY")
    print("="*60)
    print("Detailed analysis of energy efficiency performance...")
    
    runner = ExperimentRunner("energy_efficiency_results")
    runner.register_algorithm('UCT', UCT)
    
    # Test with high iteration counts for best possible performance
    algorithms = ['MCTS-RAVE', 'UCT']
    environments = ['standard', 'no_shaping']  # Focus on core environments
    
    efficiency_results = []
    
    for alg in algorithms:
        for env in environments:
            print(f"\nTesting {alg} on {env} environment...")
            
            config = runner.config_manager.create_experiment_config(
                experiment_name=f"efficiency_{alg}_{env}",
                algorithm_name=alg,
                environment_name=env,
                num_episodes=30,  # More episodes for better statistics
                iterations_per_action=5000,  # High iterations for best performance
                description=f"Energy efficiency study: {alg} on {env}"
            )
            
            try:
                result = runner.run_single_experiment(config)
                metrics = result['metrics']
                
                # Detailed efficiency statistics
                episode_results = result['results']
                efficiencies = [r.energy_efficiency for r in episode_results]
                
                efficiency_data = {
                    'algorithm': alg,
                    'environment': env,
                    'mean_efficiency': sum(efficiencies) / len(efficiencies),
                    'max_efficiency': max(efficiencies),
                    'min_efficiency': min(efficiencies),
                    'efficiency_std': (sum((x - sum(efficiencies)/len(efficiencies))**2 for x in efficiencies) / len(efficiencies))**0.5,
                    'excellent_rate': sum(1 for e in efficiencies if e >= 0.9) / len(efficiencies),
                    'good_rate': sum(1 for e in efficiencies if e >= 0.8) / len(efficiencies),
                    'success_rate': metrics.get('goal_achievement_rate', 0)
                }
                
                efficiency_results.append(efficiency_data)
                
                print(f"  Mean efficiency: {efficiency_data['mean_efficiency']:.1%}")
                print(f"  Max efficiency: {efficiency_data['max_efficiency']:.1%}")
                print(f"  Excellent rate (≥90%): {efficiency_data['excellent_rate']:.1%}")
                print(f"  Good rate (≥80%): {efficiency_data['good_rate']:.1%}")
                
            except Exception as e:
                print(f"  Failed: {e}")
    
    # Find the most energy-efficient configurations
    print(f"\nEfficiency Champions:")
    print("-" * 40)
    
    if efficiency_results:
        # Best mean efficiency
        best_mean = max(efficiency_results, key=lambda x: x['mean_efficiency'])
        print(f"Best mean efficiency: {best_mean['algorithm']} on {best_mean['environment']} "
              f"({best_mean['mean_efficiency']:.1%})")
        
        # Best maximum efficiency
        best_max = max(efficiency_results, key=lambda x: x['max_efficiency'])
        print(f"Best peak efficiency: {best_max['algorithm']} on {best_max['environment']} "
              f"({best_max['max_efficiency']:.1%})")
        
        # Most consistent
        most_consistent = min(efficiency_results, key=lambda x: x['efficiency_std'])
        print(f"Most consistent: {most_consistent['algorithm']} on {most_consistent['environment']} "
              f"(std: {most_consistent['efficiency_std']:.3f})")
        
        # Best excellent rate
        best_excellent = max(efficiency_results, key=lambda x: x['excellent_rate'])
        print(f"Best excellent rate: {best_excellent['algorithm']} on {best_excellent['environment']} "
              f"({best_excellent['excellent_rate']:.1%} episodes ≥90% efficient)")
    
    return efficiency_results


def print_detailed_comparison_results(results: Dict[str, Any]):
    """Print detailed comparison results"""
    
    print(f"\n{'='*60}")
    print(f"DETAILED COMPARISON RESULTS")
    print(f"{'='*60}")
    
    detailed_comparison = results.get('detailed_comparison', {})
    
    # Create performance table
    print(f"\nPerformance Summary:")
    print(f"{'Configuration':<25} {'Success Rate':<12} {'Efficiency':<12} {'Reward':<10}")
    print(f"{'-'*60}")
    
    for config_name, config_data in detailed_comparison.items():
        metrics = config_data['metrics']
        success_rate = metrics.get('goal_achievement_rate', 0)
        efficiency = metrics.get('avg_energy_efficiency', 0)
        reward = metrics.get('avg_total_reward', 0)
        
        print(f"{config_name:<25} {success_rate:<12.1%} {efficiency:<12.1%} {reward:<10.1f}")
    
    # Show rankings
    rankings = results.get('rankings', {})
    
    if rankings:
        print(f"\nRankings:")
        
        for metric_name, ranking_list in rankings.items():
            metric_display = metric_name.replace('_', ' ').title()
            print(f"\n{metric_display}:")
            
            for i, entry in enumerate(ranking_list[:5]):  # Top 5
                print(f"  {i+1}. {entry['algorithm_env']}: {entry['value']:.3f}")


def main():
    """Main function with experiment menu"""
    
    print("MCTS Kettle Environment - Experiment Suite")
    print("="*50)
    print("Choose an experiment to run:")
    print()
    print("1. Quick Test (3 min) - Verify pipeline works")
    print("2. Comprehensive Comparison (20-30 min) - Full algorithm comparison") 
    print("3. Parameter Optimization (30-45 min) - Optimize MCTS-RAVE parameters")
    print("4. Scalability Test (15-20 min) - Test computational budget vs performance")
    print("5. Robustness Test (20-30 min) - Test across challenging environments")
    print("6. Energy Efficiency Study (25-35 min) - Focus on energy performance")
    print("7. All Studies (2-3 hours) - Run everything")
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-7): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                run_quick_test()
                break
            elif choice == '2':
                run_comprehensive_comparison()
                break
            elif choice == '3':
                run_parameter_optimization()
                break
            elif choice == '4':
                run_scalability_test()
                break
            elif choice == '5':
                run_robustness_test()
                break
            elif choice == '6':
                run_energy_efficiency_study()
                break
            elif choice == '7':
                print("Running all studies... This will take 2-3 hours!")
                confirm = input("Are you sure? (y/N): ").strip().lower()
                if confirm == 'y':
                    run_quick_test()
                    run_comprehensive_comparison()
                    run_parameter_optimization()
                    run_scalability_test()
                    run_robustness_test()
                    run_energy_efficiency_study()
                    print("\nAll studies completed!")
                break
            else:
                print("Invalid choice. Please enter 0-7.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()