# utils/metrics_calculator.py
"""
Performance metrics calculation for MCTS experiments.
Computes various performance indicators and statistical measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings
from models.base_mcts import EpisodeResult


class MetricsCalculator:
    """Calculates performance metrics for MCTS experiments"""
    
    def __init__(self):
        self.target_temp = 100.0
        self.temp_tolerance = 2.5
        self.time_tolerance = 15
        self.default_deadline = 200
    
    def calculate_episode_metrics(self, result: EpisodeResult) -> Dict[str, Any]:
        """Calculate metrics for a single episode"""
        
        metrics = {
            # Goal achievement
            'goal_achieved': result.goal_achieved,
            'temperature_error': abs(result.final_temperature - self.target_temp),
            'time_error': abs(result.final_step - self.default_deadline),
            'within_temp_tolerance': abs(result.final_temperature - self.target_temp) <= self.temp_tolerance,
            'within_time_tolerance': abs(result.final_step - self.default_deadline) <= self.time_tolerance,
            
            # Energy efficiency
            'energy_consumed_wh': result.energy_consumed_wh,
            'energy_efficiency': result.energy_efficiency,
            'energy_waste_wh': max(0, result.energy_consumed_wh - self._theoretical_min_energy()),
            'energy_efficiency_category': self._categorize_efficiency(result.energy_efficiency),
            
            # Performance
            'total_reward': result.total_reward,
            'final_temperature': result.final_temperature,
            'final_step': result.final_step,
            
            # Control strategy analysis
            'heating_actions': self._count_heating_actions(result.actions),
            'heating_percentage': self._calculate_heating_percentage(result.actions),
            'longest_heating_streak': self._longest_heating_streak(result.actions),
            'longest_coasting_streak': self._longest_coasting_streak(result.actions),
            'control_switches': self._count_control_switches(result.actions),
            
            # Timing analysis
            'first_heating_step': self._find_first_heating_step(result.actions),
            'last_heating_step': self._find_last_heating_step(result.actions),
            'heating_duration': self._calculate_heating_duration(result.actions),
            'used_delayed_strategy': self._detect_delayed_strategy(result.actions),
            
            # Temperature trajectory analysis
            'max_temperature': max(result.temperatures) if result.temperatures else 0,
            'temperature_overshoot': max(0, max(result.temperatures) - self.target_temp) if result.temperatures else 0,
            'temperature_variance': np.var(result.temperatures) if result.temperatures else 0,
            'steady_state_steps': self._count_steady_state_steps(result.temperatures),
            
            # MCTS performance
            'mcts_iterations_total': result.mcts_stats.total_iterations,
            'mcts_time_total': result.mcts_stats.total_time,
            'mcts_tree_size': result.mcts_stats.tree_size,
            'mcts_max_depth': result.mcts_stats.max_depth,
            'mcts_rollouts': result.mcts_stats.rollouts_performed,
            'mcts_nodes_expanded': result.mcts_stats.nodes_expanded,
            'mcts_efficiency': self._calculate_mcts_efficiency(result.mcts_stats),
            
            # Timing
            'total_time': result.total_time,
            'avg_time_per_step': result.avg_time_per_step,
            'time_per_mcts_iteration': (result.mcts_stats.total_time / max(1, result.mcts_stats.total_iterations)),
        }
        
        return metrics
    
    def calculate_experiment_metrics(self, results: List[EpisodeResult]) -> Dict[str, Any]:
        """Calculate aggregated metrics for an experiment"""
        
        if not results:
            return {}
        
        # Calculate individual episode metrics
        episode_metrics = [self.calculate_episode_metrics(result) for result in results]
        
        # Aggregate metrics
        metrics = {
            'num_episodes': len(results),
            
            # Goal achievement rates
            'goal_achievement_rate': np.mean([m['goal_achieved'] for m in episode_metrics]),
            'temp_tolerance_rate': np.mean([m['within_temp_tolerance'] for m in episode_metrics]),
            'time_tolerance_rate': np.mean([m['within_time_tolerance'] for m in episode_metrics]),
            
            # Energy efficiency statistics
            'avg_energy_consumed': np.mean([m['energy_consumed_wh'] for m in episode_metrics]),
            'std_energy_consumed': np.std([m['energy_consumed_wh'] for m in episode_metrics]),
            'min_energy_consumed': np.min([m['energy_consumed_wh'] for m in episode_metrics]),
            'max_energy_consumed': np.max([m['energy_consumed_wh'] for m in episode_metrics]),
            'avg_energy_efficiency': np.mean([m['energy_efficiency'] for m in episode_metrics]),
            'std_energy_efficiency': np.std([m['energy_efficiency'] for m in episode_metrics]),
            'energy_efficiency_75th': np.percentile([m['energy_efficiency'] for m in episode_metrics], 75),
            'energy_efficiency_90th': np.percentile([m['energy_efficiency'] for m in episode_metrics], 90),
            
            # Performance statistics
            'avg_total_reward': np.mean([m['total_reward'] for m in episode_metrics]),
            'std_total_reward': np.std([m['total_reward'] for m in episode_metrics]),
            'avg_final_temperature': np.mean([m['final_temperature'] for m in episode_metrics]),
            'std_final_temperature': np.std([m['final_temperature'] for m in episode_metrics]),
            'avg_temperature_error': np.mean([m['temperature_error'] for m in episode_metrics]),
            'avg_time_error': np.mean([m['time_error'] for m in episode_metrics]),
            
            # Control strategy statistics
            'avg_heating_percentage': np.mean([m['heating_percentage'] for m in episode_metrics]),
            'avg_control_switches': np.mean([m['control_switches'] for m in episode_metrics]),
            'delayed_strategy_rate': np.mean([m['used_delayed_strategy'] for m in episode_metrics]),
            'avg_heating_duration': np.mean([m['heating_duration'] for m in episode_metrics]),
            
            # Temperature control quality
            'avg_max_temperature': np.mean([m['max_temperature'] for m in episode_metrics]),
            'avg_temperature_overshoot': np.mean([m['temperature_overshoot'] for m in episode_metrics]),
            'avg_temperature_variance': np.mean([m['temperature_variance'] for m in episode_metrics]),
            'avg_steady_state_steps': np.mean([m['steady_state_steps'] for m in episode_metrics]),
            
            # MCTS performance
            'avg_mcts_iterations': np.mean([m['mcts_iterations_total'] for m in episode_metrics]),
            'avg_mcts_time': np.mean([m['mcts_time_total'] for m in episode_metrics]),
            'avg_tree_size': np.mean([m['mcts_tree_size'] for m in episode_metrics]),
            'avg_max_depth': np.mean([m['mcts_max_depth'] for m in episode_metrics]),
            'avg_mcts_efficiency': np.mean([m['mcts_efficiency'] for m in episode_metrics]),
            
            # Timing statistics
            'avg_total_time': np.mean([m['total_time'] for m in episode_metrics]),
            'avg_time_per_step': np.mean([m['avg_time_per_step'] for m in episode_metrics]),
            'avg_time_per_iteration': np.mean([m['time_per_mcts_iteration'] for m in episode_metrics]),
            
            # Reliability metrics
            'success_consistency': self._calculate_success_consistency(episode_metrics),
            'energy_consistency': self._calculate_energy_consistency(episode_metrics),
            'temperature_consistency': self._calculate_temperature_consistency(episode_metrics),
            
            # Performance categories
            'excellent_performance_rate': self._calculate_excellence_rate(episode_metrics),
            'good_performance_rate': self._calculate_good_performance_rate(episode_metrics),
            'poor_performance_rate': self._calculate_poor_performance_rate(episode_metrics),
        }
        
        # Add confidence intervals for key metrics
        confidence_metrics = [
            'goal_achievement_rate', 'avg_energy_efficiency', 'avg_total_reward'
        ]
        
        for metric in confidence_metrics:
            if metric in metrics:
                values = [m[metric.replace('avg_', '')] for m in episode_metrics if metric.replace('avg_', '') in m]
                if values:
                    ci_lower, ci_upper = self._calculate_confidence_interval(values)
                    metrics[f'{metric}_ci_lower'] = ci_lower
                    metrics[f'{metric}_ci_upper'] = ci_upper
        
        return metrics
    
    def compare_experiments(self, 
                          results_a: List[EpisodeResult], 
                          results_b: List[EpisodeResult],
                          name_a: str = "Algorithm A",
                          name_b: str = "Algorithm B") -> Dict[str, Any]:
        """Compare two sets of experiment results"""
        
        metrics_a = self.calculate_experiment_metrics(results_a)
        metrics_b = self.calculate_experiment_metrics(results_b)
        
        comparison = {
            'algorithm_a': name_a,
            'algorithm_b': name_b,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'statistical_tests': {},
            'effect_sizes': {},
            'recommendations': []
        }
        
        # Key metrics to compare
        key_metrics = [
            ('goal_achievement_rate', 'Goal Achievement Rate'),
            ('avg_energy_efficiency', 'Energy Efficiency'),
            ('avg_total_reward', 'Total Reward'),
            ('avg_final_temperature', 'Final Temperature'),
            ('avg_energy_consumed', 'Energy Consumed')
        ]
        
        for metric_key, metric_name in key_metrics:
            if metric_key in metrics_a and metric_key in metrics_b:
                # Extract values for statistical testing
                values_a = self._extract_metric_values(results_a, metric_key)
                values_b = self._extract_metric_values(results_b, metric_key)
                
                if values_a and values_b:
                    # Perform statistical tests
                    test_results = self._perform_statistical_tests(values_a, values_b)
                    comparison['statistical_tests'][metric_key] = {
                        'metric_name': metric_name,
                        'mean_a': np.mean(values_a),
                        'mean_b': np.mean(values_b),
                        'difference': np.mean(values_b) - np.mean(values_a),
                        'percent_change': ((np.mean(values_b) - np.mean(values_a)) / np.mean(values_a)) * 100 if np.mean(values_a) != 0 else 0,
                        **test_results
                    }
                    
                    # Calculate effect size
                    effect_size = self._calculate_effect_size(values_a, values_b)
                    comparison['effect_sizes'][metric_key] = effect_size
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_comparison_recommendations(comparison)
        
        return comparison
    
    def calculate_parameter_sensitivity(self, 
                                      results: List[Dict[str, Any]], 
                                      parameter_name: str) -> Dict[str, Any]:
        """Calculate sensitivity of performance to parameter changes"""
        
        # Group results by parameter value
        parameter_groups = {}
        for result in results:
            param_value = self._extract_parameter_value(result, parameter_name)
            if param_value is not None:
                if param_value not in parameter_groups:
                    parameter_groups[param_value] = []
                parameter_groups[param_value].extend(result['results'])
        
        if len(parameter_groups) < 2:
            return {'error': 'Insufficient parameter variation for sensitivity analysis'}
        
        # Calculate metrics for each parameter value
        sensitivity_data = {}
        parameter_values = sorted(parameter_groups.keys())
        
        for param_value in parameter_values:
            episode_results = parameter_groups[param_value]
            metrics = self.calculate_experiment_metrics(episode_results)
            sensitivity_data[param_value] = metrics
        
        # Calculate sensitivity measures
        sensitivity_analysis = {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'sensitivity_data': sensitivity_data,
            'correlations': {},
            'trends': {},
            'optimal_value': None,
            'sensitivity_score': 0.0
        }
        
        # Key metrics to analyze
        key_metrics = ['goal_achievement_rate', 'avg_energy_efficiency', 'avg_total_reward']
        
        for metric in key_metrics:
            metric_values = [sensitivity_data[pv].get(metric, 0) for pv in parameter_values]
            
            # Calculate correlation
            if len(parameter_values) > 2:
                correlation, p_value = stats.pearsonr(parameter_values, metric_values)
                sensitivity_analysis['correlations'][metric] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            
            # Find optimal parameter value
            best_idx = np.argmax(metric_values)
            optimal_param = parameter_values[best_idx]
            
            sensitivity_analysis['trends'][metric] = {
                'values': metric_values,
                'optimal_parameter': optimal_param,
                'optimal_value': metric_values[best_idx],
                'range': max(metric_values) - min(metric_values),
                'coefficient_of_variation': np.std(metric_values) / np.mean(metric_values) if np.mean(metric_values) != 0 else 0
            }
        
        # Calculate overall sensitivity score
        cv_scores = [sensitivity_analysis['trends'][metric]['coefficient_of_variation'] 
                    for metric in key_metrics]
        sensitivity_analysis['sensitivity_score'] = np.mean(cv_scores)
        
        # Find overall optimal parameter value
        # Weight metrics equally for now
        weighted_scores = []
        for i, param_value in enumerate(parameter_values):
            score = 0
            for metric in key_metrics:
                metric_values = sensitivity_analysis['trends'][metric]['values']
                normalized_value = (metric_values[i] - min(metric_values)) / (max(metric_values) - min(metric_values)) if max(metric_values) != min(metric_values) else 0
                score += normalized_value
            weighted_scores.append(score / len(key_metrics))
        
        best_idx = np.argmax(weighted_scores)
        sensitivity_analysis['optimal_value'] = parameter_values[best_idx]
        
        return sensitivity_analysis
    
    def _theoretical_min_energy(self) -> float:
        """Calculate theoretical minimum energy (Wh)"""
        # 1kg water, 80°C rise, 4184 J/kg·K
        return (1.0 * 4184 * 80) / 3600
    
    def _categorize_efficiency(self, efficiency: float) -> str:
        """Categorize energy efficiency"""
        if efficiency >= 0.9:
            return "excellent"
        elif efficiency >= 0.8:
            return "good"
        elif efficiency >= 0.6:
            return "fair"
        else:
            return "poor"
    
    def _count_heating_actions(self, actions: List[int]) -> int:
        """Count number of heating actions (assuming action 1 is heating)"""
        return sum(1 for action in actions if action == 1)
    
    def _calculate_heating_percentage(self, actions: List[int]) -> float:
        """Calculate percentage of heating actions"""
        if not actions:
            return 0.0
        heating_count = self._count_heating_actions(actions)
        return (heating_count / len(actions)) * 100
    
    def _longest_heating_streak(self, actions: List[int]) -> int:
        """Find longest consecutive heating period"""
        max_streak = 0
        current_streak = 0
        
        for action in actions:
            if action == 1:  # Heating
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _longest_coasting_streak(self, actions: List[int]) -> int:
        """Find longest consecutive coasting period"""
        max_streak = 0
        current_streak = 0
        
        for action in actions:
            if action == 0:  # Coasting
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _count_control_switches(self, actions: List[int]) -> int:
        """Count number of control switches"""
        if len(actions) < 2:
            return 0
        
        switches = 0
        for i in range(1, len(actions)):
            if actions[i] != actions[i-1]:
                switches += 1
        
        return switches
    
    def _find_first_heating_step(self, actions: List[int]) -> Optional[int]:
        """Find first heating step"""
        for i, action in enumerate(actions):
            if action == 1:
                return i + 1  # 1-indexed
        return None
    
    def _find_last_heating_step(self, actions: List[int]) -> Optional[int]:
        """Find last heating step"""
        for i in range(len(actions) - 1, -1, -1):
            if actions[i] == 1:
                return i + 1  # 1-indexed
        return None
    
    def _calculate_heating_duration(self, actions: List[int]) -> int:
        """Calculate total heating duration"""
        first_heat = self._find_first_heating_step(actions)
        last_heat = self._find_last_heating_step(actions)
        
        if first_heat is None or last_heat is None:
            return 0
        
        return last_heat - first_heat + 1
    
    def _detect_delayed_strategy(self, actions: List[int]) -> bool:
        """Detect if delayed heating strategy was used"""
        first_heat = self._find_first_heating_step(actions)
        return first_heat is not None and first_heat > 50
    
    def _count_steady_state_steps(self, temperatures: List[float]) -> int:
        """Count steps where temperature is near target"""
        if not temperatures:
            return 0
        
        steady_count = 0
        for temp in temperatures:
            if abs(temp - self.target_temp) <= self.temp_tolerance:
                steady_count += 1
        
        return steady_count
    
    def _calculate_mcts_efficiency(self, mcts_stats) -> float:
        """Calculate MCTS computational efficiency"""
        if mcts_stats.total_time == 0:
            return 0.0
        
        # Iterations per second
        return mcts_stats.total_iterations / mcts_stats.total_time
    
    def _calculate_success_consistency(self, episode_metrics: List[Dict[str, Any]]) -> float:
        """Calculate consistency of goal achievement"""
        successes = [m['goal_achieved'] for m in episode_metrics]
        return 1.0 - np.std(successes)  # Lower std = higher consistency
    
    def _calculate_energy_consistency(self, episode_metrics: List[Dict[str, Any]]) -> float:
        """Calculate consistency of energy consumption"""
        energies = [m['energy_consumed_wh'] for m in episode_metrics]
        if not energies:
            return 0.0
        
        cv = np.std(energies) / np.mean(energies) if np.mean(energies) != 0 else 0
        return max(0, 1.0 - cv)  # Lower CV = higher consistency
    
    def _calculate_temperature_consistency(self, episode_metrics: List[Dict[str, Any]]) -> float:
        """Calculate consistency of final temperature"""
        temps = [m['final_temperature'] for m in episode_metrics]
        if not temps:
            return 0.0
        
        # Consistency based on temperature variance
        variance = np.var(temps)
        return max(0, 1.0 - (variance / 100))  # Normalize by reasonable temperature range
    
    def _calculate_excellence_rate(self, episode_metrics: List[Dict[str, Any]]) -> float:
        """Calculate rate of excellent performance"""
        excellent_count = 0
        for m in episode_metrics:
            if (m['goal_achieved'] and 
                m['energy_efficiency'] > 0.9 and 
                m['temperature_error'] < 1.0):
                excellent_count += 1
        
        return excellent_count / len(episode_metrics) if episode_metrics else 0.0
    
    def _calculate_good_performance_rate(self, episode_metrics: List[Dict[str, Any]]) -> float:
        """Calculate rate of good performance"""
        good_count = 0
        for m in episode_metrics:
            if (m['goal_achieved'] and 
                m['energy_efficiency'] > 0.8):
                good_count += 1
        
        return good_count / len(episode_metrics) if episode_metrics else 0.0
    
    def _calculate_poor_performance_rate(self, episode_metrics: List[Dict[str, Any]]) -> float:
        """Calculate rate of poor performance"""
        poor_count = 0
        for m in episode_metrics:
            if (not m['goal_achieved'] or 
                m['energy_efficiency'] < 0.6):
                poor_count += 1
        
        return poor_count / len(episode_metrics) if episode_metrics else 0.0
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a metric"""
        if len(values) < 2:
            return 0.0, 0.0
        
        mean = np.mean(values)
        std_err = stats.sem(values)
        h = std_err * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
        
        return mean - h, mean + h
    
    def _extract_metric_values(self, results: List[EpisodeResult], metric_key: str) -> List[float]:
        """Extract metric values from episode results"""
        values = []
        for result in results:
            episode_metrics = self.calculate_episode_metrics(result)
            if metric_key in episode_metrics:
                values.append(episode_metrics[metric_key])
            elif metric_key.startswith('avg_'):
                # Handle averaged metrics
                base_metric = metric_key.replace('avg_', '')
                if base_metric in episode_metrics:
                    values.append(episode_metrics[base_metric])
        
        return values
    
    def _perform_statistical_tests(self, values_a: List[float], values_b: List[float]) -> Dict[str, Any]:
        """Perform statistical tests comparing two groups"""
        results = {}
        
        # T-test
        try:
            t_stat, t_p_value = stats.ttest_ind(values_a, values_b)
            results['t_test'] = {
                'statistic': t_stat,
                'p_value': t_p_value,
                'significant': t_p_value < 0.05
            }
        except Exception as e:
            results['t_test'] = {'error': str(e)}
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            results['mann_whitney_u'] = {
                'statistic': u_stat,
                'p_value': u_p_value,
                'significant': u_p_value < 0.05
            }
        except Exception as e:
            results['mann_whitney_u'] = {'error': str(e)}
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p_value = stats.ks_2samp(values_a, values_b)
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p_value,
                'significant': ks_p_value < 0.05
            }
        except Exception as e:
            results['kolmogorov_smirnov'] = {'error': str(e)}
        
        return results
    
    def _calculate_effect_size(self, values_a: List[float], values_b: List[float]) -> Dict[str, float]:
        """Calculate effect size measures"""
        mean_a, mean_b = np.mean(values_a), np.mean(values_b)
        std_a, std_b = np.std(values_a, ddof=1), np.std(values_b, ddof=1)
        
        # Cohen's d
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                            (len(values_a) + len(values_b) - 2))
        
        cohens_d = (mean_b - mean_a) / pooled_std if pooled_std != 0 else 0
        
        # Glass's delta
        glass_delta = (mean_b - mean_a) / std_a if std_a != 0 else 0
        
        # Hedges' g (bias-corrected Cohen's d)
        j = 1 - (3 / (4 * (len(values_a) + len(values_b)) - 9))
        hedges_g = cohens_d * j
        
        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'hedges_g': hedges_g,
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        stats_tests = comparison.get('statistical_tests', {})
        
        for metric_key, test_data in stats_tests.items():
            metric_name = test_data.get('metric_name', metric_key)
            difference = test_data.get('difference', 0)
            percent_change = test_data.get('percent_change', 0)
            significant = test_data.get('t_test', {}).get('significant', False)
            
            if significant and abs(percent_change) > 5:  # Significant and meaningful difference
                direction = "better" if difference > 0 else "worse"
                recommendations.append(
                    f"{comparison['algorithm_b']} performs {direction} than {comparison['algorithm_a']} "
                    f"in {metric_name} ({percent_change:+.1f}% change, statistically significant)"
                )
        
        # Check effect sizes
        effect_sizes = comparison.get('effect_sizes', {})
        for metric_key, effect_data in effect_sizes.items():
            interpretation = effect_data.get('effect_size_interpretation', '')
            if interpretation in ['medium', 'large']:
                recommendations.append(
                    f"The difference in {metric_key} shows a {interpretation} effect size "
                    f"(Cohen's d = {effect_data.get('cohens_d', 0):.3f})"
                )
        
        if not recommendations:
            recommendations.append("No statistically significant or practically meaningful differences found.")
        
        return recommendations
    
    def _extract_parameter_value(self, result: Dict[str, Any], parameter_name: str) -> Any:
        """Extract parameter value from result"""
        config = result.get('config', {})
        
        # Check algorithm config
        if hasattr(config, 'algorithm'):
            if hasattr(config.algorithm, parameter_name):
                return getattr(config.algorithm, parameter_name)
        
        # Check environment config
        if hasattr(config, 'environment'):
            if hasattr(config.environment, parameter_name):
                return getattr(config.environment, parameter_name)
        
        # Check top-level config
        if hasattr(config, parameter_name):
            return getattr(config, parameter_name)
        
        return None