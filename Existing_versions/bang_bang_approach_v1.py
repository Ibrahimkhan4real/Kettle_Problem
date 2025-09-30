"""
Bang-Bang control implementation for the kettle control problem.
This is a simple threshold-based controller that serves as a baseline.
"""

import numpy as np


class BangBangAgent:
    """
    Bang-Bang controller for kettle control.
    
    This implements several variants:
    1. Simple: Turn on below threshold, turn off above
    2. Hysteresis: Add dead band to prevent chattering
    3. Time-aware: Consider deadline in decision making
    4. Optimal: Theoretical optimal bang-bang (wait then heat)
    """
    
    def __init__(self, control_type='time_aware', 
                 low_threshold=98.0, high_threshold=101.0,
                 heating_time_estimate=112):
        """
        Initialize Bang-Bang controller.
        
        Args:
            control_type: 'simple', 'hysteresis', 'time_aware', or 'optimal'
            low_threshold: Temperature below which heating starts
            high_threshold: Temperature above which heating stops
            heating_time_estimate: Estimated time needed to heat from 20°C to 100°C
        """
        self.control_type = control_type
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.heating_time_estimate = heating_time_estimate
        
        # For hysteresis
        self.last_action = 0
        
    def reset(self):
        """Reset controller state"""
        self.last_action = 0
    
    def get_action_simple(self, temp, step):
        """Simple bang-bang: on below 100°C, off above"""
        if temp < 100.0:
            return 1  # Heat
        else:
            return 0  # Don't heat
    
    def get_action_hysteresis(self, temp, step):
        """Bang-bang with hysteresis to prevent chattering"""
        if temp < self.low_threshold:
            self.last_action = 1
        elif temp > self.high_threshold:
            self.last_action = 0
        # If between thresholds, maintain last action
        return self.last_action
    
    def get_action_time_aware(self, temp, step, deadline=200):
        """Time-aware bang-bang that considers the deadline"""
        steps_to_deadline = deadline - step
        
        # Estimate time needed to reach target from current temp
        temp_diff = max(0, 100 - temp)
        time_needed = int(temp_diff * 1.3)  # Rough estimate
        
        # Don't heat if too early
        if steps_to_deadline > time_needed + 20:
            return 0
        
        # Use hysteresis when in heating window
        if temp < self.low_threshold:
            return 1
        elif temp > self.high_threshold:
            return 0
        else:
            # In dead band - maintain temperature
            return self.last_action
    
    def get_action_optimal(self, temp, step, deadline=200):
        """
        Theoretical optimal bang-bang control.
        Wait until the right moment, then heat continuously.
        """
        steps_to_deadline = deadline - step
        
        # Calculate exact time needed (simplified physics)
        if temp < 95:  # Need to heat
            if steps_to_deadline <= self.heating_time_estimate:
                return 1  # Start heating
            else:
                return 0  # Wait
        else:
            # Near target - maintain with hysteresis
            if temp < 98.5:
                return 1
            elif temp > 101.5:
                return 0
            else:
                return 0  # Prefer not heating to save energy
    
    def get_action(self, env, obs, step_num=None):
        """Get control action based on current state"""
        temp_diff, steps_to_deadline = obs
        temp = temp_diff + 100.0  # Convert to actual temperature
        step = env.target_deadline_step - int(steps_to_deadline)
        
        if self.control_type == 'simple':
            action = self.get_action_simple(temp, step)
        elif self.control_type == 'hysteresis':
            action = self.get_action_hysteresis(temp, step)
        elif self.control_type == 'time_aware':
            action = self.get_action_time_aware(temp, step, env.target_deadline_step)
            self.last_action = action  # Update for hysteresis
        elif self.control_type == 'optimal':
            action = self.get_action_optimal(temp, step, env.target_deadline_step)
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")
        
        if step_num is not None:
            print(f"\nStep {step_num} - Bang-Bang ({self.control_type}) Decision")
            print(f"Temperature: {temp:.1f}°C, Steps to deadline: {steps_to_deadline:.0f}")
            print(f"Action: {action} ({'Heat' if action == 1 else 'Wait/Cool'})")
            
            if self.control_type == 'time_aware':
                time_needed = int((100 - temp) * 1.3)
                print(f"Estimated time needed: {time_needed} steps")
                print(f"Heating window: {steps_to_deadline <= time_needed + 20}")
        
        return action
    
    def run_episode(self, env, render=False, verbose=True):
        """Run a complete episode with bang-bang control"""
        obs, info = env.reset()
        self.reset()
        
        # Episode data
        trajectory = {
            'observations': [obs],
            'actions': [],
            'rewards': [],
            'temperatures': [info['current_temperature']],
            'energy_Wh': [0],
            'steps': [0]
        }
        
        done = False
        total_reward = 0
        step = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Bang-Bang Control Episode ({self.control_type})")
            print(f"Initial temperature: {trajectory['temperatures'][0]:.1f}°C")
            print(f"Target: 100°C at step 200 (±15 steps, ±2.5°C)")
            print(f"Control thresholds: {self.low_threshold:.1f}°C - {self.high_threshold:.1f}°C")
            print(f"{'='*60}")
        
        while not done and step < env.max_steps:
            # Get action from bang-bang controller
            action = self.get_action(env, obs, step + 1 if verbose else None)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Record data
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['temperatures'].append(info['temperature'])
            trajectory['energy_Wh'].append(info['total_energy_Wh'])
            trajectory['steps'].append(info['current_step'])
            
            total_reward += reward
            
            if verbose and (step < 10 or step % 20 == 0 or abs(step - 200) < 5):
                print(f"\nStep {step+1}: Action={action} ({env.actions[action]}W)")
                print(f"  Temperature: {info['temperature']:.1f}°C")
                print(f"  Reward: {reward:.3f}, Total: {total_reward:.3f}")
                print(f"  Energy: {info['total_energy_Wh']:.1f} Wh")
            
            if render:
                env.render()
            
            step += 1
            done = done or truncated
        
        # Summary
        trajectory['total_reward'] = total_reward
        trajectory['goal_achieved'] = info.get('goal_achieved', False)
        trajectory['final_temperature'] = trajectory['temperatures'][-1]
        trajectory['final_energy_Wh'] = trajectory['energy_Wh'][-1]
        trajectory['episode_length'] = len(trajectory['actions'])
        
        if verbose:
            print(f"\n{'='*40}")
            print("Episode Summary:")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final temperature: {trajectory['final_temperature']:.1f}°C")
            print(f"Total energy: {trajectory['final_energy_Wh']:.1f} Wh")
            print(f"Goal achieved: {'YES' if trajectory['goal_achieved'] else 'NO'}")
            
            if trajectory['final_energy_Wh'] > 0:
                theoretical_min = 93.0
                efficiency = theoretical_min / trajectory['final_energy_Wh'] * 100
                print(f"Energy efficiency: {efficiency:.1f}%")
            
            # Control statistics
            on_actions = sum(1 for a in trajectory['actions'] if a == 1)
            print(f"Heating actions: {on_actions}/{len(trajectory['actions'])} "
                  f"({100*on_actions/len(trajectory['actions']):.1f}%)")
            
            # Find first and last heating
            first_heat = None
            last_heat = None
            for i, a in enumerate(trajectory['actions']):
                if a == 1:
                    if first_heat is None:
                        first_heat = i + 1
                    last_heat = i + 1
            
            if first_heat:
                print(f"First heating: step {first_heat}")
                print(f"Last heating: step {last_heat}")
                print(f"Heating duration: {last_heat - first_heat + 1} steps")
        
        return trajectory


def test_bang_bang_variants():
    """Test different bang-bang control variants"""
    from kettle_dynamic_env_v24 import KettleEnv
    
    env = KettleEnv(
        initial_temp=20.0,
        ambient_temp=25.0,
        max_steps=250,
        target_deadline_step=200
    )
    
    # Test different variants
    variants = ['simple', 'hysteresis', 'time_aware', 'optimal']
    results = {}
    
    for variant in variants:
        print(f"\n{'='*70}")
        print(f"Testing Bang-Bang variant: {variant}")
        print(f"{'='*70}")
        
        agent = BangBangAgent(control_type=variant)
        trajectory = agent.run_episode(env, render=False, verbose=True)
        results[variant] = trajectory
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON OF BANG-BANG VARIANTS")
    print(f"{'='*70}")
    print(f"{'Variant':<15} {'Total Reward':>12} {'Energy (Wh)':>12} "
          f"{'Efficiency':>10} {'Goal?':>6}")
    print("-" * 70)
    
    for variant in variants:
        traj = results[variant]
        efficiency = 93.0 / traj['final_energy_Wh'] * 100 if traj['final_energy_Wh'] > 0 else 0
        print(f"{variant:<15} {traj['total_reward']:>12.1f} "
              f"{traj['final_energy_Wh']:>12.1f} "
              f"{efficiency:>9.1f}% "
              f"{'Yes' if traj['goal_achieved'] else 'No':>6}")
    
    return results


if __name__ == "__main__":
    print("Testing Bang-Bang Control Implementation...")
    test_bang_bang_variants()