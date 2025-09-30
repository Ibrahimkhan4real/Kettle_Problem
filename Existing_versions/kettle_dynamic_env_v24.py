import gymnasium as gym
from gymnasium import spaces
import numpy as np

__version__ = "v24_energy_efficient"

class KettleEnv(gym.Env):
    """
    Energy-Efficient Kettle Environment
    
    This version includes:
    - Energy consumption penalties to encourage efficiency
    - Maintenance rewards for staying at target temperature
    - Efficiency bonuses when achieving the goal
    - Waste penalties for unnecessary heating
    
    The goal is to reach 100°C (±2.5°C) at step 200 (±15 steps) 
    while using the minimum amount of energy possible.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        water_volume=1.0,
        initial_temp=20.0,
        ambient_temp=20.0, 
        max_steps=200,
        target_deadline_step=200,
        energy_penalty_weight=0.01,  # Weight for energy consumption penalty
        goal_reward=3000.0,          # Main reward for achieving goal
        temp_maintenance_reward=5.0,  # Reward for maintaining target temp
        efficiency_bonus_scale=500.0, # Scale for efficiency bonus
        waste_penalty_scale=0.5      # Extra penalty for wasteful heating
    ):
        super(KettleEnv, self).__init__()

        # Physical parameters
        self.water_volume = water_volume
        self.m = self.water_volume           # Mass in kg
        self.c = 4184.0                      # Specific heat capacity of water (J/kg·K)
        self.hA = 5.0                        # Heat loss coefficient (W/K)

        # Target parameters
        self.T_target = 100.0                # Target temperature (°C)
        self.T_acceptable_dev = 2.5          # Acceptable deviation (±2.5°C)
        self.time_tolerance = 15             # Time window (±15 steps)

        # Initial conditions
        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp 
        self.dt = 1.0                        # Time step (seconds)

        # Energy efficiency reward parameters
        self.energy_penalty_weight = energy_penalty_weight
        self.goal_reward = goal_reward
        self.temp_maintenance_reward = temp_maintenance_reward
        self.efficiency_bonus_scale = efficiency_bonus_scale
        self.waste_penalty_scale = waste_penalty_scale

        # Action space: [0W (off), 3000W (on)]
        self.actions = [0, 3000]
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: [temperature_difference, steps_to_deadline]
        min_temp_diff = 0.0 - self.T_target
        max_temp_diff = 150.0 - self.T_target
        min_steps_diff = float(target_deadline_step - max_steps)
        max_steps_diff = float(target_deadline_step)
        low_obs = np.array([min_temp_diff, min_steps_diff], dtype=np.float32)
        high_obs = np.array([max_temp_diff, max_steps_diff], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        # State variables
        self.current_temp = 0.0
        self.steps = 0
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        
        # Episode tracking
        self.goal_achieved_this_episode = False
        self.steps_in_target_range = 0
        self.first_time_at_target = None
        
        # Calculate theoretical minimum energy needed (no heat loss)
        self.theoretical_min_energy = self.m * self.c * (self.T_target - self.initial_temp)

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_temp = round(self.initial_temp, 1) 
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        self.goal_achieved_this_episode = False
        self.steps_in_target_range = 0
        self.first_time_at_target = None
        
        # Create observation
        temp_diff = self.current_temp - self.T_target
        steps_diff = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff, steps_diff], dtype=np.float32)
        
        # Info dictionary
        info = {
            "initial_temperature": self.initial_temp, 
            "current_temperature": self.current_temp,
            "theoretical_min_energy_J": self.theoretical_min_energy,
            "theoretical_min_energy_Wh": self.theoretical_min_energy / 3600
        }
        
        return current_state, info

    def calculate_reward(self, T_after_action, T_before_action, power_W):
        """
        Calculate reward with focus on energy efficiency.
        
        Components:
        1. Energy consumption penalty (continuous negative reward)
        2. Temperature maintenance reward (positive when at target)
        3. Goal achievement reward (large positive at right time)
        4. Efficiency bonus (when goal achieved)
        5. Waste penalty (heating when already at target)
        """
        reward = 0.0
        
        # 1. Energy consumption penalty
        # Penalize energy use to encourage efficiency
        energy_used_J = power_W * self.dt
        energy_penalty = -self.energy_penalty_weight * (energy_used_J / 1000)  # Convert to kJ
        reward += energy_penalty
        
        # 2. Temperature maintenance reward
        # Reward for being in the target temperature range
        temp_error = abs(self.current_temp - self.T_target)
        if temp_error <= self.T_acceptable_dev:
            # Linear reward based on how close to perfect temperature
            maintenance_bonus = self.temp_maintenance_reward * (1 - temp_error / self.T_acceptable_dev)
            reward += maintenance_bonus
            self.steps_in_target_range += 1
            
            # Track first time reaching target
            if self.first_time_at_target is None:
                self.first_time_at_target = self.steps
        
        # 3. Goal achievement reward
        # Large reward for being at target temperature at the right time
        if not self.goal_achieved_this_episode:
            if temp_error <= self.T_acceptable_dev:
                time_error = abs(self.steps - self.target_deadline_step)
                if time_error <= self.time_tolerance:
                    # Goal achieved!
                    reward += self.goal_reward
                    self.goal_achieved_this_episode = True
                    
                    # 4. Efficiency bonus
                    # Extra reward based on energy efficiency
                    if self.total_energy_consumed_Joules > 0:
                        efficiency_ratio = self.theoretical_min_energy / self.total_energy_consumed_Joules
                        efficiency_ratio = min(efficiency_ratio, 1.0)  # Cap at 100%
                        efficiency_bonus = self.efficiency_bonus_scale * efficiency_ratio
                        reward += efficiency_bonus
        
        # 5. Waste penalty
        # Extra penalty for heating when already at target temperature
        if temp_error <= self.T_acceptable_dev and power_W > 0:
            waste_penalty = -self.waste_penalty_scale * (power_W / 1000)  # kW
            reward += waste_penalty
        
        # 6. Distance penalty near deadline
        # Encourage being close to target as deadline approaches
        if self.target_deadline_step - 30 <= self.steps <= self.target_deadline_step + 15:
            if temp_error > self.T_acceptable_dev:
                # Penalty proportional to distance from target
                distance_penalty = -0.1 * temp_error
                reward += distance_penalty
        
        return reward

    def step(self, action: int):
        """Execute one time step within the environment"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        # Get power from action
        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        
        # Track energy consumption
        energy_this_step = power_selected_W * self.dt
        self.total_energy_consumed_Joules += energy_this_step
        
        # Store temperature before action
        temp_before_action = self.current_temp
        
        # Physics simulation
        # Heat loss to environment
        heat_loss_W = self.hA * (temp_before_action - self.ambient_temp)
        
        # Temperature change rate
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c)
        
        # Update temperature
        T_next = temp_before_action + self.dt * dTemp_dt
        
        # Clip temperature to realistic bounds
        T_next = np.clip(T_next, 0.0, 150.0)
        
        # Round to 1 decimal place for stability
        self.current_temp = round(T_next, 1)
        
        # Increment time step
        self.steps += 1
        
        # Calculate reward
        current_step_reward = self.calculate_reward(
            self.current_temp, temp_before_action, power_selected_W
        )
        
        # Check termination
        terminated = self.steps >= self.max_steps
        truncated = False

        # Create observation
        temp_diff_obs = self.current_temp - self.T_target
        steps_diff_obs = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff_obs, steps_diff_obs], dtype=np.float32)
        
        # Calculate current efficiency
        if self.total_energy_consumed_Joules > 0:
            current_efficiency = self.theoretical_min_energy / self.total_energy_consumed_Joules
            current_efficiency = min(current_efficiency, 1.0)
        else:
            current_efficiency = 1.0
        
        # Create info dictionary
        info = {
            # Basic state info
            "temperature": self.current_temp,
            "power_W": power_selected_W,
            "current_step": self.steps,
            
            # Energy metrics
            "energy_used_this_step_J": energy_this_step,
            "total_energy_J": self.total_energy_consumed_Joules,
            "total_energy_Wh": self.total_energy_consumed_Joules / 3600.0,
            "theoretical_min_energy_Wh": self.theoretical_min_energy / 3600.0,
            "energy_efficiency": current_efficiency,
            
            # Goal tracking
            "is_at_deadline": self.steps == self.target_deadline_step,
            "goal_achieved": self.goal_achieved_this_episode,
            "steps_in_target_range": self.steps_in_target_range,
            "first_time_at_target": self.first_time_at_target,
            
            # Reward breakdown (for debugging)
            "reward_breakdown": {
                "energy_penalty": -self.energy_penalty_weight * (energy_this_step / 1000),
                "total_reward": current_step_reward
            },
            
            # Physics info
            "heat_loss_W": heat_loss_W,
            "temp_change_rate": dTemp_dt
        }
        
        return current_state, current_step_reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment (text output)"""
        if mode == "human":
            # Status indicators
            deadline_status = " [DEADLINE]" if self.steps == self.target_deadline_step else ""
            goal_status = " [GOAL!]" if self.goal_achieved_this_episode else ""
            temp_status = " [TARGET]" if abs(self.current_temp - self.T_target) <= self.T_acceptable_dev else ""
            
            # Calculate efficiency
            if self.total_energy_consumed_Joules > 0:
                efficiency = self.theoretical_min_energy / self.total_energy_consumed_Joules
                efficiency = min(efficiency, 1.0)
            else:
                efficiency = 1.0
            
            # Print status
            print(
                f"Step: {self.steps:3d}, "
                f"Temp: {self.current_temp:5.1f}°C{temp_status}, "
                f"Power: {self.last_power:4d}W, "
                f"Energy: {self.total_energy_consumed_Joules/3600:5.1f}Wh, "
                f"Eff: {efficiency:4.1%}"
                f"{deadline_status}{goal_status}"
            )

    def close(self):
        """Clean up resources"""
        pass

    def get_optimal_energy_info(self):
        """
        Calculate theoretical optimal energy consumption.
        This assumes perfect insulation during heating and perfect timing.
        """
        # Energy to heat water
        heating_energy = self.m * self.c * (self.T_target - self.initial_temp)
        
        # Approximate heat loss during optimal heating time
        # Assume linear temperature rise
        avg_temp_during_heating = (self.initial_temp + self.T_target) / 2
        heating_time = heating_energy / self.actions[1]  # Time at max power
        heat_loss_during_heating = self.hA * (avg_temp_during_heating - self.ambient_temp) * heating_time
        
        # Total theoretical optimal
        optimal_energy = heating_energy + heat_loss_during_heating
        
        return {
            "heating_energy_J": heating_energy,
            "heat_loss_J": heat_loss_during_heating,
            "total_optimal_J": optimal_energy,
            "total_optimal_Wh": optimal_energy / 3600,
            "optimal_heating_time_s": heating_time
        }