import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt # For testing script
import time # For testing script
import os # For testing script
# import math # Not explicitly used

__version__ = "v3_area_minimisation" # Updated version

class KettleEnv(gym.Env):
    """
    Kettle Gym Environment with a reward function to minimize
    the area under the temperature-time curve while achieving the target state.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        water_volume=1.0,
        initial_temp=20.0,
        ambient_temp=20.0, # Setting ambient to 20C for example calculation consistency
        max_steps=250,
        target_deadline_step=200
    ):
        super(KettleEnv, self).__init__()

        self.water_volume = water_volume
        self.m = self.water_volume
        self.c = 4184.0
        self.hA = 1.0

        self.T_target = 100.0
        self.T_acceptable_dev = 1.0

        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp # Crucial for area penalty calculation
        self.dt = 1.0

        self.actions = [0] + list(range(1500, 3001, 500))
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)

        min_temp_diff = 0.0 - self.T_target
        max_temp_diff = 150.0 - self.T_target
        min_steps_diff = float(target_deadline_step - max_steps)
        max_steps_diff = float(target_deadline_step)
        low_obs = np.array([min_temp_diff, min_steps_diff], dtype=np.float32)
        high_obs = np.array([max_temp_diff, max_steps_diff], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        self.current_temp = 0.0
        self.steps = 0
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

        # --- Reward function hyperparameters ---
        # Goal achievement
        self.SUCCESS_BONUS = 200.0
        self.FAILURE_PENALTY = -100.0
        
        # Energy and Temperature Profile Control
        self.PER_STEP_POWER_COST_WEIGHT = 0.01 # Cost for energy usage
        self.TEMP_AREA_PENALTY_WEIGHT = 0.005 # NEW: Penalty for (T_current - T_ambient) to minimize area
        self.TEMP_PROGRESS_WEIGHT = 0.15      # Reward for reducing temp error (may need to be stronger)
        self.OVERSHOOT_PENALTY_WEIGHT = 0.05  # Penalty for T > T_target + T_dev
        
        # This bonus aims to make idling at target (power=0) preferable to suffering the full area penalty without progress
        # It should be tuned to be slightly greater than TEMP_AREA_PENALTY_WEIGHT * (T_target - T_ambient)
        # e.g., if T_target=100, T_ambient=20, then (100-20)*0.005 = 0.4. So bonus > 0.4
        self.EFFICIENT_IDLING_BONUS = 0.42

    def reset(self, seed=None, options=None):
        # (Same as your v3_fixed_corrected version)
        super().reset(seed=seed)
        self.current_temp = self.initial_temp
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        temp_diff = self.current_temp - self.T_target
        steps_diff = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff, steps_diff], dtype=np.float32)
        return current_state, {"initial_temperature": self.current_temp}

    def calculate_reward(self, T_after_action, T_before_action, power_W):
        reward = 0.0

        # 1. Per-Step Power Consumption Cost (Direct energy cost of action)
        max_power_possible = float(self.actions[-1]) if self.actions else 3000.0
        normalized_power = float(power_W) / max_power_possible if max_power_possible > 0 else 0.0
        reward -= self.PER_STEP_POWER_COST_WEIGHT * normalized_power

        # 2. Temperature Area Minimisation Penalty (Cost of being at a certain temperature)
        # This directly penalizes the area under the T(t) curve relative to ambient temperature.
        if T_after_action > self.ambient_temp: # Only penalize if temperature is above ambient
            reward -= self.TEMP_AREA_PENALTY_WEIGHT * (T_after_action - self.ambient_temp)

        # 3. Temperature Progress Shaping Reward (Incentive to move towards target temperature)
        # This needs to be strong enough to encourage heating when necessary, despite the area penalty.
        current_error_abs = abs(T_after_action - self.T_target)
        previous_error_abs = abs(T_before_action - self.T_target)
        error_delta = previous_error_abs - current_error_abs # Positive if error reduced (got closer)
        reward += self.TEMP_PROGRESS_WEIGHT * error_delta

        # 4. Penalty for Significant Overshooting
        # This specifically discourages heating far beyond the target.
        if T_after_action > self.T_target + self.T_acceptable_dev:
            overshoot_amount = T_after_action - (self.T_target + self.T_acceptable_dev)
            # Scaled penalty, e.g., (overshoot_amount / 10.0) means penalty per 10C segment of overshoot
            reward -= self.OVERSHOOT_PENALTY_WEIGHT * (overshoot_amount / 10.0)

        # 5. Bonus for Efficient Idling when at Target Temperature (before deadline)
        # This encourages turning off power if the target is reached early.
        if self.steps < self.target_deadline_step:
            # Check if within the acceptable target temperature range
            if (self.T_target - self.T_acceptable_dev) <= T_after_action <= (self.T_target + self.T_acceptable_dev):
                if power_W == 0:
                    # This bonus should ideally make the net reward for idling at target slightly positive
                    # or less negative than continuing to apply power or cooling down.
                    reward += self.EFFICIENT_IDLING_BONUS
        
        # 6. Deadline Achievement Reward/Penalty (Dominant factor for success)
        if self.steps == self.target_deadline_step:
            if current_error_abs <= self.T_acceptable_dev:
                reward += self.SUCCESS_BONUS
            else:
                reward += self.FAILURE_PENALTY
        
        return reward

    def step(self, action: int):
        # (Same as your v3_fixed_corrected version - correctly passes temps to calculate_reward)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        self.total_energy_consumed_Joules += power_selected_W * self.dt
        
        temp_before_action = self.current_temp 
        T_current_for_physics = round(temp_before_action, 1)
        
        heat_loss_W = self.hA * (T_current_for_physics - self.ambient_temp)
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c)
        T_next_calculated = T_current_for_physics + self.dt * dTemp_dt
        temp_after_action = np.clip(T_next_calculated, 0.0, 150.0)
        self.current_temp = temp_after_action
        
        self.steps += 1
        
        current_step_reward = self.calculate_reward(temp_after_action, temp_before_action, power_selected_W)
        
        terminated = self.steps >= self.max_steps
        truncated = False

        temp_diff_obs = self.current_temp - self.T_target
        steps_diff_obs = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff_obs, steps_diff_obs], dtype=np.float32)
        
        info = {
            "temperature": self.current_temp,
            "power_W": power_selected_W,
            "total_energy_Wh": self.total_energy_consumed_Joules / 3600.0,
            "current_step": self.steps,
            "is_at_deadline": self.steps == self.target_deadline_step,
        }
        return current_state, current_step_reward, terminated, truncated, info

    def render(self, mode="human"): # No changes needed
        # ... (same as before) ...
        if mode == "human":
            deadline_status = " (DEADLINE!)" if self.steps == self.target_deadline_step else ""
            print(
                f"Step: {self.steps:3d}, Temp: {self.current_temp:6.2f}°C, "
                f"Pwr: {self.last_power:4d}W{deadline_status}"
            )

    def close(self): # No changes needed
        pass