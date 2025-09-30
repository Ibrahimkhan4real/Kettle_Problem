import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import matplotlib.pyplot as plt # Not used directly in KettleEnv class
# import time # Not used directly in KettleEnv class
# import os # Not used directly in KettleEnv class
# import math # Not explicitly used

__version__ = "v3_smart_action_ دقیق_physics" # More precise physics calculation

class KettleEnv(gym.Env):
    """
    Kettle Gym Environment with:
    - Reward function balanced for optimal and efficient action.
    - Physics calculation sensitive to smaller temperature differences.
    - State temperature self.current_temp maintained at 1 decimal place.
    Observation space is [temp_diff, steps_diff].
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        water_volume=1.0,
        initial_temp=20.0,
        ambient_temp=20.0, 
        max_steps=250,
        target_deadline_step=200
    ):
        super(KettleEnv, self).__init__()

        self.water_volume = water_volume
        self.m = self.water_volume
        self.c = 4184.0
        self.hA = 5.0

        self.T_target = 100.0
        self.T_acceptable_dev = 2.5
        self.time_tolerance = 15

        self.initial_temp = initial_temp # Can be more precise initially
        self.ambient_temp = ambient_temp 
        self.dt = 1.0

        #self.actions = [0] + list(range(1500, 3001, 500))
        self.actions = [0, 3000]
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)

        min_temp_diff = 0.0 - self.T_target
        max_temp_diff = 150.0 - self.T_target # Max physical temp, not necessarily max state temp
        min_steps_diff = float(target_deadline_step - max_steps)
        max_steps_diff = float(target_deadline_step)
        low_obs = np.array([min_temp_diff, min_steps_diff], dtype=np.float32)
        high_obs = np.array([max_temp_diff, max_steps_diff], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        self.current_temp = 0.0 # Will be rounded to 1 d.p.
        self.steps = 0
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        
        self.goal_achieved_this_episode = False

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Ensure self.current_temp starts with 1 decimal place precision
        self.current_temp = round(self.initial_temp, 1) 
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        
        temp_diff = self.current_temp - self.T_target # Based on 1 d.p. current_temp
        steps_diff = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff, steps_diff], dtype=np.float32)
        return current_state, {"initial_temperature": self.initial_temp, "current_temperature": self.current_temp}

    def calculate_reward(self, T_after_action_1dp, T_before_action_1dp, power_W):
        
        MINIMISATION_COMPONENT = 0.002
        
        reward = - MINIMISATION_COMPONENT * (self.current_temp - self.initial_temp)
        
        if abs(self.current_temp - self.T_target) <= self.T_acceptable_dev:
            if abs(self.steps - self.target_deadline_step) <= self.time_tolerance:
                reward += 3000
        
        return reward

    # In your KettleEnv class:

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        self.total_energy_consumed_Joules += power_selected_W * self.dt
        
        temp_before_action_1dp = self.current_temp # This is the state (1 d.p. from previous step or reset)
        
        # Calculate physical heat loss based on the 1 d.p. starting temperature
        heat_loss_W = self.hA * (temp_before_action_1dp - self.ambient_temp)
        
        # Calculate the physical rate of temperature change
        dTemp_dt_physical = (power_selected_W - heat_loss_W) / (self.m * self.c)
        
        # --- MODIFICATION START: Ensure visible cooling if applicable ---
        dTemp_dt_effective = dTemp_dt_physical

        # --- MODIFICATION END ---
        
        T_next_calculated_precise = temp_before_action_1dp + self.dt * dTemp_dt_effective
        
        temp_after_action_clipped = np.clip(T_next_calculated_precise, 0.0, 150.0)
        
        # Update self.current_temp to be stored with 1 decimal place precision
        self.current_temp = round(temp_after_action_clipped, 1) 
        
        self.steps += 1
        
        current_step_reward = self.calculate_reward(self.current_temp, temp_before_action_1dp, power_selected_W)
        
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
             # You could add dTemp_dt_physical and dTemp_dt_effective to info for debugging
            "dTemp_dt_physical": dTemp_dt_physical,
            "dTemp_dt_effective": dTemp_dt_effective
        }
        return current_state, current_step_reward, terminated, truncated, info

    # The __init__, reset, calculate_reward, render, and close methods would remain as they were
    # in your "v3_smart_action_دقیق_physics" version with the tolerance updates,
    # assuming the reward hyperparameters there are what you intend to use.
    # I'm only showing the modified step function as per your direct request.

    def render(self, mode="human"):
        if mode == "human":
            deadline_status = " (DEADLINE!)" if self.steps == self.target_deadline_step else ""
            print(
                f"Step: {self.steps:3d}, Temp: {self.current_temp:.1f}°C, " # Already formatted to .1f
                f"Pwr: {self.last_power:4d}W{deadline_status}"
            )

    def close(self):
        pass