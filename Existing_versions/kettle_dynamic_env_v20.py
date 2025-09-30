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
        self.hA = 1.0

        self.T_target = 100.0
        self.T_acceptable_dev = 1.0

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

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

        # Using the "Balanced for Optimal & Efficient Action" hyperparameters
        self.SUCCESS_BONUS = 10000.0
        self.FAILURE_PENALTY = -100.0
        self.PER_STEP_POWER_COST_WEIGHT = 0.00005
        self.TEMP_AREA_PENALTY_WEIGHT = 0.00005
        self.EFFICIENT_IDLING_BONUS = 0.42     
        self.TEMP_PROGRESS_WEIGHT = 0.15
        self.OVERSHOOT_PENALTY_WEIGHT = 0.05

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
        # This function expects T_after_action and T_before_action to represent the state
        # at 1 decimal place precision, as per the model's view of the state.
        reward = 0.0
        current_error_abs = abs(T_after_action_1dp - self.T_target)

        max_power_possible = float(self.actions[-1]) if self.actions and len(self.actions) > 0 else 3000.0
        if max_power_possible == 0 and len(self.actions) > 1:
            positive_powers = [p for p in self.actions if p > 0]
            if positive_powers: max_power_possible = float(max(positive_powers))
            else: max_power_possible = 3000.0 
        normalized_power = float(power_W) / max_power_possible if max_power_possible > 0 else 0.0
        reward -= self.PER_STEP_POWER_COST_WEIGHT * normalized_power

        if T_after_action_1dp > self.ambient_temp:
            reward -= self.TEMP_AREA_PENALTY_WEIGHT * (T_after_action_1dp - self.ambient_temp)

        previous_error_abs = abs(T_before_action_1dp - self.T_target)
        error_delta = previous_error_abs - current_error_abs 
        reward += self.TEMP_PROGRESS_WEIGHT * error_delta 

        if T_after_action_1dp > self.T_target + self.T_acceptable_dev:
            overshoot_amount = T_after_action_1dp - (self.T_target + self.T_acceptable_dev)
            reward -= self.OVERSHOOT_PENALTY_WEIGHT * (overshoot_amount / 10.0) 

        if self.steps < self.target_deadline_step:
            if (self.T_target - self.T_acceptable_dev) <= T_after_action_1dp <= (self.T_target + self.T_acceptable_dev):
                if power_W == 0:
                    reward += self.EFFICIENT_IDLING_BONUS
        
        if self.steps == self.target_deadline_step:
            if current_error_abs <= self.T_acceptable_dev:
                reward += self.SUCCESS_BONUS
            else:
                reward += self.FAILURE_PENALTY
        
        return reward

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        self.total_energy_consumed_Joules += power_selected_W * self.dt
        
        # temp_before_action_precise holds the internal state before this step's physics.
        # If self.current_temp is always rounded at end of step, this will be 1 d.p.
        # However, to ensure the dTemp/dt calculation is as sensitive as possible,
        # we should use the internal, potentially more precise, self.current_temp if it wasn't already rounded.
        # To be explicit: the state *is* 1 d.p. So physics should evolve from that 1 d.p. state.
        # The original issue was `round(possibly_precise_temp, 1)` causing the *difference* to become zero.
        # If `self.current_temp` *is* the 1 d.p. state, then `temp_before_action_1dp` IS `self.current_temp`.

        temp_before_action_1dp = self.current_temp # This is the state (1 d.p. from previous step or reset)
        
        # Physics calculation uses this 1 d.p. state representation as its starting point.
        # The heat loss is calculated from this state.
        heat_loss_W = self.hA * (temp_before_action_1dp - self.ambient_temp)
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c) # dTemp_dt can be small & precise
        
        # The change is applied to the 1 d.p. starting temperature.
        T_next_calculated_precise = temp_before_action_1dp + self.dt * dTemp_dt
        
        temp_after_action_clipped = np.clip(T_next_calculated_precise, 0.0, 150.0)
        
        # Update self.current_temp to be stored with 1 decimal place precision
        self.current_temp = round(temp_after_action_clipped, 1) 
        
        self.steps += 1
        
        # Reward is calculated based on the 1 d.p. state before and after the action
        current_step_reward = self.calculate_reward(self.current_temp, temp_before_action_1dp, power_selected_W)
        
        terminated = self.steps >= self.max_steps
        truncated = False

        temp_diff_obs = self.current_temp - self.T_target # Based on 1 d.p. current_temp
        steps_diff_obs = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff_obs, steps_diff_obs], dtype=np.float32)
        
        info = {
            "temperature": self.current_temp, # This is now the 1 d.p. state value
            "power_W": power_selected_W,
            "total_energy_Wh": self.total_energy_consumed_Joules / 3600.0,
            "current_step": self.steps,
            "is_at_deadline": self.steps == self.target_deadline_step,
        }
        return current_state, current_step_reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            deadline_status = " (DEADLINE!)" if self.steps == self.target_deadline_step else ""
            print(
                f"Step: {self.steps:3d}, Temp: {self.current_temp:.1f}°C, " # Already formatted to .1f
                f"Pwr: {self.last_power:4d}W{deadline_status}"
            )

    def close(self):
        pass