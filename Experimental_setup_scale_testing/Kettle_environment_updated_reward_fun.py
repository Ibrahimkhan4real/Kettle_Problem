import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

class SimpleKettleEnv(gym.Env):
    """
    The original, normalised kettle environment.
    Reward signals are scaled to be approximately within (0, -1).
    """

    def __init__(self, alpha=0.000001, beta=0.01):
        super(SimpleKettleEnv, self).__init__()

        # === Environment Parameters ===
        self.initial_temp = 20.0
        self.target_temp = 100.0
        self.ambient_temp = 20.0
        self.max_temp = 150.0
        self.target_step = 200
        self.max_steps = 200
        self.time_tolerance = 1

        self.mass = 1.0
        self.specific_heat = 4184
        self.heat_loss_coeff = 5.0
        self.dt = 1.0

        self.alpha = alpha
        self.beta = beta

        # === Action and Observation Spaces ===
        self.actions = [0, 3000]
        self.action_space = spaces.Discrete(len(self.actions))
        low = np.array([0.0, 0.0, 0.0, -100.0], dtype=np.float32)
        high = np.array([150.0, 110.0, 60.0, self.max_steps], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.step_count = 0
        self.temp = self.initial_temp
        self.total_energy_used = 0
        self.last_action = 0 # Store last action for reward calculation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temp = self.initial_temp
        self.step_count = 0
        self.total_energy_used = 0.0
        self.last_action = 0
        return self._get_obs(), {}

    def _get_obs(self):
        steps_remaining = self.target_step - self.step_count
        return np.array([self.temp, self.target_temp, self.ambient_temp, steps_remaining], dtype=np.float32)

    def step(self, action):
        power = self.actions[action]
        self.step_count += 1
        self.last_action = action

        # === Physics ===
        self.total_energy_used += power * self.dt
        heat_loss = self.heat_loss_coeff * (self.temp - self.ambient_temp)
        net_power = power - heat_loss
        temp_change = (net_power * self.dt) / (self.mass * self.specific_heat)
        self.temp += temp_change
        self.temp = np.clip(self.temp, 0.0, self.max_temp)

        # === Reward and Termination ===
        reward, done = self.reward_function()
        
        truncated = False
        if self.step_count >= self.max_steps:
            done = True
            truncated = True

        return self._get_obs(), reward, done, truncated, {}

    def reward_function(self):
        done = False
        reward = 0.0
        
        power = self.actions[self.last_action]
        energy_used = power * self.dt
        reward += -self.alpha * energy_used

        if abs(self.step_count - self.target_step) <= self.time_tolerance or self.step_count >= self.max_steps:
            done = True
            temp_penalty = -self.beta * abs(self.target_temp - self.temp)
            reward += temp_penalty

        return reward, done

    def render(self, mode="human"):
        print(f"Step {self.step_count:03d} | Temp: {self.temp:.2f}°C")

    def close(self):
        pass

class SimpleKettleEnvUnscaled100(SimpleKettleEnv):
    """
    An unscaled version of the kettle environment with a 100x scale factor.
    """
    def __init__(self):
        super().__init__()
        self.alpha *= 100.0
        self.beta *= 100.0

class SimpleKettleEnvUnscaled10000(SimpleKettleEnv):
    """
    An unscaled version of the kettle environment with a 10,000x scale factor.
    """
    def __init__(self):
        super().__init__()
        self.alpha *= 10000.0
        self.beta *= 10000.0
