import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv
import os

class SimpleKettleEnv(gym.Env):
    """
    A simplified and readable kettle environment with continuous action space.
    Goal: Reach target temperature at the target time while minimizing energy use.
    """

    def __init__(self):
        super(SimpleKettleEnv, self).__init__()

        # === Environment Parameters ===
        self.initial_temp = 20.0           # Starting temperature in °C
        self.target_temp = 100.0           # Target temperature in °C
        self.ambient_temp = 20.0           # Ambient temperature
        self.max_temp = 150.0              # Physical upper limit
        self.target_step = 200             # Desired step to reach target temperature
        self.max_steps = 200               # Max steps per episode
        self.temp_tolerance = 2.0          # Acceptable range around target temperature
        self.time_tolerance = 1            # Acceptable time window around target step

        self.mass = 1.0                    # Mass of water in kg
        self.specific_heat = 4184          # J/kg·K
        self.heat_loss_coeff = 5.0         # W/°C (simplified linear cooling)
        self.dt = 1.0                      # Timestep duration in seconds
        self.shaping_gemma = 1             # Reward shaping factor

        self.alpha = 0.000001  # Energy penalty weight
        self.beta = 0.01       # Temperature penalty weight

        # === Action Space ===
        # Continuous: any heating power between 0 and 3000W
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([3000.0]), dtype=np.float32)

        # === Observation Space ===
        # Observation: [current_temperature, target_temperature, ambient_temperature, steps_remaining]
        low = np.array([0.0, 0.0, 0.0, -100.0], dtype=np.float32)
        high = np.array([150.0, 110.0, 60.0, self.max_steps], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # === Logging Setup ===
        os.makedirs("logs", exist_ok=True)
        self.log_path = "logs/kettle_log.csv"
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Step", "Action (W)", "Temperature (°C)", "Reward", "Total Energy (Wh)", "Total Reward"])

        self.episode_counter = 0
        self.episode_total_reward = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temp = self.initial_temp
        self.step_count = 0
        self.total_energy_used = 0.0
        self.episode_counter += 1
        self.episode_total_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        """Observation includes current temperature, target temperature, ambient temperature, and steps remaining."""
        steps_remaining = self.target_step - self.step_count
        return np.array([self.temp, self.target_temp, self.ambient_temp, steps_remaining], dtype=np.float32)

    def step(self, action):
        state = self._get_obs()

        # Ensure action is valid (continuous within bounds)
        power = float(np.clip(action, self.action_space.low, self.action_space.high)[0])
        self.step_count += 1
        previous_temp = self.temp

        # === Energy used this step ===
        self.energy_joules = power * self.dt
        self.total_energy_used += self.energy_joules

        # === Heat loss and temperature change ===
        heat_loss = self.heat_loss_coeff * (self.temp - self.ambient_temp)
        net_power = power - heat_loss
        temp_change = (net_power * self.dt) / (self.mass * self.specific_heat)
        self.temp += temp_change
        self.temp = np.clip(self.temp, 0.0, self.max_temp)

        next_state = self._get_obs()
        reward, done = self.reward_function(state, power, next_state)
        self.episode_total_reward += reward

        # === Logging ===
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_counter,
                self.step_count,
                f"{power:.4f}",
                f"{self.temp:.10f}",
                f"{reward:.10f}",
                f"{self.total_energy_used / 3600.0:.10f}",  # Joules → Wh
                f"{self.episode_total_reward:.10f}"
            ])
        

        if np.isnan(reward) or np.any(np.isnan(self._get_obs())):
            print(f"NaN detected in reward or observation: reward={reward}, obs={self._get_obs()}")
        
        return self._get_obs(), reward, done, False, {
            "temperature": self.temp,
            "step": self.step_count,
            "power_used": power,
            "energy_kJ": self.total_energy_used / 1000,
        }

    def reward_function(self, state, power, next_state):
        done = False
        reward = 0.0

        current_temp, target_temp, ambient_temp, steps_remaining = state
        next_temp, _, _, next_steps_remaining = next_state

        # Energy penalty
        energy_used = power * self.dt
        reward += -self.alpha * energy_used

        # Temperature penalty if at/near target step
        if abs(self.step_count - self.target_step) <= self.time_tolerance:
            reward += -self.beta * abs(target_temp - next_temp)
            done = True
    
        # End if max steps reached
        if self.step_count >= self.max_steps:
            if not done:
                temp_penalty = -self.beta * abs(self.target_temp - next_state[0])
                reward += temp_penalty
            done = True

        return reward, done

    def render(self, mode="human"):
        print(f"Step {self.step_count:03d} | Temp: {self.temp:.2f}°C | Energy: {self.total_energy_used / 3600:.2f} Wh")

    def close(self):
        pass
