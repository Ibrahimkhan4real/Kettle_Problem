import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleKettleEnv(gym.Env):
    """
    A simplified and readable kettle environment.
    Goal: Reach target temperature at the target time while minimizing energy use.
    """

    def __init__(self):
        super(SimpleKettleEnv, self).__init__()

        # === Environment Parameters ===
        self.initial_temp = 20.0           # Starting temperature in °C
        self.target_temp = 100.0           # Target temperature in °C
        self.ambient_temp = 20.0           # Ambient temperature
        self.max_temp = 150.0              # Physical upper limit
        self.target_step = 150              # Desired step to reach target temperature
        self.max_steps = 200               # Max steps per episode
        self.temp_tolerance = 2.0          # Acceptable range around target temperature
        self.time_tolerance = 5            # Acceptable time window around target step

        self.mass = 1.0                    # Mass of water in kg
        self.specific_heat = 4184          # J/kg·K
        self.heat_loss_coeff = 5.0         # W/°C (simplified linear cooling)
        self.dt = 1.0                      # Timestep duration in seconds
        self.shaping_gemma = 1          # Reward shaping factor

        # === Action Space ===
        self.actions = [0, 3000]           # Discrete: 0W (off), 3000W (on)
        self.action_space = spaces.Discrete(len(self.actions))

        # === Observation Space ===
        # Observation: [current_temperature, steps_remaining]
        low = np.array([0.0, 0.0, 0.0, -100.0, 0.0], dtype=np.float32)
        high = np.array([150.0, 110.0, 60.0, self.max_steps, 3000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temp = self.initial_temp
        self.step_count = 0
        self.total_energy_used = 0.0
        #print("-------------------------------------------------------------------")

        return self._get_obs(), {}

    def _get_obs(self): # Observation space inlcudes current temperature, steps remaining, and total energy used
        steps_remaining = self.target_step - self.step_count
        return np.array([self.temp, self.target_temp, self.ambient_temp, steps_remaining, self.total_energy_used], dtype=np.float32)

    def step(self, action):
        power = self.actions[action]
        self.step_count += 1
        previous_temp = self.temp

        # === Energy used this step ===
        energy_joules = power * self.dt
        self.total_energy_used += energy_joules

        # === Heat loss and temperature change ===
        heat_loss = self.heat_loss_coeff * (self.temp - self.ambient_temp)
        net_power = power - heat_loss
        temp_change = (net_power * self.dt) / (self.mass * self.specific_heat)
        self.temp += temp_change
        self.temp = np.clip(self.temp, 0.0, self.max_temp)

        # === Reward Function ===
        reward = 0.0
        done = False

        # === Proximity bonus (REWARD SHAPING) ===#
        phi_before = -abs(previous_temp - self.target_temp)
        phi_after = -abs(self.temp - self.target_temp)
        reward += self.shaping_gemma * phi_after - phi_before

        

        # 1. Goal reward
        if abs(self.temp - self.target_temp) <= self.temp_tolerance:
            if abs(self.step_count - self.target_step) <= self.time_tolerance:
                reward += 1000.0  # Successfully hit goal
                done = True

        # 2. Energy penalty
        reward -= energy_joules / 6000  # Convert to kJ

        # # 3. Distance-to-goal penalty (optional, helps early learning)
        # temp_error = abs(self.temp - self.target_temp)
        # time_error = abs(self.step_count - self.target_step)
        # reward -= 0.1 * (temp_error + time_error)

        # 4. Episode termination
        if self.step_count >= self.max_steps:
            done = True
        
        #print(f"Step {self.step_count:03d} | Difference in Phi: {self.shaping_gemma * phi_after - phi_before:.2f} | ")
        #print(f"Step {self.step_count}: Action={action} (Power: {power}W), Temp: {self.temp:.2f}°C, Reward: {reward:.2f}, Total Energy: {self.total_energy_used / 3600:.2f} Wh")
        

        return self._get_obs(), reward, done, False, {
            "temperature": self.temp,
            "step": self.step_count,
            "power_used": power,
            "energy_kJ": self.total_energy_used / 1000,
        }

    def render(self, mode="human"):
        print(f"Step {self.step_count:03d} | Temp: {self.temp:.2f}°C | Energy: {self.total_energy_used / 3600:.2f} Wh")

    def close(self):
        pass
