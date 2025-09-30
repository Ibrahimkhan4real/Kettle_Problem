import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

__version__ = "v3_simplified_mcts"

class KettleEnv(gym.Env):
    """
    Simplified Kettle Gym Environment for MCTS.
    Focuses on reaching a target temperature at a specific time with energy efficiency.
    Reduced dynamism and parameters for easier understanding.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        water_volume=1.0,       # Liters
        initial_temp=20.0,      # Degrees C
        ambient_temp=25.0,      # Degrees C (fixed per episode)
        max_steps=250,          # Max steps in an episode
        target_deadline_step=200 # The critical step for evaluation
    ):
        super(KettleEnv, self).__init__()

        # Physical properties
        self.water_volume = water_volume
        self.m = self.water_volume  # mass in kg
        self.c = 4184.0  # Specific heat capacity of water (J/kg°C)
        self.hA = 1.0    # Combined heat transfer coefficient * Area (W/°C), simplified (h=4, A=0.25)

        self.T_target = 100  # Target temperature (°C)
        self.T_acceptable_dev = 1 # Acceptable deviation for being "at target"

        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp # Fixed ambient temperature

        self.dt = 1  # Time step in seconds

        # Actions: 0W (off) and power levels from 1500W to 3000W in 50W increments
        self.actions = [0] + list(range(1500, 3001, 500))
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: [temperature, water_volume, ambient_temperature, current_step]
        # Water volume & ambient_temp are constant within an episode in this simplified version
        low_obs = np.array([0.0, self.T_target, 0.0], dtype=np.float32)
        high_obs = np.array([150.0, self.T_target, float(max_steps)], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        self.current_temp = 0.0
        self.steps = 0
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        self.steps_diff = self.max_steps - self.steps
        
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        self.target_reached_ever = False # General status: was target ever reached

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

    def reset(self, seed=None, options=None): # Options removed for simplicity
        super().reset(seed=seed)
        self.current_temp = self.initial_temp
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        self.target_reached_ever = abs(self.current_temp - self.T_target) <= self.T_acceptable_dev
        
        # State: [temperature, water_volume, ambient_temperature, current_step]
        # water_volume and ambient_temp are fixed per episode in this version
        current_state = np.array([self.current_temp, self.T_target, float(self.steps_diff)], dtype=np.float32)
        return current_state, {"initial_temperature": self.current_temp}

    def calculate_reward(self, T_next, power_W):
        reward = 0.0
        temp_diff_abs_next = abs(T_next - self.T_target)

        # --- Reward Coefficients (can be tuned directly here) ---
        DEADLINE_TEMP_BONUS = 100.0
        DEADLINE_TEMP_PENALTY = 100.0
        ENERGY_COST_FACTOR_AT_DEADLINE = 0.00001 # Cost per Joule
        INTERMEDIATE_ENERGY_PENALTY_WEIGHT = 0.000519 # Per normalized Watt
        PROXIMITY_REWARD_FACTOR = 0.05
        OVERSHOOT_PENALTY_FACTOR = 0.1
        COLD_AND_OFF_PENALTY = 0.2

        # 1. Intermediate energy penalty (per step)
        if power_W > 0:
            max_power_action = self.actions[-1]
            if max_power_action > 0:
                 normalized_power = power_W / max_power_action
                 reward -= INTERMEDIATE_ENERGY_PENALTY_WEIGHT * normalized_power

        # 2. Intermediate proximity reward (Gaussian-like)
        sigma_temp_guidance = 20.0 # Wider sigma for gentler pull
        reward += PROXIMITY_REWARD_FACTOR * np.exp(-(temp_diff_abs_next**2) / (2 * sigma_temp_guidance**2))
        
        # 3. Intermediate penalty for significant overshooting
        if T_next > self.T_target + (2 * self.T_acceptable_dev) and power_W > 0:
            reward -= OVERSHOOT_PENALTY_FACTOR * (T_next - (self.T_target + self.T_acceptable_dev))
        
        # 4. Intermediate penalty for being cold, heater OFF, and before deadline
        if T_next < self.T_target - (15 * self.T_acceptable_dev) and power_W == 0 and self.steps < self.target_deadline_step:
            reward -= COLD_AND_OFF_PENALTY

        # 5. Terminal reward/penalty at the target_deadline_step
        if self.steps == self.target_deadline_step:
            if temp_diff_abs_next <= self.T_acceptable_dev:
                reward += DEADLINE_TEMP_BONUS
                reward -= ENERGY_COST_FACTOR_AT_DEADLINE * self.total_energy_consumed_Joules
            else:
                penalty_scaled = min(1.0, temp_diff_abs_next / (10 * self.T_acceptable_dev))
                reward -= DEADLINE_TEMP_PENALTY * penalty_scaled
                # Still penalize energy, perhaps less if target missed
                reward -= (ENERGY_COST_FACTOR_AT_DEADLINE * self.total_energy_consumed_Joules) / 2.0
        return reward

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        
        energy_this_step_J = power_selected_W * self.dt
        self.total_energy_consumed_Joules += energy_this_step_J

        T_current = float(math.ceil(self.current_temp*10)/10)
        
        # Physics: dQ/dt = P_in - P_out
        # P_out = hA * (T_water - T_env)
        # m * c * dT/dt = P_in - hA * (T_water - T_env)
        # dT/dt = (P_in - hA * (T_water - T_env)) / (m * c)
        heat_loss_W = self.hA * (T_current - self.ambient_temp)
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c)
        
        T_next = T_current + self.dt * dTemp_dt
        T_next = np.clip(T_next, self.observation_space.low[0], self.observation_space.high[0])
        T_next = float(math.ceil(T_next*10)/10)
        
        self.current_temp = T_next
        self.steps += 1
        
        current_step_reward = self.calculate_reward(T_next, power_selected_W)
        
        terminated = self.steps >= self.max_steps
        truncated = False 

        if not self.target_reached_ever and abs(T_next - self.T_target) <= self.T_acceptable_dev:
            self.target_reached_ever = True

        current_state = np.array([self.current_temp, self.T_target, float(self.steps_diff)], dtype=np.float32)
        info = {
            "temperature": self.current_temp,
            "power_W": power_selected_W,
            "total_energy_Wh": self.total_energy_consumed_Joules / 3600,
            "current_step": self.steps,
            "is_at_deadline": self.steps == self.target_deadline_step,
        }
        return current_state, current_step_reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            deadline_status = " (DEADLINE!)" if self.steps == self.target_deadline_step else ""
            print(
                f"Step: {self.steps:3d}, Temp: {self.current_temp:6.2f}°C, statespace: {current_obs} "
                f"Pwr: {self.last_power:4d}W{deadline_status}"
            )

    def close(self):
        pass

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    # --- Simplified Test Parameters ---
    test_initial_temp = 15.0 
    test_ambient_temp = 20.0
    test_target_deadline_step = 60  # 1 minute for quicker test
    test_max_steps = test_target_deadline_step + 20 # Test a bit beyond deadline

    print(f"--- Testing KettleEnv ({__version__}) ---")
    env = KettleEnv(
        initial_temp=test_initial_temp,
        ambient_temp=test_ambient_temp,
        max_steps=test_max_steps,
        target_deadline_step=test_target_deadline_step,
    )
    
    print(f"Target: {env.T_target}°C @ Step {env.target_deadline_step} (Max Steps: {env.max_steps})")
    print(f"Initial: {env.initial_temp}°C, Ambient: {env.ambient_temp}°C")


    current_obs, info = env.reset() 
    terminated = False
    total_reward_acc = 0.0

    log_steps = []
    log_temps = []
    log_powers = []
    log_rewards = []

    # Simple deterministic policy for testing: heat until near target, then try to hold/off
    for step_count in range(env.max_steps):
        action_idx = 0 # Default off
        current_temp_from_obs = current_obs[0] # Temp is the first element of state
        
        if env.steps < env.target_deadline_step:
            if current_temp_from_obs < env.T_target - 10:
                action_idx = env.num_actions - 1 # Max power
            elif current_temp_from_obs < env.T_target - 2:
                 action_idx = env.num_actions // 2 + 5 # Medium-high power
            elif current_temp_from_obs > env.T_target + env.T_acceptable_dev:
                action_idx = 0 # Off if overshot
        elif env.steps == env.target_deadline_step:
             action_idx = 0 # Turn off at deadline if at target, or try to correct
             if not (env.T_target - env.T_acceptable_dev <= current_temp_from_obs <= env.T_target + env.T_acceptable_dev):
                 action_idx = env.num_actions -1 if current_temp_from_obs < env.T_target else 0 # one last push if cold
        else: # Post-deadline
            action_idx = 0 # Conserve

        current_obs, reward, terminated, truncated, info = env.step(action_idx)
        total_reward_acc += reward

        log_steps.append(info["current_step"])
        log_temps.append(info["temperature"])
        log_powers.append(info["power_W"])
        log_rewards.append(reward)

        if info["current_step"] % 10 == 0 or terminated or info["is_at_deadline"]:
            env.render()
            print(f"  Action: {env.actions[action_idx]}W, Step R: {reward:.3f}, Total R: {total_reward_acc:.2f}\n")
        
        if terminated or truncated:
            break
            
    print(f"\nEpisode finished after {info['current_step']} steps. Total Reward: {total_reward_acc:.2f}")
    if info["current_step"] >= env.target_deadline_step:
        temp_at_deadline = log_temps[env.target_deadline_step -1] # -1 because steps are 1-indexed in log
        energy_at_deadline = env.total_energy_consumed_Joules # Assuming this is the final total if episode ends at/after deadline
        
        print(f"At Deadline (Step {env.target_deadline_step}): Temp={temp_at_deadline:.2f}°C, Total Energy Consumed: {energy_at_deadline/3600:.3f} Wh")


    env.close()

    # Simplified Plotting
    if not os.path.exists("Graphs"): os.makedirs("Graphs")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:red'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(log_steps, log_temps, color=color, marker='.', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(env.T_target, color='gray', linestyle='--', label=f'Target Temp ({env.T_target}°C)')
    ax1.axvline(env.target_deadline_step, color='black', linestyle=':', label=f'Deadline (Step {env.target_deadline_step})')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Power (W)', color=color)
    ax2.step(log_steps, log_powers, color=color, where='post', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"Simplified KettleEnv Test ({__version__})", pad=20)
    ax1.legend(loc='upper left')
    ax2.legend(['Power Applied'], loc='upper right')
    plot_filename = f"Graphs/KettleEnv_Test_{__version__}_Deadline{env.target_deadline_step}.png"
    plt.savefig(plot_filename, dpi=200)
    print(f"\nTest plot saved to {plot_filename}")
    
    end_time = time.perf_counter()
    print(f"\nKettleEnv test script finished in {end_time - start_time:.3f} seconds.")