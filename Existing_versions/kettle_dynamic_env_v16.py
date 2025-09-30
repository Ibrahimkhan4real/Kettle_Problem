import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

__version__ = "v3_fixed"

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


        min_temp_diff = 0.0 - self.T_target
        max_temp_diff = 150.0 - self.T_target # Max possible temp - target
        low_obs = np.array([min_temp_diff, self.water_volume, self.ambient_temp, 0.0], dtype=np.float32)
        high_obs = np.array([max_temp_diff, self.water_volume, self.ambient_temp, float(max_steps)], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )
        

        self.current_temp = 0.0
        self.steps = 0
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

    def reset(self, seed=None, options=None): # Options removed for simplicity
        super().reset(seed=seed)
        self.current_temp = self.initial_temp
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        
        # FIXED: Calculate diffs here, at the moment of state creation.
        temp_diff = self.current_temp - self.T_target
        steps_diff = self.target_deadline_step - self.steps

        current_state = np.array([temp_diff, self.water_volume, self.ambient_temp, float(steps_diff)], dtype=np.float32)
        return current_state, {"initial_temperature": self.current_temp}

    def calculate_reward(self, T_next, power_W):
        # This function is logically sound, no changes needed.
        reward = 0.0
        temp_diff_abs_next = abs(T_next - self.T_target)
        DEADLINE_TEMP_BONUS = 100.0
        DEADLINE_TEMP_PENALTY = 100.0
        ENERGY_COST_FACTOR_AT_DEADLINE = 0.00001
        INTERMEDIATE_ENERGY_PENALTY_WEIGHT = 0.00225
        PROXIMITY_REWARD_FACTOR = 0.05
        OVERSHOOT_PENALTY_FACTOR = 0.1
        COLD_AND_OFF_PENALTY = 0.2
        if power_W > 0:
            max_power_action = self.actions[-1]
            if max_power_action > 0:
                normalized_power = power_W / max_power_action
                reward -= INTERMEDIATE_ENERGY_PENALTY_WEIGHT * normalized_power
        sigma_temp_guidance = 20.0
        reward += PROXIMITY_REWARD_FACTOR * np.exp(-(temp_diff_abs_next**2) / (2 * sigma_temp_guidance**2))
        if T_next > self.T_target + (2 * self.T_acceptable_dev) and power_W > 0:
            reward -= OVERSHOOT_PENALTY_FACTOR * (T_next - (self.T_target + self.T_acceptable_dev))
        if T_next < self.T_target - (15 * self.T_acceptable_dev) and power_W == 0 and self.steps < self.target_deadline_step:
            reward -= COLD_AND_OFF_PENALTY
        if self.steps == self.target_deadline_step:
            if temp_diff_abs_next <= self.T_acceptable_dev:
                reward += DEADLINE_TEMP_BONUS
                reward -= ENERGY_COST_FACTOR_AT_DEADLINE * self.total_energy_consumed_Joules
            else:
                penalty_scaled = min(1.0, temp_diff_abs_next / (10 * self.T_acceptable_dev))
                reward -= DEADLINE_TEMP_PENALTY * penalty_scaled
                reward -= (ENERGY_COST_FACTOR_AT_DEADLINE * self.total_energy_consumed_Joules) / 2.0
        return reward

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        
        self.total_energy_consumed_Joules += power_selected_W * self.dt
        
        # Using round() is a more standard way to handle floating point issues
        T_current = round(self.current_temp, 1)
        
        heat_loss_W = self.hA * (T_current - self.ambient_temp)
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c)
        
        T_next = T_current + self.dt * dTemp_dt
        
        # FIXED: Clipping is done to absolute temperature bounds, not observation space bounds.
        self.current_temp = np.clip(T_next, 0.0, 150.0)
        self.steps += 1
        
        current_step_reward = self.calculate_reward(self.current_temp, power_selected_W)
        
        terminated = self.steps >= self.max_steps
        truncated = False 

        # FIXED: Calculate diffs here, at the moment of state creation.
        temp_diff = self.current_temp - self.T_target
        steps_diff = self.target_deadline_step - self.steps

        current_state = np.array([temp_diff, self.water_volume, self.ambient_temp, float(steps_diff)], dtype=np.float32)
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
            # FIXED: Removed the undefined 'current_obs' variable for printing.
            print(
                f"Step: {self.steps:3d}, Temp: {self.current_temp:6.2f}°C, "
                f"Pwr: {self.last_power:4d}W{deadline_status}"
            )

    def close(self):
        pass

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    test_initial_temp = 15.0 
    test_ambient_temp = 20.0
    test_target_deadline_step = 60
    test_max_steps = test_target_deadline_step + 20

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

    log_steps, log_temps, log_powers, log_rewards = [], [], [], []

    # Simple deterministic policy for testing
    for step_count in range(env.max_steps):
        action_idx = 0 # Default off
        
        # FIXED: Test logic now correctly uses temp_diff from the observation
        temp_diff_from_obs = current_obs[0] # is now current_temp - target_temp
        
        if env.steps < env.target_deadline_step:
            if temp_diff_from_obs < -10:  # If more than 10C below target
                action_idx = env.num_actions - 1 # Max power
            elif temp_diff_from_obs < -2: # If less than 2C below target
                # FIXED: This action is now safe and won't go out of bounds
                action_idx = env.num_actions // 2 
            elif temp_diff_from_obs > env.T_acceptable_dev: # If overshot
                action_idx = 0 # Off if overshot
        else: # At or after deadline
            action_idx = 0 # Conserve energy

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
        temp_at_deadline = log_temps[env.target_deadline_step -1]
        energy_at_deadline_Wh = info["total_energy_Wh"]
        print(f"At Deadline (Step {env.target_deadline_step}): Temp={temp_at_deadline:.2f}°C, Total Energy Consumed: {energy_at_deadline_Wh:.3f} Wh")

    env.close()

    # --- Plotting (no changes needed here) ---
    if not os.path.exists("Graphs"): os.makedirs("Graphs")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:red'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(log_steps, log_temps, color=color, marker='.', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(env.T_target, color='gray', linestyle='--', label=f'Target Temp ({env.T_target}°C)')
    ax1.axvline(env.target_deadline_step, color='black', linestyle=':', label=f'Deadline (Step {env.target_deadline_step})')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Power (W)', color=color)
    ax2.step(log_steps, log_powers, color=color, where='post', alpha=0.7, label='Power Applied')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title(f"Simplified KettleEnv Test ({__version__})", pad=20)
    plot_filename = f"Graphs/KettleEnv_Test_{__version__}_Deadline{env.target_deadline_step}.png"
    plt.savefig(plot_filename, dpi=200)
    print(f"\nTest plot saved to {plot_filename}")
    
    end_time = time.perf_counter()
    print(f"\nKettleEnv test script finished in {end_time - start_time:.3f} seconds.")