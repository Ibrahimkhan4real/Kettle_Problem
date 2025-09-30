import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import os
# import math # Not explicitly used, can be removed if not needed for other parts

__version__ = "v3_fixed_corrected" # Updated version to reflect these fixes

class KettleEnv(gym.Env):
    """
    Simplified Kettle Gym Environment for MCTS.
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
        self.m = self.water_volume      # mass in kg (assuming water density 1kg/L)
        self.c = 4184.0                 # Specific heat capacity of water (J/kg°C)
        self.hA = 1.0                   # Combined heat transfer coefficient * Area (W/°C)

        self.T_target = 100.0           # Target temperature (°C)
        self.T_acceptable_dev = 1.0     # Acceptable deviation for being "at target"

        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp

        self.dt = 1.0                   # Time step in seconds

        # Actions: 0W (off) and power levels from 1500W to 3000W in 500W increments
        self.actions = [0] + list(range(1500, 3001, 500))
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: [temp_diff_to_target, water_volume, ambient_temp, steps_to_deadline]
        # temp_diff = current_temp - T_target
        min_temp_diff = 0.0 - self.T_target   # Min possible temp (0C) - target
        max_temp_diff = 150.0 - self.T_target # Max allowed temp (150C) - target

        # steps_diff = target_deadline_step - current_steps
        # Min value: target_deadline_step - max_steps (can be negative if max_steps > target_deadline_step)
        # Max value: target_deadline_step (at step 0)
        min_steps_diff = float(target_deadline_step - max_steps)
        max_steps_diff = float(target_deadline_step)

        low_obs = np.array([min_temp_diff, min_steps_diff], dtype=np.float32)
        high_obs = np.array([max_temp_diff, max_steps_diff], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )
        
        self.current_temp = 0.0 # Will be set in reset
        self.steps = 0          # Will be set in reset
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        
        self.last_power = 0     # Will be set in reset
        self.total_energy_consumed_Joules = 0.0 # Will be set in reset

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = self.initial_temp
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        
        temp_diff = self.current_temp - self.T_target
        steps_diff = float(self.target_deadline_step - self.steps) # Ensure float for consistency

        current_state = np.array([temp_diff, steps_diff], dtype=np.float32)
        return current_state, {"initial_temperature": self.current_temp}

    def calculate_reward(self, T_after_action, power_W):
        # Hyperparameters for the reward function
        PER_STEP_POWER_COST_WEIGHT = 0.01
        SUCCESS_BONUS = 100.0
        FAILURE_PENALTY = -50.0
        
        # Determine max power from actions for normalization; provide a fallback if actions list is unusual.
        if self.actions and self.actions[-1] > 0:
            max_power_val = float(self.actions[-1])
        else: # Fallback if actions list is empty or max power is 0 (should not happen in normal use)
            max_power_val = 3000.0 
            if self.actions and len(self.actions) > 1 and self.actions[-1] == 0: # e.g. if actions = [0, 500, 0] incorrectly
                # Try to find a positive max power if available
                positive_powers = [p for p in self.actions if p > 0]
                if positive_powers:
                    max_power_val = float(max(positive_powers))

        normalized_power = float(power_W) / max_power_val if max_power_val > 0 else 0.0
        reward = - (PER_STEP_POWER_COST_WEIGHT * normalized_power)

        if self.steps == self.target_deadline_step:
            final_temp_error = abs(T_after_action - self.T_target)
            if final_temp_error <= self.T_acceptable_dev:
                reward += SUCCESS_BONUS
            else:
                reward += FAILURE_PENALTY
        return reward

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        
        self.total_energy_consumed_Joules += power_selected_W * self.dt
        
        # temp_before_action = self.current_temp # For clarity if needed by a more complex reward
        T_current_for_physics = round(self.current_temp, 1) # Temp at start of this step's physics calc
        
        heat_loss_W = self.hA * (T_current_for_physics - self.ambient_temp)
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c)
        
        T_next_calculated = T_current_for_physics + self.dt * dTemp_dt
        
        self.current_temp = np.clip(T_next_calculated, 0.0, 150.0) # Update system state
        self.steps += 1
        
        # The reward is calculated based on the temperature *after* the step (self.current_temp)
        current_step_reward = self.calculate_reward(self.current_temp, power_selected_W)
        
        terminated = self.steps >= self.max_steps
        truncated = False # Not using truncation for early stops other than max_steps

        temp_diff_obs = self.current_temp - self.T_target
        steps_diff_obs = float(self.target_deadline_step - self.steps) # Ensure float

        current_state = np.array([temp_diff_obs, steps_diff_obs], dtype=np.float32)
        info = {
            "temperature": self.current_temp,
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
                f"Step: {self.steps:3d}, Temp: {self.current_temp:6.2f}°C, "
                f"Pwr: {self.last_power:4d}W{deadline_status}"
            )

    def close(self):
        pass

# --- Testing Script (with minor refinement for energy logging) ---
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
    print(f"Obs Space Low: {env.observation_space.low}, High: {env.observation_space.high}") # Print obs space for verification
    print(f"Initial: {env.initial_temp}°C, Ambient: {env.ambient_temp}°C")
    print(f"Actions: {env.actions}")


    current_obs, info = env.reset() 
    terminated = False
    truncated = False # Explicitly track truncated
    total_reward_acc = 0.0

    log_steps, log_temps, log_powers, log_rewards, log_energy_Wh = [], [], [], [], []

    for step_count in range(env.max_steps):
        action_idx = 0 
        
        temp_diff_from_obs = current_obs[0]
        
        if env.steps < env.target_deadline_step:
            if temp_diff_from_obs < -10:
                action_idx = env.num_actions - 1
            elif temp_diff_from_obs < -2:
                action_idx = env.num_actions // 2 
            elif temp_diff_from_obs > env.T_acceptable_dev:
                action_idx = 0
        else: 
            action_idx = 0

        current_obs, reward, terminated, truncated, info = env.step(action_idx)
        total_reward_acc += reward

        log_steps.append(info["current_step"])
        log_temps.append(info["temperature"])
        log_powers.append(info["power_W"])
        log_rewards.append(reward)
        log_energy_Wh.append(info["total_energy_Wh"]) # Log energy at each step

        if info["current_step"] % 10 == 0 or terminated or truncated or info["is_at_deadline"]:
            env.render()
            print(f"  Obs: {[f'{x:.2f}' for x in current_obs]}, Action: {env.actions[action_idx]}W, Step R: {reward:.3f}, Total R: {total_reward_acc:.2f}\n")
        
        if terminated or truncated:
            break
            
    print(f"\nEpisode finished after {info['current_step']} steps. Total Reward: {total_reward_acc:.2f}")
    if info["current_step"] >= env.target_deadline_step:
        # Ensure deadline_step is a valid index for logs (step numbers are 1-based in log_steps)
        deadline_log_idx = env.target_deadline_step - 1 
        if 0 <= deadline_log_idx < len(log_temps):
            temp_at_deadline = log_temps[deadline_log_idx]
            energy_at_deadline_Wh = log_energy_Wh[deadline_log_idx] # Use logged energy
            print(f"At Deadline (Step {env.target_deadline_step}): Temp={temp_at_deadline:.2f}°C, Total Energy Consumed: {energy_at_deadline_Wh:.3f} Wh")
        else:
            print(f"Deadline step {env.target_deadline_step} not reached or log index out of bounds.")


    env.close()

    # --- Plotting ---
    plot_dir = "Graphs_KettleEnv" # Changed directory name slightly
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    
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
    plot_filename = os.path.join(plot_dir, f"KettleEnv_Test_{__version__}_Deadline{env.target_deadline_step}.png")
    plt.savefig(plot_filename, dpi=200)
    print(f"\nTest plot saved to {plot_filename}")
    
    end_time = time.perf_counter()
    print(f"\nKettleEnv test script finished in {end_time - start_time:.3f} seconds.")