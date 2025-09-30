import gymnasium as gym
from gymnasium import spaces
import numpy as np

__version__ = "v26_policy_invariant_reward_shaping"

class KettleEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        water_volume=1.0,
        initial_temp=20.0,
        ambient_temp=20.0, 
        max_steps=250,
        target_deadline_step=200,
        # MDP parameters
        gamma=0.99,                      # Discount factor (must be in [0,1) for policy invariance)
        # Base reward parameters (sparse)
        goal_reward=1000.0,              # Sparse reward for achieving goal
        energy_cost_per_kJ=0.01,         # Cost per kJ of energy used
        efficiency_bonus_scale=200.0,    # Bonus for energy efficiency at goal
        # Potential function parameters
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

        # MDP and reward shaping parameters
        self.gamma = gamma
        assert 0 <= self.gamma < 1, "Discount factor must be in [0,1) for policy invariance"
        
        # Base reward parameters
        self.goal_reward = goal_reward
        self.energy_cost_per_kJ = energy_cost_per_kJ
        self.efficiency_bonus_scale = efficiency_bonus_scale


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
        self.first_goal_step = None
        
        # Calculate theoretical minimum energy and optimal heating time
        self.theoretical_min_energy = self.m * self.c * (self.T_target - self.initial_temp)
        
        # Approximate heating time considering heat loss
        avg_temp_during_heating = (self.initial_temp + self.T_target) / 2
        avg_heat_loss = self.hA * (avg_temp_during_heating - self.ambient_temp)
        net_heating_power = self.actions[1] - avg_heat_loss
        self.approx_heating_time = (self.T_target - self.initial_temp) * self.m * self.c / net_heating_power

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

    def calculate_base_reward(self, state_before, state_after, action):
        temp_after, step_after, energy_after = state_after
        
        reward = 0.0
        
        # Energy cost (continuous small penalty)
        power_W = self.actions[action]
        energy_cost = -self.energy_cost_per_kJ * (power_W * self.dt / 1000)
        reward += energy_cost
        
        # Goal achievement (sparse large reward)
        temp_error = abs(temp_after - self.T_target)
        if not self.goal_achieved_this_episode:
            if temp_error <= self.T_acceptable_dev:
                time_error = abs(step_after - self.target_deadline_step)
                if time_error <= self.time_tolerance:
                    # Goal achieved!
                    reward += self.goal_reward
                    self.goal_achieved_this_episode = True
                    self.first_goal_step = step_after
                    
                    # Efficiency bonus
                    if energy_after > 0:
                        efficiency = self.theoretical_min_energy / energy_after
                        efficiency_bonus = self.efficiency_bonus_scale * min(efficiency, 1.0)
                        reward += efficiency_bonus
        
        return reward

    def reset(self, seed=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_temp = round(self.initial_temp, 1) 
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        self.goal_achieved_this_episode = False
        self.first_goal_step = None
        
        # Create observation
        temp_diff = self.current_temp - self.T_target
        steps_diff = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff, steps_diff], dtype=np.float32)
        
        # Calculate initial potential
        initial_potential = self.potential_function(
            self.current_temp, self.steps, self.total_energy_consumed_Joules
        )
        
        # Info dictionary
        info = {
            "initial_temperature": self.initial_temp,
            "current_temperature": self.current_temp,
            "theoretical_min_energy_J": self.theoretical_min_energy,
            "theoretical_min_energy_Wh": self.theoretical_min_energy / 3600,
            "approx_heating_time": self.approx_heating_time,
            "optimal_start_step": self.target_deadline_step - int(self.approx_heating_time),
            "initial_potential": initial_potential,
            "reward_shaping_enabled": self.enable_reward_shaping
        }
        
        return current_state, info

    def step(self, action: int):
        """Execute one time step within the environment"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}")

        # Store state before action
        state_before = (self.current_temp, self.steps, self.total_energy_consumed_Joules)

        # Get power from action
        power_selected_W = self.actions[action]
        self.last_power = power_selected_W
        
        # Track energy consumption
        energy_this_step = power_selected_W * self.dt
        self.total_energy_consumed_Joules += energy_this_step
        
        # Physics simulation
        heat_loss_W = self.hA * (self.current_temp - self.ambient_temp)
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c)
        T_next = self.current_temp + self.dt * dTemp_dt
        T_next = np.clip(T_next, 0.0, 150.0)
        self.current_temp = round(T_next, 1)
        
        # Increment time step
        self.steps += 1
        
        # State after action
        state_after = (self.current_temp, self.steps, self.total_energy_consumed_Joules)
        
        # Calculate rewards
        base_reward = self.calculate_base_reward(state_before, state_after, action)
        shaping_reward = self.calculate_shaping_reward(state_before, state_after)
        total_reward = base_reward + shaping_reward
        
        # Check termination
        terminated = self.steps >= self.max_steps
        truncated = False

        # Create observation
        temp_diff_obs = self.current_temp - self.T_target
        steps_diff_obs = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff_obs, steps_diff_obs], dtype=np.float32)
        
        # Calculate metrics
        current_efficiency = self.theoretical_min_energy / max(1.0, self.total_energy_consumed_Joules)
        current_potential = self.potential_function(
            self.current_temp, self.steps, self.total_energy_consumed_Joules
        )
        
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
            "first_goal_step": self.first_goal_step,
            
            # Reward components (for analysis)
            "base_reward": base_reward,
            "shaping_reward": shaping_reward,
            "total_reward": total_reward,
            "current_potential": current_potential,
            
            # Physics info
            "heat_loss_W": heat_loss_W,
            "temp_change_rate": dTemp_dt,
            
            # State info for debugging
            "state_before": state_before,
            "state_after": state_after
        }
        
        return current_state, total_reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment (text output)"""
        if mode == "human":
            # Status indicators
            deadline_status = " [DEADLINE]" if self.steps == self.target_deadline_step else ""
            goal_status = " [GOAL!]" if self.goal_achieved_this_episode else ""
            temp_status = " [TARGET]" if abs(self.current_temp - self.T_target) <= self.T_acceptable_dev else ""
            
            # Calculate efficiency
            efficiency = self.theoretical_min_energy / max(1.0, self.total_energy_consumed_Joules)
            
            # Current potential
            current_potential = self.potential_function(
                self.current_temp, self.steps, self.total_energy_consumed_Joules
            )
            
            # Print status
            print(
                f"Step: {self.steps:3d}, "
                f"Temp: {self.current_temp:5.1f}°C{temp_status}, "
                f"Power: {self.last_power:4d}W, "
                f"Energy: {self.total_energy_consumed_Joules/3600:5.1f}Wh, "
                f"Eff: {efficiency:4.1%}, "
                f"Φ: {current_potential:7.2f}"
                f"{deadline_status}{goal_status}"
            )

    def close(self):
        """Clean up resources"""
        pass

    def get_state_value_info(self, temp=None, step=None, energy_J=None):
        """
        Get potential value and reward information for analysis.
        Useful for understanding the shaped reward structure.
        """
        if temp is None:
            temp = self.current_temp
        if step is None:
            step = self.steps
        if energy_J is None:
            energy_J = self.total_energy_consumed_Joules
        
        potential = self.potential_function(temp, step, energy_J)
        
        # Calculate components separately for analysis
        temp_normalized = (temp - self.initial_temp) / (self.T_target - self.initial_temp)
        steps_remaining = self.target_deadline_step - step
        energy_normalized = energy_J / max(1.0, self.theoretical_min_energy)
        
        return {
            "potential": potential,
            "temp": temp,
            "step": step,
            "energy_J": energy_J,
            "temp_normalized": temp_normalized,
            "steps_remaining": steps_remaining,
            "energy_normalized": energy_normalized,
            "is_goal_achievable": steps_remaining >= 0 and steps_remaining <= self.approx_heating_time + 20
        }

    def verify_policy_invariance(self, policy1, policy2, n_episodes=100, seed=42):
        """
        Verify that two policies have the same performance with and without shaping.
        This demonstrates the policy invariance property.
        """
        np.random.seed(seed)
        results = {}
        
        for policy_name, policy in [("Policy1", policy1), ("Policy2", policy2)]:
            for use_shaping in [False, True]:
                self.enable_reward_shaping = use_shaping
                
                total_base_rewards = []
                total_shaped_rewards = []
                goal_achievements = []
                
                for _ in range(n_episodes):
                    obs, _ = self.reset()
                    done = False
                    base_reward_sum = 0
                    shaped_reward_sum = 0
                    
                    while not done:
                        action = policy(obs)
                        obs, reward, done, truncated, info = self.step(action)
                        base_reward_sum += info['base_reward']
                        shaped_reward_sum += reward
                        done = done or truncated
                    
                    total_base_rewards.append(base_reward_sum)
                    total_shaped_rewards.append(shaped_reward_sum)
                    goal_achievements.append(info['goal_achieved'])
                
                key = f"{policy_name}_shaping_{use_shaping}"
                results[key] = {
                    'mean_base_reward': np.mean(total_base_rewards),
                    'mean_shaped_reward': np.mean(total_shaped_rewards),
                    'goal_rate': np.mean(goal_achievements)
                }
        
        return results