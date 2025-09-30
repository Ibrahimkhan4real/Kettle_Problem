import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KettleEnv(gym.Env):
    """
    Energy-Efficient Kettle Environment with Policy-Invariant Reward Shaping
    
    Implements potential-based reward shaping following Ng, Harada, Russell (1999):
    "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping"
    
    Key theoretical guarantee:
    If R(s,a,s') is the original reward and Φ:S→ℝ is a potential function, then:
    R'(s,a,s') = R(s,a,s') + F(s,a,s') where F(s,a,s') = γΦ(s') - Φ(s)
    has the same optimal policies as R for all γ ∈ [0,1).
    
    Goal: Reach target temperature (100°C ±2.5°C) at target time (step 200 ±15 steps) 
          with minimal energy consumption through delayed heating strategy.
    """
    
    def __init__(
        self,
        water_volume=1.0,
        initial_temp=20.0,
        ambient_temp=20.0,
        target_temp=100.0,
        max_steps=250,
        target_deadline_step=200,
        temp_tolerance=2.5,
        time_tolerance=15,
        # Reward shaping parameters
        enable_reward_shaping=True,
        discount_factor=0.99,
        potential_scale=50.0,
        temp_potential_weight=1.0,
        time_potential_weight=0.8,
        energy_potential_weight=0.5,
        urgency_factor=2.0
    ):
        super().__init__()
        
        # Physical parameters
        self.water_volume = water_volume
        self.m = water_volume  # Mass in kg
        self.c = 4184.0        # Specific heat capacity of water (J/kg·K)
        self.hA = 5.0          # Heat loss coefficient (W/K)
        self.dt = 1.0          # Time step (seconds)
        
        # Environment parameters
        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp
        self.target_temp = target_temp
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        self.temp_tolerance = temp_tolerance
        self.time_tolerance = time_tolerance
        
        # Reward shaping parameters (Ng, Harada, Russell 1999)
        self.enable_reward_shaping = enable_reward_shaping
        self.gamma = discount_factor  # Must be in [0,1) for policy invariance guarantee
        self.potential_scale = potential_scale
        self.temp_potential_weight = temp_potential_weight
        self.time_potential_weight = time_potential_weight
        self.energy_potential_weight = energy_potential_weight
        self.urgency_factor = urgency_factor
        
        assert 0 <= self.gamma < 1, "Discount factor must be in [0,1) for policy invariance"
        
        # Action space: [0W (off), 3000W (on)]
        self.actions = [0, 3000]
        self.action_space = spaces.Discrete(len(self.actions))
        
        # Observation space: [temperature_difference, steps_to_deadline]
        min_temp_diff = initial_temp - target_temp
        max_temp_diff = 150.0 - target_temp
        min_steps_diff = target_deadline_step - max_steps
        max_steps_diff = target_deadline_step
        
        self.observation_space = spaces.Box(
            low=np.array([min_temp_diff, min_steps_diff], dtype=np.float32),
            high=np.array([max_temp_diff, max_steps_diff], dtype=np.float32),
            dtype=np.float32
        )
        
        # Calculate theoretical minimum energy needed and optimal timing
        self.theoretical_min_energy = self.m * self.c * (self.target_temp - self.initial_temp)
        
        # Estimate optimal heating time (considering heat loss)
        avg_temp_during_heating = (self.initial_temp + self.target_temp) / 2
        avg_heat_loss = self.hA * (avg_temp_during_heating - self.ambient_temp)
        net_heating_power = self.actions[1] - avg_heat_loss
        self.optimal_heating_duration = int((self.target_temp - self.initial_temp) * self.m * self.c / net_heating_power)
        self.optimal_start_step = self.target_deadline_step - self.optimal_heating_duration
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_temp = float(self.initial_temp)
        self.steps = 0
        self.total_energy_consumed = 0.0  # Joules
        self.goal_achieved = False
        self.goal_achieved_step = None
        
        obs = self._get_observation()
        info = {
            "theoretical_min_energy_J": self.theoretical_min_energy,
            "theoretical_min_energy_Wh": self.theoretical_min_energy / 3600,
            "initial_temp": self.initial_temp,
            "target_temp": self.target_temp,
            "target_step": self.target_deadline_step
        }
        
        return obs, info
    
    def _calculate_base_reward(self, action, achieved_goal_this_step):
        """
        Calculate base (original) reward R(s,a,s') - the true task objective
        
        This represents the actual task without shaping:
        1. Small energy cost for power consumption
        2. Large sparse reward for achieving goal efficiently
        """
        reward = 0.0
        
        # Energy cost (continuous small penalty)
        power = self.actions[action]
        energy_cost = -power * self.dt / 1000.0  # Small penalty per kJ
        reward += energy_cost
        
        # Sparse goal reward (only when goal first achieved)
        if achieved_goal_this_step:
            # Large reward scaled by energy efficiency
            efficiency = self.theoretical_min_energy / max(1.0, self.total_energy_consumed)
            goal_reward = 1000.0 * min(efficiency, 1.0)  # Cap at perfect efficiency
            reward += goal_reward
        
        # Small penalty for missing temperature target at deadline
        if self.steps == self.target_deadline_step:
            temp_error = abs(self.current_temp - self.target_temp)
            if temp_error > self.temp_tolerance:
                reward -= 50.0
        
        return reward
    
    def step(self, action):
        """Execute one environment step with potential-based reward shaping"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Store state before action for shaping calculation
        state_before = (self.current_temp, self.steps, self.total_energy_consumed)
        
        # Get power from action
        power = self.actions[action]
        
        # Track energy consumption
        energy_this_step = power * self.dt
        self.total_energy_consumed += energy_this_step
        
        # Physics simulation: temperature change
        heat_loss = self.hA * (self.current_temp - self.ambient_temp)
        net_heat_input = power - heat_loss
        temp_change = net_heat_input * self.dt / (self.m * self.c)
        
        self.current_temp += temp_change
        self.current_temp = np.clip(self.current_temp, 0.0, 150.0)
        
        # Increment time
        self.steps += 1
        
        # Check goal achievement
        temp_error = abs(self.current_temp - self.target_temp)
        time_error = abs(self.steps - self.target_deadline_step)
        achieved_goal_this_step = False
        
        if (temp_error <= self.temp_tolerance and 
            time_error <= self.time_tolerance and 
            not self.goal_achieved):
            self.goal_achieved = True
            self.goal_achieved_step = self.steps
            achieved_goal_this_step = True
        
        # State after action for shaping calculation
        state_after = (self.current_temp, self.steps, self.total_energy_consumed)
        
        # Calculate rewards
        base_reward = self._calculate_base_reward(action, achieved_goal_this_step)
        shaping_reward = self.calculate_shaping_reward(state_before, state_after)
        total_reward = base_reward + shaping_reward
        
        # Check termination
        terminated = self.steps >= self.max_steps
        truncated = False
        
        # Create observation
        obs = self._get_observation()
        
        # Enhanced info dictionary
        info = {
            # Basic state
            "temperature": self.current_temp,
            "power": power,
            "energy_this_step": energy_this_step,
            "total_energy_J": self.total_energy_consumed,
            "total_energy_Wh": self.total_energy_consumed / 3600,
            "energy_efficiency": self.theoretical_min_energy / max(1.0, self.total_energy_consumed),
            "goal_achieved": self.goal_achieved,
            "goal_achieved_step": self.goal_achieved_step,
            "steps_to_deadline": self.target_deadline_step - self.steps,
            "temp_error": temp_error,
            "at_target": temp_error <= self.temp_tolerance,
            
            # Reward components (for analysis)
            "base_reward": base_reward,
            "shaping_reward": shaping_reward, 
            "total_reward": total_reward,
            
            # Shaping analysis
            "reward_shaping_enabled": self.enable_reward_shaping,
            "potential_before": self.potential_function(*state_before) if self.enable_reward_shaping else 0.0,
            "potential_after": self.potential_function(*state_after) if self.enable_reward_shaping else 0.0,

            
            # Physics
            "heat_loss": heat_loss,
            "temp_change": temp_change
        }
        
        return obs, total_reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation"""
        temp_diff = self.current_temp - self.target_temp
        steps_to_deadline = self.target_deadline_step - self.steps
        return np.array([temp_diff, steps_to_deadline], dtype=np.float32)
    
    def potential_function(self, temp, step, energy_consumed):
        """
        Compute potential function Φ(s) for policy-invariant reward shaping.
        
        Based on Ng, Harada, Russell (1999): The potential function should encode
        domain knowledge about "good" states without changing the optimal policy.
        
        Design principles:
        1. Higher potential for states closer to achieving the goal efficiently
        2. Encourage delayed heating (energy-efficient strategy)  
        3. Create urgency as deadline approaches
        4. Bounded function to ensure convergence
        
        Args:
            temp: Current temperature
            step: Current time step
            energy_consumed: Total energy consumed so far (Joules)
            
        Returns:
            Potential value Φ(s)
        """
        if not self.enable_reward_shaping:
            return 0.0
        
        # Normalize inputs for stable computation
        temp_progress = (temp - self.initial_temp) / (self.target_temp - self.initial_temp)
        temp_progress = np.clip(temp_progress, -0.2, 1.5)  # Allow some overshoot
        
        steps_remaining = self.target_deadline_step - step
        steps_remaining_norm = steps_remaining / self.target_deadline_step
        
        energy_efficiency = self.theoretical_min_energy / max(energy_consumed, 1.0)
        energy_efficiency = np.clip(energy_efficiency, 0.0, 2.0)
        
        # Component 1: Temperature achievement potential
        # Only reward temperature progress when timing is appropriate
        if steps_remaining > self.optimal_heating_duration + 30:
            # Too early to heat - no temperature reward (encourages waiting)
            temp_potential = 0.0
        elif steps_remaining > 0:
            # Within reasonable heating window
            if abs(temp - self.target_temp) <= self.temp_tolerance:
                # At target temperature
                temp_potential = self.temp_potential_weight * 1.0
            else:
                # Approaching target - smooth reward based on progress
                temp_achievement = max(0, temp_progress)
                # Add urgency multiplier as deadline approaches
                urgency_mult = 1 + (self.urgency_factor * max(0, 1 - steps_remaining / self.optimal_heating_duration))
                temp_potential = self.temp_potential_weight * temp_achievement * urgency_mult
        else:
            # Past deadline
            if abs(temp - self.target_temp) <= self.temp_tolerance:
                temp_potential = self.temp_potential_weight * 0.7  # Partial credit
            else:
                temp_potential = -self.temp_potential_weight * 0.3  # Penalty
        
        # Component 2: Timing potential  
        # Create appropriate urgency without encouraging premature heating
        if steps_remaining > self.optimal_heating_duration:
            # Plenty of time - slight positive potential for waiting
            time_potential = self.time_potential_weight * 0.1
        elif steps_remaining > 0:
            # Time pressure building - increase potential smoothly
            urgency = (self.optimal_heating_duration - steps_remaining) / self.optimal_heating_duration
            time_potential = self.time_potential_weight * urgency
        else:
            # Past deadline - moderate penalty
            time_potential = self.time_potential_weight * 0.5
        
        # Component 3: Energy efficiency potential
        # Always encourage energy conservation
        energy_potential = self.energy_potential_weight * (energy_efficiency - 1.0)
        
        # Combine components
        total_potential = self.potential_scale * (
            temp_potential + time_potential + energy_potential
        )
        
        # Ensure bounded potential (important for theoretical guarantees)
        total_potential = np.clip(total_potential, -200, 200)
        
        return total_potential
    
    def calculate_shaping_reward(self, state_before, state_after):
        """
        Calculate policy-invariant shaping reward: F(s,a,s') = γΦ(s') - Φ(s)
        
        This is the core formula from Ng, Harada, Russell (1999) that guarantees
        the shaped reward has the same optimal policy as the original reward.
        
        Args:
            state_before: (temp, step, energy) before action
            state_after: (temp, step, energy) after action
            
        Returns:
            Shaping reward F(s,a,s')
        """
        if not self.enable_reward_shaping:
            return 0.0
        
        temp_before, step_before, energy_before = state_before
        temp_after, step_after, energy_after = state_after
        
        # Calculate potentials
        phi_before = self.potential_function(temp_before, step_before, energy_before)
        phi_after = self.potential_function(temp_after, step_after, energy_after)
        
        # Policy-invariant shaping reward formula
        shaping_reward = self.gamma * phi_after - phi_before
        
        return shaping_reward
    
    def render(self, mode="human"):
        """Render environment state"""
        if mode == "human":
            efficiency = self.theoretical_min_energy / max(1.0, self.total_energy_consumed)
            goal_status = " [GOAL!]" if self.goal_achieved else ""
            deadline_status = " [DEADLINE]" if self.steps == self.target_deadline_step else ""
            
            print(f"Step {self.steps:3d}: Temp={self.current_temp:5.1f}°C, "
                  f"Energy={self.total_energy_consumed/3600:5.1f}Wh, "
                  f"Eff={efficiency:.1%}{goal_status}{deadline_status}")
    
    def close(self):
        """Clean up resources"""
        pass