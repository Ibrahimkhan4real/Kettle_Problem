import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
from math import log, sqrt
import os # For creating directory for plots

# Removed: from kettle_dynamic_env_v20 import KettleEnv
# KettleEnv class will be defined directly below

# --- KettleEnv Class Definition (with aggressive reward function) ---
class KettleEnv(gym.Env):
    """
    Kettle Gym Environment with an aggressively reshaped reward function
    to encourage proactive heating and aim for area minimization.
    Version: v3_aggressive_reward_embedded
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        water_volume=1.0,
        initial_temp=20.0,
        ambient_temp=20.0, # Default ambient for hyperparameter calibration
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

        self.initial_temp = initial_temp
        self.ambient_temp = ambient_temp
        self.dt = 1.0

        self.actions = [0] + list(range(1500, 3001, 500))
        self.num_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.num_actions)

        min_temp_diff = 0.0 - self.T_target
        max_temp_diff = 150.0 - self.T_target
        min_steps_diff = float(target_deadline_step - max_steps)
        max_steps_diff = float(target_deadline_step)
        low_obs = np.array([min_temp_diff, self.water_volume, self.ambient_temp, min_steps_diff], dtype=np.float32)
        high_obs = np.array([max_temp_diff, self.water_volume, self.ambient_temp, max_steps_diff], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        self.current_temp = 0.0
        self.steps = 0
        self.max_steps = max_steps
        self.target_deadline_step = target_deadline_step
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0

        if not (0 <= self.target_deadline_step <= self.max_steps):
            raise ValueError("target_deadline_step must be between 0 and max_steps.")

        # === Reward function hyperparameters (Aggressively Reshaped) ===
        self.SUCCESS_BONUS = 200.0
        self.FAILURE_PENALTY = -100.0
        
        self.PER_STEP_POWER_COST_WEIGHT = 0.001  # DRASTICALLY REDUCED
        self.TEMP_AREA_PENALTY_WEIGHT = 0.001  # DRASTICALLY REDUCED
        
        # Calibrated for T_target=100, T_ambient=20. Area penalty = -0.001 * 80 = -0.08. Bonus 0.10 gives +0.02 net.
        # If run_mcts_controlled_episodes uses different T_ambient, this bonus's relative effect changes.
        self.EFFICIENT_IDLING_BONUS = 0.10     
        
        self.TEMP_PROGRESS_WEIGHT = 0.75       # MASSIVELY INCREASED
        self.OVERSHOOT_PENALTY_WEIGHT = 0.05

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = self.initial_temp
        self.steps = 0
        self.last_power = 0
        self.total_energy_consumed_Joules = 0.0
        temp_diff = self.current_temp - self.T_target
        steps_diff = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff, self.water_volume, self.ambient_temp, steps_diff], dtype=np.float32)
        return current_state, {"initial_temperature": self.current_temp, "current_temperature": self.current_temp}

    def calculate_reward(self, T_after_action, T_before_action, power_W):
        reward = 0.0
        current_error_abs = abs(T_after_action - self.T_target)

        max_power_possible = float(self.actions[-1]) if self.actions and len(self.actions) > 0 else 3000.0
        if max_power_possible == 0 and len(self.actions) > 1:
            positive_powers = [p for p in self.actions if p > 0]
            if positive_powers: max_power_possible = float(max(positive_powers))
            else: max_power_possible = 3000.0
        normalized_power = float(power_W) / max_power_possible if max_power_possible > 0 else 0.0
        reward -= self.PER_STEP_POWER_COST_WEIGHT * normalized_power

        if T_after_action > self.ambient_temp:
            reward -= self.TEMP_AREA_PENALTY_WEIGHT * (T_after_action - self.ambient_temp)

        previous_error_abs = abs(T_before_action - self.T_target)
        error_delta = previous_error_abs - current_error_abs 
        reward += self.TEMP_PROGRESS_WEIGHT * error_delta

        if T_after_action > self.T_target + self.T_acceptable_dev:
            overshoot_amount = T_after_action - (self.T_target + self.T_acceptable_dev)
            reward -= self.OVERSHOOT_PENALTY_WEIGHT * (overshoot_amount / 10.0) 

        if self.steps < self.target_deadline_step:
            if (self.T_target - self.T_acceptable_dev) <= T_after_action <= (self.T_target + self.T_acceptable_dev):
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
        
        temp_before_action = self.current_temp 
        T_current_for_physics = round(temp_before_action, 1) # Physics calculation based on 1 d.p. start temp
        
        heat_loss_W = self.hA * (T_current_for_physics - self.ambient_temp)
        dTemp_dt = (power_selected_W - heat_loss_W) / (self.m * self.c)
        T_next_calculated = T_current_for_physics + self.dt * dTemp_dt
        temp_after_action = np.clip(T_next_calculated, 0.0, 150.0)
        self.current_temp = temp_after_action # Internal state can be more precise
        
        self.steps += 1
        
        current_step_reward = self.calculate_reward(temp_after_action, temp_before_action, power_selected_W)
        
        terminated = self.steps >= self.max_steps
        truncated = False

        temp_diff_obs = self.current_temp - self.T_target
        steps_diff_obs = float(self.target_deadline_step - self.steps)
        current_state = np.array([temp_diff_obs, self.water_volume, self.ambient_temp, steps_diff_obs], dtype=np.float32)
        
        info = {
            "temperature": self.current_temp, # Actual current temperature
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
                f"Step: {self.steps:3d}, Temp: {self.current_temp:.1f}°C, " # Formatted to .1f
                f"Pwr: {self.last_power:4d}W{deadline_status}"
            )

    def close(self):
        pass
# --- End of KettleEnv Class Definition ---

# --- Configuration ---
EPISODES_TO_RUN = 1
MAX_TIMESTEPS_PER_EPISODE = 250

# MCTS Hyperparameters
UCB_C = 6.5
MCTS_ITERATIONS_PER_ACTION = 2000
MAX_ROLLOUT_DEPTH = 250 # CRITICAL CHANGE: Increased to allow reaching default deadline (200) and max_steps (250)


class Node:
    # ... (Node class remains unchanged from your provided script) ...
    def __init__(self, env_state_copy, done_flag, parent_node, observation, action_idx_leading_to_node):
        self.children = {}
        self.total_simulation_reward = 0.0
        self.num_visits = 0
        self.env_state = env_state_copy
        self.observation = observation
        self.is_terminal = done_flag
        self.parent = parent_node
        self.action_that_led_here = action_idx_leading_to_node

    def get_ucb1_score(self) -> float:
        if self.num_visits == 0:
            return float("inf")
        if not self.parent:
            parent_total_visits = self.num_visits
        else:
            parent_total_visits = self.parent.num_visits
        exploitation_term = self.total_simulation_reward / self.num_visits
        exploration_term = UCB_C * sqrt(log(max(1, parent_total_visits)) / self.num_visits)
        return exploitation_term + exploration_term

    def detach_from_parent(self):
        self.parent = None

    def expand(self):
        if self.is_terminal or self.children:
            return
        num_possible_actions = self.env_state.action_space.n
        for action_idx in range(num_possible_actions):
            child_env_state_copy = deepcopy(self.env_state)
            obs, _, done, truncated, _ = child_env_state_copy.step(action_idx)
            child_is_terminal = done or truncated
            self.children[action_idx] = Node(child_env_state_copy, child_is_terminal, self, obs, action_idx)

    def rollout(self) -> float:
        if self.is_terminal:
            return 0.0
        rollout_env = deepcopy(self.env_state)
        accumulated_rollout_reward = 0.0
        current_depth = 0
        is_rollout_done = False
        # Store initial observation for heuristic rollout if needed later
        # current_rollout_obs = self.observation 

        while not is_rollout_done and current_depth < MAX_ROLLOUT_DEPTH:
            # --- Standard Random Rollout Policy ---
            random_action = rollout_env.action_space.sample()
            
            # --- Placeholder for a Potential Heuristic Rollout Policy (Advanced) ---
            # To implement a heuristic policy, you would get info from `rollout_env` or `current_rollout_obs`
            # For example:
            # temp_in_rollout = rollout_env.current_temp # Direct access if env structure allows
            # steps_remaining_rollout = rollout_env.target_deadline_step - rollout_env.steps
            # if temp_in_rollout < rollout_env.T_target - 10 and steps_remaining_rollout > 10:
            #     chosen_action_rollout = rollout_env.num_actions - 1 # Max power
            # elif temp_in_rollout > rollout_env.T_target + 5:
            #     chosen_action_rollout = 0 # Off
            # else: # Near target or not clear what to do, could use a mix or smaller power
            #     # For simplicity, stick to random or a simpler heuristic if above gets complex
            #     chosen_action_rollout = rollout_env.action_space.sample() 
            # --- End of Placeholder ---

            obs_rollout, reward, term, trunc, _ = rollout_env.step(random_action) # Use random_action for now
            # current_rollout_obs = obs_rollout # Update obs if heuristic depends on it
            is_rollout_done = term or trunc
            accumulated_rollout_reward += reward
            current_depth += 1
        return accumulated_rollout_reward

    def backpropagate(self, reward_from_simulation):
        current_node_in_path = self
        while current_node_in_path is not None:
            current_node_in_path.num_visits += 1
            current_node_in_path.total_simulation_reward += reward_from_simulation
            current_node_in_path = current_node_in_path.parent

    def select_best_child_for_action(self):
        if self.is_terminal:
            # print("Warning: Trying to select action from a terminal node.") # Kept for debug
            return self, None
        if not self.children:
            # print("Warning: Node has no children to select from...") # Kept for debug
            if not self.is_terminal:
                self.expand()
                if not self.children:
                    return self, None
            else: # Is terminal and no children (correct)
                return self, None

        max_visits = -1
        best_children_nodes = []
        for child_node in self.children.values():
            if child_node.num_visits > max_visits:
                max_visits = child_node.num_visits
                best_children_nodes = [child_node]
            elif child_node.num_visits == max_visits:
                best_children_nodes.append(child_node)
        
        if not best_children_nodes:
            # This case should be rare if children exist and MCTS ran.
            # Fallback to UCB scores if no visits (e.g. if all children are new)
            # or even a random choice among existing children.
            # print("Warning: No children found with visits. Trying UCB for selection or random.")
            if self.children: # If children exist but none were visited (e.g. first few iterations)
                max_ucb = -float('inf')
                # best_children_nodes list will be repopulated by UCB scores
                for child_node_ucb in self.children.values():
                    ucb = child_node_ucb.get_ucb1_score() # unvisited will be inf
                    if ucb > max_ucb:
                        max_ucb = ucb
                        best_children_nodes = [child_node_ucb]
                    elif ucb == max_ucb:
                        best_children_nodes.append(child_node_ucb)
                if not best_children_nodes: # Still no best children (highly unlikely)
                    chosen_child_node = random.choice(list(self.children.values()))
                else:
                    chosen_child_node = random.choice(best_children_nodes) # Tie-break UCB
            else: # No children at all
                return self, None
        else: # Standard case: choose from most visited
            chosen_child_node = random.choice(best_children_nodes)

        if not chosen_child_node:
            return self, None
        action_to_take_idx = chosen_child_node.action_that_led_here
        # chosen_child_node.detach_from_parent() # Not detaching if tree is rebuilt each step
        return chosen_child_node, action_to_take_idx


def mcts_single_iteration(root_node: Node):
    # ... (mcts_single_iteration remains unchanged from your provided script) ...
    current_selection = root_node
    while current_selection.children:
        child_nodes = current_selection.children
        best_child_node = None
        max_ucb_score = -float('inf')
        child_action_keys = list(child_nodes.keys())
        random.shuffle(child_action_keys)
        for action_key in child_action_keys:
            node = child_nodes[action_key]
            ucb_score = node.get_ucb1_score()
            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_child_node = node
        current_selection = best_child_node if best_child_node else random.choice(list(child_nodes.values()))

    node_to_simulate_from = current_selection
    if not current_selection.is_terminal:
        if current_selection.num_visits > 0: 
            current_selection.expand()
            if current_selection.children: 
                unvisited_children = [child for child in current_selection.children.values() if child.num_visits == 0]
                if unvisited_children:
                    node_to_simulate_from = random.choice(unvisited_children)
                else: 
                    node_to_simulate_from = random.choice(list(current_selection.children.values()))
    simulation_reward = node_to_simulate_from.rollout()
    node_to_simulate_from.backpropagate(simulation_reward)


def get_action_via_mcts(mcts_root_node: Node, num_iterations: int):
    # ... (get_action_via_mcts structure remains unchanged) ...
    if mcts_root_node.is_terminal:
        print("MCTS: Root node is terminal. No action to select.")
        return mcts_root_node, None # Return original root if terminal

    for i in range(num_iterations):
        mcts_single_iteration(mcts_root_node)

    # Select best child from the original root_node passed to this function
    # The first return from select_best_child_for_action is the child node that will be the next root
    # IF we were reusing the tree. Since we rebuild the tree, we only need chosen_action_idx here.
    # However, the original code returned next_mcts_root_node, chosen_action_idx and used next_mcts_root_node.
    # For tree rebuilding strategy, we don't need to return the child as the next root.
    # But select_best_child_for_action might be used elsewhere, so keep its signature.
    # The crucial part is that run_mcts_controlled_episodes IGNORES the returned node for tree reuse.
    
    # The print statements below are for the root node of the *current* MCTS search
    # This mcts_root_node is the one passed into this function.
    
    # Ensure child stats are for the children of the *current MCTS search root*
    # mcts_root_node.select_best_child_for_action() selects based on visits.
    # For printing and choosing action, we operate on mcts_root_node.children.

    # The `select_best_child_for_action` method in your Node class already handles the logic
    # of picking the best action (most visited child). It also returns the child node itself.
    # We should call this once to get the action and the (not-to-be-reused) next root.
    
    # Store the original root's observation if needed for printing before selection potentially alters it
    # (though select_best_child_for_action doesn't alter the root's observation)
    root_obs_temp_diff = mcts_root_node.observation[0]

    # Select the best action using the defined strategy (e.g. most visited)
    # This also expands the root if it had no children and was not terminal
    # chosen_child_as_next_root, chosen_action_idx = mcts_root_node.select_best_child_for_action()
    
    # For printing, iterate through children of the current mcts_root_node.
    # Then, separately, select the best action.

    print(f"MCTS Root (Temp Diff: {root_obs_temp_diff:.1f}, Child Stats:") # Formatted to .1f
    if mcts_root_node.children:
        action_meanings = {idx: f"{power}W" for idx, power in enumerate(mcts_root_node.env_state.actions)}
        for action_idx, child_node in sorted(mcts_root_node.children.items()):
            avg_q_value = (child_node.total_simulation_reward / child_node.num_visits) if child_node.num_visits > 0 else -float('inf')
            ucb_score = child_node.get_ucb1_score() 
            print(f"  Action {action_idx} ({action_meanings.get(action_idx, 'N/A')}): "
                  f"Visits={child_node.num_visits}, AvgQ={avg_q_value:.3f}, UCB={ucb_score:.3f}")
    else:
        print("  MCTS Root has no children after iterations (or was not expanded).")
    
    # Now, select the best action. The select_best_child_for_action can be called here.
    # The first return is the child node itself, the second is the action index.
    # Since tree is rebuilt, we mainly care about chosen_action_idx.
    # The `_` indicates the returned child node isn't used to propagate the tree.
    _ , chosen_action_idx = mcts_root_node.select_best_child_for_action()
    
    # The original code returned next_mcts_root_node, chosen_action_idx.
    # If tree is rebuilt, the first return value is effectively ignored by the caller loop.
    # We return None for the next_mcts_root_node to signify no tree reuse from this function's PoV.
    return None, chosen_action_idx


def run_mcts_controlled_episodes():
    all_episodes_temperatures = []
    all_episodes_actions = []
    all_episodes_rewards = []

    for episode_idx in range(EPISODES_TO_RUN):
        episode_initial_temp = 20.0
        episode_ambient_temp = 25.0 # Example ambient temp for this run
        kettle_env = KettleEnv(
            initial_temp=episode_initial_temp,
            ambient_temp=episode_ambient_temp,
            # Using default max_steps=250, target_deadline_step=200 from KettleEnv
        )

        current_obs, reset_info = kettle_env.reset()
        # Correctly log initial temperature
        current_episode_temps = [reset_info.get("current_temperature", episode_initial_temp)] 
        
        is_done_from_env = False
        is_truncated_from_env = False
        total_episode_reward_accumulated = 0.0
        current_episode_actions_idx = []

        print(f"\n{'='*60}")
        print(f"Starting Episode {episode_idx+1}/{EPISODES_TO_RUN} (MCTS Tree Rebuilt Each Step)")
        print(f"{'='*60}")
        try: T_target_log = kettle_env.T_target
        except AttributeError: T_target_log = 100.0
        
        # Print actual initial temp, not temp_diff from obs
        print(f"Initial Temp: {kettle_env.current_temp:.1f}°C, Target: {T_target_log:.1f}°C, Ambient: {kettle_env.ambient_temp:.1f}°C") # Formatted
        print(f"MCTS Iterations/Action: {MCTS_ITERATIONS_PER_ACTION}, Max Rollout Depth: {MAX_ROLLOUT_DEPTH}")

        for timestep_num in range(MAX_TIMESTEPS_PER_EPISODE):
            # current_obs[0] is temp_diff, current_obs[1] is water_vol, current_obs[2] is ambient_temp, current_obs[3] is steps_to_deadline
            # For printing current state temp, use kettle_env.current_temp
            print(f"\n--- Ep {episode_idx+1}, Timestep {timestep_num+1}/{MAX_TIMESTEPS_PER_EPISODE} ---")
            print(f"Current State (Temp: {kettle_env.current_temp:.1f}°C, Steps to Deadline: {current_obs[3]:.0f})") # Formatted

            if is_done_from_env or is_truncated_from_env:
                term_reason_loop_check = "terminated (done)" if is_done_from_env else "truncated"
                print(f"Loop check: Episode ended at start of timestep {timestep_num+1} due to: {term_reason_loop_check}")
                break

            mcts_root_for_this_step = Node(
                env_state_copy=deepcopy(kettle_env),
                done_flag=is_done_from_env, 
                parent_node=None,
                observation=current_obs,
                action_idx_leading_to_node=None
            )
            
            # The first returned value from get_action_via_mcts is now None (or ignored)
            _, chosen_action_index = get_action_via_mcts(mcts_root_for_this_step, MCTS_ITERATIONS_PER_ACTION)

            if chosen_action_index is None:
                print(f"MCTS did not return a valid action at timestep {timestep_num+1}. Ending episode.")
                if mcts_root_for_this_step and mcts_root_for_this_step.is_terminal:
                    print("  Reason: MCTS root node (freshly created) indicates a terminal state.")
                break

            action_power_W = kettle_env.actions[chosen_action_index]
            print(f"MCTS chose Action Index: {chosen_action_index} (Power: {action_power_W}W)")
            current_episode_actions_idx.append(chosen_action_index)

            next_obs, reward, is_done_from_env, is_truncated_from_env, info = kettle_env.step(chosen_action_index)
            
            # Correctly log temperature after action
            current_episode_temps.append(info.get("temperature", kettle_env.current_temp)) 
            print(f"Env after action: New Temp = {info.get('temperature', 'N/A'):.1f}°C, Reward: {reward:.3f}, Env Step: {info.get('current_step', 'N/A')}") # Formatted

            total_episode_reward_accumulated += reward
            current_obs = next_obs

            if is_done_from_env or is_truncated_from_env:
                term_reason = "terminated (done)" if is_done_from_env else "truncated"
                print(f"Episode finished by environment at timestep {timestep_num+1} (Env step: {info.get('current_step', 'N/A')}) due to: {term_reason}.")
                break
        
        if not (is_done_from_env or is_truncated_from_env) and timestep_num + 1 >= MAX_TIMESTEPS_PER_EPISODE:
            print(f"Episode finished by reaching MCTS MAX_TIMESTEPS_PER_EPISODE ({MAX_TIMESTEPS_PER_EPISODE}).")

        print(f"\n{'*'*60}")
        print(f"Episode {episode_idx+1} Finished Summary:")
        print(f"{'*'*60}")
        print(f"Total Reward Accumulated: {total_episode_reward_accumulated:.2f}")
        print(f"Number of Timesteps Taken by MCTS agent: {len(current_episode_actions_idx)}")
        
        # Ensure final_temp is actual temperature and correctly accessed
        final_temp = current_episode_temps[-1] if current_episode_temps else episode_initial_temp 
        try: T_target_log_final = kettle_env.T_target
        except AttributeError: T_target_log_final = 100.0
        print(f"Initial Temperature: {current_episode_temps[0]:.1f}°C") # Formatted
        print(f"Final Temperature: {final_temp:.1f}°C (Target: {T_target_log_final:.1f}°C)") # Formatted
        
        try: deadline_step_log = kettle_env.target_deadline_step
        except AttributeError: deadline_step_log = "N/A"
        print(f"Environment's Target Deadline Step: {deadline_step_log}")

        print("\nAction Sequence (Index: Power W):")
        try: action_map = kettle_env.actions
        except AttributeError: action_map = {i: f"idx_{i}" for i in range(kettle_env.action_space.n)} 

        for t, act_idx in enumerate(current_episode_actions_idx):
            power_val_str = str(action_map[act_idx]) + "W" # Assumes action_map values are numbers
            temp_after_action_log = current_episode_temps[t+1] # t+1 because current_episode_temps[0] is initial temp
            temp_before_action_log = current_episode_temps[t]
            temp_change = temp_after_action_log - temp_before_action_log
            print(f"  MCTS Step {t+1}: Act Idx {act_idx} ({power_val_str}) -> Temp: {temp_after_action_log:6.1f}°C (ΔT: {temp_change:+.1f}°C)") # Formatted

        on_actions_count = sum(1 for act_idx in current_episode_actions_idx if action_map[act_idx] > 0)
        off_actions_count = len(current_episode_actions_idx) - on_actions_count
        if current_episode_actions_idx:
            on_percent = (on_actions_count / len(current_episode_actions_idx)) * 100
            print(f"\nAction Summary: {on_actions_count} ON ({on_percent:.1f}%), "
                  f"{off_actions_count} OFF ({(100-on_percent):.1f}%)")

        all_episodes_temperatures.append(current_episode_temps)
        all_episodes_actions.append(current_episode_actions_idx)
        all_episodes_rewards.append(total_episode_reward_accumulated)

        kettle_env.close()

    return all_episodes_temperatures, all_episodes_actions, all_episodes_rewards


def plot_simulation_results(ep_temps_data, ep_actions_data, ep_rewards_data, target_temp_plot=100.0):
    # ... (plot_simulation_results structure remains unchanged) ...
    # Only check formatting if any print statements were missed
    if not ep_temps_data:
        print("No data available to plot.")
        return

    num_episodes_plotted = len(ep_temps_data)
    plot_dir = "Experiments/MCTS_Plots"
    os.makedirs(plot_dir, exist_ok=True)

    fig1, ax_temp = plt.subplots(figsize=(15, 7))
    fig1.suptitle("Kettle Temperature Control via MCTS", fontsize=16)
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, num_episodes_plotted)))

    for i in range(num_episodes_plotted):
        temps = ep_temps_data[i] # These are actual temperatures
        actions_indices = ep_actions_data[i]
        reward_val = ep_rewards_data[i]
        time_steps_temps = range(len(temps))
        ax_temp.plot(
            time_steps_temps, temps, marker='o', markersize=3, linestyle='-', linewidth=1.5,
            color=colors[i if num_episodes_plotted > 1 else 0],
            label=f"Ep {i+1} Temp (Total R: {reward_val:.2f})"
        )
        for t_idx, action_idx_val in enumerate(actions_indices):
            if action_idx_val > 0: 
                ax_temp.axvspan(t_idx, t_idx + 1, alpha=0.15, color=colors[i if num_episodes_plotted > 1 else 0], ymin=0, ymax=0.6)

    ax_temp.axhline(y=target_temp_plot, color="black", linestyle="--", linewidth=1.5, label=f"Target ({target_temp_plot:.1f}°C)") # Formatted
    ax_temp.set_xlabel("Timestep (MCTS Agent decisions)", fontsize=12)
    ax_temp.set_ylabel("Temperature (°C)", fontsize=12)
    ax_temp.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax_temp.grid(True, linestyle=':', alpha=0.7)
    all_temp_flat = [t for ep_t in ep_temps_data for t in ep_t]
    if all_temp_flat:
        y_min_temp = min(all_temp_flat) - 5
        y_max_temp = max(all_temp_flat) + 10
        ax_temp.set_ylim(max(0, y_min_temp), y_max_temp)
    fig1.tight_layout(rect=[0, 0, 0.83, 0.95])
    plot1_path = os.path.join(plot_dir, "mcts_kettle_temperatures_actions.png")
    fig1.savefig(plot1_path, dpi=300)
    print(f"\nTemperature & Actions plot saved to '{plot1_path}'")
    plt.close(fig1)

    if any(ep_actions_data):
        fig2, ax_actions = plt.subplots(figsize=(15, 4))
        fig2.suptitle("MCTS Agent Actions (Heater Power Index)", fontsize=14)
        for i in range(num_episodes_plotted):
            actions_indices = ep_actions_data[i]
            if actions_indices:
                time_steps_actions = range(len(actions_indices))
                ax_actions.step(
                    time_steps_actions, actions_indices, where="post",
                    color=colors[i if num_episodes_plotted > 1 else 0],
                    linewidth=1.5, label=f"Ep {i+1} Action Idx"
                )
        ax_actions.set_ylabel("Action Index", fontsize=12)
        ax_actions.set_xlabel("Timestep (MCTS Agent decisions)", fontsize=12)
        ax_actions.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax_actions.grid(True, axis='x', linestyle=':', alpha=0.6)
        fig2.tight_layout(rect=[0, 0, 0.85, 0.93])
        plot2_path = os.path.join(plot_dir, "mcts_kettle_action_indices.png")
        fig2.savefig(plot2_path, dpi=300)
        print(f"Action indices plot saved to '{plot2_path}'")
        plt.close(fig2)


if __name__ == "__main__":
    print("--- Starting MCTS Control for Kettle Environment ---")
    temps_data, actions_data, rewards_data = run_mcts_controlled_episodes()

    if temps_data:
        plot_target_temp = 100.0 
        plot_simulation_results(temps_data, actions_data, rewards_data, target_temp_plot=plot_target_temp)

        print("\n--- Overall MCTS Run Summary ---")
        for i in range(len(rewards_data)):
            print(f"Episode {i+1}:")
            final_temp_val_str = "N/A"
            if temps_data[i] and len(temps_data[i]) > 0:
                final_temp_val_str = f"{temps_data[i][-1]:.1f}" # Formatted
            print(f"  - Final Temp: {final_temp_val_str}°C (Target: {plot_target_temp:.1f}°C)") # Formatted
            print(f"  - Total Reward: {rewards_data[i]:.2f}")
            if actions_data[i]:
                on_c = sum(1 for act_idx in actions_data[i] if act_idx > 0) 
                off_c = len(actions_data[i]) - on_c
                total_actions = len(actions_data[i])
                if total_actions > 0:
                    print(f"  - Actions: {on_c} ON ({on_c/total_actions*100:.1f}%), "
                          f"{off_c} OFF ({off_c/total_actions*100:.1f}%)")
            else:
                print("  - Actions: No actions recorded.")
    else:
        print("No episodes were run or no data was recorded by MCTS.")

    print("\n--- MCTS Script Finished ---")