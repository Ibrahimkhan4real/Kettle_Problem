import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from math import log, sqrt
import os # For creating directory for plots

# IMPORTANT: This script now assumes 'kettle_dynamic_env_v13.py' contains
# a KettleEnv version compatible with the simplified interface (e.g., v3_simplified_mcts),
# meaning it takes 'initial_temp', 'ambient_temp' in __init__ and has a simplified reward.
from kettle_dynamic_env_v23 import KettleEnv

# --- Configuration ---
EPISODES_TO_RUN = 1  # Number of episodes to run and plot
MAX_TIMESTEPS_PER_EPISODE = 250  # Max interaction steps for the MCTS agent per episode

# MCTS Hyperparameters
UCB_C = 32.5  # Exploration constant for UCB1. sqrt(2) is common, can be tuned.
MCTS_ITERATIONS_PER_ACTION = 1000  # Number of MCTS simulations (REDUCED from 30000)
MAX_ROLLOUT_DEPTH = 250


class Node:
    """
    Represents a node in the Monte Carlo Search Tree
    Each node stores statistics for a particular environment state.
    """
    def __init__(self, env_state_copy, done_flag, parent_node, observation, action_idx_leading_to_node):
        self.children = {}  # Dictionary of child nodes: {action_index: Node_object}
        self.total_simulation_reward = 0.0  # Sum of rewards from rollouts (Q-value numerator)
        self.num_visits = 0  # Number of times this node has been visited (N)

        self.env_state = env_state_copy  # A deepcopy of the Gym environment at this node's state
        self.observation = observation # The observation corresponding to env_state_copy
        self.is_terminal = done_flag  # Flag indicating if this node's state is terminal

        self.parent = parent_node  # Reference to the parent node
        self.action_that_led_here = action_idx_leading_to_node

    def get_ucb1_score(self) -> float:
        """Calculates the UCB1 score for this node."""
        if self.num_visits == 0:
            return float("inf")  # Prioritize unvisited nodes

        if not self.parent: # Should ideally not be called on root for UCB selection against siblings
            parent_total_visits = self.num_visits
        else:
            parent_total_visits = self.parent.num_visits
        
        # If self.num_visits > 0, parent_total_visits should also be > 0.
        # max(1, parent_total_visits) ensures log argument is >=1
        exploitation_term = self.total_simulation_reward / self.num_visits
        exploration_term = UCB_C * sqrt(log(max(1, parent_total_visits)) / self.num_visits)

        return exploitation_term + exploration_term

    def detach_from_parent(self):
        """Detaches this node from its parent (e.g., when it becomes the new root)."""
        self.parent = None

    def expand(self):
        """
        Expands this node by creating all possible child nodes.
        """
        if self.is_terminal or self.children:  # Do not expand if terminal or already has children
            return

        num_possible_actions = self.env_state.action_space.n
        for action_idx in range(num_possible_actions):
            child_env_state_copy = deepcopy(self.env_state)
            obs, _, done, truncated, _ = child_env_state_copy.step(action_idx)
            child_is_terminal = done or truncated
            self.children[action_idx] = Node(child_env_state_copy, child_is_terminal, self, obs, action_idx)

    def rollout(self) -> float:
        """
        Performs a random simulation (rollout) from this node's state.
        """
        if self.is_terminal:
            return 0.0

        rollout_env = deepcopy(self.env_state)
        accumulated_rollout_reward = 0.0
        current_depth = 0
        is_rollout_done = False

        while not is_rollout_done and current_depth < MAX_ROLLOUT_DEPTH:
            random_action = rollout_env.action_space.sample()
            _, reward, term, trunc, _ = rollout_env.step(random_action)
            is_rollout_done = term or trunc
            accumulated_rollout_reward += reward
            current_depth += 1
        return accumulated_rollout_reward

    def backpropagate(self, reward_from_simulation):
        """
        Propagates the simulation result up the tree
        """
        current_node_in_path = self
        while current_node_in_path is not None:
            current_node_in_path.num_visits += 1
            current_node_in_path.total_simulation_reward += reward_from_simulation
            current_node_in_path = current_node_in_path.parent

    def select_best_child_for_action(self):
        """
        Selects the best action from children of this node (typically the root after MCTS).
        Chooses the child with the highest visit count.
        """
        if self.is_terminal:
            print("Warning: Trying to select action from a terminal node.")
            return self, None

        if not self.children:
            print("Warning: Node has no children to select from. MCTS might not have run or expanded sufficiently.")
            if not self.is_terminal: # Attempt to expand if it's a leaf, though ideally MCTS iterations handle this
                self.expand()
                if not self.children:
                    return self, None # Still no children
            else:
                return self, None

        max_visits = -1
        best_children_nodes = []
        for child_node in self.children.values():
            if child_node.num_visits > max_visits:
                max_visits = child_node.num_visits
                best_children_nodes = [child_node]
            elif child_node.num_visits == max_visits:
                best_children_nodes.append(child_node)

        if not best_children_nodes: # Should not happen if children exist and MCTS ran
            print("Error: No children found with visits. Fallback to random if any children exist.")
            chosen_child_node = random.choice(list(self.children.values())) if self.children else None
        else:
            chosen_child_node = random.choice(best_children_nodes) # Random tie-breaking

        if not chosen_child_node:
            return self, None # Could not select a child

        action_to_take_idx = chosen_child_node.action_that_led_here
        chosen_child_node.detach_from_parent()
        return chosen_child_node, action_to_take_idx


def mcts_single_iteration(root_node: Node):
    """Performs one full MCTS iteration: Selection, Expansion, Simulation, Backpropagation."""

    # 1. Selection
    current_selection = root_node
    while current_selection.children:
        child_nodes = current_selection.children
        best_child_node = None
        max_ucb_score = -float('inf')
        child_action_keys = list(child_nodes.keys())
        random.shuffle(child_action_keys) # For random tie-breaking

        for action_key in child_action_keys:
            node = child_nodes[action_key]
            ucb_score = node.get_ucb1_score()
            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_child_node = node
        
        current_selection = best_child_node if best_child_node else random.choice(list(child_nodes.values()))


    # 2. Expansion & Selection for Rollout
    node_to_simulate_from = current_selection
    if not current_selection.is_terminal:
        if current_selection.num_visits > 0: # If visited before (and not terminal), expand it
            current_selection.expand()
            if current_selection.children: # If expansion created children
                # Prefer an unvisited child for rollout
                unvisited_children = [child for child in current_selection.children.values() if child.num_visits == 0]
                if unvisited_children:
                    node_to_simulate_from = random.choice(unvisited_children)
                else: # All children visited, pick one (e.g. current_selection has become internal, pick any child)
                    node_to_simulate_from = random.choice(list(current_selection.children.values()))
        # If current_selection.num_visits == 0, it's a new leaf, simulate from it.

    # 3. Simulation (Rollout)
    simulation_reward = node_to_simulate_from.rollout()

    # 4. Backpropagation
    node_to_simulate_from.backpropagate(simulation_reward)


def get_action_via_mcts(mcts_root_node: Node, num_iterations: int):
    """
    Runs MCTS simulations from mcts_root_node to determine the best action.
    """
    if mcts_root_node.is_terminal:
        print("MCTS: Root node is terminal. No action to select.")
        return mcts_root_node, None

    for i in range(num_iterations):
        mcts_single_iteration(mcts_root_node)
        # if (i + 1) % (num_iterations // 10 or 1) == 0 and num_iterations >= 100 :
        #     print(f"   MCTS progress: {i+1}/{num_iterations} iterations done.")


    next_mcts_root_node, chosen_action_idx = mcts_root_node.select_best_child_for_action()
    
    
    
    # Inside get_action_via_mcts(mcts_root_node: Node, num_iterations: int),
    # AFTER the for loop for iterations:

    print(f"MCTS Root (Temp: {mcts_root_node.observation[0]:.2f}, Child Stats:")
    if mcts_root_node.children:
        # You'll need access to the env's action meanings if possible, or just use index
        # temp_env_for_actions = mcts_root_node.env_state # The env copy in the node
        action_meanings = {idx: f"{power}W" for idx, power in enumerate(mcts_root_node.env_state.actions)}

        for action_idx, child_node in sorted(mcts_root_node.children.items()):
            avg_q_value = (child_node.total_simulation_reward / child_node.num_visits) if child_node.num_visits > 0 else -float('inf')
            ucb_score = child_node.get_ucb1_score() # Recalculate for display if needed, or store during selection
            print(f"  Action {action_idx} ({action_meanings.get(action_idx, 'N/A')}): "
                f"Visits={child_node.num_visits}, AvgQ={avg_q_value:.3f}, UCB={ucb_score:.3f}") # UCB score might be inf for unvisited
    else:
        print("  MCTS Root has no children after iterations (should not happen if iterations > 0 and not terminal).")
    
    
    
    return next_mcts_root_node, chosen_action_idx


def run_mcts_controlled_episodes():
    """Runs episodes of the KettleEnv controlled by MCTS and logs data.
    MCTS tree is rebuilt at each timestep.
    """
    all_episodes_temperatures_diff = [] # Will store temperature differences
    all_episodes_actions = []
    all_episodes_total_rewards = [] # Stores total reward per episode
    all_episodes_step_rewards = []  # NEW: To store per-step rewards for each episode

    for episode_idx in range(EPISODES_TO_RUN):
        # --- Environment Setup ---
        episode_initial_temp = 20.0
        episode_ambient_temp = 25.0 # As in your MCTS script
        kettle_env = KettleEnv(
            initial_temp=episode_initial_temp,
            ambient_temp=episode_ambient_temp,
            # It will use its internal defaults for max_steps, target_deadline_step, T_target etc.
        )

        current_obs, _ = kettle_env.reset()
        is_done_from_env = False
        is_truncated_from_env = False 
        total_episode_reward_accumulated = 0.0

        # current_episode_temps now stores temp_diff from observations
        current_episode_temp_diffs = [current_obs[0]] 
        current_episode_actions_idx = []
        current_episode_single_step_rewards = [] # NEW: For this episode's per-step rewards

        # ... (print statements for episode start, etc. from your script) ...
        print(f"\n{'='*60}")
        print(f"Starting Episode {episode_idx+1}/{EPISODES_TO_RUN} (MCTS Tree Rebuilt Each Step)")
        print(f"{'='*60}")
        try: 
            T_target_log = kettle_env.T_target
            # For converting temp_diff back to actual temp, we need T_target
            # It's better to get it from the env instance.
        except AttributeError: 
            T_target_log = 100.0 # Fallback
        
        # The current_obs[0] is temp_diff, current_obs[1] is steps_diff
        initial_actual_temp = current_obs[0] + T_target_log 
        print(f"Initial Temp (Actual): {initial_actual_temp:.1f}°C (Diff: {current_obs[0]:.1f}), Target: {T_target_log:.1f}°C") # Updated print
        print(f"MCTS Iterations/Action: {MCTS_ITERATIONS_PER_ACTION}, Max Rollout Depth: {MAX_ROLLOUT_DEPTH}")


        for timestep_num in range(MAX_TIMESTEPS_PER_EPISODE):
            # ... (print statements for timestep start, etc. from your script) ...
            # Current State print should show actual temp for clarity
            current_actual_temp_for_print = current_obs[0] + T_target_log
            print(f"\n--- Ep {episode_idx+1}, Timestep {timestep_num+1}/{MAX_TIMESTEPS_PER_EPISODE} ---")
            print(f"Current State (Actual Temp: {current_actual_temp_for_print:.1f}°C, TempDiff: {current_obs[0]:.1f}, StepsToDeadline: {current_obs[1]:.0f}s)")


            if is_done_from_env or is_truncated_from_env:
                # ... (existing break logic) ...
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
            
            _, chosen_action_index = get_action_via_mcts(mcts_root_for_this_step, MCTS_ITERATIONS_PER_ACTION)

            if chosen_action_index is None:
                # ... (existing break logic) ...
                print(f"MCTS did not return a valid action at timestep {timestep_num+1}. Ending episode.")
                if mcts_root_for_this_step and mcts_root_for_this_step.is_terminal:
                    print("  Reason: MCTS root node (freshly created) indicates a terminal state.")
                break

            action_power_W = kettle_env.actions[chosen_action_index]
            # ... (existing print for chosen action) ...
            print(f"MCTS chose Action Index: {chosen_action_index} (Power: {action_power_W}W)")
            current_episode_actions_idx.append(chosen_action_index)

            next_obs, reward, is_done_from_env, is_truncated_from_env, info = kettle_env.step(chosen_action_index)

            current_episode_temp_diffs.append(next_obs[0]) # Log temp_diff
            current_episode_single_step_rewards.append(reward) # NEW: Log per-step reward
            
            # Env after action print should show actual temp for clarity
            next_actual_temp_for_print = next_obs[0] + T_target_log
            print(f"Env after action: New Actual Temp = {next_actual_temp_for_print:.1f}°C (Diff: {next_obs[0]:.1f}), Reward: {reward:.3f}, Env Step: {info.get('current_step', 'N/A')}")


            total_episode_reward_accumulated += reward
            current_obs = next_obs

            if is_done_from_env or is_truncated_from_env:
                # ... (existing break logic and print) ...
                term_reason = "terminated (done)" if is_done_from_env else "truncated"
                print(f"Episode finished by environment at timestep {timestep_num+1} (Env step: {info.get('current_step', 'N/A')}) due to: {term_reason}.")
                break
        
        # ... (existing end of episode logic and prints for MAX_TIMESTEPS_PER_EPISODE) ...
        if not (is_done_from_env or is_truncated_from_env) and timestep_num +1 >= MAX_TIMESTEPS_PER_EPISODE :
            print(f"Episode finished by reaching MCTS MAX_TIMESTEPS_PER_EPISODE ({MAX_TIMESTEPS_PER_EPISODE}).")

        # --- Log episode data ---
        # Store temp_diffs, actions, total reward, and per-step rewards
        all_episodes_temperatures_diff.append(current_episode_temp_diffs)
        all_episodes_actions.append(current_episode_actions_idx)
        all_episodes_total_rewards.append(total_episode_reward_accumulated)
        all_episodes_step_rewards.append(current_episode_single_step_rewards) # NEW

        # --- Print Episode Summary (using actual temperatures) ---
        print(f"\n{'*'*60}")
        print(f"Episode {episode_idx+1} Finished Summary:")
        print(f"{'*'*60}")
        print(f"Total Reward Accumulated: {total_episode_reward_accumulated:.2f}")
        print(f"Number of Timesteps Taken by MCTS agent: {len(current_episode_actions_idx)}")
        
        initial_actual_temp_summary = current_episode_temp_diffs[0] + T_target_log
        final_actual_temp_summary = current_episode_temp_diffs[-1] + T_target_log
        
        print(f"Initial Temperature (Actual): {initial_actual_temp_summary:.1f}°C")
        print(f"Final Temperature (Actual): {final_actual_temp_summary:.1f}°C (Target: {T_target_log:.1f}°C)")
        
        try: deadline_step_log = kettle_env.target_deadline_step
        except AttributeError: deadline_step_log = "N/A"
        print(f"Environment's Target Deadline Step: {deadline_step_log}")

        print("\nAction Sequence (Index: Power W):")
        try: action_map_summary = kettle_env.actions
        except AttributeError: action_map_summary = {i: f"idx_{i}" for i in range(kettle_env.action_space.n)} 

        for t, act_idx in enumerate(current_episode_actions_idx):
            power_val_str = str(action_map_summary[act_idx]) + "W"
            # temp_after and temp_before for delta calculation should use actual temps
            temp_before_actual = current_episode_temp_diffs[t] + T_target_log
            temp_after_actual = current_episode_temp_diffs[t+1] + T_target_log
            temp_change_actual = temp_after_actual - temp_before_actual
            print(f"  MCTS Step {t+1}: Act Idx {act_idx} ({power_val_str}) -> Actual Temp: {temp_after_actual:6.1f}°C (ΔT: {temp_change_actual:+.1f}°C)")

        # ... (on_actions_count, off_actions_count, etc. from your script) ...
        on_actions_count = sum(1 for act_idx_val in current_episode_actions_idx if action_map_summary[act_idx_val] > 0)
        off_actions_count = len(current_episode_actions_idx) - on_actions_count
        if current_episode_actions_idx:
            on_percent = (on_actions_count / len(current_episode_actions_idx)) * 100
            print(f"\nAction Summary: {on_actions_count} ON ({on_percent:.1f}%), "
                  f"{off_actions_count} OFF ({(100-on_percent):.1f}%)")


        # Gather env parameters for plotting before closing the env
        env_action_map = kettle_env.actions
        env_target_deadline = kettle_env.target_deadline_step
        env_dt = kettle_env.dt 
        env_T_target = kettle_env.T_target # Get actual T_target from env

        kettle_env.close()

    # Pass the new parameters to the plotting function
    # Note: the first argument is now temp_diff data
    return all_episodes_temperatures_diff, all_episodes_actions, all_episodes_total_rewards, all_episodes_step_rewards, \
           env_action_map, env_target_deadline, env_dt, env_T_target

def plot_simulation_results(
    ep_temps_diff_data,       # List of lists of (temp_diff)
    ep_actions_data,          # List of lists of action_indices
    ep_total_rewards_data,    # List of total rewards per episode
    ep_step_rewards_data,     # NEW: List of lists of per-step rewards
    env_action_map,           # New: list of power values for actions
    env_target_deadline,      # New: target deadline step from env
    env_dt,                   # New: time step duration from env
    env_T_target              # New: Actual target temperature from env
    ):
    """Plots 4 meaningful graphs from the MCTS controlled episodes."""
    if not ep_temps_diff_data:
        print("No data available to plot.")
        return

    num_episodes_plotted = len(ep_temps_diff_data)
    plot_dir = "Experiments/MCTS_Analysis_Plots" # New directory name
    os.makedirs(plot_dir, exist_ok=True)

    # Common colors for multi-episode plots if needed, or use default cycling
    # colors = plt.cm.viridis(np.linspace(0, 1, max(1, num_episodes_plotted)))

    # --- Plot 1: Actual Temperature & Action Shading vs. Time Step ---
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    fig1.suptitle("Kettle Actual Temperature & Actions vs. Time", fontsize=16)
    for i in range(num_episodes_plotted):
        temp_diffs = ep_temps_diff_data[i]
        actual_temps = [td + env_T_target for td in temp_diffs] # Convert to actual temp
        actions_indices = ep_actions_data[i]
        total_ep_reward = ep_total_rewards_data[i]
        time_steps = range(len(actual_temps))

        ax1.plot(
            time_steps, actual_temps, marker='.', markersize=4, linestyle='-', linewidth=1.5,
            label=f"Ep {i+1} Actual Temp (Total R: {total_ep_reward:.2f})"
        )
        # Action shading
        for t_idx, action_idx_val in enumerate(actions_indices):
            if env_action_map[action_idx_val] > 0: # Heater ON
                ax1.axvspan(t_idx + 0.5 , t_idx + 1.5, alpha=0.15, color='orange', ymin=0, ymax=0.7)
                # Shading between step t and t+1 where action at step t takes effect
                # The X-axis for actions_indices is 0 to N-1. The X-axis for temps is 0 to N.
                # Action at t_idx influences temp from time_steps[t_idx] to time_steps[t_idx+1]
                # So shade from t_idx to t_idx+1 is good.
    ax1.axhline(y=env_T_target, color="red", linestyle="--", linewidth=2, label=f"Target ({env_T_target:.1f}°C)")
    ax1.axvline(x=env_target_deadline, color="magenta", linestyle="--", linewidth=2, label=f"Deadline (Step {env_target_deadline})")
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Actual Temperature (°C)", fontsize=12)
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=':', alpha=0.7)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    plot1_path = os.path.join(plot_dir, f"plot1_actual_temp_vs_time.png")
    fig1.savefig(plot1_path, dpi=300)
    print(f"\n1. Actual Temperature plot saved to '{plot1_path}'")
    plt.close(fig1)

    # --- Plot 2: Cumulative Energy Consumed (Wh) vs. Time Step ---
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    fig2.suptitle("Cumulative Energy Consumption vs. Time", fontsize=16)
    for i in range(num_episodes_plotted):
        actions_indices = ep_actions_data[i]
        cumulative_energy_Wh = [0.0]
        current_energy_J = 0.0
        for action_idx_val in actions_indices:
            power_W = env_action_map[action_idx_val]
            current_energy_J += power_W * env_dt 
            cumulative_energy_Wh.append(current_energy_J / 3600.0)
        
        time_steps_energy = range(len(cumulative_energy_Wh)) # Should be N_actions + 1
        ax2.plot(
            time_steps_energy, cumulative_energy_Wh, marker='.', markersize=4, linestyle='-', linewidth=1.5,
            label=f"Ep {i+1} Cum. Energy"
        )
    ax2.set_xlabel("Time Step", fontsize=12)
    ax2.set_ylabel("Cumulative Energy Consumed (Wh)", fontsize=12)
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=':', alpha=0.7)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    plot2_path = os.path.join(plot_dir, f"plot2_cumulative_energy_vs_time.png")
    fig2.savefig(plot2_path, dpi=300)
    print(f"2. Cumulative Energy plot saved to '{plot2_path}'")
    plt.close(fig2)

    # --- Plot 3: Action Power Levels (W) vs. Time Step ---
    fig3, ax3 = plt.subplots(figsize=(15, 7))
    fig3.suptitle("Heater Power Applied vs. Time", fontsize=14)
    for i in range(num_episodes_plotted):
        actions_indices = ep_actions_data[i]
        if actions_indices:
            time_steps_actions = range(len(actions_indices))
            power_values = [env_action_map[idx] for idx in actions_indices]
            ax3.step(
                time_steps_actions, power_values, where="post",
                linewidth=1.5, label=f"Ep {i+1} Power"
            )
    ax3.set_ylabel("Heater Power (W)", fontsize=12)
    ax3.set_xlabel("Time Step", fontsize=12)
    # Set y-ticks to actual unique power levels for clarity
    unique_power_levels = sorted(list(set(env_action_map)))
    if len(unique_power_levels) > 1 : # Avoid setting ticks if only one power level (e.g. only 0W if never heats)
         ax3.set_yticks(unique_power_levels)
    ax3.legend(loc="best")
    ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    plot3_path = os.path.join(plot_dir, f"plot3_action_power_vs_time.png")
    fig3.savefig(plot3_path, dpi=300)
    print(f"3. Action Power plot saved to '{plot3_path}'")
    plt.close(fig3)

    # --- Plot 4: Per-Step Reward vs. Time Step ---
    fig4, ax4 = plt.subplots(figsize=(15, 7))
    fig4.suptitle("Per-Step Reward vs. Time", fontsize=16)
    for i in range(num_episodes_plotted):
        step_rewards = ep_step_rewards_data[i]
        if step_rewards:
            time_steps_rewards = range(len(step_rewards))
            ax4.plot(
                time_steps_rewards, step_rewards, marker='.', markersize=4, linestyle='-', linewidth=1.0, alpha=0.8,
                label=f"Ep {i+1} Step Reward (Total: {ep_total_rewards_data[i]:.2f})"
            )
    ax4.set_xlabel("Time Step", fontsize=12)
    ax4.set_ylabel("Reward Received", fontsize=12)
    ax4.legend(loc="best")
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.axhline(y=0, color="grey", linestyle="--", linewidth=0.8) # Zero reward line
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    plot4_path = os.path.join(plot_dir, f"plot4_step_reward_vs_time.png")
    fig4.savefig(plot4_path, dpi=300)
    print(f"4. Per-Step Reward plot saved to '{plot4_path}'")
    plt.close(fig4)

    # Optional: Plot 5 - Mean Temperature Profile (if multiple episodes)
    if num_episodes_plotted > 1:
        fig5, ax_mean_temp = plt.subplots(figsize=(15, 7))
        fig5.suptitle("Mean Actual Temperature Profile Across Episodes", fontsize=16)

        max_len = 0
        for ep_td in ep_temps_diff_data: # ep_td is a list of temp_diffs for one episode
            if len(ep_td) > max_len:
                max_len = len(ep_td)
        
        padded_actual_temps = np.full((num_episodes_plotted, max_len), np.nan)
        for i, ep_td in enumerate(ep_temps_diff_data):
            actual_temps_ep = [td_val + env_T_target for td_val in ep_td]
            padded_actual_temps[i, :len(actual_temps_ep)] = actual_temps_ep
            
        mean_temps = np.nanmean(padded_actual_temps, axis=0)
        std_temps = np.nanstd(padded_actual_temps, axis=0)
        time_steps_mean = range(max_len)

        ax_mean_temp.plot(time_steps_mean, mean_temps, color='black', linewidth=2, label="Mean Actual Temperature")
        ax_mean_temp.fill_between(
            time_steps_mean, mean_temps - std_temps, mean_temps + std_temps, 
            color='skyblue', alpha=0.4, label="Std. Deviation"
        )
        ax_mean_temp.axhline(y=env_T_target, color="red", linestyle="--", linewidth=1.5, label=f"Target Temp ({env_T_target:.1f}°C)")
        ax_mean_temp.axvline(x=env_target_deadline, color="magenta", linestyle="--", linewidth=1.5, label=f"Deadline (Step {env_target_deadline})")
        ax_mean_temp.set_xlabel("Timestep", fontsize=12)
        ax_mean_temp.set_ylabel("Actual Temperature (°C)", fontsize=12)
        ax_mean_temp.legend(loc="best")
        ax_mean_temp.grid(True, linestyle=':', alpha=0.7)
        fig5.tight_layout(rect=[0, 0, 1, 0.96])
        plot5_path = os.path.join(plot_dir, f"plot5_mean_temp_profile.png")
        fig5.savefig(plot5_path, dpi=300)
        print(f"5. Mean temperature profile plot saved to '{plot5_path}'")
        plt.close(fig5)


if __name__ == "__main__":
    print("--- Starting MCTS Control for Kettle Environment ---")

    # run_mcts_controlled_episodes now returns more data
    temps_diff_data, actions_data, total_rewards_data, step_rewards_data, \
    env_action_map, env_target_deadline, env_dt, env_T_target = run_mcts_controlled_episodes()

    if temps_diff_data:
        plot_simulation_results(
            temps_diff_data,
            actions_data,
            total_rewards_data,
            step_rewards_data, # Pass new data
            env_action_map,
            env_target_deadline,
            env_dt,
            env_T_target # Pass actual target temp from env
            # REMOVE: target_temp_plot=env_T_target
        )

        print("\n--- Overall MCTS Run Summary ---")
        for i in range(len(total_rewards_data)): # Iterate based on total_rewards_data
            print(f"Episode {i+1}:")
            # Ensure temps_diff_data[i] is not empty before accessing
            final_temp_diff = temps_diff_data[i][-1] if temps_diff_data[i] else float('nan')
            final_actual_temp_str = f"{(final_temp_diff + env_T_target):.1f}" if not np.isnan(final_temp_diff) else "N/A"
            
            print(f"  - Final Actual Temp: {final_actual_temp_str}°C (Target: {env_T_target:.1f}°C)")
            print(f"  - Total Reward: {total_rewards_data[i]:.2f}")
            if actions_data[i]:
                # Ensure env_action_map is valid for indexing
                on_c = sum(1 for act_idx in actions_data[i] if env_action_map[act_idx] > 0) 
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