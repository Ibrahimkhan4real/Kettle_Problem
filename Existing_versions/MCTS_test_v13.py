import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from math import log, sqrt
import os # For creating directory for plots

# IMPORTANT: This script now assumes 'kettle_dynamic_env_v13.py' contains
# a KettleEnv version compatible with the simplified interface (e.g., v3_simplified_mcts),
# meaning it takes 'initial_temp', 'ambient_temp' in __init__ and has a simplified reward.
from kettle_dynamic_env_v20 import KettleEnv

# --- Configuration ---
EPISODES_TO_RUN = 1  # Number of episodes to run and plot
MAX_TIMESTEPS_PER_EPISODE = 250  # Max interaction steps for the MCTS agent per episode

# MCTS Hyperparameters
UCB_C = 6.5  # Exploration constant for UCB1. sqrt(2) is common, can be tuned.
MCTS_ITERATIONS_PER_ACTION = 1000  # Number of MCTS simulations (REDUCED from 30000)
MAX_ROLLOUT_DEPTH = 250


class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
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
        Propagates the simulation result up the tree.
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
    all_episodes_temperatures = []
    all_episodes_actions = []
    all_episodes_rewards = []

    for episode_idx in range(EPISODES_TO_RUN):
        # --- Environment Setup ---
        episode_initial_temp = 20.0
        episode_ambient_temp = 25.0
        kettle_env = KettleEnv(
            initial_temp=episode_initial_temp,
            ambient_temp=episode_ambient_temp,
        )

        current_obs, _ = kettle_env.reset()
        is_done_from_env = False
        is_truncated_from_env = False # Explicitly track truncation
        total_episode_reward_accumulated = 0.0

        current_episode_temps = [current_obs[0]]
        current_episode_actions_idx = []

        # No persistent mcts_current_root across timesteps anymore

        print(f"\n{'='*60}")
        print(f"Starting Episode {episode_idx+1}/{EPISODES_TO_RUN} (MCTS Tree Rebuilt Each Step)")
        print(f"{'='*60}")
        try: T_target_log = kettle_env.T_target
        except AttributeError: T_target_log = 100.0 # Default
        print(f"Initial Temp: {current_obs[0]:.2f}°C, Target: {T_target_log:.1f}°C")
        print(f"MCTS Iterations/Action: {MCTS_ITERATIONS_PER_ACTION}, Max Rollout Depth: {MAX_ROLLOUT_DEPTH}")

        for timestep_num in range(MAX_TIMESTEPS_PER_EPISODE):
            print(f"\n--- Ep {episode_idx+1}, Timestep {timestep_num+1}/{MAX_TIMESTEPS_PER_EPISODE} ---")
            print(f"Current State (Temp: {current_obs[0]:.2f}°C, TimeRem: {current_obs[1]:.0f}s)")

            # If the environment is already in a terminal state from the previous action, break.
            if is_done_from_env or is_truncated_from_env:
                term_reason_loop_check = "terminated (done)" if is_done_from_env else "truncated"
                print(f"Loop check: Episode ended at start of timestep {timestep_num+1} due to: {term_reason_loop_check}")
                break

            # 1. Create a NEW MCTS root node for the current environment state at each timestep.
            # The 'is_done_from_env' flag here refers to the status of 'current_obs'.
            # If current_obs is terminal, get_action_via_mcts will handle it.
            mcts_root_for_this_step = Node(
                env_state_copy=deepcopy(kettle_env), # Fresh copy of the current env state
                done_flag=is_done_from_env, # This is the done status OF current_obs
                parent_node=None,
                observation=current_obs,
                action_idx_leading_to_node=None
            )

            # 2. Perform MCTS to get an action.
            # The first returned value (the chosen child node) is not reused for the next timestep.
            _, chosen_action_index = get_action_via_mcts(mcts_root_for_this_step, MCTS_ITERATIONS_PER_ACTION)

            if chosen_action_index is None:
                print(f"MCTS did not return a valid action at timestep {timestep_num+1}. Ending episode.")
                if mcts_root_for_this_step and mcts_root_for_this_step.is_terminal:
                    print("  Reason: MCTS root node (freshly created) indicates a terminal state.")
                break # End episode if no action can be chosen

            action_power_W = kettle_env.actions[chosen_action_index]
            print(f"MCTS chose Action Index: {chosen_action_index} (Power: {action_power_W}W)")
            current_episode_actions_idx.append(chosen_action_index)

            # 3. Take the action in the actual environment
            next_obs, reward, is_done_from_env, is_truncated_from_env, info = kettle_env.step(chosen_action_index)

            current_episode_temps.append(next_obs[0])
            print(f"Env after action: New Temp = {next_obs[0]:.2f}°C, Reward: {reward:.3f}, Env Step: {info.get('current_step', 'N/A')}")

            total_episode_reward_accumulated += reward
            current_obs = next_obs # Update current_obs for the next iteration

            # The MCTS tree (mcts_root_for_this_step) and its statistics are discarded here.
            # A new one will be created in the next iteration based on the new 'current_obs'.

            if is_done_from_env or is_truncated_from_env:
                term_reason = "terminated (done)" if is_done_from_env else "truncated"
                print(f"Episode finished by environment at timestep {timestep_num+1} (Env step: {info.get('current_step', 'N/A')}) due to: {term_reason}.")
                break
        
        # Check if MAX_TIMESTEPS_PER_EPISODE was hit if not done/truncated
        if not (is_done_from_env or is_truncated_from_env) and timestep_num +1 >= MAX_TIMESTEPS_PER_EPISODE :
            print(f"Episode finished by reaching MCTS MAX_TIMESTEPS_PER_EPISODE ({MAX_TIMESTEPS_PER_EPISODE}).")

        # ... (rest of the episode summary and logging remains the same) ...
        print(f"\n{'*'*60}")
        print(f"Episode {episode_idx+1} Finished Summary:")
        print(f"{'*'*60}")
        print(f"Total Reward Accumulated: {total_episode_reward_accumulated:.2f}")
        print(f"Number of Timesteps Taken by MCTS agent: {len(current_episode_actions_idx)}")
        final_temp = current_episode_temps[-1] if current_episode_temps else current_obs[0]
        try: T_target_log_final = kettle_env.T_target 
        except AttributeError: T_target_log_final = 100.0
        print(f"Initial Temperature: {current_episode_temps[0]:.2f}°C")
        print(f"Final Temperature: {final_temp:.2f}°C (Target: {T_target_log_final:.1f}°C)")
        
        try: deadline_step_log = kettle_env.target_deadline_step
        except AttributeError: deadline_step_log = "N/A"
        print(f"Environment's Target Deadline Step: {deadline_step_log}")

        print("\nAction Sequence (Index: Power W):")
        try: action_map = kettle_env.actions
        except AttributeError: action_map = {i: f"idx_{i}" for i in range(kettle_env.action_space.n)} 

        for t, act_idx in enumerate(current_episode_actions_idx):
            power_val_str = str(action_map[act_idx]) + "W" if isinstance(action_map, list) else action_map.get(act_idx, "Unknown")
            temp_after = current_episode_temps[t+1]
            temp_before = current_episode_temps[t]
            temp_change = temp_after - temp_before
            print(f"  MCTS Step {t+1}: Act Idx {act_idx} ({power_val_str}) -> Temp: {temp_after:6.2f}°C (ΔT: {temp_change:+.2f}°C)")

        on_actions_count = sum(1 for act_idx in current_episode_actions_idx if (isinstance(action_map, list) and action_map[act_idx] > 0) or (not isinstance(action_map, list) and act_idx > 0) )
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
    """Plots the results from the MCTS controlled episodes."""
    if not ep_temps_data:
        print("No data available to plot.")
        return

    num_episodes_plotted = len(ep_temps_data)
    plot_dir = "Experiments/MCTS_Plots"
    os.makedirs(plot_dir, exist_ok=True) # Ensure directory exists

    # --- Plot 1: Temperatures and Actions ---
    fig1, ax_temp = plt.subplots(figsize=(15, 7))
    fig1.suptitle("Kettle Temperature Control via MCTS", fontsize=16)

    colors = plt.cm.viridis(np.linspace(0, 1, max(1, num_episodes_plotted)))

    for i in range(num_episodes_plotted):
        temps = ep_temps_data[i]
        actions_indices = ep_actions_data[i]
        reward_val = ep_rewards_data[i]
        time_steps_temps = range(len(temps))

        ax_temp.plot(
            time_steps_temps, temps, marker='o', markersize=3, linestyle='-', linewidth=1.5,
            color=colors[i if num_episodes_plotted > 1 else 0],
            label=f"Ep {i+1} Temp (Total R: {reward_val:.2f})"
        )
        for t_idx, action_idx_val in enumerate(actions_indices):
            if action_idx_val > 0: # Assuming action index 0 is OFF
                ax_temp.axvspan(t_idx, t_idx + 1, alpha=0.15, color=colors[i if num_episodes_plotted > 1 else 0], ymin=0, ymax=0.6)

    ax_temp.axhline(y=target_temp_plot, color="black", linestyle="--", linewidth=1.5, label=f"Target ({target_temp_plot}°C)")
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

    # --- Plot 2: Action Indices ---
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
        # Attempt to get target_temp from the last used env configuration for plotting, if possible.
        # This is a bit tricky as the env is closed. We'll use a default or a hardcoded value.
        plot_target_temp = 100.0 # Default or get from a config if stored globally
        # If you know the specific KettleEnv class has T_target as a class variable:
        # plot_target_temp = KettleEnv.T_target
        plot_simulation_results(temps_data, actions_data, rewards_data, target_temp_plot=plot_target_temp)

        print("\n--- Overall MCTS Run Summary ---")
        for i in range(len(rewards_data)):
            print(f"Episode {i+1}:")
            final_temp_val = temps_data[i][-1] if temps_data[i] and len(temps_data[i]) > 0 else "N/A"
            print(f"  - Final Temp: {final_temp_val}°C (Target: {plot_target_temp}°C)")
            print(f"  - Total Reward: {rewards_data[i]:.2f}")
            if actions_data[i]:
                on_c = sum(1 for act_idx in actions_data[i] if act_idx > 0) # Assuming index 0 is OFF
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