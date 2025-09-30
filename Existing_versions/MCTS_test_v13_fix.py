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
        # Mark timesteps when heater is ON with shaded regions
        for t_idx, action_idx_val in enumerate(actions_indices):
            if action_idx_val > 0: # Assuming action index 0 is OFF
                ax_temp.axvspan(t_idx, t_idx + 1, alpha=0.15, color=colors[i if num_episodes_plotted > 1 else 0], ymin=0, ymax=0.6)

    # Add target temperature line
    ax_temp.axhline(y=target_temp_plot, color="black", linestyle="--", linewidth=1.5, label=f"Target ({target_temp_plot}°C)")
    
    # Set clear X and Y axis labels
    ax_temp.set_xlabel("Timestep", fontsize=12)
    ax_temp.set_ylabel("Kettle Temperature (°C)", fontsize=12)
    
    # Add tick marks at regular intervals on X-axis
    max_timesteps = max([len(temps) for temps in ep_temps_data])
    tick_interval = max(1, max_timesteps // 10)  # Ensure at least 10 ticks if possible
    ax_temp.set_xticks(range(0, max_timesteps, tick_interval))
    
    ax_temp.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax_temp.grid(True, linestyle=':', alpha=0.7)
    
    # Add horizontal grid lines at regular temperature intervals
    all_temp_flat = [t for ep_t in ep_temps_data for t in ep_t]
    if all_temp_flat:
        # Make sure y-axis extends at least to the target temperature
        y_min_temp = min(all_temp_flat) - 5
        y_max_temp = max(max(all_temp_flat) + 10, target_temp_plot + 5)
        ax_temp.set_ylim(max(0, y_min_temp), y_max_temp)
        
        # Add horizontal grid lines at regular temperature intervals
        temp_interval = 10  # Every 10°C
        temp_ticks = range(int(max(0, y_min_temp)), int(y_max_temp) + temp_interval, temp_interval)
        ax_temp.set_yticks(temp_ticks)

    fig1.tight_layout(rect=[0, 0, 0.83, 0.95])
    plot1_path = os.path.join(plot_dir, "mcts_kettle_temperatures_actions.png")
    fig1.savefig(plot1_path, dpi=300)
    print(f"\nTemperature & Actions plot saved to '{plot1_path}'")
    plt.close(fig1)
    
    # --- Plot 2: Action Indices ---
    if any(ep_actions_data):
        fig2, ax_actions = plt.subplots(figsize=(15, 4))
        fig2.suptitle("MCTS Agent Actions (Heater Power Settings)", fontsize=14)
        
        # Get the action power values from the environment
        try:
            # Try to create a temporary environment to get the action values
            temp_env = KettleEnv()
            action_power_values = temp_env.actions
            temp_env.close()
        except:
            # If it fails, use action indices
            action_power_values = None
        
        for i in range(num_episodes_plotted):
            actions_indices = ep_actions_data[i]
            if actions_indices:
                time_steps_actions = range(len(actions_indices))
                
                if action_power_values is not None:
                    # Convert action indices to power values for y-axis
                    power_values = [action_power_values[idx] for idx in actions_indices]
                    ax_actions.step(
                        time_steps_actions, power_values, where="post",
                        color=colors[i if num_episodes_plotted > 1 else 0],
                        linewidth=1.5, label=f"Ep {i+1} Power (W)"
                    )
                    ax_actions.set_ylabel("Power (Watts)", fontsize=12)
                else:
                    # Use action indices if power values not available
                    ax_actions.step(
                        time_steps_actions, actions_indices, where="post",
                        color=colors[i if num_episodes_plotted > 1 else 0],
                        linewidth=1.5, label=f"Ep {i+1} Action Idx"
                    )
                    ax_actions.set_ylabel("Action Index", fontsize=12)
                
                # Add tick marks at regular intervals on X-axis
                max_timesteps = max([len(actions) for actions in ep_actions_data])
                tick_interval = max(1, max_timesteps // 10)  # Ensure at least 10 ticks if possible
                ax_actions.set_xticks(range(0, max_timesteps, tick_interval))
        
        ax_actions.set_xlabel("Timestep", fontsize=12)
        ax_actions.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax_actions.grid(True, linestyle=':', alpha=0.6)
        fig2.tight_layout(rect=[0, 0, 0.85, 0.93])
        plot2_path = os.path.join(plot_dir, "mcts_kettle_action_indices.png")
        fig2.savefig(plot2_path, dpi=300)
        print(f"Action indices plot saved to '{plot2_path}'")
        plt.close(fig2)
