import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch as th
import os

# Import your fixed environment
from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv  # Replace with your actual module name


import numpy as np
import torch as th
import os
from stable_baselines3.common.callbacks import BaseCallback


class InstabilityMonitorCallback(BaseCallback):
    """
    Fixed version. Key changes:
    1. Jacobian computed without epsilon cap — uses float64 for accuracy
    2. log_ratio computed by comparing current policy to a snapshot
       taken at the START of each rollout, not from the buffer
    3. Entropy fallback for StateDependentNoiseDistribution
    """

    def __init__(self, log_freq=500, save_path="results/instability_monitor", verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.data = {
            "timestep":            [],
            "policy_mean_abs_max": [],
            "log_std_mean":        [],
            "log_std_min":         [],
            "jacobian_mean":       [],
            "jacobian_max":        [],
            "jacobian_true_max":   [],  # float64 version, no epsilon cap
            "log_ratio_max":       [],
            "entropy":             [],
            "approx_kl":           [],
            "pg_loss":             [],
        }

        # Snapshot of log-probs taken at rollout start for ratio computation
        self._snapshot_log_probs = None
        self._snapshot_obs       = None
        self._snapshot_actions   = None
        self._last_log_step      = 0

    # ------------------------------------------------------------------
    def _on_rollout_start(self):
        """
        Take a snapshot of the current policy's log-probs at the
        START of each rollout. Used to compute log-ratio at log time.
        """
        buf = self.model.rollout_buffer
        if buf.pos == 0 and not buf.full:
            return

        n = buf.buffer_size if buf.full else buf.pos
        obs_np = buf.observations[:n].reshape(-1, buf.observations.shape[-1])
        act_np = buf.actions[:n].reshape(-1, buf.actions.shape[-1])

        policy = self.model.policy
        obs_t = th.tensor(obs_np, dtype=th.float32, device=policy.device)
        act_t = th.tensor(act_np, dtype=th.float32, device=policy.device)

        with th.no_grad():
            _, log_prob, _ = policy.evaluate_actions(obs_t, act_t)
            self._snapshot_log_probs = log_prob.cpu().numpy()
            self._snapshot_obs       = obs_np
            self._snapshot_actions   = act_np

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_log_step) < self.log_freq:
            return True
        self._last_log_step = self.num_timesteps

        buf = self.model.rollout_buffer
        if buf.pos == 0 and not buf.full:
            return True

        try:
            self._log_signals()
        except Exception as e:
            if self.verbose > 0:
                print(f"[Monitor] Warning at step {self.num_timesteps}: {e}")
        return True

    # ------------------------------------------------------------------
    def _log_signals(self):
        buf    = self.model.rollout_buffer
        policy = self.model.policy

        n       = buf.buffer_size if buf.full else buf.pos
        obs_np  = buf.observations[:n].reshape(-1, buf.observations.shape[-1])
        act_np  = buf.actions[:n].reshape(-1, buf.actions.shape[-1])

        obs_t = th.tensor(obs_np, dtype=th.float32, device=policy.device)
        act_t = th.tensor(act_np, dtype=th.float32, device=policy.device)

        with th.no_grad():

            # --- Forward pass ---
            values, log_prob, entropy = policy.evaluate_actions(obs_t, act_t)

            # --- 1. Pre-squash policy mean ---
            features  = policy.extract_features(obs_t)
            latent_pi, _ = policy.mlp_extractor(features)
            mu = policy.action_net(latent_pi)          # pre-squash mean, shape (B, d)
            policy_mean_abs_max = mu.abs().max().item()

            # --- 2. Jacobian correction — float32 version (will saturate) ---
            tanh_mu = th.tanh(mu)
            one_minus_sq_f32 = 1.0 - tanh_mu ** 2 + 1e-8
            jac_f32 = -th.log(one_minus_sq_f32).sum(dim=-1)
            jacobian_mean = jac_f32.mean().item()
            jacobian_max  = jac_f32.max().item()

            # --- 3. TRUE Jacobian — float64 for accurate measurement ---
            # Uses the algebraically stable identity from Eq.4 of setup:
            # -log(1 - tanh²(u)) = 2u - 2*log2 + 2*log(1 + exp(-2u))
            # This does not saturate and gives the true value at large u
            mu_f64 = mu.double()
            log2   = th.log(th.tensor(2.0, dtype=th.float64))
            jac_true = (
                2.0 * mu_f64.abs()
                - 2.0 * log2
                + 2.0 * th.log1p(th.exp(-2.0 * mu_f64.abs()))
            ).sum(dim=-1)
            jacobian_true_max = jac_true.max().item()

            # --- 4. log_std ---
            log_std_mean = float("nan")
            log_std_min  = float("nan")
            dist = policy.action_dist
            if hasattr(dist, "log_std") and dist.log_std is not None:
                ls = dist.log_std.detach()
                log_std_mean = ls.mean().item()
                log_std_min  = ls.min().item()
            elif hasattr(policy, "log_std"):
                ls = policy.log_std.detach()
                log_std_mean = ls.mean().item()
                log_std_min  = ls.min().item()

            # --- 5. Log-ratio using rollout-start snapshot ---
            # This compares current policy to what it was at rollout start
            log_ratio_max = float("nan")
            if self._snapshot_log_probs is not None:
                snap_lp = th.tensor(
                    self._snapshot_log_probs,
                    dtype=th.float32,
                    device=policy.device
                )
                # Only compare where shapes match
                min_len = min(len(log_prob), len(snap_lp))
                log_ratio = (log_prob[:min_len] - snap_lp[:min_len]).abs()
                log_ratio_max = log_ratio.max().item()

            # --- 6. Entropy ---
            if entropy is not None:
                ent = entropy.mean().item()
            else:
                # Manual entropy for squashed Gaussian when SB3 returns None
                # H = 0.5*(1 + log(2π)) + log_std_mean - E[J(u)]
                # Approximation using current mu and log_std
                if not np.isnan(log_std_mean):
                    ent = (
                        0.5 * (1.0 + np.log(2 * np.pi))
                        + log_std_mean
                        - jacobian_mean
                    )
                else:
                    ent = float("nan")

            # --- 7. approx_kl and pg_loss from SB3 logger ---
            approx_kl = self._read_logger_value("train/approx_kl")
            pg_loss   = self._read_logger_value("train/policy_gradient_loss")

        # --- Store ---
        self.data["timestep"].append(self.num_timesteps)
        self.data["policy_mean_abs_max"].append(policy_mean_abs_max)
        self.data["log_std_mean"].append(log_std_mean)
        self.data["log_std_min"].append(log_std_min)
        self.data["jacobian_mean"].append(jacobian_mean)
        self.data["jacobian_max"].append(jacobian_max)
        self.data["jacobian_true_max"].append(jacobian_true_max)
        self.data["log_ratio_max"].append(log_ratio_max)
        self.data["entropy"].append(ent)
        self.data["approx_kl"].append(approx_kl)
        self.data["pg_loss"].append(pg_loss)

        if self.verbose > 0:
            print(
                f"[Monitor] step={self.num_timesteps:>8d} | "
                f"mu_max={policy_mean_abs_max:7.2f} | "
                f"log_std_min={log_std_min:6.3f} | "
                f"J_true_max={jacobian_true_max:8.1f} | "
                f"ratio_max={log_ratio_max:6.3f} / 88.72 | "
                f"KL={approx_kl:.4f}"
            )

    # ------------------------------------------------------------------
    def _read_logger_value(self, key):
        try:
            return float(
                self.model.logger.name_to_value.get(key, float("nan"))
            )
        except Exception:
            return float("nan")

    def _on_rollout_end(self):
        self._save()

    def _on_training_end(self):
        self._save()

    def _save(self):
        path = os.path.join(self.save_path, "monitor_data.npz")
        arrays = {k: np.array(v, dtype=np.float64) for k, v in self.data.items()}
        np.savez(path, **arrays)


# Instability Plot

import matplotlib.pyplot as plt
import numpy as np


def plot_instability_traces(
    npz_path: str,
    crash_step: int = None,
    save_path: str = "results/instability_traces.png"
):
    """
    Plot all six instability signals on a shared time axis.
    Pass crash_step (the timestep where NaN occurred) to draw
    a vertical crash marker.

    This produces Figure 2 of the paper.
    """
    d = np.load(npz_path)
    steps = d["timestep"]

    overflow_threshold = 88.72  # float32 log-overflow threshold

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    panels = [
        # (key, ylabel, title, reference_line_value, reference_label)
        ("policy_mean_abs_max", "|μ(s)|_max",
         "Policy mean abs max (boundary-seeking signal)",
         None, None),

        ("log_std_min",         "log σ_min",
         "gSDE log-std minimum (variance collapse signal)",
         None, None),

        ("jacobian_true_max",        "J(u)_true_max",
         "Jacobian correction max (singularity signal)",
         None, None),

        ("log_ratio_max",       "|Δ log π|_max",
         "Log-ratio max (proximity to float32 overflow)",
         overflow_threshold, "float32 overflow = 88.72"),

        ("entropy",             "H(π)",
         "Policy entropy (entropy vs variance competition)",
         None, None),

        ("approx_kl",           "approx KL",
         "Approx KL divergence",
         None, None),
    ]

    for ax, (key, ylabel, title, ref, ref_label) in zip(axes, panels):
        vals = d[key]
        # Mask NaN for clean plotting
        valid = ~np.isnan(vals)
        ax.plot(steps[valid], vals[valid], linewidth=1.5, color="#1f77b4")

        if ref is not None:
            ax.axhline(ref, color="red", linestyle="--",
                       linewidth=1.2, label=ref_label)
            ax.legend(fontsize=9)

        if crash_step is not None:
            ax.axvline(crash_step, color="crimson", linestyle=":",
                       linewidth=2.0, label="NaN crash")
            ax.legend(fontsize=9)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("Timestep", fontsize=10)

    fig.suptitle(
        "Pre-crash instability traces — PPO + gSDE + squash_output=True",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_path}")






class EpisodeRewardCallback(BaseCallback):
    """Callback to track episode rewards during training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            # Get episode info from the monitor
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_reward = self.locals['infos'][0]['episode']['r']
                episode_length = self.locals['infos'][0]['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
        return True

def make_monitored_env():
    """Create a monitored environment for proper episode tracking"""
    env = SimpleKettleEnv()
    env = Monitor(env)
    return env

def train_models(total_timesteps=4000_000, seed=71):
    th.autograd.set_detect_anomaly(True)

    np.random.seed(seed)
    
    env_ppo = DummyVecEnv([lambda: make_monitored_env()])
    env_sac = DummyVecEnv([lambda: make_monitored_env()])
    env_ppo.seed(seed)
    env_sac.seed(seed)
    
    callback_ppo = EpisodeRewardCallback()
    instability_mon  = InstabilityMonitorCallback(
        log_freq=500,
        save_path="results/instability_monitor",
        verbose=1)

    callback_sac = EpisodeRewardCallback()
    
    print("Training PPO...")
    model_ppo = PPO(
        "MlpPolicy",
        env_ppo,
        verbose=1,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.03,
        learning_rate=3e-4,
        seed=seed,
        use_sde=True,
        policy_kwargs=dict(
        net_arch=[128, 128],  # Larger network
        activation_fn=torch.nn.Tanh,
        squash_output=True,  # Ensure proper action scaling
    ),
    normalize_advantage=True,  # Better gradient scaling
    clip_range=0.2,  # More aggressive clipping
    )
    #model_ppo.learn(total_timesteps=total_timesteps, callback=callback_ppo)
    
    try:
        model_ppo.learn(
            total_timesteps=total_timesteps,
            # Pass both callbacks as a list
            callback=[callback_ppo, instability_mon]
        )
    except RuntimeError as e:
        if "nan" in str(e).lower():
            crash_step = instability_mon.num_timesteps
            print(f"\n[!] NaN crash at step {crash_step}")
            # Save whatever data was collected before crash
            instability_mon._save()
        else:
            raise

    # Plot immediately after training (crashed or not)
    plot_instability_traces(
        npz_path="results/instability_monitor/monitor_data.npz",
        crash_step=crash_step,
        save_path="results/instability_traces.png"
    )

    print(f"PPO completed {len(callback_ppo.episode_rewards)} episodes")
    
    print("\nTraining SAC...")
    model_sac = SAC(
        'MlpPolicy', 
        env_sac, 
        verbose=1,
        seed=seed  # ✅ Added seed
    )
    model_sac.learn(total_timesteps=total_timesteps, callback=callback_sac)
    
    print(f"SAC completed {len(callback_sac.episode_rewards)} episodes")
    
    # Save models
    os.makedirs("models", exist_ok=True)
    model_ppo.save("models/ppo_kettle")
    model_sac.save("models/sac_kettle")
    
    return model_ppo, model_sac, callback_ppo.episode_rewards, callback_sac.episode_rewards

def plot_training_comparison(ppo_rewards, sac_rewards):
    """Create comprehensive training comparison plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Episode rewards over time
    ax1.plot(range(len(ppo_rewards)), ppo_rewards, label='PPO', alpha=0.7, color='blue')
    ax1.plot(range(len(sac_rewards)), sac_rewards, label='SAC', alpha=0.7, color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards Over Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Boxplot for stability comparison
    reward_data = [ppo_rewards, sac_rewards]
    ax2.boxplot(reward_data, labels=['PPO', 'SAC'])
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Distribution (Training Stability)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Standard deviation over training (bar graph)
    window_size = max(10, min(len(ppo_rewards), len(sac_rewards)) // 20)
    
    # Calculate rolling standard deviation
    ppo_rolling_std = pd.Series(ppo_rewards).rolling(window=window_size).std().fillna(0)
    sac_rolling_std = pd.Series(sac_rewards).rolling(window=window_size).std().fillna(0)
    
    # Create bar positions
    episodes_ppo = np.arange(len(ppo_rolling_std))
    episodes_sac = np.arange(len(sac_rolling_std))
    
    width = 0.4
    ax3.bar(episodes_ppo - width/2, ppo_rolling_std, width=width, alpha=0.7, label='PPO', color='blue')
    ax3.bar(episodes_sac + width/2, sac_rolling_std, width=width, alpha=0.7, label='SAC', color='red')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title(f'Rolling Standard Deviation (Window={window_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Moving average for clearer trends
    ppo_ma = pd.Series(ppo_rewards).rolling(window=window_size).mean()
    sac_ma = pd.Series(sac_rewards).rolling(window=window_size).mean()
    
    ax4.plot(range(len(ppo_ma)), ppo_ma, label='PPO (Moving Avg)', color='blue', linewidth=2)
    ax4.plot(range(len(sac_ma)), sac_ma, label='SAC (Moving Avg)', color='red', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.set_title(f'Moving Average Rewards (Window={window_size})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_temperature_control(model_ppo, model_sac):
    """Evaluate both models and plot temperature control performance"""
    
    results = {}
    
    for name, model in [('PPO', model_ppo), ('SAC', model_sac)]:
        # Create fresh environment for evaluation
        eval_env = SimpleKettleEnv()
        
        obs, _ = eval_env.reset()
        temperatures = [eval_env.temp]
        actions = []
        rewards = []
        
        total_reward = 0
        
        print(f"\nEvaluating {name}...")
        for step in range(eval_env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action[0])
            obs, reward, done, _, info = eval_env.step(action)
            temperatures.append(eval_env.temp)
            rewards.append(reward)
            total_reward += reward
            
            if step < 5 or step % 50 == 0:  # Debug output
                print(f"  Step {step}: Action={action[0]:.1f}, Temp={eval_env.temp:.1f}, Reward={reward:.4f}")
            
            if done:
                print(f"  {name} episode ended at step {step}")
                break
        
        results[name] = {
            'temperatures': temperatures,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'final_temp': temperatures[-1],
            'steps': len(actions)
        }
        
        print(f"  {name} completed {len(actions)} steps, final temp: {temperatures[-1]:.1f}°C")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Temperature profiles
    for name, data in results.items():
        timesteps = range(len(data['temperatures']))
        color = 'blue' if name == 'PPO' else 'red'
        ax1.plot(timesteps, data['temperatures'], label=f'{name} (Steps: {data["steps"]})', linewidth=2, color=color)
    
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.8, label='Target Temperature')
    ax1.axvline(x=200, color='orange', linestyle='--', alpha=0.8, label='Target Time')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Control Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Control actions - ensure both have same length for comparison
    max_steps = max(len(data['actions']) for data in results.values())
    for name, data in results.items():
        timesteps = range(len(data['actions']))
        color = 'blue' if name == 'PPO' else 'red'
        ax2.plot(timesteps, data['actions'], label=f'{name} Actions', linewidth=2, color=color)
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Control Actions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_steps)
    
    plt.tight_layout()
    plt.savefig('results/temperature_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def print_training_statistics(ppo_rewards, sac_rewards):
    """Print comprehensive training statistics"""
    
    print("\n" + "="*60)
    print("TRAINING STATISTICS COMPARISON")
    print("="*60)
    
    stats = {}
    for name, rewards in [('PPO', ppo_rewards), ('SAC', sac_rewards)]:
        stats[name] = {
            'episodes': len(rewards),
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'final_10_avg': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        }
    
    for name, s in stats.items():
        print(f"\n{name} Results:")
        print(f"  Episodes: {s['episodes']}")
        print(f"  Mean Reward: {s['mean']:.4f}")
        print(f"  Std Reward: {s['std']:.4f}")
        print(f"  Min Reward: {s['min']:.4f}")
        print(f"  Max Reward: {s['max']:.4f}")
        print(f"  Final 10 Episodes Avg: {s['final_10_avg']:.4f}")
    
    print("="*60)
    
    return stats

def print_evaluation_results(eval_results):
    """Print evaluation results"""
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for name, results in eval_results.items():
        print(f"\n{name} Evaluation:")
        print(f"  Total Reward: {results['total_reward']:.4f}")
        print(f"  Final Temperature: {results['final_temp']:.2f}°C")
        print(f"  Steps Completed: {results['steps']}")
        print(f"  Target Achievement: {abs(results['final_temp'] - 100):.2f}°C from target")
    
    print("="*60)

def main():
    """Main execution function"""
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    print("Starting PPO vs SAC comparison on Kettle Environment")
    print("="*60)
    
    # Train models
    model_ppo, model_sac, ppo_rewards, sac_rewards = train_models(total_timesteps=4000_000)
    
    # Print training statistics
    training_stats = print_training_statistics(ppo_rewards, sac_rewards)
    
    # Plot training comparison
    print("\nGenerating training comparison plots...")
    plot_training_comparison(ppo_rewards, sac_rewards)
    
    # Evaluate models
    print("Evaluating trained models...")
    eval_results = evaluate_temperature_control(model_ppo, model_sac)
    
    # Print evaluation results
    print_evaluation_results(eval_results)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- results/training_comparison.png")
    print("- results/temperature_evaluation.png")
    print("- models/ppo_kettle.zip")
    print("- models/sac_kettle.zip")

if __name__ == "__main__":
    main()