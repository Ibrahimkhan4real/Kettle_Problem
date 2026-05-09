import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class InstabilityMonitorCallback(BaseCallback):
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
            "jacobian_true_max":   [],
            "log_ratio_max":       [],
            "entropy":             [],
            "approx_kl":           [],
            "pg_loss":             [],
        }

        self._snapshot_log_probs = None
        self._snapshot_obs       = None
        self._snapshot_actions   = None
        self._last_log_step      = 0

    def _on_rollout_start(self):
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

    def _log_signals(self):
        buf    = self.model.rollout_buffer
        policy = self.model.policy

        n       = buf.buffer_size if buf.full else buf.pos
        obs_np  = buf.observations[:n].reshape(-1, buf.observations.shape[-1])
        act_np  = buf.actions[:n].reshape(-1, buf.actions.shape[-1])

        obs_t = th.tensor(obs_np, dtype=th.float32, device=policy.device)
        act_t = th.tensor(act_np, dtype=th.float32, device=policy.device)

        with th.no_grad():
            values, log_prob, entropy = policy.evaluate_actions(obs_t, act_t)

            features  = policy.extract_features(obs_t)
            latent_pi, _ = policy.mlp_extractor(features)
            mu = policy.action_net(latent_pi)
            policy_mean_abs_max = mu.abs().max().item()

            tanh_mu = th.tanh(mu)
            one_minus_sq_f32 = 1.0 - tanh_mu ** 2 + 1e-8
            jac_f32 = -th.log(one_minus_sq_f32).sum(dim=-1)
            jacobian_mean = jac_f32.mean().item()
            jacobian_max  = jac_f32.max().item()

            mu_f64 = mu.double()
            log2   = th.log(th.tensor(2.0, dtype=th.float64))
            jac_true = (
                2.0 * mu_f64.abs()
                - 2.0 * log2
                + 2.0 * th.log1p(th.exp(-2.0 * mu_f64.abs()))
            ).sum(dim=-1)
            jacobian_true_max = jac_true.max().item()

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

            log_ratio_max = float("nan")
            if self._snapshot_log_probs is not None:
                snap_lp = th.tensor(
                    self._snapshot_log_probs,
                    dtype=th.float32,
                    device=policy.device,
                )
                min_len = min(len(log_prob), len(snap_lp))
                log_ratio = (log_prob[:min_len] - snap_lp[:min_len]).abs()
                log_ratio_max = log_ratio.max().item()

            if entropy is not None:
                ent = entropy.mean().item()
            else:
                if not np.isnan(log_std_mean):
                    ent = (
                        0.5 * (1.0 + np.log(2 * np.pi))
                        + log_std_mean
                        - jacobian_mean
                    )
                else:
                    ent = float("nan")

            approx_kl = self._read_logger_value("train/approx_kl")
            pg_loss   = self._read_logger_value("train/policy_gradient_loss")

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

    def _read_logger_value(self, key):
        try:
            return float(self.model.logger.name_to_value.get(key, float("nan")))
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


def plot_instability_traces(npz_path: str, crash_step: int = None,
                            save_path: str = "results/instability_traces.png"):
    d = np.load(npz_path)
    steps = d["timestep"]
    overflow_threshold = 88.72

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    panels = [
        ("policy_mean_abs_max", "|μ(s)|_max",
         "Policy mean abs max (boundary-seeking signal)", None, None),
        ("log_std_min", "log σ_min",
         "gSDE log-std minimum (variance collapse signal)", None, None),
        ("jacobian_true_max", "J(u)_true_max",
         "Jacobian correction max (singularity signal)", None, None),
        ("log_ratio_max", "|Δ log π|_max",
         "Log-ratio max (proximity to float32 overflow)",
         overflow_threshold, "float32 overflow = 88.72"),
        ("entropy", "H(π)", "Policy entropy", None, None),
        ("approx_kl", "approx KL", "Approx KL divergence", None, None),
    ]

    for ax, (key, ylabel, title, ref, ref_label) in zip(axes, panels):
        vals = d[key]
        valid = ~np.isnan(vals)
        ax.plot(steps[valid], vals[valid], linewidth=1.5, color="#1f77b4")
        if ref is not None:
            ax.axhline(ref, color="red", linestyle="--", linewidth=1.2, label=ref_label)
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
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {save_path}")