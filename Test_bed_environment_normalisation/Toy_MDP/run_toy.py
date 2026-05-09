import os
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from toy_bangbang_env import BangBangToyEnv
# Assumes InstabilityMonitorCallback + plot_instability_traces are importable
# from the file you already have.
from instability_monitor import InstabilityMonitorCallback, plot_instability_traces


def run(use_sde: bool, squash_output: bool, seed: int,
        total_timesteps: int = 200_000, tag: str = "ablation"):
    th.autograd.set_detect_anomaly(True)
    np.random.seed(seed)

    env = DummyVecEnv([lambda: Monitor(BangBangToyEnv())])
    env.seed(seed)

    save_dir = f"results/toy/{tag}_sde{int(use_sde)}_sq{int(squash_output)}_seed{seed}"
    os.makedirs(save_dir, exist_ok=True)

    monitor = InstabilityMonitorCallback(log_freq=500, save_path=save_dir, verbose=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        seed=seed,
        use_sde=use_sde,
        policy_kwargs=dict(
            net_arch=[16],              # minimal net: one hidden layer, 16 units
            activation_fn=th.nn.Tanh,
            squash_output=squash_output,
        ),
    )

    crash_step = None
    try:
        model.learn(total_timesteps=total_timesteps, callback=monitor)
    except RuntimeError as e:
        if "nan" in str(e).lower():
            crash_step = monitor.num_timesteps
            print(f"[!] NaN crash at step {crash_step} "
                  f"(use_sde={use_sde}, squash={squash_output}, seed={seed})")
            monitor._save()
        else:
            raise

    plot_instability_traces(
        npz_path=f"{save_dir}/monitor_data.npz",
        crash_step=crash_step,
        save_path=f"{save_dir}/traces.png",
    )
    return {"use_sde": use_sde, "squash": squash_output,
            "seed": seed, "crash_step": crash_step}


if __name__ == "__main__":
    configs = [
        (False, False),  # standard Gaussian + env clipping
        (True,  False),  # gSDE, no squash
        (True,  True),   # gSDE + squash — expected crash
    ]

    results = []
    for seed in [0, 1, 2, 3, 4]:
        for use_sde, squash_output in configs:
            print(f"\n=== seed={seed}  use_sde={use_sde}  squash={squash_output} ===")
            results.append(run(use_sde, squash_output, seed))

    print("\n\n=== ablation summary (toy MDP) ===")
    for r in results:
        status = f"crash@{r['crash_step']}" if r["crash_step"] else "stable"
        print(f"sde={int(r['use_sde'])} sq={int(r['squash'])} "
              f"seed={r['seed']} : {status}")