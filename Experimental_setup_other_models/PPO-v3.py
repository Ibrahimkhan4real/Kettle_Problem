import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# Import the env class from the same module/file
from Kettle_environment_updated_reward_fun_continuous import SimpleKettleEnv  # adjust import to your file structure

def make_env():
    return Monitor(SimpleKettleEnv())

if __name__ == "__main__":
    # Sanity-check the custom env against the Gymnasium API that SB3 expects
    env = SimpleKettleEnv()
    check_env(env, warn=True)

    # Vectorized environments for PPO
    vec_env = make_vec_env(make_env, n_envs=4)

    # Separate eval env
    eval_env = make_vec_env(make_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./ppo_kettle_best",
        log_path="./ppo_kettle_eval",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # PPO on a Box action space uses MlpPolicy
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        seed=42,
        use_sde=True,
        policy_kwargs=dict(
        net_arch=[256, 256],  # Larger network
        activation_fn=torch.nn.Tanh,
        squash_output=True,  # Ensure proper action scaling
    ),
    normalize_advantage=True,  # Better gradient scaling
    clip_range=0.3,  # More aggressive clipping
    )

    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save("ppo_kettle")

    # Quick test rollout
    test_env = SimpleKettleEnv()
    obs, info = test_env.reset(seed=123)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
