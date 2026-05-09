import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BangBangToyEnv(gym.Env):
    """
    Minimal 1-state bandit MDP for reproducing PPO + gSDE + squash_output NaN.

    - Observation: constant (no state dynamics)
    - Action: continuous in [-1, 1]
    - Reward: +1 if a > 0.9 else 0
    - Single-step episodes

    Optimal policy is deterministic at a = 1 (the right boundary).
    The only confounder this env contains is the policy's own parameterisation —
    there is no transition dynamics, no long-horizon credit assignment,
    no advantage estimator noise beyond a constant-state baseline.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        a = float(np.asarray(action).flatten()[0])
        reward = 1.0 if a > 0.9 else 0.0
        return np.zeros(1, dtype=np.float32), reward, True, False, {}