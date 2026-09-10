import numpy as np
from typing import List, Any


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total_count
        ) / total_count
        self.count = total_count

    def get_state(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def set_state(self, state: dict):
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        if mean.shape != self.mean.shape:
            raise ValueError(
                f"shape mismatch: got {mean.shape}, expected {self.mean.shape}"
            )
        self.mean = mean.copy()
        self.var = var.copy()
        self.count = int(state["count"])


class VecNormalize:
    """
    Wraps a list of OpenAI Gym-style environments, normalizing
    observations and optionally rewards using running statistics.

    Args:
        envs: list of gym-compatible environments (must share obs/action space)
        norm_obs: normalize observations
        norm_reward: normalize rewards
        clip_obs: clip normalized observations to [-clip_obs, clip_obs]
        clip_reward: clip normalized rewards to [-clip_reward, clip_reward]
        gamma: discount factor used for reward running variance
        epsilon: small constant for numerical stability
    """

    def __init__(
        self,
        envs: List[Any],
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        self.envs = envs
        self.num_envs = len(envs)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        obs_shape = envs[0].observation_space.shape
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.ret_rms = RunningMeanStd(shape=())
        self._returns = np.zeros(self.num_envs)  # discounted return per env
        self.training = True  # if False, stats are frozen

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self):
        obs = np.array([env.reset()[0] for env in self.envs])
        self._returns = np.zeros(self.num_envs)
        return self._normalize_obs(obs)

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rewards, dones, _, infos = zip(*results)
        obs = list(obs)
        infos = [dict(info) for info in infos]

        rewards = np.array(rewards, dtype=np.float64)
        dones = np.array(dones)

        # Update running stats on terminal obs (before reset)
        if self.training:
            self.obs_rms.update(np.array(obs))
            self._returns = self._returns * self.gamma + rewards
            self.ret_rms.update(self._returns)
            self._returns[dones] = 0.0

        # Auto-reset finished envs; stash terminal obs in info
        for i, done in enumerate(dones):
            if done:
                infos[i]["terminal_observation"] = obs[i]
                s, _ = envs.envs[i].reset()
                obs[i] = s

        for i, info in enumerate(infos):
            info['reward'] = rewards[i]

        norm_obs = self._normalize_obs(np.array(obs))
        # for i, info in enumerate(infos):
        #     infos[i]['normed_unnoisy_state'] = self._normalize_obs(np.array(info['state']))
        if self.training:
            norm_rew = self._normalize_reward(rewards)
        else:
            norm_rew = rewards
        return norm_obs, norm_rew, dones, list(infos)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.norm_obs:
            return obs
        normed = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)

    def _normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        if not self.norm_reward:
            return rewards
        normed = rewards / np.sqrt(self.ret_rms.var + self.epsilon)
        return np.clip(normed, -self.clip_reward, self.clip_reward)

    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs * np.sqrt(self.obs_rms.var + self.epsilon) + self.obs_rms.mean

    # ------------------------------------------------------------------
    # Passthrough helpers
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)

    def close(self):
        for env in self.envs:
            env.close()

    def eval(self):
        """Freeze running stats (e.g. at test time)."""
        self.training = False

    def train(self):
        self.training = True

    # ------------------------------------------------------------------
    # Stat synchronization
    # ------------------------------------------------------------------

    def get_norm_state(self) -> dict:
        """Snapshot of the running statistics (deep-copied, safe to hold)."""
        return {
            "obs_rms": self.obs_rms.get_state(),
            "ret_rms": self.ret_rms.get_state(),
        }

    def set_norm_state(self, state: dict):
        """Load running statistics from a snapshot. Does not touch _returns."""
        if "obs_rms" in state:
            self.obs_rms.set_state(state["obs_rms"])
        if "ret_rms" in state:
            self.ret_rms.set_state(state["ret_rms"])

    def sync_from(self, other: "VecNormalize", obs: bool = True, ret: bool = True):
        """Copy running stats from another VecNormalize (e.g. train -> eval)."""
        if obs:
            self.obs_rms.set_state(other.obs_rms.get_state())
        if ret:
            self.ret_rms.set_state(other.ret_rms.get_state())
        return self