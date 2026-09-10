import gymnasium as gym
import numpy as np
from collections import deque


class ActionStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_actions=1):
        super().__init__(env)
        self.num_actions = num_actions
        act_dim = np.prod(env.action_space.shape)
        self._action_history = deque(
            [np.zeros(act_dim)] * num_actions, maxlen=num_actions
        )
        low_obs = env.observation_space.low
        high_obs = env.observation_space.high
        low_act = np.tile(env.action_space.low, num_actions)
        high_act = np.tile(env.action_space.high, num_actions)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low_obs, low_act]),
            high=np.concatenate([high_obs, high_act]),
            dtype=np.float32,
        )

    def observation(self, obs):
        actions = np.concatenate(list(self._action_history))
        return np.concatenate([obs, actions]).astype(np.float32)

    def step(self, action):
        self._action_history.append(np.asarray(action).flatten())
        return super().step(action)

    def reset(self, **kwargs):
        for i in range(self.num_actions):
            self._action_history[i] = np.zeros_like(self._action_history[i])
        return super().reset(**kwargs)