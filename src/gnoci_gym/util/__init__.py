import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from ..env import GnociGymEnv, test_cfg, dom_rnd_cfg
from .action_stack import ActionStackWrapper

CONTROL_HZ = 50
FRAME_STACK = 8
ACTION_STACK = 8
NUM_STEPS = 2048
NUM_ENVS = 8
seed=1
device='cpu'


def make_env(seed=1, max_steps=NUM_STEPS, test: bool=False):
    config = test_cfg if test else dom_rnd_cfg
    kwargs = {
        **config,

        "obs_noise_level": 1.0,

        "target_velocity": 0.1,
        "target_velocity_band": 0.1,
        "action_filter_alpha": 0.75,
        "control_hz": CONTROL_HZ,
        "task": 'walk',
        "action_scale": 0.125,
        "reward_coefs": {
            'stand':            1.0,
            'velocity':         2.5,
            'rotation':         0.05,
            'strafe':           0.05,
            'foot_swing':       1.0,
            'orientation':      1.0,
            'yoke_joint':       2.0,
            'action_magnitude': 0.0,
            'action_bounds':    1.0,
            'yoke_symmetry':    0.0,
            'action_rate':      0.5,
        }
    }
    env = GnociGymEnv(**kwargs)
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK)
    env = gym.wrappers.FlattenObservation(env)
    env = ActionStackWrapper(env, num_actions=ACTION_STACK)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = TimeLimit(env, max_episode_steps=NUM_STEPS)
    return env