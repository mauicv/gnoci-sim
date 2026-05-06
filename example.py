from gnoci_gym import GnociGymEnv
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import sys

gym.register(
    id="gnoci_gym/Gnoci-v0",
    entry_point=GnociGymEnv,
    kwargs={
        'initial_randomness': 0.0,
        'inertial_mass_range': (0.00, 0.00),
        'inertial_mass_noise': 0.00,
        'floor_tilt_range': 0.075,
    }
)

env = gym.make(
    "gnoci_gym/Gnoci-v0",
    initial_randomness=0.0,
    floor_tilt_range=0.0,
    control_hz=40,
    max_joint_vel=6.0,
)

state, *_ = env.reset(seed=0)

times = []
actions = []
control = []
responses = []
dones = []
frames = []
rewards = []

horizontal_state = []
standing_height = []

root_heights = []
root_uprights = []

for i in tqdm(range(100)):
    action = np.random.uniform(-1, 1, env.action_space.shape[0])
    # action = np.ones(env.action_space.shape[0])
    # action = env.unwrapped.data.ctrl.copy()
    state, reward, done, truncated, _ = env.step(action)

    root_height = env.unwrapped._get_root_height()
    root_upright = env.unwrapped._get_root_upright()
    root_heights.append(root_height)
    root_uprights.append(root_upright)
    
    times.append(env.unwrapped.data.time)
    # dones.append(int(done))
    # actions.append(action)
    # control.append(env.unwrapped.data.ctrl.copy())
    # responses.append(state)
    rewards.append(reward)
    frames.append(env.render())

plt.plot(times, root_heights)
plt.plot(times, root_uprights)
plt.plot(times, rewards)
plt.show()

imageio.mimsave(f'assets/animation.gif', frames, loop=0, fps=30)

