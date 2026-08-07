from gnoci_gym import GnociGymEnv
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import sys

# kwargs = {
#     "initial_randomness": 0.0,
#     "floor_tilt_range": 0.0,
#     "inertial_mass_range": (0.0, 0.0),
#     "inertial_mass_noise": 0.0,
#     "action_filter_alpha": 0.0,
#     "control_hz": 40,
# }

# env = GnociGymEnv(**kwargs)

env = GnociGymEnv(
    # initial_randomness=0.0,
    # inertial_mass_range=(0.00, 0.00),
    # inertial_mass_noise=0.00,
    # floor_tilt_range=0.0,
    action_filter_alpha=1,
    control_hz=40,
)

state, *_ = env.reset(seed=0)
                                                                                                                                                                                    
times = []
actions = []
control = []
contacts = []
dones = []
frames = []
rewards = []
touching_floor = []

horizontal_state = []
standing_height = []

root_heights = []
root_uprights = []

for i in tqdm(range(100)):

    # action = np.random.uniform(-1, 1, env.action_space.shape[0])
    action = np.random.randn(env.action_space.shape[0], 1) * 1
    # action = np.zeros(env.action_space.shape[0])
    # action[2] = 1
    # action[5+2] = 1
    # action[4] = 1
    # action[9] = 1
    # action[3] = 1
    # action[8] = 1

    # action = np.zeros(env.action_space.shape[0])
    # action = np.ones(env.action_space.shape[0])
    # action = env.unwrapped.data.ctrl.copy()
    state, reward, done, truncated, _ = env.step(action)

    root_height = env.unwrapped._get_root_height()
    root_upright = env.unwrapped._get_root_upright()
    root_heights.append(root_height)
    root_uprights.append(root_upright)
    
    times.append(env.unwrapped.data.time)
    dones.append(int(done))
    touching_floor.append(int(env.unwrapped._body_below_floor()))

    # actions.append(action)
    # control.append(env.unwrapped.data.ctrl.copy())
    contacts.append(env.unwrapped._get_contact_forces())
    rewards.append(reward)
    frames.append(env.render())

plt.plot(times, rewards)
plt.plot(times, touching_floor)
# plt.plot(times, root_uprights)
# plt.plot(times, rewards)
plt.plot(times, dones)
# plt.plot(times, contacts)
plt.show()

imageio.mimsave(f'assets/animation.gif', frames, loop=0, fps=30)

