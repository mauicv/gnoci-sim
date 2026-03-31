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
        'env_rate': 0.005,
        'initial_randomness': 0.6,
        'inertial_mass_range': (0.04, 0.06),
        'inertial_mass_noise': 0.01,
    }
)

env = gym.make(
    "gnoci_gym/Gnoci-v0",
    env_rate=0.005,
    initial_randomness=0.2,
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

for i in tqdm(range(100)):
    action = np.random.uniform(-1, 1, env.action_space.shape[0])
    # action = np.array([1,1,1,1,1,1])
    state, reward, done, truncated, _ = env.step(action)
    body_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "hor_rot_body_joint")
    xmat = env.unwrapped.data.xmat[body_id]
    z_axis = np.array([xmat[6], xmat[7], xmat[8]])
    dot = np.dot(z_axis, [0, 0, 1])
    xpos = env.unwrapped.data.xpos[body_id]
    horizontal_state.append(dot)
    standing_height.append(xpos[2])

    times.append(env.unwrapped.data.time)
    dones.append(int(done))
    actions.append(action)
    control.append(env.unwrapped.data.ctrl.copy())
    responses.append(state)
    rewards.append(reward)
    frames.append(env.render())


imageio.mimsave(f'assets/animation.gif', frames, loop=0, fps=30)