from src import GnociGymEnv
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco


gym.register(
    id="gnoci_gym/Gnoci-v0",
    entry_point=GnociGymEnv,
)

env = gym.make("gnoci_gym/Gnoci-v0")

state, *_ = env.reset(seed=0)

times = []
actions = []
control = []
responses = []
horizontal_state = []
dones = []
frames = []
rewards = []
for i in tqdm(range(100)):
    # a = 2*(2*np.pi*i)/100
    # action = np.array([
    #     np.sin(a), -np.sin(a), np.sin(a), -np.sin(a),
    #     np.sin(a), -np.sin(a), np.sin(a), -np.sin(a)
    # ])
    action = np.random.uniform(-1, 1, 6)
    state, reward, done, _ = env.step(action)
    body_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "root")
    xmat = env.unwrapped.data.xmat[body_id]
    z_axis = np.array([xmat[6], xmat[7], xmat[8]])
    dot = np.dot(z_axis, [0, 0, 1])

    times.append(env.unwrapped.data.time)
    horizontal_state.append(dot)
    dones.append(int(done))
    actions.append(action)
    control.append(env.unwrapped.data.ctrl)
    responses.append(state)
    rewards.append(reward)
    frames.append(env.render())

env.reset()

imageio.mimsave('animation.gif', frames, fps=30)

actions = np.array(actions)
responses = np.array(responses)
dones = np.array(dones)
rewards = np.array(rewards)
control = np.array(control)

fig, axs = plt.subplots(nrows=2, ncols=4)
for i in range(3):
    for j in range(2):
        axs[j, i].plot(times, actions[:, 3*j + i], label='action')
        axs[j, i].plot(times, control[:, 3*j + i], label='control')
        axs[j, i].plot(times, responses[:, 3*j + i], label='position')
        # axs[j, i].plot(times, 100*responses[:, 6 + 3*j + i], label='velocity')
        axs[j, i].legend()

axs[0, 3].plot(dones, label='done')
axs[0, 3].legend()
axs[1, 3].plot(rewards, label='reward')
axs[1, 3].legend()
plt.savefig('test.png')