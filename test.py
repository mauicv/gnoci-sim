from gnoci_gym import GnociGymEnv
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

env = gym.make(
    "gnoci_gym/Gnoci-v0",
    env_rate=0.005,
    system_rate=0.005,
    control_rate=0.01,
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
    action = np.random.uniform(-1, 1, 6)
    # action = np.array([1,1,1,1,1,1])
    state, reward, done, truncated, _ = env.step(action)
    body_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "root")
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


imageio.mimsave('assets/animation.gif', frames, loop=0, fps=30)

actions = np.array(actions)
responses = np.array(responses)
dones = np.array(dones)
rewards = np.array(rewards)
control = np.array(control)

fig, axs = plt.subplots(nrows=4, ncols=6)
for j in range(6):
    axs[0, j].plot(times, actions[:, j], label='action')
    axs[0, j].legend()


for j in range(6):
    axs[1, j].plot(times, control[:, j], label='control')
    axs[1, j].legend()


for j in range(6):
    axs[2, j].plot(times, responses[:, j], label='position')
    axs[2, j].legend()

for j in range(6):
    axs[3, j].plot(times, responses[:, 6 + j], label='velocity')
    axs[3, j].legend()

# for i in range(3):
#     axs[2, i].plot(times,responses[:, 20-3 + i])

# axs[0, 3].plot(standing_height, label='standing_height')
# axs[0, 3].plot(horizontal_state, label='horizontal_state')
# axs[0, 3].legend()
# axs[1, 3].plot(times, rewards, label='reward')
# axs[1, 3].legend()
plt.savefig('test.png')