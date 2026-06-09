"""
Generate simulated step response data for each joint × action pairing.
Each test applies a constant action (±1) to a single joint (all others zero).
Root body is pinned in place; gravity is enabled throughout.
Output is saved to simulated_response_data.json with the same structure as
real_response_data.json.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnoci_gym import GnociGymEnv
from gnoci_gym.config import CONTROL_HZ
import numpy as np
import mujoco
import imageio.v2 as imageio
import json
from tqdm import tqdm

N_STEPS = 500
RENDER_W, RENDER_H = 640, 480
RENDER_FPS = 40

rendered_dir = os.path.join(os.path.dirname(__file__), 'rendered_states')
os.makedirs(rendered_dir, exist_ok=True)

results = []
frames = []

env = GnociGymEnv(
    initial_randomness=0.0,
    inertial_mass_range=(0.0, 0.0),
    inertial_mass_noise=0.0,
    floor_tilt_range=0.0,
    floor_friction_range=(1.0, 1.0),
    joint_friction_range=(0.1, 0.1),
    joint_armature_range=(0.005, 0.005),
    actuator_gain_range=(1.0, 1.0),
    gravity_noise=0.0,
    obs_noise_scale=0.0,
    push_force_max=0.0,
    max_action_delay=0,
    action_filter_alpha=0.4,
    control_hz=CONTROL_HZ,
    max_joint_vel=6,
    fix_root_body=False,
)
env.reset(seed=0)

action = np.zeros(env.action_space.shape[0])

state_data= {
    "states": [],
    "actions": [],
    "times": [],
}
renderer = mujoco.Renderer(env.model, height=RENDER_H, width=RENDER_W)


for i in tqdm(range(N_STEPS)):
    state, *_ = env.step(action)

    action = np.zeros((10))
    action[1] = np.sin(i / 50 * 2 * np.pi) / 25
    action[6] = np.cos(i / 50 * 2 * np.pi) / 25

    state_data["states"].append(state.tolist())
    state_data["actions"].append(action.tolist())
    state_data["times"].append(float(env.data.time))

    renderer.update_scene(env.data, camera="track")
    frames.append(renderer.render())


renderer.close()

video_path = os.path.join(rendered_dir, f"state_render.mp4")
imageio.mimwrite(video_path, frames, fps=RENDER_FPS)

output_path = os.path.join(os.path.dirname(__file__), 'simulated_state_data.json')
with open(output_path, 'w') as f:
    json.dump(state_data, f, indent=4)

print(f"Saved {N_STEPS} entries to {output_path}")
