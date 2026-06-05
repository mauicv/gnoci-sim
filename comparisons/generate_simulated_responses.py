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
import numpy as np
import mujoco
import imageio.v2 as imageio
import json
from tqdm import tqdm

_JOINT_NAMES = [
    'head__left_yoke',
    'left_yoke__hip',
    'left_hip__upper_leg',
    'left_upper_leg__lower_leg',
    'left_lower_leg__foot',
    'head__right_yoke',
    'right_yoke__hip',
    'right_hip__upper_leg',
    'right_upper_leg__lower_leg',
    'right_lower_leg__foot',
]

N_STEPS = 100
CONTROL_HZ = 40
RENDER_W, RENDER_H = 640, 480
RENDER_FPS = 30

rendered_dir = os.path.join(os.path.dirname(__file__), 'rendered_responses')
os.makedirs(rendered_dir, exist_ok=True)

results = []
frames = []
pbar = tqdm(total=len(_JOINT_NAMES) * 2, desc='Generating responses')

for action_val in [-1.0, 1.0]:
    for joint_idx, joint_name in enumerate(_JOINT_NAMES):
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
            max_joint_vel=3,
            fix_root_body=True,
        )
        env.reset(seed=0)

        action = np.zeros(env.action_space.shape[0])
        action[joint_idx] = action_val

        angular_pos = []
        angular_vel = []
        times = []

        renderer = mujoco.Renderer(env.model, height=RENDER_H, width=RENDER_W)

        for _ in range(N_STEPS):
            state, *_ = env.step(action)
            angular_pos.append(float(state[joint_idx]))
            angular_vel.append(float(state[10 + joint_idx]))
            times.append(float(env.data.time))
            renderer.update_scene(env.data, camera="track")
            frames.append(renderer.render())

        renderer.close()

        results.append({
            "joint_name": joint_name,
            "action": int(action_val),
            "angular_pos": angular_pos,
            "angular_vel": angular_vel,
            "time": times,
        })


        pbar.set_postfix(joint=joint_name, action=f'{action_val:+.0f}')
        pbar.update(1)

video_path = os.path.join(rendered_dir, f"response_render.mp4")
imageio.mimwrite(video_path, frames, fps=RENDER_FPS)

pbar.close()

output_path = os.path.join(os.path.dirname(__file__), 'simulated_response_data.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Saved {len(results)} entries to {output_path}")
