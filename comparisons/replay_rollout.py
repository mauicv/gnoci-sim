"""
Replay the recorded actions from rollout.json through the sim, compare the
resulting simulated states against the recorded ones, and render a video of
the simulated rollout.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import mujoco
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import math

from gnoci_gym import GnociGymEnv
from gnoci_gym.config import CONTROL_HZ

RECORDED_COLOR = "#2563EB"
SIM_COLOR      = "#EA580C"

N_STATES = 32
COLS = 8
ROWS = N_STATES // COLS  # 4

_J = ["L head\nyoke", "L yoke\nhip", "L hip\nupper", "L upper\nlower", "L lower\nfoot",
      "R head\nyoke", "R yoke\nhip", "R hip\nupper", "R upper\nlower", "R lower\nfoot"]

STATE_LABELS = (
    [f"{j}\npos" for j in _J]
  + [f"{j}\nvel" for j in _J]
  + ["fwd L\ncontact", "bck L\ncontact", "fwd R\ncontact", "bck R\ncontact"]
  + ["gyro x", "gyro y", "gyro z"]
  + ["acc x",  "acc y",  "acc z"]
  + ["roll", "pitch"]
)

RENDER_W, RENDER_H = 640, 480
RENDER_FPS = CONTROL_HZ

comparisons_dir = os.path.dirname(__file__)
rendered_dir = os.path.join(comparisons_dir, 'rendered_rollout')
os.makedirs(rendered_dir, exist_ok=True)

with open(os.path.join(comparisons_dir, 'rollout.json')) as f:
    rollout = json.load(f)

recorded_states = np.array(rollout['states'], dtype=float)
actions = np.array(rollout['actions'], dtype=float)

env = GnociGymEnv(
    initial_randomness=0.0,
    inertial_mass_range=(0.0, 0.0),
    inertial_mass_noise=0.0,
    floor_tilt_range=0.0,
    floor_friction_range=(1.0, 1.0),
    gravity_noise=0.0,
    obs_noise_level=0.0,
    push_force_max=0.0,
    max_action_delay=0,
    action_filter_alpha=1.0,
    control_hz=CONTROL_HZ,
    fix_root_body=False,
)
state, _ = env.reset(seed=0)

sim_states = [state]
frames = []
renderer = mujoco.Renderer(env.model, height=RENDER_H, width=RENDER_W)
frames.append(renderer.render())  # matches the pre-first-action state above

for action in tqdm(actions, desc="Replaying rollout"):
    action = ((math.pi*0.75)/0.2) * action
    state, *_ = env.step(action)
    sim_states.append(state)
    renderer.update_scene(env.data, camera="track")
    frames.append(renderer.render())

renderer.close()

sim_states = np.array(sim_states, dtype=float)

video_path = os.path.join(rendered_dir, "rollout_render.mp4")
imageio.mimwrite(video_path, frames, fps=RENDER_FPS)
print(f"Saved video to {video_path}")

n_steps = min(len(recorded_states), len(sim_states))
recorded_states = recorded_states[:n_steps]
sim_states = sim_states[:n_steps]
times = np.arange(n_steps) / CONTROL_HZ

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "axes.labelcolor": "#374151",
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.titlesize": 7.5,
    "axes.titlecolor": "#111827",
    "axes.titlepad": 6,
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
})

fig, axs = plt.subplots(ROWS, COLS, figsize=(22, 9), sharex=False)
fig.suptitle("State Comparison — Recorded (rollout.json) vs Simulated Replay",
             fontsize=12, fontweight="bold", color="#111827", y=0.999)

for i in range(N_STATES):
    row, col = divmod(i, COLS)
    ax = axs[row, col]

    ax.plot(times, recorded_states[:, i], color=RECORDED_COLOR, linewidth=1.0, zorder=3)
    ax.plot(times, sim_states[:, i], color=SIM_COLOR, linewidth=1.0,
            linestyle="--", zorder=2)

    ax.set_title(STATE_LABELS[i], pad=4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.tick_params(length=2)
    ax.spines["top"].set_visible(False)

    if row == ROWS - 1:
        ax.set_xlabel("time (s)", labelpad=3)

handles = [
    plt.Line2D([0], [0], color=RECORDED_COLOR, linewidth=1.6, label="Recorded"),
    plt.Line2D([0], [0], color=SIM_COLOR, linewidth=1.6, linestyle="--", label="Sim replay"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           frameon=True, framealpha=0.9, edgecolor="#D1D5DB",
           fontsize=9, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.997])
out_path = os.path.join(comparisons_dir, "rollout_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
plt.show()
