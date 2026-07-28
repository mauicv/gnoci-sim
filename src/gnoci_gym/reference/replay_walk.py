#!/usr/bin/env python3
"""
Replay an interpolated walk trajectory in the MuJoCo viewer.

Loops the trajectory continuously until the viewer window is closed.
Use --period to control playback speed (duration of one gait cycle in seconds).

Usage:
    python replay_walk.py
    python replay_walk.py --traj walk_traj.npy --period 1.5
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from gnoci_gym.load_xml import _load_xml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj",   default=str(HERE / "npy" / "walk_traj.npy"), help="Trajectory .npy")
    parser.add_argument("--period", type=float, default=1.0,             help="Gait cycle duration (seconds)")
    args = parser.parse_args()

    xml = _load_xml("scene")
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    traj = np.load(args.traj)
    n_frames = len(traj)
    dt = args.period / n_frames
    print(f"Replaying {n_frames} frames  period={args.period}s  dt={dt*1000:.1f}ms")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame in traj:
                if not viewer.is_running():
                    break
                data.qpos[:len(frame)] = frame
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)


if __name__ == "__main__":
    main()
