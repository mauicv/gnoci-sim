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

from gnoci_gym.sysid.sysid_ds_interface import SysidDSInterface

ds = SysidDSInterface()

from gnoci_gym.load_xml import _load_and_perturb_basic_xml


def main(rollout_idx):
    xml = _load_and_perturb_basic_xml("scene", fix_root_body=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    rollout = ds.data['data'][rollout_idx]
    traj_states = np.array(rollout['measured_states'])
    rollout_type = rollout['type']
    dt = ds.config['physics_dt']*ds.config['n_substeps']
    
    print('dt: ', dt)
    print('rollout_idx: ', rollout_idx)
    print('traj_states.shape: ', traj_states.shape)
    print('rollout_type: ', rollout_type)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame in traj_states:
                if not viewer.is_running():
                    break
                data.qpos[:10] = frame[:10] * np.pi
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout', type=int, default=0)
    args = parser.parse_args()
    main(args.rollout)
