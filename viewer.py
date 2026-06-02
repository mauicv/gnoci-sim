#!/usr/bin/env python3
"""
Interactive viewer for testing position actuators.

Uses the MuJoCo built-in viewer controls:
  Space      — pause / resume
  Backspace  — reset simulation
  Ctrl+A     — toggle actuator controls panel
  F1         — help overlay

Examples:
  python3 viewer.py
  python3 viewer.py --no-gravity
  python3 viewer.py --gravity 0 0 -4.9
  python3 viewer.py --timestep 0.001
"""

import argparse
import mujoco
import mujoco.viewer
from src.gnoci_gym import GnociGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-gravity", action="store_true",
                        help="Disable gravity (sets to 0 0 0)")
    parser.add_argument("--gravity", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Set gravity vector, e.g. --gravity 0 0 -4.9")
    parser.add_argument("--timestep", type=float,
                        help="Override simulation timestep (seconds)")
    return parser.parse_args()


def apply_overrides(model, args):
    if args.no_gravity:
        model.opt.gravity[:] = [0, 0, 0]
        print("Gravity: disabled")
    elif args.gravity:
        model.opt.gravity[:] = args.gravity
        print(f"Gravity: {args.gravity}")

    if args.timestep:
        model.opt.timestep = args.timestep
        print(f"Timestep: {args.timestep}")


def main():
    args = parse_args()
    env = GnociGymEnv(
        initial_randomness=0.0,
        inertial_mass_range=(0.00, 0.00),
        inertial_mass_noise=0.00,
        floor_tilt_range=0.0,
        action_filter_alpha=0.0,
        control_hz=40,
        max_joint_vel=6.0,
        fix_root_body=True,
    )
    env.reset(seed=0)
    apply_overrides(env.model, args)
    mujoco.viewer.launch(env.model, env.data)


if __name__ == '__main__':
    main()
