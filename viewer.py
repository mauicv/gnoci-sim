#!/usr/bin/env python3
"""
Interactive viewer for testing position actuators.

Open the "Controls" panel in the viewer to drag actuator sliders.
Press R to reset the simulation.
"""

import mujoco
import mujoco.viewer
from src.gnoci_gym import GnociGymEnv


def main():
    env = GnociGymEnv()
    env.reset(seed=0)

    reset_requested = [False]

    def key_callback(keycode):
        if chr(keycode) == 'R':
            reset_requested[0] = True

    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            if reset_requested[0]:
                mujoco.mj_resetData(env.model, env.data)
                reset_requested[0] = False

            mujoco.mj_step(env.model, env.data)
            viewer.sync()


if __name__ == '__main__':
    main()
