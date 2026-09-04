from pathlib import Path

import mujoco
import numpy as np

from .config import CONTROL_HZ
from .env import (
    PHYSICS_DT,
    _JOINT_NAMES,
)
from .load_xml import _load_xml
from .reference.interpolate_walk import _load_keyframes, interpolate

_REFERENCE_XML = {
    'walk': Path(__file__).parent / 'reference' / 'xml' / 'walk.xml',
}

_N_JOINTS = len(_JOINT_NAMES)
# ReferenceEnv only exposes the joint-position part of the observation; the
# velocity / contact / IMU / pitch-roll components of GnociGymEnv's obs are
# deliberately ignored here.
_OBS_DIM  = _N_JOINTS


class ReferenceEnv:

    def __init__(self, task='walk', control_hz=CONTROL_HZ, period=1.0, frame_stack=1):
        if task not in _REFERENCE_XML:
            raise ValueError(f"No reference trajectory for task '{task}'. Supported: {list(_REFERENCE_XML)}")
        self.task = task
        self.control_hz = control_hz
        self.period = period
        self.frame_stack = frame_stack
        self.n_substeps = int(round(1.0 / (control_hz * PHYSICS_DT)))
        # Sim time between consecutive rows of self._dataset. n_coarse below is
        # chosen so this equals exactly one control step (1 / control_hz), i.e.
        # one GnociGymEnv.step() worth of sim time.
        self.dt = self.n_substeps * PHYSICS_DT
        self._initialize_model()
        self._dataset = self._build_dataset()

    def _initialize_model(self):
        xml = _load_xml('scene')
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)

        self.joint_pos_sensor_addrs = [
            self.model.sensor_adr[self.model.sensor(f'{j}-pos').id]
            for j in _JOINT_NAMES
        ]

    def _build_dataset(self) -> np.ndarray:
        # One row per control step over `period` seconds of the gait cycle, so
        # consecutive rows are spaced by exactly one GnociGymEnv.step() worth of
        # sim time (self.dt == 1 / control_hz). Each row holds only the joint
        # positions, matching the leading _N_JOINTS entries of the policy obs
        # (self.data.sensordata[joint_pos_sensor_addrs] / pi — see
        # GnociGymEnv._get_policy_obs).
        n_coarse = max(1, int(round(self.period * self.control_hz)))

        keys = _load_keyframes(_REFERENCE_XML[self.task])
        traj = interpolate(keys, n_frames=n_coarse)  # (n_coarse, 17) qpos

        saved_qpos = self.data.qpos.copy()

        dataset = np.zeros((n_coarse, _OBS_DIM), dtype=np.float32)

        for k in range(n_coarse):
            self.data.qpos[:traj.shape[1]] = traj[k]
            mujoco.mj_forward(self.model, self.data)
            dataset[k] = self.data.sensordata[self.joint_pos_sensor_addrs] / np.pi

        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)

        if self.frame_stack > 1:
            stacked = np.zeros((n_coarse, _OBS_DIM * self.frame_stack), dtype=np.float32)
            for i in range(n_coarse):
                for k in range(self.frame_stack):
                    # Match gym.wrappers.FrameStackObservation ordering: oldest
                    # frame first, newest frame last.
                    idx = (i - (self.frame_stack - 1 - k)) % n_coarse
                    stacked[i, k * _OBS_DIM:(k + 1) * _OBS_DIM] = dataset[idx]
            dataset = stacked

        return dataset

    def sample_pairs(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample consecutive state pairs (s_i, s_{i+1}) from the reference trajectory.

        Consecutive rows are spaced by ``self.dt`` (one control step), so the
        returned pairs have the same time delta as consecutive
        GnociGymEnv.step() observations.

        Returns:
            Tuple of (s_i, s_next), each of shape (batch_size, *obs_shape).
        """
        n_ref = len(self._dataset)
        inds = np.random.randint(0, n_ref, size=batch_size)
        s_i = self._dataset[inds]
        s_next = self._dataset[(inds + 1) % n_ref]
        return s_i, s_next

    def get_reference(self) -> np.ndarray:
        return self._dataset
