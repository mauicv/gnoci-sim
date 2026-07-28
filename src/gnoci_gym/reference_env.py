from pathlib import Path

import mujoco
import numpy as np
import torch

from .config import CONTROL_HZ
from .env import (
    PHYSICS_DT,
    IMU_GYRO_SCALE,
    IMU_ACC_SCALE,
    _JOINT_NAMES,
    _TOUCH_SENSOR_NAMES,
    _DEFAULT_JOINT_POS_ARRAY,
    _OBS_NORM,
)
from .filters import ComplementaryFilter
from .load_xml import _load_xml
from .reference.interpolate_walk import _load_keyframes, interpolate

_REFERENCE_XML = {
    'walk': Path(__file__).parent / 'reference' / 'xml' / 'walk.xml',
}

_N_JOINTS = len(_JOINT_NAMES)
_N_TOUCH  = len(_TOUCH_SENSOR_NAMES)
_OBS_DIM  = _N_JOINTS + _N_JOINTS + _N_TOUCH + 6 + 2  # 32


class ReferenceEnv:

    def __init__(self, task='walk', control_hz=CONTROL_HZ, period=1.0, frame_stack=1):
        if task not in _REFERENCE_XML:
            raise ValueError(f"No reference trajectory for task '{task}'. Supported: {list(_REFERENCE_XML)}")
        self.task = task
        self.control_hz = control_hz
        self.period = period
        self.frame_stack = frame_stack
        self.n_substeps = int(round(1.0 / (control_hz * PHYSICS_DT)))
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
        self.joint_dof_addrs = [
            self.model.jnt_dofadr[self.model.joint(j).id]
            for j in _JOINT_NAMES
        ]
        self.touch_sensor_addrs = [
            self.model.sensor_adr[self.model.sensor(n).id]
            for n in _TOUCH_SENSOR_NAMES
        ]
        self.imu_sensor_addrs = [
            self.model.sensor_adr[self.model.sensor(n).id]
            for n in ['imu-gyro', 'imu-acc']
        ]
        self.joint_qpos_addrs = [
            self.model.jnt_qposadr[self.model.joint(j).id]
            for j in _JOINT_NAMES
        ]
        self.comp_filter = ComplementaryFilter()

    def _build_dataset(self) -> np.ndarray:
        n_ref = int(round(self.period / PHYSICS_DT))
        keys = _load_keyframes(_REFERENCE_XML[self.task])
        traj = interpolate(keys, n_frames=n_ref)  # (n_ref, 17)

        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        self.comp_filter.reset()

        dataset = np.zeros((n_ref, _OBS_DIM), dtype=np.float32)
        vel_window = 2.0 / self.control_hz  # time span of centered difference

        for i in range(n_ref):
            self.data.qpos[:traj.shape[1]] = traj[i]

            i_fwd = (i + self.n_substeps) % n_ref
            i_bwd = (i - self.n_substeps) % n_ref
            vel = (traj[i_fwd, 7:] - traj[i_bwd, 7:]) / vel_window
            for j, dof_addr in enumerate(self.joint_dof_addrs):
                self.data.qvel[dof_addr] = vel[j]

            mujoco.mj_forward(self.model, self.data)

            gyro = self.data.sensordata[self.imu_sensor_addrs[0]:self.imu_sensor_addrs[0] + 3]
            acc  = self.data.sensordata[self.imu_sensor_addrs[1]:self.imu_sensor_addrs[1] + 3]
            self.comp_filter.update(acc, gyro, dt=PHYSICS_DT)

            joint_pos = [jp/np.pi - _DEFAULT_JOINT_POS_ARRAY[i] for i, jp in enumerate(self.data.sensordata[self.joint_pos_sensor_addrs])]
            joint_vel = self.data.qvel[self.joint_dof_addrs]

            dataset[i] = np.array([
                *joint_pos,
                *joint_vel,
                *(self.data.sensordata[self.touch_sensor_addrs] > 0).astype(np.float32),
                *(gyro / IMU_GYRO_SCALE),
                *(acc / IMU_ACC_SCALE),
                self.comp_filter.pitch,
                self.comp_filter.roll,
            ], dtype=np.float32)

        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        self.comp_filter.reset()

        # Match GnociGymEnv._get_obs per-dimension normalisation so AMP compares
        # like-for-like.
        dataset /= _OBS_NORM

        if self.frame_stack > 1:
            stacked = np.zeros((n_ref, _OBS_DIM * self.frame_stack), dtype=np.float32)
            for i in range(n_ref):
                for k in range(self.frame_stack):
                    # Match gym.wrappers.FrameStackObservation ordering: oldest
                    # frame first, newest frame last.
                    idx = (i - (self.frame_stack - 1 - k)) % n_ref
                    stacked[i, k * _OBS_DIM:(k + 1) * _OBS_DIM] = dataset[idx]
            dataset = stacked

        return dataset

    def sample_pairs(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample consecutive state pairs (s_i, s_{i+1}) from the reference trajectory.

        Returns:
            Tuple of (s_i, s_next), each of shape (batch_size, *obs_shape).
        """
        n_ref = len(self._dataset)
        inds = torch.randint(0, n_ref, (batch_size,))
        s_i = torch.from_numpy(self._dataset[inds])
        s_next = torch.from_numpy(self._dataset[(inds + 1) % n_ref])
        return s_i, s_next

    def get_reference(self) -> np.ndarray:
        return self._dataset
