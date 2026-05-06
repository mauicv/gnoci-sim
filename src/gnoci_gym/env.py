import mujoco
import gymnasium as gym
import numpy as np
import os
from .utils import tolerance
from .load_xml import _load_and_perturb_basic_xml
from .filters import ComplementaryFilter, EMAFilter

_STANDING_HEIGHT = 0.235  # TODO: calibrate once robot is simulating

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

_DEFAULT_JOINT_POSITIONS: dict[str, float] = {
    "head__left_yoke":            None,
    "left_yoke__hip":             None,
    "left_hip__upper_leg":        -0.6,
    "left_upper_leg__lower_leg":  1.4,
    "left_lower_leg__foot":       0.75,
    "head__right_yoke":           None,
    "right_yoke__hip":            None,
    "right_hip__upper_leg":        0.6,
    "right_upper_leg__lower_leg":  -1.4,
    "right_lower_leg__foot":       -0.75,
}

_TOUCH_SENSOR_NAMES = [
    "forward_left_c_sense-touch",
    "back_left_c_sense-touch",
    "forward_right_c_sense-touch",
    "back_right_c_sense-touch",
]

_N_JOINTS = len(_JOINT_NAMES)   # 10
_N_TOUCH  = len(_TOUCH_SENSOR_NAMES)  # 4

PHYSICS_DT    = 0.002  # MuJoCo integration timestep (500 Hz)
MAX_JOINT_VEL = 3.0   # max joint angular velocity (rad/s) — scales action deltas


class GnociGymEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
            self,
            camera='track',
            render_mode='rgb_array',
            control_hz=60,
            max_joint_vel=MAX_JOINT_VEL,
            initial_randomness=0.1,
            inertial_mass_range=(0.04, 0.06),
            inertial_mass_noise=0.01,
            floor_tilt_range=0.0,
            action_filter_alpha=0.4,
        ):
        self.camera = camera
        self.render_mode = render_mode
        self.done = False
        self.control_hz = control_hz
        self.n_substeps = int(round(1.0 / (control_hz * PHYSICS_DT)))
        self.action_scale = max_joint_vel / control_hz
        self.initial_randomness = initial_randomness
        self.inertial_mass_range = inertial_mass_range
        self.inertial_mass_noise = inertial_mass_noise
        self.floor_tilt_range = floor_tilt_range
        self.action_filter_alpha = action_filter_alpha
        self.metadata['render_fps'] = control_hz

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(_N_JOINTS + _N_JOINTS + _N_TOUCH + 6 + 2,),  # joint pos, joint vel, touch, imu, pitch+roll
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(_N_JOINTS,), dtype=np.float32
        )
        self.initialize_model()
        self._set_joint_positions(_DEFAULT_JOINT_POSITIONS)
        self._randomize_joint_positions(randomness=self.initial_randomness)
        mujoco.mj_forward(self.model, self.data)

    def initialize_model(self):
        xml_content = _load_and_perturb_basic_xml(
            'scene',
            inertial_mass_range=self.inertial_mass_range,
            inertial_mass_noise=self.inertial_mass_noise,
            floor_tilt_range=self.floor_tilt_range,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_content)
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
        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "head_base"
        )
        self.floor_geom_id = self.model.geom("floor").id
        self.floor_z = self.model.geom_pos[self.floor_geom_id][2]
        self.joint_qpos_addrs = [
            self.model.jnt_qposadr[self.model.joint(j).id]
            for j in _JOINT_NAMES
        ]
        self.joint_ranges = [
            self.model.jnt_range[self.model.joint(j).id]
            for j in _JOINT_NAMES
        ]
        self.imu_sensor_addrs = [
            self.model.sensor_adr[self.model.sensor(n).id]
            for n in ["imu-gyro", "imu-acc"]
        ]
        self.comp_filter = ComplementaryFilter()
        self.action_filters = [EMAFilter(alpha=self.action_filter_alpha) for _ in range(_N_JOINTS)]

    def _set_joint_positions(self, joint_positions):
        for jnt_name, pos in joint_positions.items():
            if pos is None:
                continue
            qpos_addr = self.model.jnt_qposadr[self.model.joint(jnt_name).id]
            self.data.qpos[qpos_addr] = pos
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.ctrl[i] = self.data.qpos[qpos_addr]

    def _randomize_joint_positions(self, randomness):
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            range_min, range_max = self.model.jnt_range[joint_id]
            adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[adr] += np.clip(np.random.normal(0, randomness), range_min, range_max)
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.ctrl[i] = self.data.qpos[qpos_addr]

    def reset(self, seed=None, **kwargs):
        self.initialize_model()
        mujoco.mj_resetData(self.model, self.data)
        self._set_joint_positions(_DEFAULT_JOINT_POSITIONS)
        self._randomize_joint_positions(randomness=self.initial_randomness)
        mujoco.mj_forward(self.model, self.data)
        self.comp_filter.reset()
        self.done = False
        return self._get_obs(), {}

    def _get_joint_positions(self):
        return self.data.sensordata[self.joint_pos_sensor_addrs]

    def _get_joint_velocities(self):
        return self.data.qvel[self.joint_dof_addrs]

    def _get_contact_forces(self):
        return self.data.sensordata[self.touch_sensor_addrs]

    def _get_obs(self):
        gyro, acc = self._get_imu_data()
        obs = np.array([
            *self._get_joint_positions(),
            *self._get_joint_velocities(),
            *self._get_contact_forces(),
            *gyro,
            *acc,
            *self._get_pitch_and_roll(gyro, acc),
        ])
        return obs.astype(np.float32)

    def _get_info(self):
        return {}

    def overturned(self):
        return self._get_root_upright() < 0

    def _get_root_upright(self):
        xmat = self.data.xmat[self.body_id]
        z_axis = np.array([xmat[6], xmat[7], xmat[8]])
        return np.dot(z_axis, [0, 0, 1])

    def _get_root_height(self):
        _, _, z = self.data.xpos[self.body_id]
        return z - self.floor_z

    def _get_imu_data(self):
        gyro = self.data.sensordata[self.imu_sensor_addrs[0]:self.imu_sensor_addrs[0] + 3]
        acc  = self.data.sensordata[self.imu_sensor_addrs[1]:self.imu_sensor_addrs[1] + 3]
        return gyro, acc

    def _get_pitch_and_roll(self, gyro, acc):
        self.comp_filter.update(acc, gyro, dt=1.0 / self.control_hz)
        return np.array([self.comp_filter.pitch, self.comp_filter.roll], dtype=np.float32)

    def _get_velocity(self):
        return self.data.cvel[self.body_id][3:6]

    def _get_reward(self):
        upright = self._get_root_upright()
        height = self._get_root_height()
        standing = tolerance(
            height,
            bounds=(_STANDING_HEIGHT, float('inf')),
            margin=_STANDING_HEIGHT / 2
        )
        upright = (1 + upright) / 2
        stand_reward = (3 * standing + upright) / 4
        # velocity = self._get_velocity()
        # side_v = abs(velocity[1])
        # lateral_penalty = max(1.0 - 0.5 * side_v, 0.0)

        # velocity_reward = tolerance(
        #     -velocity[0],
        #     bounds=(1, 2),
        #     margin=1
        # )

        # total_reward = stand_reward * (5 * velocity_reward + 1) / 6
        # return total_reward * lateral_penalty
        return stand_reward

    def step(self, action):
        action = action.clip(-1, 1)
        for i in range(self.model.nu):
            delta = self.action_filters[i].update(action[i]) * self.action_scale
            lo, hi = self.joint_ranges[i]
            self.data.ctrl[i] = float(np.clip(self.data.ctrl[i] + delta, lo, hi))
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        state = self._get_obs()
        reward = self._get_reward()
        if self.overturned():
            self.done = True
        return (state, reward, self.done, self.done, {})

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            with mujoco.Renderer(self.model) as renderer:
                renderer.update_scene(self.data, camera=self.camera)
                return renderer.render()
        else:
            raise ValueError(f"Invalid render mode: {mode}")
