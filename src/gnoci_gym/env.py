import mujoco
import gymnasium as gym
import numpy as np
import os
from collections import deque
from .utils import tolerance
from .load_xml import _load_and_perturb_basic_xml
from .filters import ComplementaryFilter, EMAFilter

_STANDING_HEIGHT = 0.235

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

_DEFAULT_JOINT_POS_ARRAY = np.array(
    [_DEFAULT_JOINT_POSITIONS[j] if _DEFAULT_JOINT_POSITIONS[j] is not None else 0.0
     for j in _JOINT_NAMES],
    dtype=np.float32,
)

PHYSICS_DT    = 0.002   # MuJoCo integration timestep (500 Hz)
_CONTACT_GRACE_PERIOD = 0.2  # seconds — grace window for single-foot contact reward
MAX_JOINT_VEL = 3.0    # max joint angular velocity (rad/s) — scales action deltas
IMU_GYRO_SCALE = 10.0  # rad/s — clips to [-1, 1] at this angular velocity
IMU_ACC_SCALE  = 19.62 # m/s² (2g) — clips to [-1, 1] at 2g


test_cfg = dict(
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
)

dom_rnd_cfg = dict(
    initial_randomness=0.05,
    inertial_mass_range=(0.02, 0.04),
    inertial_mass_noise=0.01,
    floor_tilt_range=0.02,
    floor_friction_range=(0.7, 1.3),
    joint_friction_range=(0.07, 0.15),
    joint_armature_range=(0.004, 0.008),
    actuator_gain_range=(0.9, 1.1),
    gravity_noise=0.1,
    obs_noise_scale=0.01,
    push_force_max=1.0,
    push_interval_range=(3.0, 6.0),
    max_action_delay=1,
)


class GnociGymEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    DEFAULT_REWARD_COEFS = {
        'stand':        1.0,
        'velocity':     2.5,
        'foot_contact': 0.0,
        'foot_airtime': 0.5,
        'orientation':  1.0,
        'heading':      0.3,
        'yoke_joint':   1.0,
    }

    def __init__(
            self,
            camera='track',
            render_mode='rgb_array',
            control_hz=80,
            max_joint_vel=MAX_JOINT_VEL,
            initial_randomness=0.1,
            inertial_mass_range=(0.04, 0.06),
            inertial_mass_noise=0.03,
            floor_tilt_range=0.0,
            floor_friction_range=(1.0, 1.0),
            joint_friction_range=(0.1, 0.1),
            joint_armature_range=(0.005, 0.005),
            actuator_gain_range=(1.0, 1.0),
            gravity_noise=0.0,
            obs_noise_scale=0.0,
            push_force_max=0.0,
            push_interval_range=(2.0, 5.0),
            max_action_delay=0,
            action_filter_alpha=0.4,
            task='stand',
            reward_coefs=None,
        ):
        self.camera = camera
        self.render_mode = render_mode
        self.task = task
        self.reward_coefs = {**self.DEFAULT_REWARD_COEFS, **(reward_coefs or {})}
        self.done = False
        self.control_hz = control_hz
        self.n_substeps = int(round(1.0 / (control_hz * PHYSICS_DT)))
        self.action_scale = max_joint_vel / control_hz
        self.initial_randomness = initial_randomness
        self.inertial_mass_range = inertial_mass_range
        self.inertial_mass_noise = inertial_mass_noise
        self.floor_tilt_range = floor_tilt_range
        self.floor_friction_range = floor_friction_range
        self.joint_friction_range = joint_friction_range
        self.joint_armature_range = joint_armature_range
        self.actuator_gain_range = actuator_gain_range
        self.gravity_noise = gravity_noise
        self.obs_noise_scale = obs_noise_scale
        self.push_force_max = push_force_max
        self.push_interval_range = push_interval_range
        self.max_action_delay = max_action_delay
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
        self._sync_action_filters()
        mujoco.mj_forward(self.model, self.data)

        self._push_step = 0
        self._push_interval = self._sample_push_interval()
        self._action_delay = 0
        self._action_buffer = deque([np.zeros(_N_JOINTS, dtype=np.float32)], maxlen=1)

    def initialize_model(self):
        xml_content = _load_and_perturb_basic_xml(
            'scene',
            inertial_mass_range=self.inertial_mass_range,
            inertial_mass_noise=self.inertial_mass_noise,
            floor_tilt_range=self.floor_tilt_range,
            floor_friction_range=self.floor_friction_range,
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

        for dof_addr in self.joint_dof_addrs:
            self.model.dof_frictionloss[dof_addr] = np.random.uniform(*self.joint_friction_range)
            self.model.dof_armature[dof_addr] = np.random.uniform(*self.joint_armature_range)

        for i in range(self.model.nu):
            scale = np.random.uniform(*self.actuator_gain_range)
            self.model.actuator_gainprm[i, 0] *= scale
            self.model.actuator_biasprm[i, 1] *= scale

        if self.gravity_noise > 0:
            self.model.opt.gravity[2] += np.random.normal(0, self.gravity_noise)

        self.comp_filter = ComplementaryFilter()
        self.action_filters = [EMAFilter(alpha=self.action_filter_alpha) for _ in range(_N_JOINTS)]
        grace_steps = max(1, int(_CONTACT_GRACE_PERIOD * self.control_hz))
        self._contact_buffer = deque([False] * grace_steps, maxlen=grace_steps)
        self._foot_airtime = [0.0, 0.0]
        self._foot_was_contact = [False, False]

    def _set_joint_positions(self, joint_positions):
        for jnt_name, pos in joint_positions.items():
            if pos is None:
                continue
            qpos_addr = self.model.jnt_qposadr[self.model.joint(jnt_name).id]
            self.data.qpos[qpos_addr] = pos
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.ctrl[i] = self.data.qpos[qpos_addr]

    def _sync_action_filters(self):
        for i, f in enumerate(self.action_filters):
            f.value = float(self.data.ctrl[i])

    def _randomize_joint_positions(self, randomness):
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            range_min, range_max = self.model.jnt_range[joint_id]
            adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[adr] += np.clip(np.random.normal(0, randomness), range_min, range_max)
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.ctrl[i] = self.data.qpos[qpos_addr]

    def _sample_push_interval(self):
        lo = int(self.push_interval_range[0] * self.control_hz)
        hi = int(self.push_interval_range[1] * self.control_hz)
        return np.random.randint(lo, hi + 1)

    def reset(self, seed=None, **kwargs):
        self.initialize_model()
        mujoco.mj_resetData(self.model, self.data)
        self._set_joint_positions(_DEFAULT_JOINT_POSITIONS)
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self._sync_action_filters()
        mujoco.mj_forward(self.model, self.data)
        self.comp_filter.reset()
        self.done = False

        self._push_step = 0
        self._push_interval = self._sample_push_interval()

        if self.max_action_delay > 0:
            self._action_delay = np.random.randint(0, self.max_action_delay + 1)
            self._action_buffer = deque(
                [np.zeros(_N_JOINTS, dtype=np.float32)] * (self._action_delay + 1),
                maxlen=self._action_delay + 1,
            )

        noisey_state, state = self._get_obs()
        return noisey_state, {'state': state}

    def _get_joint_positions(self):
        return self.data.sensordata[self.joint_pos_sensor_addrs]

    def _get_joint_velocities(self):
        return self.data.qvel[self.joint_dof_addrs]

    def _get_contact_forces(self):
        return (self.data.sensordata[self.touch_sensor_addrs] > 0).astype(np.float32)

    def _get_obs(self):
        gyro, acc = self._get_imu_data()
        joint_pos = (self._get_joint_positions() - _DEFAULT_JOINT_POS_ARRAY) / np.pi
        joint_vel = self._get_joint_velocities() / np.pi
        obs = np.array([
            *joint_pos,
            *joint_vel,
            *self._get_contact_forces(),
            *(gyro / IMU_GYRO_SCALE),
            *(acc / IMU_ACC_SCALE),
            *self._get_pitch_and_roll(gyro, acc),
        ])
        if self.obs_noise_scale > 0:
            noisey_obs = obs + np.random.normal(0, self.obs_noise_scale, obs.shape)
        else:
            noisey_obs = obs
        return noisey_obs.astype(np.float32), obs.astype(np.float32)

    def _get_info(self):
        return {}

    def overturned(self):
        return self._get_root_upright() < 0.3

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

    def _get_foot_airtime_reward(self):
        dt = 1.0 / self.control_hz
        contacts = self.data.sensordata[self.touch_sensor_addrs]
        current = [
            contacts[0] > 0 or contacts[1] > 0,  # left foot
            contacts[2] > 0 or contacts[3] > 0,  # right foot
        ]
        reward = 0.0
        for i, (in_contact, was_contact) in enumerate(zip(current, self._foot_was_contact)):
            if not in_contact:
                self._foot_airtime[i] += dt
            elif not was_contact:  # touchdown
                reward += self._foot_airtime[i] - 0.12
                self._foot_airtime[i] = 0.0
        self._foot_was_contact = current
        return reward

    def _get_stand_reward(self):
        upright = self._get_root_upright()
        height  = self._get_root_height()
        standing = tolerance(
            height,
            bounds=(_STANDING_HEIGHT, float('inf')),
            margin=_STANDING_HEIGHT / 2,
        )
        upright = (1 + upright) / 2
        return (3 * standing + upright) / 4

    def _get_velocity_reward(self):
        velocity = self._get_velocity()
        side_v = abs(velocity[0])
        lateral_penalty = max(1.0 - 2.0 * side_v, 0.0)
        forward_reward = tolerance(-velocity[1], bounds=(0.4, 0.6), margin=0.4)
        return forward_reward * lateral_penalty

    def _get_orientation_reward(self):
        pitch = float(self.comp_filter.pitch)
        roll  = float(self.comp_filter.roll)
        return (
            tolerance(pitch, bounds=(0, 0), margin=0.2) *
            tolerance(roll,  bounds=(0, 0), margin=0.2)
        )

    def _get_yoke_joint_reward(self):
        # Encourage head__left_yoke, left_yoke__hip, head__right_yoke, right_yoke__hip
        # to stay near their default position of 0.0
        positions = self._get_joint_positions()
        yoke_indices = [0, 1, 5, 6]
        return float(np.mean([
            tolerance(float(positions[i]), bounds=(0.0, 0.0), margin=0.2)
            for i in yoke_indices
        ]))

    def _get_heading_reward(self):
        xmat = self.data.xmat[self.body_id]
        # Body forward direction in world XY plane (-Y body axis, rewarded motion is -Y world)
        body_forward = np.array([-xmat[3], -xmat[4]])
        body_forward_norm = np.linalg.norm(body_forward)
        if body_forward_norm < 1e-6:
            return 0.0
        body_forward = body_forward / body_forward_norm
        return float((np.dot(body_forward, [0.0, -1.0]) + 1.0) / 2.0)

    def _get_foot_contact_reward(self):
        contacts = self.data.sensordata[self.touch_sensor_addrs]
        left  = contacts[0] > 0 or contacts[1] > 0
        right = contacts[2] > 0 or contacts[3] > 0
        single = left ^ right
        self._contact_buffer.append(single)
        return 1.0 if any(self._contact_buffer) else 0.0

    def _get_reward(self):
        c = self.reward_coefs
        stand_reward = self._get_stand_reward()

        if self.task == 'walk':
            velocity_reward     = self._get_velocity_reward()
            foot_contact_reward = self._get_foot_contact_reward()
            foot_airtime_reward = self._get_foot_airtime_reward()
            orientation_reward  = self._get_orientation_reward()
            heading_reward      = self._get_heading_reward()
            yoke_joint_reward   = self._get_yoke_joint_reward()
            return (
                c['stand'] * stand_reward * (1 + c['velocity'] * velocity_reward)
                + c['foot_contact'] * foot_contact_reward
                + c['foot_airtime'] * foot_airtime_reward
                + c['orientation']  * orientation_reward
                + c['heading']      * heading_reward
                + c['yoke_joint']   * yoke_joint_reward
            )

        return c['stand'] * stand_reward

    def step(self, action):
        action = action.clip(-1, 1)

        if self.max_action_delay > 0:
            self._action_buffer.append(action.copy())
            action = self._action_buffer[0]

        if self.push_force_max > 0:
            self.data.xfrc_applied[self.body_id] = 0
            self._push_step += 1
            if self._push_step >= self._push_interval:
                fx = np.random.uniform(-self.push_force_max, self.push_force_max)
                fy = np.random.uniform(-self.push_force_max, self.push_force_max)
                self.data.xfrc_applied[self.body_id, 3] = fx
                self.data.xfrc_applied[self.body_id, 4] = fy
                self._push_step = 0
                self._push_interval = self._sample_push_interval()

        for i in range(self.model.nu):
            delta = self.action_filters[i].update(action[i]) * self.action_scale
            lo, hi = self.joint_ranges[i]
            self.data.ctrl[i] = float(np.clip(self.data.ctrl[i] + delta, lo, hi))
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        state, noisey_state = self._get_obs()
        reward = self._get_reward()
        if self.overturned():
            self.done = True
        return (noisey_state, reward, self.done, self.done, {'state': state})

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            with mujoco.Renderer(self.model) as renderer:
                renderer.update_scene(self.data, camera=self.camera)
                return renderer.render()
        else:
            raise ValueError(f"Invalid render mode: {mode}")
