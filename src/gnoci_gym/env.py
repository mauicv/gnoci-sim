import mujoco
import gymnasium as gym
import numpy as np
import os
from collections import deque
from .utils import tolerance
from .load_xml import _load_xml
from .filters import ComplementaryFilter, EMAFilter, Debouncer
from .config import (
    CONTROL_HZ,
    ACC_FILTER_ALPHA,
    JOINT_VEL_FILTER_ALPHA,
)
import math

_STANDING_HEIGHT = 0.23

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

_TOUCH_SENSOR_NAMES = [
    "forward_left_c_sense-touch",
    "back_left_c_sense-touch",
    "forward_right_c_sense-touch",
    "back_right_c_sense-touch",
]

_N_JOINTS = len(_JOINT_NAMES)   # 10
_N_TOUCH  = len(_TOUCH_SENSOR_NAMES)  # 4

PHYSICS_DT    = 0.002   # MuJoCo integration timestep (500 Hz)
_CONTACT_GRACE_PERIOD = 0.2  # seconds — grace window for single-foot contact reward
_CONTACT_DEBOUNCE_PERIOD = 0.05  # seconds — a contact change must persist this long to register
# Hysteresis thresholds (N) mimicking the limit switches' actuation/release
# forces: contact turns on above ON, off below OFF.  Standing loads are
# ~2.7 N on the lightest (front) sites, so ON stays well below stance forces.
_CONTACT_FORCE_ON  = 1.0
_CONTACT_FORCE_OFF = 0.5

IMU_GYRO_SCALE = ((180 / np.pi) / 250.0)
IMU_ACC_SCALE  = 9.81 # m/s² (2g) — clips to [-1, 1]

# Sysid-derived per-joint dynamics — must stay equal to the frictionloss /
# armature attributes on the named joints in desc/gnoci.xml. They have to be
# threaded through here because _randomize_dynamics() stamps
# dof_frictionloss/dof_armature on every reset, so the MJCF values alone
# never survive past construction.
SYSID_JOINT_FRICTIONLOSS = 0.0072408653310164755
SYSID_JOINT_ARMATURE     = 0.014643797951585229

# Per-sensor observation noise scales, in raw sensor units (rad, rad/s, m/s²).
# Sampled uniform(-scale, scale) and converted into obs units with the same
# transforms _get_obs applies before being added to the obs.
OBS_NOISE_SCALES = dict(
    hip_pos=0.03,   # rad, for each hip joint
    knee_pos=0.05,  # rad, for each knee joint
    ankle_pos=0.08, # rad, for each ankle joint
    joint_vel=2.5,  # rad/s # Was 1.5
    gravity=0.1,    # rad, applied to the pitch/roll estimate
    linvel=0.1,     # unused: no linvel in the obs
    gyro=0.1,       # rad/s
    accelerometer=0.05,  # m/s²
)


def _build_obs_noise_vec(scales):
    joint_pos = []
    for name in _JOINT_NAMES:
        if 'lower_leg__foot' in name:
            s = scales['ankle_pos']
        elif 'upper_leg__lower_leg' in name:
            s = scales['knee_pos']
        else:
            s = scales['hip_pos']
        joint_pos.append(s / np.pi)  # obs joint pos are rad / pi
    return np.concatenate([
        joint_pos,
        np.full(_N_JOINTS, scales['joint_vel']),                # obs vel is raw rad/s
        np.zeros(_N_TOUCH),                                     # binary contacts stay clean
        np.full(3, scales['gyro'] * IMU_GYRO_SCALE),
        np.full(3, scales['accelerometer'] / IMU_ACC_SCALE),
        np.full(2, scales['gravity']),                          # pitch/roll in rad
    ])

test_cfg = dict(
    initial_randomness=0.0,
    inertial_mass_range=(0.0, 0.0),
    inertial_mass_noise=0.0,
    floor_tilt_range=0.0,
    floor_friction_range=(1.0, 1.0),
    joint_friction_range=(SYSID_JOINT_FRICTIONLOSS, SYSID_JOINT_FRICTIONLOSS),
    joint_armature_range=(SYSID_JOINT_ARMATURE, SYSID_JOINT_ARMATURE),
    actuator_gain_range=(1.0, 1.0),
    gravity_noise=0.0,
    obs_noise_level=0.0,
    push_force_max=0.0,
    max_action_delay=0,
)

dom_rnd_cfg = dict(
    initial_randomness=0.05,
    inertial_mass_range=(0.02, 0.04),
    inertial_mass_noise=0.01,
    floor_tilt_range=0.02,
    floor_friction_range=(0.7, 1.3),
    # same relative spread the old ranges had around their (pre-sysid)
    # nominals: friction x0.7-1.5, armature x0.8-1.6
    joint_friction_range=(0.7 * SYSID_JOINT_FRICTIONLOSS, 1.5 * SYSID_JOINT_FRICTIONLOSS),
    joint_armature_range=(0.8 * SYSID_JOINT_ARMATURE, 1.6 * SYSID_JOINT_ARMATURE),
    actuator_gain_range=(0.9, 1.1),
    gravity_noise=0.1,
    obs_noise_level=1.0,
    push_force_max=1.0,
    push_interval_range=(3.0, 6.0),
    max_action_delay=1,
)


class GnociGymEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    DEFAULT_REWARD_COEFS = {
        'stand':         1.0,
        'both_feet':     0.5,
        'default_pose':  0.5,
        'velocity':      2.5,
        'foot_contact':  0.75,
        'foot_airtime':  0.5,
        'foot_clearance': 0.5,
        'fall':          0.5,
        'orientation':   0.1,
        'heading':       0.3,
        'yoke_joint':    0.0,
        'yoke_symmetry': 0.1,
        'action_magnitude': 0.05,
    }

    def __init__(
            self,
            camera='track',
            render_mode='rgb_array',
            control_hz=CONTROL_HZ,
            initial_randomness=0.1,
            inertial_mass_range=(0.04, 0.06),
            inertial_mass_noise=0.03,
            floor_tilt_range=0.0,
            floor_friction_range=(1.0, 1.0),
            joint_friction_range=(SYSID_JOINT_FRICTIONLOSS, SYSID_JOINT_FRICTIONLOSS),
            joint_armature_range=(SYSID_JOINT_ARMATURE, SYSID_JOINT_ARMATURE),
            actuator_gain_range=(1.0, 1.0),
            gravity_noise=0.0,
            obs_noise_level=0.0,
            obs_noise_scales=None,
            push_force_max=0.0,
            push_interval_range=(2.0, 5.0),
            max_action_delay=0,
            action_filter_alpha=0.4,
            action_scale=0.25,
            task='stand',
            reward_coefs=None,
            fix_root_body=False,
            survival_bonus=0.2,
            target_velocity=0.1,
            target_velocity_band=0.1,
            foot_clearance_height=0.02,
        ):
        # Curriculum-controlled knobs. These are plain attributes so an external
        # trainer can ramp them between phases (e.g. SB3 env_method/set_attr) via
        # set_curriculum(). Defaults give a small standing floor + slow target
        # that the trainer is expected to anneal: survival_bonus -> 0 and
        # target_velocity upward.
        self.survival_bonus = float(survival_bonus)
        self.target_velocity = float(target_velocity)
        self.target_velocity_band = float(target_velocity_band)
        self.foot_clearance_height = float(foot_clearance_height)
        self.camera = camera
        self.render_mode = render_mode
        self.task = task
        self.reward_coefs = {**self.DEFAULT_REWARD_COEFS, **(reward_coefs or {})}
        self.done = False
        self.control_hz = control_hz
        self.n_substeps = int(round(1.0 / (control_hz * PHYSICS_DT)))
        self.initial_randomness = initial_randomness
        self.inertial_mass_range = inertial_mass_range
        self.inertial_mass_noise = inertial_mass_noise
        self.floor_tilt_range = floor_tilt_range
        self.floor_friction_range = floor_friction_range
        self.joint_friction_range = joint_friction_range
        self.joint_armature_range = joint_armature_range
        self.actuator_gain_range = actuator_gain_range
        self.gravity_noise = gravity_noise
        self.obs_noise_level = obs_noise_level
        self.obs_noise_scales = {**OBS_NOISE_SCALES, **(obs_noise_scales or {})}
        self._obs_noise_vec = _build_obs_noise_vec(self.obs_noise_scales)
        self.push_force_max = push_force_max
        self.push_interval_range = push_interval_range
        self.max_action_delay = max_action_delay
        self.action_filter_alpha = action_filter_alpha
        self.action_scale = action_scale
        self.fix_root_body = fix_root_body
        self.metadata['render_fps'] = control_hz

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(_N_JOINTS + _N_JOINTS + _N_TOUCH + 6 + 2,),  # joint pos, joint vel, touch, imu, pitch+roll
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(_N_JOINTS,), dtype=np.float32
        )
        self._build_model()
        self._randomize_dynamics()
        self._set_joint_positions()
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self._sync_action_filters()
        mujoco.mj_forward(self.model, self.data)

        self._push_step = 0
        self._push_interval = self._sample_push_interval()
        self._action_delay = 0
        self._action_buffer = deque([np.zeros(_N_JOINTS, dtype=np.float32)], maxlen=1)
        self._last_raw_action = np.zeros(_N_JOINTS, dtype=np.float32)

    def _build_model(self):
        """One-time model compile + cache. Not called on reset() — only the
        MjData (via mj_resetData) and the per-episode dynamics randomization
        (via _randomize_dynamics) change between episodes. Recompiling from
        XML and recreating the Renderer on every reset used to dominate PPO
        wall-clock (~180x the cost of a single physics step)."""
        xml_content = _load_xml('scene', fix_root_body=self.fix_root_body)
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
        self.left_foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_base"
        )
        self.right_foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_base"
        )
        try:
            self.floor_geom_id = self.model.geom("floor").id
            self.floor_z = self.model.geom_pos[self.floor_geom_id][2]
        except KeyError:
            self.floor_geom_id = -1
            self.floor_z = 0.0
        # Episode ends when any of these geom centres passes below the floor
        # plane; the feet and world geoms (floor) are excluded.
        foot_body_ids = {self.left_foot_body_id, self.right_foot_body_id}
        self._ground_termination_geoms = np.array([
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] != 0
            and self.model.geom_bodyid[g] not in foot_body_ids
        ])
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

        # Base (unrandomized) values, snapshotted once so per-episode dynamics
        # randomization in _randomize_dynamics() can be resampled from a fixed
        # reference each reset instead of compounding on the previous episode's
        # already-randomized values.
        self._base_body_mass = self.model.body_mass.copy()
        self._base_body_inertia = self.model.body_inertia.copy()
        self._randomizable_body_ids = np.nonzero(self._base_body_mass > 0)[0]
        self._base_actuator_gainprm = self.model.actuator_gainprm.copy()
        self._base_actuator_biasprm = self.model.actuator_biasprm.copy()
        self._base_gravity_z = float(self.model.opt.gravity[2])

        self.comp_filter = ComplementaryFilter()
        self.action_filters = [EMAFilter(alpha=self.action_filter_alpha) for _ in range(_N_JOINTS)]
        self.acc_filters = [EMAFilter(alpha=ACC_FILTER_ALPHA, warm_start=True) for _ in range(3)]
        self.joint_vel_filters = [EMAFilter(alpha=JOINT_VEL_FILTER_ALPHA, warm_start=True) for _ in range(_N_JOINTS)]
        self._grace_steps = max(1, int(_CONTACT_GRACE_PERIOD * self.control_hz))
        self._contact_buffer = deque([False] * self._grace_steps, maxlen=self._grace_steps)
        self._debounce_steps = max(1, int(_CONTACT_DEBOUNCE_PERIOD * self.control_hz))
        self.contact_debouncers = [Debouncer(self._debounce_steps) for _ in range(_N_TOUCH)]
        self._contact_states = np.zeros(_N_TOUCH, dtype=np.float32)
        self._foot_airtime = [0.0, 0.0]
        self._foot_was_contact = [False, False]

        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model)

    def _randomize_dynamics(self):
        """Per-episode domain randomization, applied directly to the already-
        compiled model (no XML/recompile involved)."""
        for body_id in self._randomizable_body_ids:
            scale = (
                1.0
                + np.random.uniform(*self.inertial_mass_range)
                + np.random.normal(0, self.inertial_mass_noise)
            )
            self.model.body_mass[body_id] = self._base_body_mass[body_id] * scale
            self.model.body_inertia[body_id] = self._base_body_inertia[body_id] * scale

        if self.floor_geom_id != -1:
            self.model.geom_friction[self.floor_geom_id, 0] = np.random.uniform(*self.floor_friction_range)

            if self.floor_tilt_range > 0:
                roll  = np.random.uniform(-self.floor_tilt_range, self.floor_tilt_range)
                pitch = np.random.uniform(-self.floor_tilt_range, self.floor_tilt_range)
                qr = [np.cos(roll  / 2), np.sin(roll  / 2), 0.0, 0.0]
                qp = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
                w = qr[0]*qp[0] - qr[1]*qp[1] - qr[2]*qp[2] - qr[3]*qp[3]
                x = qr[0]*qp[1] + qr[1]*qp[0] + qr[2]*qp[3] - qr[3]*qp[2]
                y = qr[0]*qp[2] - qr[1]*qp[3] + qr[2]*qp[0] + qr[3]*qp[1]
                z = qr[0]*qp[3] + qr[1]*qp[2] - qr[2]*qp[1] + qr[3]*qp[0]
                self.model.geom_quat[self.floor_geom_id] = [w, x, y, z]

        for dof_addr in self.joint_dof_addrs:
            self.model.dof_frictionloss[dof_addr] = np.random.uniform(*self.joint_friction_range)
            self.model.dof_armature[dof_addr] = np.random.uniform(*self.joint_armature_range)

        for i in range(self.model.nu):
            scale = np.random.uniform(*self.actuator_gain_range)
            self.model.actuator_gainprm[i, 0] = self._base_actuator_gainprm[i, 0] * scale
            self.model.actuator_biasprm[i, 1] = self._base_actuator_biasprm[i, 1] * scale

        if self.gravity_noise > 0:
            self.model.opt.gravity[2] = self._base_gravity_z + np.random.normal(0, self.gravity_noise)

        # Recompute compile-time-derived constants (e.g. body/dof invweight)
        # that depend on the mass/inertia values just edited above. Far
        # cheaper than a full XML recompile.
        mujoco.mj_setConst(self.model, self.data)

    def _set_joint_positions(self):
        for jnt_name in _JOINT_NAMES:
            qpos_addr = self.model.jnt_qposadr[self.model.joint(jnt_name).id]
            self.data.qpos[qpos_addr] = 0.0
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.ctrl[i] = self.data.qpos[qpos_addr]

    def _sync_action_filters(self):
        for i, f in enumerate(self.action_filters):
            f.value = 0.0

    def _randomize_joint_positions(self, randomness):
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            range_min, range_max = self.model.jnt_range[joint_id]
            adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[adr] += np.clip(np.random.normal(0, randomness), range_min, range_max)
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.ctrl[i] = self.data.qpos[qpos_addr]

    def set_curriculum(
            self,
            *,
            survival_bonus=None,
            target_velocity=None,
            target_velocity_band=None,
            foot_clearance_height=None,
    ):
        """Update curriculum knobs mid-training.

        Intended to be driven by the trainer (e.g. via SB3 ``env_method``) to
        anneal the standing floor and exploration noise away while ramping the
        forward-speed target up. Only provided values are changed.
        """
        if survival_bonus is not None:
            self.survival_bonus = max(0.0, float(survival_bonus))
        if target_velocity is not None:
            self.target_velocity = max(0.0, float(target_velocity))
        if target_velocity_band is not None:
            self.target_velocity_band = max(0.0, float(target_velocity_band))
        if foot_clearance_height is not None:
            self.foot_clearance_height = max(0.0, float(foot_clearance_height))

    def _sample_push_interval(self):
        lo = int(self.push_interval_range[0] * self.control_hz)
        hi = int(self.push_interval_range[1] * self.control_hz)
        return np.random.randint(lo, hi + 1)

    def reset(self, seed=None, **kwargs):
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_dynamics()
        self._set_joint_positions()
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self._sync_action_filters()
        mujoco.mj_forward(self.model, self.data)
        self.comp_filter.reset()
        for f in self.acc_filters + self.joint_vel_filters:
            f.reset()
        self.done = False

        self._push_step = 0
        self._push_interval = self._sample_push_interval()

        self._contact_buffer = deque([False] * self._grace_steps, maxlen=self._grace_steps)
        for d in self.contact_debouncers:
            d.reset()
        self._foot_airtime = [0.0, 0.0]
        self._foot_was_contact = [False, False]
        self._last_raw_action = np.zeros(_N_JOINTS, dtype=np.float32)

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

    def _update_contact_states(self):
        """Threshold (with hysteresis) and debounce the touch readings; called
        once per control step (from _get_obs).  All contact consumers read the
        cached result."""
        forces = self.data.sensordata[self.touch_sensor_addrs]
        states = []
        for d, f in zip(self.contact_debouncers, forces):
            threshold = _CONTACT_FORCE_OFF if d.state else _CONTACT_FORCE_ON
            states.append(d.update(f > threshold))
        self._contact_states = np.array(states, dtype=np.float32)

    def _get_contact_forces(self):
        return self._contact_states

    def _get_obs(self):
        self._update_contact_states()
        gyro, acc = self._get_imu_data()
        joint_pos = self._get_joint_positions() / np.pi
        joint_vel = self._get_joint_velocities()
        # Hardware low-passes the accelerometer and joint velocities before
        # building the obs, but feeds the complementary filter the *raw*
        # accelerometer — mirror both, so pitch/roll below use unfiltered acc.
        acc_filtered = [f.update(a) for f, a in zip(self.acc_filters, acc)]
        joint_vel_filtered = [f.update(v) for f, v in zip(self.joint_vel_filters, joint_vel)]
        obs = np.array([
            *joint_pos,
            *joint_vel_filtered,
            *self._get_contact_forces(),
            *gyro,
            *acc_filtered,
            *self._get_pitch_and_roll(gyro * 250, acc),
        ])
        if self.obs_noise_level > 0:
            noise = np.random.uniform(-1, 1, obs.shape) * self._obs_noise_vec
            noisey_obs = obs + self.obs_noise_level * noise
        else:
            noisey_obs = obs
        return noisey_obs.astype(np.float32), obs.astype(np.float32)

    def _get_info(self):
        return {}

    def _body_below_floor(self):
        """True when any non-foot geom centre passes below the floor plane.

        Uses the floor's live position and normal so it stays correct under
        the tilted-floor randomization.
        """
        if self.floor_geom_id == -1:
            return False
        normal = self.data.geom_xmat[self.floor_geom_id].reshape(3, 3)[:, 2]
        offsets = self.data.geom_xpos[self._ground_termination_geoms] \
            - self.data.geom_xpos[self.floor_geom_id]
        return bool((offsets @ normal < 0.0).any())

    def _get_root_upright(self):
        xmat = self.data.xmat[self.body_id]
        z_axis = np.array([xmat[6], xmat[7], xmat[8]])
        return np.dot(z_axis, [0, 0, 1])

    def _get_root_height(self):
        _, _, z = self.data.xpos[self.body_id]
        return z - self.floor_z

    def permute_imu_data(self, data):
        x, y, z = data[0], data[1], data[2]
        return [y, x, -z]

    def _get_imu_data(self):
        gyro = self.data.sensordata[self.imu_sensor_addrs[0]:self.imu_sensor_addrs[0] + 3]
        acc  = self.data.sensordata[self.imu_sensor_addrs[1]:self.imu_sensor_addrs[1] + 3]
        gyro = [r * IMU_GYRO_SCALE for r in gyro]
        acc = [a / IMU_ACC_SCALE for a in acc]
        gyro = self.permute_imu_data(gyro)
        acc = self.permute_imu_data(acc)
        return np.array(gyro), np.array(acc)

    def _get_pitch_and_roll(self, gyro, acc):
        self.comp_filter.update(acc, gyro, dt=1.0 / self.control_hz)
        return np.array([self.comp_filter.pitch, self.comp_filter.roll], dtype=np.float32)

    def _get_velocity(self):
        return self.data.cvel[self.body_id][3:6]

    def _get_foot_airtime_reward(self):
        dt = 1.0 / self.control_hz
        contacts = self._contact_states
        current = [
            contacts[0] > 0 or contacts[1] > 0,  # left foot
            contacts[2] > 0 or contacts[3] > 0,  # right foot
        ]
        reward = 0.0
        for i, (in_contact, was_contact) in enumerate(zip(current, self._foot_was_contact)):
            if not in_contact:
                self._foot_airtime[i] += dt
            elif not was_contact:  # touchdown
                reward += self._foot_airtime[i] - 0.075
                self._foot_airtime[i] = 0.0
        self._foot_was_contact = current
        return reward

    def _get_stand_gate(self):
        """Posture quality in [0, 1]: 1 when upright at standing height, decaying
        as the robot tips or sinks. Used as a multiplicative gate on the shaped
        locomotion reward (and, inverted, as the fall penalty) rather than as a
        standalone reward term, so it provides no standing reward floor."""
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
        forward = -velocity[1]  # rewarded direction is -Y world
        target = self.target_velocity
        if forward <= 0.0 or target <= 0.0:
            # Exactly zero at (or below) zero forward speed — no deadband leak,
            # so standing still earns nothing from velocity.
            forward_reward = 0.0
        elif forward < target:
            # Linear ramp from 0 at v=0 to 1 at the (curriculum) target speed.
            forward_reward = forward / target
        else:
            # At/above target: full credit within the band, fading beyond it.
            forward_reward = tolerance(
                forward,
                bounds=(target, target + self.target_velocity_band),
                margin=target,
            )
        return forward_reward * lateral_penalty

    def _get_foot_clearance_reward(self):
        """Reward committing to a step: credit the height difference between the
        feet so one foot must clearly leave the ground. Exactly 0 when both feet
        are at the same height (e.g. planted), ramping to 1 once the swing foot
        is ``2 * foot_clearance_height`` above the stance foot."""
        threshold = self.foot_clearance_height
        if threshold <= 0.0:
            return 0.0
        lz = self.data.xpos[self.left_foot_body_id][2]
        rz = self.data.xpos[self.right_foot_body_id][2]
        clearance = abs(lz - rz)
        return float(np.clip((clearance - threshold) / threshold, 0.0, 1.0))

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

    def _get_yoke_symmetry_reward(self):
        # Encourage the yoke joints to mirror left/right without pulling them
        # toward 0: head__left_yoke ~ head__right_yoke and left_yoke__hip ~
        # right_yoke__hip. The left joint axes are -z and the right +z in the
        # MJCF, so a mirrored posture means *equal* joint values.
        positions = self._get_joint_positions()
        return float(np.mean([
            tolerance(float(positions[l] - positions[r]), bounds=(0.0, 0.0), margin=0.2)
            for l, r in ((0, 5), (1, 6))  # (head__yoke, yoke__hip) pairs
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
        contacts = self._contact_states
        left  = contacts[0] > 0 or contacts[1] > 0
        right = contacts[2] > 0 or contacts[3] > 0
        single = left ^ right
        self._contact_buffer.append(single)
        return 1.0 if any(self._contact_buffer) else 0.0

    def _get_both_feet_contact_reward(self):
        contacts = self._contact_states
        left  = contacts[0] > 0 or contacts[1] > 0
        right = contacts[2] > 0 or contacts[3] > 0
        return 1.0 if (left and right) else 0.0

    def _get_action_magnitude_reward(self):
        # Mean squared *raw* (pre-clip) action. Using the unclipped value
        # means a policy that saturates keeps getting corrective gradient —
        # clip() has zero derivative outside [-1, 1], so penalising the
        # clipped action can't push an already-saturating policy back down.
        return float(np.mean(np.square(self._last_raw_action)))

    def _get_default_pose_reward(self):
        # Every joint's default is 0.0 (see _set_joint_positions).
        positions = self._get_joint_positions()
        return float(np.mean([
            tolerance(float(p), bounds=(0.0, 0.0), margin=0.3)
            for p in positions
        ]))

    def _get_reward(self):
        c = self.reward_coefs
        stand_gate = self._get_stand_gate()
        components = {'stand_gate': stand_gate}

        if self.task == 'walk':
            velocity_reward      = self._get_velocity_reward()
            foot_contact_reward  = self._get_foot_contact_reward()
            foot_airtime_reward  = self._get_foot_airtime_reward()
            foot_clearance_reward = self._get_foot_clearance_reward()
            orientation_reward   = self._get_orientation_reward()
            heading_reward       = self._get_heading_reward()
            yoke_joint_reward    = self._get_yoke_joint_reward()
            yoke_symmetry_reward = self._get_yoke_symmetry_reward()
            action_magnitude_reward = self._get_action_magnitude_reward()

            # Motion-only locomotion terms. velocity/airtime/clearance are ~0
            # while still; foot_contact only pays on single-foot support.
            locomotion = (
                c['velocity']       * velocity_reward
                + c['foot_contact']  * foot_contact_reward
                + c['foot_airtime']  * foot_airtime_reward
                + c['foot_clearance'] * foot_clearance_reward
            )
            # Posture shaping, gated by forward motion so a motionless-but-tidy
            # robot earns ~0 from it (no alternate standing floor).
            posture = velocity_reward * (
                c['orientation'] * orientation_reward
                + c['heading']   * heading_reward
                + c['yoke_joint']    * yoke_joint_reward
                + c['yoke_symmetry'] * yoke_symmetry_reward
            )
            fall_term = -c['fall'] * (1.0 - stand_gate)
            # Ungated: penalise large/jerky commands regardless of posture, so
            # the signal survives even while falling.
            action_magnitude_term = -c['action_magnitude'] * action_magnitude_reward
            # The shaped reward is gated by posture quality (so falling throttles
            # it toward 0). The only thing payable while still is the decaying
            # survival_bonus; falling is penalised rather than rewarded.
            reward = (
                stand_gate * (locomotion + posture)
                + self.survival_bonus
                + fall_term
                + action_magnitude_term
            )

            # Each value here is the final, weighted/gated contribution to
            # `reward` (not the raw [0, 1] signal) — they sum exactly to the
            # total, so they can be plotted as a stacked breakdown.
            components.update({
                'velocity':       stand_gate * c['velocity']       * velocity_reward,
                'foot_contact':   stand_gate * c['foot_contact']   * foot_contact_reward,
                'foot_airtime':   stand_gate * c['foot_airtime']   * foot_airtime_reward,
                'foot_clearance': stand_gate * c['foot_clearance'] * foot_clearance_reward,
                'orientation':    stand_gate * velocity_reward * c['orientation'] * orientation_reward,
                'heading':        stand_gate * velocity_reward * c['heading']    * heading_reward,
                'yoke_joint':     stand_gate * velocity_reward * c['yoke_joint']    * yoke_joint_reward,
                'yoke_symmetry':  stand_gate * velocity_reward * c['yoke_symmetry'] * yoke_symmetry_reward,
                'survival_bonus': self.survival_bonus,
                'fall':           fall_term,
                'action_magnitude': action_magnitude_term,
            })
            return reward, components

        both_feet_reward    = self._get_both_feet_contact_reward()
        default_pose_reward = self._get_default_pose_reward()
        action_magnitude_reward = self._get_action_magnitude_reward()
        fall_term = -c['fall'] * (1.0 - stand_gate)
        action_magnitude_term = -c['action_magnitude'] * action_magnitude_reward
        # Contact and pose shaping are gated by posture quality so a fallen
        # robot whose feet still graze the floor earns nothing from them.
        # The action-magnitude penalty stays ungated, same reasoning as fall_term.
        reward = (
            c['stand'] * stand_gate
            + stand_gate * (
                c['both_feet']      * both_feet_reward
                + c['default_pose'] * default_pose_reward
            )
            + self.survival_bonus
            + fall_term
            + action_magnitude_term
        )
        components.update({
            'stand':          c['stand'] * stand_gate,
            'both_feet':      stand_gate * c['both_feet']    * both_feet_reward,
            'default_pose':   stand_gate * c['default_pose'] * default_pose_reward,
            'survival_bonus': self.survival_bonus,
            'fall':           fall_term,
            'action_magnitude': action_magnitude_term,
        })
        return reward, components

    def step(self, action):
        self._last_raw_action = np.asarray(action, dtype=np.float32)
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
            # Absolute position control: action in [-1, 1] scales to a target
            # offset (in radians) from the joint's default pose (qpos == 0,
            # see _set_joint_positions), then clips to the joint's range as a
            # safety bound rather than an amplitude the policy is meant to hit.
            filtered = self.action_filters[i].update(action[i])
            lo, hi = self.joint_ranges[i]
            target = self.action_scale * filtered
            self.data.ctrl[i] = float(np.clip(target, lo, hi))
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        noisey_state, state = self._get_obs()
        reward, reward_components = self._get_reward()
        if self._body_below_floor():
            self.done = True
        return (noisey_state, reward, self.done, self.done, {'state': state, 'reward_components': reward_components})

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        raise ValueError(f"Invalid render mode: {mode}")

    def close(self):
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
            self._renderer = None
