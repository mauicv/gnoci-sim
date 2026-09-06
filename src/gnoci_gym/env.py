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

# Offset of the [pitch, roll] pair within the policy obs, matching the
# concatenation order _get_policy_obs() builds: joint_pos, joint_vel,
# contacts, gyro(3), acc(3), then pitch/roll.
_PITCH_ROLL_OBS_OFFSET = 2 * _N_JOINTS + _N_TOUCH + 6

PHYSICS_DT    = 0.002   # MuJoCo integration timestep (500 Hz)
_CONTACT_DEBOUNCE_PERIOD = 0.05  # seconds — a contact change must persist this long to register
# Hysteresis thresholds (N) mimicking the limit switches' actuation/release
# forces: contact turns on above ON, off below OFF.  Standing loads are
# ~2.7 N on the lightest (front) sites, so ON stays well below stance forces.
_CONTACT_FORCE_ON  = 1.0
_CONTACT_FORCE_OFF = 0.5

# Denominator of the exp() falloff in _get_velocity_reward — smaller sigma
# means the reward drops off faster as forward speed misses target_velocity.
TRACKING_SIGMA = 0.01

# --- Raibert foot-placement heuristic (_get_raibert_reward) -----------------
# Feedback gain k in the Raibert fore-aft step-placement target
#   fwd_offset* = 0.5 * T_stance * v_forward + k * (v_forward - target_velocity)
# The 0.5 * T_stance * v term is the "neutral" foot placement (half the
# distance the body covers in one stance phase); k * velocity_error steps
# the foot further ahead when the body is under-speed (push harder) and
# shorter when it's over-speed (brake). Small robot at low speed — a light
# gain keeps the correction from swamping the neutral term. Tunable.
_RAIBERT_FEEDBACK_GAIN = 0.1
# exp() falloff on the squared fore-aft foot-placement error (m²): a ~5 cm
# miss costs one e-fold at 0.0025. Tunable.
_RAIBERT_TRACKING_SIGMA = 0.0025

IMU_GYRO_SCALE = ((180 / np.pi) / 250.0)
IMU_ACC_SCALE  = 9.81 # m/s² (2g) — clips to [-1, 1]

# Sysid-derived per-joint dynamics — must stay equal to the frictionloss /
# armature / damping attributes on the named joints in desc/gnoci.xml. They
# have to be threaded through here because _randomize_dynamics() stamps
# dof_frictionloss/dof_armature/dof_damping on every reset, so the MJCF
# values alone never survive past construction.

SYSID_JOINT_FRICTIONLOSS = 0.006426057652042094
SYSID_JOINT_ARMATURE     = 0.012995945315907826
SYSID_JOINT_DAMPING      = 0.17889452739994438

# Per-sensor observation noise scales, in raw sensor units (rad, rad/s, m/s²).
# Sampled uniform(-scale, scale) and converted into obs units with the same
# transforms _get_obs applies before being added to the obs.
OBS_NOISE_SCALES = dict(
    hip_pos=0.03,   # rad, for each hip joint
    knee_pos=0.05,  # rad, for each knee joint
    ankle_pos=0.08, # rad, for each ankle joint
    joint_vel=0.5,  # rad/s # Was 1.5
    gravity=0.01,    # rad, applied to the pitch/roll estimate
    linvel=0.1,     # unused: no linvel in the obs
    gyro=0.5,       # rad/s
    accelerometer=0.075,  # m/s²
)


def _clamp_nonneg(value):
    return max(0.0, float(value))


def _clamp_nonneg_int(value):
    return max(0, int(value))


def _as_range(value):
    lo, hi = value
    return (float(lo), float(hi))


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
        np.zeros(_N_JOINTS),                                    # commanded target is known exactly, stays clean
    ])

test_cfg = dict(
    initial_randomness=0.0,
    joint_zero_offset_range=0.0,
    joint_cmd_gain_range=(1.0, 1.0),
    inertial_mass_range=(0.0, 0.0),
    torso_com_offset_range=0.0,
    torso_mass_range=(1.0, 1.0),
    floor_tilt_range=0.0,
    floor_friction_range=(1.0, 1.0),
    joint_friction_range=(SYSID_JOINT_FRICTIONLOSS, SYSID_JOINT_FRICTIONLOSS),
    joint_armature_range=(SYSID_JOINT_ARMATURE, SYSID_JOINT_ARMATURE),
    joint_damping_range=(SYSID_JOINT_DAMPING, SYSID_JOINT_DAMPING),
    # actuator_gain_range already scales kp (gainprm[0]/biasprm[1]) by a
    # percentage of its size each reset; kv_range does the same for kv
    # (biasprm[2]), independently — see _randomize_dynamics().
    actuator_gain_range=(1.0, 1.0),
    kv_range=(1.0, 1.0),
    gravity_noise=0.0,
    obs_noise_level=0.0,
    imu_bias_range=0.0,
    push_force_max=0.0,
    max_action_delay=0,
    max_obs_delay=0,
)

dom_rnd_cfg = dict(
    initial_randomness=0.05,
    # per-joint servo zero-point (encoder calibration) error — a fixed
    # per-episode offset between the joint angle the policy commands/observes
    # and the physical qpos the servo actually holds. ~0.05 rad (~2.9deg) is
    # in line with hobby-servo horn-spline / assembly miscalibration.
    joint_zero_offset_range=0.05,
    # per-joint multiplicative command-gain (servo travel) tolerance — the
    # transmission-side sibling of joint_zero_offset_range: horn-spline seating,
    # gear backlash and PWM->angle calibration make actual travel a few percent
    # off the commanded target. Left observable (not compensated in
    # _get_joint_positions), so the policy corrects it via feedback.
    joint_cmd_gain_range=(0.9, 1.1),
    # widened from the old (0.02, 0.04) range to fold in what the removed
    # inertial_mass_noise (std 0.01) used to contribute
    inertial_mass_range=(0.01, 0.05),
    # torso (head_base) CoM position uncertainty (battery/wiring/mounting
    # placement) and mass uncertainty, wider than the generic per-body
    # inertial_mass_range since payload variation concentrates there
    torso_com_offset_range=0.012,  # metres, +/- per axis (~12mm)
    torso_mass_range=(0.85, 1.15),
    floor_tilt_range=0.02,
    floor_friction_range=(0.7, 1.3),
    # same relative spread the old ranges had around their (pre-sysid)
    # nominals: friction x0.7-1.5, armature x0.8-1.6; damping mirrors the
    # friction spread since it's likewise a passive resistive term
    joint_friction_range=(0.7 * SYSID_JOINT_FRICTIONLOSS, 1.5 * SYSID_JOINT_FRICTIONLOSS),
    joint_armature_range=(0.8 * SYSID_JOINT_ARMATURE, 1.6 * SYSID_JOINT_ARMATURE),
    joint_damping_range=(0.7 * SYSID_JOINT_DAMPING, 1.5 * SYSID_JOINT_DAMPING),
    actuator_gain_range=(0.9, 1.1),
    kv_range=(0.9, 1.1),
    gravity_noise=0.1,
    obs_noise_level=1.0,
    imu_bias_range=3.0,
    push_force_max=1.0,
    push_interval_range=(3.0, 6.0),
    max_action_delay=2,
    max_obs_delay=2,
)


class GnociGymEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    DEFAULT_REWARD_COEFS = {
        'stand':         1.0,
        'both_feet':     0.5,
        'default_pose':  0.5,
        'velocity':      2.5,
        'foot_swing':    1.0,
        'raibert':       1.0,
        'fall':          0.5,
        'orientation':   0.1,
        'rotation':      0.5,
        'strafe':        2.5,
        'yoke_joint':    0.0,
        'yoke_symmetry': 0.1,
        'action_magnitude': 0.05,
        'action_bounds': 1.0,
        'action_rate': 0.02,
    }

    # Curriculum knobs set_curriculum() can update mid-training, mapped to
    # the caster applied to each new value before setattr. Scalars are
    # clamped to >= 0 since they're all physical magnitudes (noise scales,
    # gains, counts, seconds); *_range tuples are just type-coerced, same as
    # __init__ takes them (no clamping there either). ``kp``/``kv`` are
    # handled separately in set_curriculum() since they need _apply_kp() /
    # _apply_kv(), not a plain setattr.
    _CURRICULUM_CASTERS = {
        'survival_bonus':        _clamp_nonneg,
        'target_velocity':       _clamp_nonneg,
        'foot_clearance_height': _clamp_nonneg,
        'swing_time':            _clamp_nonneg,
        'max_actuator_velocity': _clamp_nonneg,
        'initial_randomness':    _clamp_nonneg,
        'joint_zero_offset_range': _clamp_nonneg,
        'joint_cmd_gain_range':   _as_range,
        'inertial_mass_range':   _as_range,
        'torso_com_offset_range': _clamp_nonneg,
        'torso_mass_range':      _as_range,
        'floor_tilt_range':      _clamp_nonneg,
        'floor_friction_range':  _as_range,
        'joint_friction_range':  _as_range,
        'joint_armature_range':  _as_range,
        'joint_damping_range':   _as_range,
        'actuator_gain_range':   _as_range,
        'kv_range':              _as_range,
        'gravity_noise':         _clamp_nonneg,
        'obs_noise_level':       _clamp_nonneg,
        'imu_bias_range':        _clamp_nonneg,
        'push_force_max':        _clamp_nonneg,
        'push_interval_range':   _as_range,
        'max_action_delay':      _clamp_nonneg_int,
        'max_obs_delay':         _clamp_nonneg_int,
    }

    def __init__(
            self,
            camera='track',
            render_mode='rgb_array',
            control_hz=CONTROL_HZ,
            initial_randomness=0.1,
            joint_zero_offset_range=0.0,
            joint_cmd_gain_range=(1.0, 1.0),
            # widened from the old (0.04, 0.06) range to fold in what the
            # removed inertial_mass_noise (std 0.03) used to contribute
            inertial_mass_range=(0.0, 0.1),
            torso_com_offset_range=0.0,
            torso_mass_range=(1.0, 1.0),
            floor_tilt_range=0.0,
            floor_friction_range=(1.0, 1.0),
            joint_friction_range=(SYSID_JOINT_FRICTIONLOSS, SYSID_JOINT_FRICTIONLOSS),
            joint_armature_range=(SYSID_JOINT_ARMATURE, SYSID_JOINT_ARMATURE),
            joint_damping_range=(SYSID_JOINT_DAMPING, SYSID_JOINT_DAMPING),
            actuator_gain_range=(1.0, 1.0),
            kv_range=(1.0, 1.0),
            gravity_noise=0.0,
            obs_noise_level=0.0,
            obs_noise_scales=None,
            imu_bias_range=3.0,
            push_force_max=0.0,
            push_interval_range=(2.0, 5.0),
            max_action_delay=0,
            max_obs_delay=0,
            action_filter_alpha=0.4,
            action_scale=0.25,
            task='stand',
            reward_coefs=None,
            fix_root_body=False,
            survival_bonus=0.2,
            target_velocity=0.2,
            foot_clearance_height=0.04,
            swing_time=0.4,
            kp=25.0,
            kv=None,
            max_actuator_velocity=6.5,
        ):
        # Curriculum-controlled knobs. These are plain attributes so an external
        # trainer can ramp them between phases (e.g. SB3 env_method/set_attr) via
        # set_curriculum(). Defaults give a small standing floor + slow target
        # that the trainer is expected to anneal: survival_bonus -> 0 and
        # target_velocity upward.
        self.survival_bonus = float(survival_bonus)
        self.target_velocity = float(target_velocity)
        # Swing-foot height (metres) that earns full height-credit in
        # _get_foot_swing_reward — not a fixed constant so it can be
        # curriculum-annealed (e.g. start low, close to what the policy can
        # already reach, and ramp toward the real target height).
        self.foot_clearance_height = float(foot_clearance_height)
        # Target single-support swing duration (seconds) that earns full
        # time-credit in _get_foot_swing_reward — likewise curriculum-
        # annealable rather than fixed.
        self.swing_time = float(swing_time)
        # Actuator position gain (matches the XML's <position kp="..."/> default,
        # 50). Kept as an attribute so it can be re-baselined via set_curriculum();
        # _apply_kp() is (re-)applied in _build_model() once the model compiles.
        self.kp = float(kp)
        # Actuator velocity gain (the position servo's damping term, biasprm[2]).
        # None (default) means "keep whatever the XML's dampratio compiled to" —
        # unlike kp there's no separately-tuned override value, so _build_model()
        # reads the compiled value back into self.kv rather than stamping one.
        # Passing a value (here or via set_curriculum) re-baselines it via
        # _apply_kv(), same mechanism as kp/_apply_kp().
        self.kv = None if kv is None else float(kv)
        # Actuator slew-rate limit (rad/s), applied in step() to the commanded
        # target before it reaches the position servo. Default is the
        # miuzei_25kg no-load speed rating (~0.16s/60°  ~6.5 rad/s).
        self.max_actuator_velocity = float(max_actuator_velocity)
        self.camera = camera
        self.render_mode = render_mode
        self.task = task
        self.reward_coefs = {**self.DEFAULT_REWARD_COEFS, **(reward_coefs or {})}
        self.done = False
        self.control_hz = control_hz
        self.n_substeps = int(round(1.0 / (control_hz * PHYSICS_DT)))
        self.initial_randomness = initial_randomness
        # Symmetric bound (radians) on a fixed per-joint, per-episode servo
        # zero-point offset — the sim analogue of a joint's MJCF ``ref`` /
        # ``model.qpos0`` being miscalibrated on the real robot. MuJoCo's
        # position actuators and jointpos sensors both ignore qpos0, so this
        # is applied by hand: step() drives the joint to
        # ``gain * policy_target + offset`` while _get_joint_positions()
        # subtracts the offset back off, so the policy's control loop stays
        # self-consistent (an encoder-zero error is unobservable) but the
        # physical pose — and hence contact geometry / balance — is shifted.
        # Resampled each reset() (see self._joint_zero_offset).
        self.joint_zero_offset_range = float(joint_zero_offset_range)
        self._joint_zero_offset = np.zeros(_N_JOINTS, dtype=np.float32)
        # Per-joint multiplicative gain on the commanded target — servo travel
        # tolerance (transmission side), the sibling of the additive zero-point
        # offset above: physical qpos settles to ``gain * policy_target +
        # offset``. Unlike the offset this is left uncompensated in
        # _get_joint_positions(), so the policy sees the joint under/overshoot
        # and closes the loop against it. Resampled each reset() (see
        # self._joint_cmd_gain).
        self.joint_cmd_gain_range = _as_range(joint_cmd_gain_range)
        self._joint_cmd_gain = np.ones(_N_JOINTS, dtype=np.float32)
        self.inertial_mass_range = inertial_mass_range
        self.torso_com_offset_range = torso_com_offset_range
        self.torso_mass_range = torso_mass_range
        self.floor_tilt_range = floor_tilt_range
        self.floor_friction_range = floor_friction_range
        self.joint_friction_range = joint_friction_range
        self.joint_armature_range = joint_armature_range
        self.joint_damping_range = joint_damping_range
        self.actuator_gain_range = actuator_gain_range
        self.kv_range = kv_range
        self.gravity_noise = gravity_noise
        self.obs_noise_level = obs_noise_level
        self.obs_noise_scales = {**OBS_NOISE_SCALES, **(obs_noise_scales or {})}
        self._obs_noise_vec = _build_obs_noise_vec(self.obs_noise_scales)
        # Symmetric bound (degrees) on a fixed per-episode IMU pitch/roll
        # miscalibration bias, mimicking a real IMU's fixed mounting offset
        # rather than the per-step gravity noise above. Resampled once each
        # reset() (see self._imu_bias) and added on top of that noise in
        # _get_policy_obs() — only to the noisy policy obs, never to the
        # clean copy the critic sees.
        self.imu_bias_range = float(imu_bias_range)
        self._imu_bias = np.zeros(2, dtype=np.float32)
        self.push_force_max = push_force_max
        self.push_interval_range = push_interval_range
        self.max_action_delay = max_action_delay
        # Same idea as max_action_delay but for the policy's (noisy) obs
        # slice — see reset()/_get_obs(). The critic's clean_policy_obs is
        # never delayed, only what the actor sees.
        self.max_obs_delay = max_obs_delay
        self._obs_delay = 0
        self._obs_buffer = None
        self.action_filter_alpha = action_filter_alpha
        self.action_scale = action_scale
        self.fix_root_body = fix_root_body
        self.metadata['render_fps'] = control_hz

        self.policy_observation_space_size = _N_JOINTS + _N_JOINTS + _N_TOUCH + 6 + 2 + _N_JOINTS # joint pos, joint vel, touch, imu (accelerometer (3) and gyroscope (3)), pitch+roll (2), prev commanded target
        # critic sees a clean (noise-free) copy of the policy obs plus privileged
        # extras: gravity vector (3), base lin v (3), base angular v (3), base
        # height (1), foot lin v (2 feet x 3), foot air time (2). Total +18.
        self.critic_observation_space_size = self.policy_observation_space_size + 18

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(self.policy_observation_space_size + self.critic_observation_space_size,),
            dtype=np.float32
        )
        # Index vectors into the flat observation returned by step()/reset(),
        # so callers can do policy_obs = obs[env.policy_obs_idx] and
        # critic_obs = obs[env.critic_obs_idx] instead of hardcoding the
        # split point. See _get_obs for the concatenation order these mirror.
        self.policy_obs_idx = np.arange(self.policy_observation_space_size)
        self.critic_obs_idx = np.arange(
            self.policy_observation_space_size,
            self.policy_observation_space_size + self.critic_observation_space_size,
        )
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(_N_JOINTS,), dtype=np.float32
        )
        self._build_model()
        self._randomize_dynamics()
        self._randomize_servo_calibration()
        self._set_joint_positions()
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self._sync_action_filters()
        mujoco.mj_forward(self.model, self.data)

        self._push_step = 0
        self._push_interval = self._sample_push_interval()
        self._action_delay = 0
        self._action_buffer = deque([np.zeros(_N_JOINTS, dtype=np.float32)], maxlen=1)
        self._last_raw_action = np.zeros(_N_JOINTS, dtype=np.float32)
        self._prev_raw_action = np.zeros(_N_JOINTS, dtype=np.float32)
        # Rate-limiter state (see step()): starts at the actuators' initial
        # commanded position so the first step isn't slew-limited relative to
        # 0. Held in the policy's target space — invert step()'s
        # ``gain * target + offset`` mapping on the seeded ctrl — matching
        # what step() compares against.
        self._prev_target = (
            (self.data.ctrl - self._joint_zero_offset) / self._joint_cmd_gain
        ).astype(np.float32)

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
        # Leg roots (attachment to the torso) — the "hip" reference point the
        # Raibert foot-placement target is measured from (see _get_raibert_reward).
        self.left_hip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_yoke_lower_frame"
        )
        self.right_hip_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_yoke_lower_frame"
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
        self._joint_lo = np.array([r[0] for r in self.joint_ranges], dtype=np.float32)
        self._joint_hi = np.array([r[1] for r in self.joint_ranges], dtype=np.float32)
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
        self._base_body_ipos = self.model.body_ipos.copy()
        # Torso (self.body_id) is excluded here since it gets its own,
        # separately-ranged randomization below (torso_mass_range /
        # torso_com_offset_range) — including it in both would mean
        # inertial_mass_range's contribution to the torso is silently
        # overwritten and wasted.
        self._randomizable_body_ids = np.nonzero(self._base_body_mass > 0)[0]
        self._randomizable_body_ids = self._randomizable_body_ids[
            self._randomizable_body_ids != self.body_id
        ]
        self._base_actuator_gainprm = self.model.actuator_gainprm.copy()
        self._base_actuator_biasprm = self.model.actuator_biasprm.copy()
        self._apply_kp(self.kp)
        if self.kv is None:
            # No override given: read back whatever the XML's dampratio
            # compiled biasprm[2] to, so self.kv reflects the live value.
            self.kv = float(-self.model.actuator_biasprm[0, 2])
        else:
            self._apply_kv(self.kv)
        self._base_gravity_z = float(self.model.opt.gravity[2])

        self.comp_filter = ComplementaryFilter()
        self.action_filters = [EMAFilter(alpha=self.action_filter_alpha) for _ in range(_N_JOINTS)]
        self.acc_filters = [EMAFilter(alpha=ACC_FILTER_ALPHA, warm_start=True) for _ in range(3)]
        self.joint_vel_filters = [EMAFilter(alpha=JOINT_VEL_FILTER_ALPHA, warm_start=True) for _ in range(_N_JOINTS)]
        self._debounce_steps = max(1, int(_CONTACT_DEBOUNCE_PERIOD * self.control_hz))
        self.contact_debouncers = [Debouncer(self._debounce_steps) for _ in range(_N_TOUCH)]
        self._contact_states = np.zeros(_N_TOUCH, dtype=np.float32)
        self._foot_airtime = [0.0, 0.0]

        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model)

    def _apply_kp(self, kp):
        """(Re-)baseline the actuators' position gain that _randomize_dynamics()
        scales by actuator_gain_range each reset. biasprm[1] is the matching
        position term (-kp); see _apply_kv() for the velocity term
        (biasprm[2]/kv), baselined and randomized the same way."""
        self.kp = float(kp)
        self._base_actuator_gainprm[:, 0] = self.kp
        self._base_actuator_biasprm[:, 1] = -self.kp

    def _apply_kv(self, kv):
        """(Re-)baseline the actuators' velocity gain — biasprm[2], the
        position servo's damping term. Mirrors _apply_kp(): writes the base
        array that _randomize_dynamics() scales by kv_range each reset, so
        (like kp) this takes effect on the next reset(), not immediately."""
        self.kv = float(kv)
        self._base_actuator_biasprm[:, 2] = -self.kv

    def _randomize_dynamics(self):
        """Per-episode domain randomization, applied directly to the already-
        compiled model (no XML/recompile involved)."""
        for body_id in self._randomizable_body_ids:
            scale = 1.0 + np.random.uniform(*self.inertial_mass_range)
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
            self.model.dof_damping[dof_addr] = np.random.uniform(*self.joint_damping_range)

        for i in range(self.model.nu):
            # kp (gainprm[0]/biasprm[1]) and kv (biasprm[2]) each get their
            # own independent per-episode scale — actuator_gain_range for kp,
            # kv_range for kv — rather than sharing one draw, since gain and
            # damping uncertainty need not track each other.
            kp_scale = np.random.uniform(*self.actuator_gain_range)
            self.model.actuator_gainprm[i, 0] = self._base_actuator_gainprm[i, 0] * kp_scale
            self.model.actuator_biasprm[i, 1] = self._base_actuator_biasprm[i, 1] * kp_scale
            kv_scale = np.random.uniform(*self.kv_range)
            self.model.actuator_biasprm[i, 2] = self._base_actuator_biasprm[i, 2] * kv_scale

        if self.gravity_noise > 0:
            self.model.opt.gravity[2] = self._base_gravity_z + np.random.normal(0, self.gravity_noise)

        # Torso (head_base) CoM offset + mass — excluded from the generic
        # inertial_mass_range loop above and randomized with its own, wider
        # range since payload/wiring placement uncertainty concentrates in
        # the main body. body_inertia is rescaled by the same mass factor to
        # stay physically consistent (density-preserving approximation,
        # ignores the CoM shift's effect via the parallel axis theorem —
        # fine for domain randomization).
        torso_id = self.body_id
        self.model.body_ipos[torso_id] = self._base_body_ipos[torso_id] + np.random.uniform(
            -self.torso_com_offset_range, self.torso_com_offset_range, 3
        )
        torso_mass_scale = np.random.uniform(*self.torso_mass_range)
        self.model.body_mass[torso_id] = self._base_body_mass[torso_id] * torso_mass_scale
        self.model.body_inertia[torso_id] = self._base_body_inertia[torso_id] * torso_mass_scale

        # Recompute compile-time-derived constants (e.g. body/dof invweight)
        # that depend on the mass/inertia values just edited above. Far
        # cheaper than a full XML recompile.
        mujoco.mj_setConst(self.model, self.data)

    def _randomize_servo_calibration(self):
        """Resample the per-joint servo calibration for the episode: the
        additive zero-point offset (joint_zero_offset_range) and the
        multiplicative command gain (joint_cmd_gain_range). Both are one fixed
        draw per episode and combine in step() as
        ``ctrl = gain * policy_target + offset``; only the offset is undone in
        _get_joint_positions()."""
        r = self.joint_zero_offset_range
        if r > 0:
            self._joint_zero_offset = np.random.uniform(
                -r, r, _N_JOINTS
            ).astype(np.float32)
        else:
            self._joint_zero_offset = np.zeros(_N_JOINTS, dtype=np.float32)

        lo, hi = self.joint_cmd_gain_range
        if (lo, hi) != (1.0, 1.0):
            self._joint_cmd_gain = np.random.uniform(
                lo, hi, _N_JOINTS
            ).astype(np.float32)
        else:
            self._joint_cmd_gain = np.ones(_N_JOINTS, dtype=np.float32)

    def _set_joint_positions(self):
        # Seed each actuated joint at its zero-point offset: this is the
        # physical pose the servo settles to for a zero command, so the
        # episode starts already consistent with the miscalibration rather
        # than snapping to it over the first few steps.
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[qpos_addr] = self._joint_zero_offset[i]
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

    def set_curriculum(self, *, kp=None, kv=None, **kwargs):
        """Update curriculum / domain-randomization knobs mid-training.

        Intended to be driven by the trainer (e.g. via SB3 ``env_method``) to
        anneal the standing floor and exploration noise away while ramping
        the forward-speed target and dom-rand ranges up. Accepts any of the
        keys in ``_CURRICULUM_CASTERS`` (the same names as the matching
        ``__init__`` params) plus ``kp``/``kv``; unset or ``None`` values are
        left unchanged, and an unknown keyword raises ``TypeError``.

        ``kp``/``kv`` and the dom-rand ranges (``inertial_mass_range``,
        ``floor_friction_range``, ``joint_damping_range``, ``kv_range``,
        etc.) re-baseline knobs that _randomize_dynamics() only reads at
        reset() time, so they take effect on the next reset(), not
        mid-episode.

        ``max_actuator_velocity`` (see step()) is read directly every step,
        so unlike those it takes effect immediately, mid-episode included.
        """
        unknown = kwargs.keys() - self._CURRICULUM_CASTERS.keys()
        if unknown:
            raise TypeError(
                f"set_curriculum() got unexpected keyword argument(s): {sorted(unknown)}"
            )
        for name, value in kwargs.items():
            if value is not None:
                setattr(self, name, self._CURRICULUM_CASTERS[name](value))
        if kp is not None:
            self._apply_kp(_clamp_nonneg(kp))
        if kv is not None:
            self._apply_kv(_clamp_nonneg(kv))

    def _sample_push_interval(self):
        lo = int(self.push_interval_range[0] * self.control_hz)
        hi = int(self.push_interval_range[1] * self.control_hz)
        return np.random.randint(lo, hi + 1)

    def reset(self, seed=None, **kwargs):
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_dynamics()
        self._randomize_servo_calibration()
        self._set_joint_positions()
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self._sync_action_filters()
        mujoco.mj_forward(self.model, self.data)
        self.comp_filter.reset()
        for f in self.acc_filters + self.joint_vel_filters:
            f.reset()
        # Fresh per-episode [pitch, roll] miscalibration bias (see
        # imu_bias_range), converted from degrees into the same units
        # ComplementaryFilter.pitch/roll are expressed in (degrees / 180).
        self._imu_bias = (
            np.random.uniform(-self.imu_bias_range, self.imu_bias_range, size=2) / 180.0
        ).astype(np.float32)
        self.done = False

        self._push_step = 0
        self._push_interval = self._sample_push_interval()

        for d in self.contact_debouncers:
            d.reset()
        self._foot_airtime = [0.0, 0.0]
        self._last_raw_action = np.zeros(_N_JOINTS, dtype=np.float32)
        self._prev_raw_action = np.zeros(_N_JOINTS, dtype=np.float32)
        # policy target space, matching step() (see the __init__ seeding).
        self._prev_target = (
            (self.data.ctrl - self._joint_zero_offset) / self._joint_cmd_gain
        ).astype(np.float32)

        if self.max_action_delay > 0:
            self._action_delay = np.random.randint(0, self.max_action_delay + 1)
            self._action_buffer = deque(
                [np.zeros(_N_JOINTS, dtype=np.float32)] * (self._action_delay + 1),
                maxlen=self._action_delay + 1,
            )

        # Fresh per-episode observation-delay length. Unlike the action
        # buffer above, this isn't zero-filled here: there's no "no reading
        # yet" placeholder that's physically meaningful for sensor data, so
        # _get_obs() lazily seeds _obs_buffer with the real first reading
        # (warm start) the first time it runs after this reset.
        self._obs_delay = np.random.randint(0, self.max_obs_delay + 1) if self.max_obs_delay > 0 else 0
        self._obs_buffer = None

        noisey_state, state = self._get_obs()
        return noisey_state, {'state': state}

    def _get_joint_positions(self):
        # Subtract only the per-episode servo zero-point offset (an encoder-zero
        # error is unobservable): the policy and the joint-based reward terms
        # then see ``t`` back when the joint has settled at its commanded
        # target, and the offset shows up only in the physics (contact
        # geometry, balance). The command-gain error (_joint_cmd_gain) is
        # deliberately *not* undone here — it's a transmission-side error the
        # policy is meant to observe as under/overshoot and correct.
        return self.data.sensordata[self.joint_pos_sensor_addrs] - self._joint_zero_offset

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

    def _get_policy_obs(self):
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
            *self._prev_target,
        ])
        if self.obs_noise_level > 0:
            noise = np.random.uniform(-1, 1, obs.shape) * self._obs_noise_vec
            noisey_obs = obs + self.obs_noise_level * noise
        else:
            noisey_obs = obs.copy()
        # Per-episode IMU bias (see reset()) — added after the per-step noise,
        # only to the pitch/roll pair of the noisy obs the policy sees.
        noisey_obs[_PITCH_ROLL_OBS_OFFSET:_PITCH_ROLL_OBS_OFFSET + 2] += self._imu_bias
        return noisey_obs.astype(np.float32), obs.astype(np.float32)

    def _get_critic_only_obs(self):
        """Privileged features only the critic sees (asymmetric actor-critic):
        no real-hardware equivalent, so never exposed to the policy slice of
        the observation. All noise-free, and expressed in the robot's own
        body frame (rather than world frame) for the same reason forward
        velocity/rotation/strafe rewards were switched to body-relative —
        orientation-invariant and not tied to a fixed global heading."""
        xmat = self.data.xmat[self.body_id].reshape(3, 3)

        gravity_body = xmat.T @ np.array([0.0, 0.0, -1.0])
        base_lin_v_body = xmat.T @ self._get_velocity()

        gyro_addr = self.imu_sensor_addrs[0]
        base_ang_v_body = self.data.sensordata[gyro_addr:gyro_addr + 3]  # gyro is already body-frame

        base_height = self._get_root_height()

        left_foot_v_body = xmat.T @ self.data.cvel[self.left_foot_body_id][3:6]
        right_foot_v_body = xmat.T @ self.data.cvel[self.right_foot_body_id][3:6]

        foot_airtime = np.array(self._foot_airtime, dtype=np.float32)

        return np.concatenate([
            gravity_body,
            base_lin_v_body,
            base_ang_v_body,
            [base_height],
            left_foot_v_body,
            right_foot_v_body,
            foot_airtime,
        ]).astype(np.float32)

    def _get_obs(self):
        noise_policy_obs, clean_policy_obs = self._get_policy_obs()
        if self.max_obs_delay > 0:
            # Warm start: on the first call after reset() (see _obs_delay),
            # seed the buffer with the real current reading rather than
            # zeros, so a delayed episode doesn't start from a fake all-zero
            # observation.
            if self._obs_buffer is None:
                self._obs_buffer = deque(
                    [noise_policy_obs.copy()] * (self._obs_delay + 1),
                    maxlen=self._obs_delay + 1,
                )
            else:
                self._obs_buffer.append(noise_policy_obs.copy())
            noise_policy_obs = self._obs_buffer[0]
        critic_only_obs = self._get_critic_only_obs()

        noisey_state = np.concatenate([noise_policy_obs, clean_policy_obs, critic_only_obs], axis=0)
        state = np.concatenate([clean_policy_obs, clean_policy_obs, critic_only_obs], axis=0)
        return noisey_state.astype(np.float32), state.astype(np.float32)

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

    def _get_yaw_rate(self):
        # Raw gyro sensordata in rad/s, permuted into the hardware axis
        # convention (see permute_imu_data). Index 2 is the axis the
        # complementary filter deliberately ignores for pitch/roll (see
        # ComplementaryFilter.update's `x_gyro, y_gyro, _ = gyro_data`) —
        # i.e. yaw rate, which is directly measurable on real hardware unlike
        # world-frame lateral drift.
        gyro = self.data.sensordata[self.imu_sensor_addrs[0]:self.imu_sensor_addrs[0] + 3]
        return float(self.permute_imu_data(gyro)[2])

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
        # Forward speed along the robot's own body-forward axis, not a fixed
        # global direction — so turning to walk a curve/heading still earns
        # forward credit, rather than only ever crediting motion toward -Y.
        forward = float(np.dot(velocity[:2], self._get_body_forward_xy()))
        lin_vel_error = (self.target_velocity - forward) ** 2
        return float(np.nan_to_num(np.exp(-lin_vel_error / TRACKING_SIGMA)))

    def _get_rotation_penalty_reward(self):
        # Squared yaw rate (rad/s) — same gyro axis the complementary filter
        # ignores for pitch/roll, so it's measurable on real hardware unlike
        # world-frame lateral drift (see _get_yaw_rate). A separate ungated
        # term rather than a gate on forward velocity, so spinning is always
        # penalised on its own terms instead of just zeroing out forward credit.
        # Normalised by target_velocity, same reference _get_velocity_reward
        # uses to normalise forward_reward to [0, 1] — raw rad/s is tiny next
        # to a bounded [0, 1] reward, so without this the coefficients aren't
        # actually comparable regardless of what they're set to.
        yaw_rate = self._get_yaw_rate()
        if self.target_velocity <= 0.0:
            return float(np.square(yaw_rate))
        return float(np.square(yaw_rate / self.target_velocity))

    def _get_strafe_penalty_reward(self):
        # Sideways speed relative to the robot's own forward axis (not world
        # X) — so it stays meaningful under turns, same reasoning as
        # _get_velocity_reward's forward term. Still privileged sim state
        # (world-frame CoM velocity), same as forward speed itself; unlike
        # rotation there's no gyro-measurable substitute for this one, it's a
        # training-time-only shaping term.
        # Normalised by target_velocity for the same reason as rotation above:
        # expresses drift as a fraction of target speed, comparable to
        # forward_reward's [0, 1] scale instead of raw (tiny) m/s.
        velocity = self._get_velocity()
        forward_xy = self._get_body_forward_xy()
        right_xy = np.array([forward_xy[1], -forward_xy[0]])
        strafe = float(abs(np.dot(velocity[:2], right_xy)))
        if self.target_velocity <= 0.0:
            return strafe
        return strafe / self.target_velocity

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

    def _get_body_forward_xy(self):
        """Unit body-forward direction (-Y body axis) projected into the world
        XY plane. Zero vector when the body is on its side (axis has no XY
        component)."""
        xmat = self.data.xmat[self.body_id]
        body_forward = np.array([-xmat[3], -xmat[4]])
        norm = np.linalg.norm(body_forward)
        if norm < 1e-6:
            return np.zeros(2)
        return body_forward / norm

    def _get_foot_swing_reward(self):
        """Dense single-support swing-quality reward.

        For whichever foot is airborne while the other is planted, credit is
        the product of two fractions: how far into the target swing duration
        (self.swing_time) the current continuous airtime is (ramping linearly
        from 0, then dropping to 0 once airtime exceeds swing_time — no
        credit for dawdling past the target duration), and how close to the
        target swing height (self.foot_clearance_height) the foot currently
        is (ramping linearly from 0 and holding at 1 past its cap).
        Multiplying the two means neither a long-but-flat lift nor a
        high-but-momentary tap earns much on its own — both a real duration
        and a real height are required together. Both caps are plain
        attributes (rather than module constants) so they can be
        curriculum-annealed via set_curriculum().

        0 during double support, while both feet are airborne at once (no
        single-support foot to credit, e.g. a hop/stumble), and once a swing
        has overrun swing_time.
        """
        dt = 1.0 / self.control_hz
        contacts = self._contact_states
        in_contact = [
            contacts[0] > 0 or contacts[1] > 0,  # left foot
            contacts[2] > 0 or contacts[3] > 0,  # right foot
        ]
        foot_z = [
            self.data.xpos[self.left_foot_body_id][2] - self.floor_z,
            self.data.xpos[self.right_foot_body_id][2] - self.floor_z,
        ]

        reward = 0.0
        for i, other in ((0, 1), (1, 0)):
            if in_contact[i]:
                self._foot_airtime[i] = 0.0
                continue
            self._foot_airtime[i] += dt
            if not in_contact[other] or self.swing_time <= 0.0 or self.foot_clearance_height <= 0.0:
                continue
            time_frac = self._foot_airtime[i] / self.swing_time
            if time_frac > 1.0:
                continue
            height_frac = min(1.0, max(0.0, foot_z[i]) / self.foot_clearance_height)
            reward += time_frac * height_frac
        return reward

    def _get_raibert_reward(self):
        """Raibert foot-placement heuristic — rewards the swing foot for
        heading toward where a Raibert-style step planner would put it.

        Raibert's rule is a sagittal-plane (fore-aft) speed-regulation law:
        for a periodic gait the neutral touchdown point of a foot, measured
        forward from its hip, is half the distance the body travels in one
        stance phase, 0.5 * T_stance * v_forward, plus a velocity-error
        feedback term k * (v_forward - v_desired) that steps the foot further
        ahead when the body is under-speed (to push harder) and shorter when
        it's over-speed (to brake):

            fwd_offset* = 0.5 * T_stance * v_forward + k * (v_forward - target_velocity)

        Credit is exp(-error² / sigma) on the fore-aft distance between the
        swing foot's actual hip-relative forward offset and this target, with
        v_forward and the offset both taken along the body-forward axis. Only
        the airborne foot during single support is scored (the planted foot's
        placement is already committed); 0 during double support or a
        two-foot flight. T_stance is approximated by self.swing_time — the
        single-support swing of one foot roughly equals the stance of the
        other for a symmetric gait — so it follows the same curriculum knob
        as _get_foot_swing_reward. Lateral placement is left to the strafe
        penalty and leg kinematics; only the fore-aft axis is scored here.

        This is soft shaping: rewarding the swing foot to sit at its
        touchdown target for the whole swing (not just at the end) is a
        deliberate simplification, kept mild by its coefficient.
        """
        if self.swing_time <= 0.0:
            return 0.0
        forward_xy = self._get_body_forward_xy()
        if not np.any(forward_xy):
            return 0.0

        contacts = self._contact_states
        in_contact = [
            contacts[0] > 0 or contacts[1] > 0,  # left foot
            contacts[2] > 0 or contacts[3] > 0,  # right foot
        ]

        v_fwd = float(np.dot(self._get_velocity()[:2], forward_xy))
        target_fwd = (
            0.5 * self.swing_time * v_fwd
            + _RAIBERT_FEEDBACK_GAIN * (v_fwd - self.target_velocity)
        )

        feet = (
            (self.left_foot_body_id, self.left_hip_body_id),
            (self.right_foot_body_id, self.right_hip_body_id),
        )
        reward = 0.0
        for i, other in ((0, 1), (1, 0)):
            if in_contact[i] or not in_contact[other]:
                continue  # score only the swing foot in single support
            foot_id, hip_id = feet[i]
            d_xy = self.data.xpos[foot_id][:2] - self.data.xpos[hip_id][:2]
            off_fwd = float(np.dot(d_xy, forward_xy))
            err = (off_fwd - target_fwd) ** 2
            reward += math.exp(-err / _RAIBERT_TRACKING_SIGMA)
        return reward

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

    def _get_action_bounds_reward(self):
        # Barrier on the pre-clip action mu, scaled to the same units as the
        # actuation target (see step()): 0 while action_scale * mu stays
        # inside the joint's physical [lo, hi] range, growing with the square
        # of the overshoot once it doesn't. Distinct from action_magnitude
        # (which penalises size regardless of bounds) — this specifically
        # discourages commands that would saturate the per-joint safety clip.
        target = self.action_scale * self._last_raw_action
        excess = np.maximum(0.0, target - self._joint_hi) + np.maximum(0.0, self._joint_lo - target)
        return float(np.mean(np.square(excess)))

    def _get_action_rate_reward(self):
        # Penalises jerky commands: squared step-to-step change in the raw
        # (pre-clip, pre-filter) action, independent of the EMA filter that
        # smooths what's actually sent to the actuators.
        return float(np.mean(np.square(self._last_raw_action - self._prev_raw_action)))

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
            foot_swing_reward    = self._get_foot_swing_reward()
            raibert_reward       = self._get_raibert_reward()
            orientation_reward   = self._get_orientation_reward()
            yoke_joint_reward    = self._get_yoke_joint_reward()
            yoke_symmetry_reward = self._get_yoke_symmetry_reward()
            action_magnitude_reward = self._get_action_magnitude_reward()
            action_bounds_reward = self._get_action_bounds_reward()
            action_rate_reward = self._get_action_rate_reward()
            rotation_penalty_reward = self._get_rotation_penalty_reward()
            strafe_penalty_reward = self._get_strafe_penalty_reward()

            # Motion-only locomotion terms: both are ~0 while still, and
            # foot_swing only pays during a genuine single-support swing that
            # gets real duration and real height (see _get_foot_swing_reward).
            locomotion = (
                c['velocity']    * velocity_reward
                + c['foot_swing'] * foot_swing_reward
                + c['raibert']    * raibert_reward
            )
            # Posture shaping, gated by forward motion so a motionless-but-tidy
            # robot earns ~0 from it (no alternate standing floor).
            posture = velocity_reward * (
                c['orientation'] * orientation_reward
                + c['yoke_joint']    * yoke_joint_reward
                + c['yoke_symmetry'] * yoke_symmetry_reward
            )
            fall_term = -c['fall'] * (1.0 - stand_gate)
            # Ungated: penalise large/jerky/out-of-bounds commands regardless
            # of posture, so the signal survives even while falling.
            action_magnitude_term = -c['action_magnitude'] * action_magnitude_reward
            action_bounds_term = -c['action_bounds'] * action_bounds_reward
            action_rate_term = -c['action_rate'] * action_rate_reward
            rotation_term = -c['rotation'] * rotation_penalty_reward
            strafe_term = -c['strafe'] * strafe_penalty_reward
            # The shaped reward is gated by posture quality (so falling throttles
            # it toward 0). The only thing payable while still is the decaying
            # survival_bonus; falling is penalised rather than rewarded.
            reward = (
                stand_gate + (locomotion + posture)
                + self.survival_bonus
                + fall_term
                + action_magnitude_term
                + action_bounds_term
                + action_rate_term
                + rotation_term
                + strafe_term
            )

            # Each value here is the final, weighted/gated contribution to
            # `reward` (not the raw [0, 1] signal) — they sum exactly to the
            # total, so they can be plotted as a stacked breakdown.
            components.update({
                'velocity':       stand_gate * c['velocity']       * velocity_reward,
                'foot_swing':     stand_gate * c['foot_swing']     * foot_swing_reward,
                'raibert':        stand_gate * c['raibert']        * raibert_reward,
                'orientation':    stand_gate * velocity_reward * c['orientation'] * orientation_reward,
                'yoke_joint':     stand_gate * velocity_reward * c['yoke_joint']    * yoke_joint_reward,
                'yoke_symmetry':  stand_gate * velocity_reward * c['yoke_symmetry'] * yoke_symmetry_reward,
                'survival_bonus': self.survival_bonus,
                'fall':           fall_term,
                'action_magnitude': action_magnitude_term,
                'action_bounds':  action_bounds_term,
                'action_rate':    action_rate_term,
                'rotation':       rotation_term,
                'strafe':         strafe_term,
            })
            return reward, components

        both_feet_reward    = self._get_both_feet_contact_reward()
        default_pose_reward = self._get_default_pose_reward()
        action_magnitude_reward = self._get_action_magnitude_reward()
        action_bounds_reward = self._get_action_bounds_reward()
        action_rate_reward = self._get_action_rate_reward()
        fall_term = -c['fall'] * (1.0 - stand_gate)
        action_magnitude_term = -c['action_magnitude'] * action_magnitude_reward
        action_bounds_term = -c['action_bounds'] * action_bounds_reward
        action_rate_term = -c['action_rate'] * action_rate_reward
        # Contact and pose shaping are gated by posture quality so a fallen
        # robot whose feet still graze the floor earns nothing from them.
        # The action penalties stay ungated, same reasoning as fall_term.
        reward = (
            c['stand'] * stand_gate
            + stand_gate * (
                c['both_feet']      * both_feet_reward
                + c['default_pose'] * default_pose_reward
            )
            + self.survival_bonus
            + fall_term
            + action_magnitude_term
            + action_bounds_term
            + action_rate_term
        )
        components.update({
            'stand':          c['stand'] * stand_gate,
            'both_feet':      stand_gate * c['both_feet']    * both_feet_reward,
            'default_pose':   stand_gate * c['default_pose'] * default_pose_reward,
            'survival_bonus': self.survival_bonus,
            'fall':           fall_term,
            'action_magnitude': action_magnitude_term,
            'action_bounds':  action_bounds_term,
            'action_rate':    action_rate_term,
        })
        return reward, components

    def step(self, action):
        self._prev_raw_action = self._last_raw_action
        self._last_raw_action = np.asarray(action, dtype=np.float32)
        action = self._last_raw_action

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

        target_actions = []
        max_step = self.max_actuator_velocity / self.control_hz
        for i in range(self.model.nu):
            # Absolute position control: action in [-1, 1] scales to a target
            # offset (in radians) from the joint's default pose (qpos == 0,
            # see _set_joint_positions). The raw target is then slew-rate
            # limited to the actuator's max velocity (mirrors the physical
            # servo, which can't jump instantly to a new position) before
            # clipping to the joint's range as a safety bound rather than an
            # amplitude the policy is meant to hit.
            filtered = self.action_filters[i].update(action[i])
            lo, hi = self.joint_ranges[i]
            raw_target = self.action_scale * filtered
            new_target = np.clip(
                raw_target,
                self._prev_target[i] - max_step,
                self._prev_target[i] + max_step,
            )
            self._prev_target[i] = new_target
            # The policy commands, and _prev_target/slew-limiting track, a
            # target in the policy's own space; the physical servo drives the
            # joint to ``gain * target + offset`` — its per-episode command-gain
            # and zero-point miscalibration (see _randomize_servo_calibration).
            # Range-clip in physical space since it's a bound on the real joint.
            self.data.ctrl[i] = float(np.clip(
                self._joint_cmd_gain[i] * new_target + self._joint_zero_offset[i], lo, hi
            ))
            target_actions.append(float(new_target))
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        noisey_state, state = self._get_obs()
        reward, reward_components = self._get_reward()
        if self._body_below_floor():
            self.done = True
        return (noisey_state, reward, self.done, self.done, {'state': state, 'reward_components': reward_components, 'target_action': target_actions})

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        raise ValueError(f"Invalid render mode: {mode}")

    def close(self):
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
            self._renderer = None
