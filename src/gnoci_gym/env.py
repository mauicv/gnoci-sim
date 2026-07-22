import mujoco
import gymnasium as gym
import numpy as np
from collections import deque
from .utils import tolerance
from .load_xml import _load_and_perturb_basic_xml
from .filters import ComplementaryFilter
from .servo import Servo
from .config import CONTROL_HZ, FREQ

_STANDING_HEIGHT = 0.225
_MIN_STANDING_HEIGHT = 0.175

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

IMU_GYRO_SCALE = ((180 / np.pi) / 250.0)
IMU_ACC_SCALE  = 9.81 # m/s² (2g) — clips to [-1, 1]

# Per-dimension observation normalisation divisors. Each value is the ~99th
# percentile of |obs| measured from random-action rollouts, chosen so every
# channel lands at roughly unit scale before entering the encoder. Without this
# raw joint velocities (~1.3 std) dominate joint positions / pitch-roll (~0.1
# std) in both the encoder gradients and the world-model reconstruction loss.
# These MUST be mirrored in the real-robot obs pipeline (gnoci-control) and are
# applied identically in ReferenceEnv so AMP compares like-for-like.
_OBS_NORM = np.array(
    [0.52] * _N_JOINTS      # joint positions  (already /pi, offset-removed)
    + [11.0] * _N_JOINTS    # joint velocities (rad/s)
    + [1.0] * _N_TOUCH      # binary foot contacts
    + [2.5] * 3             # gyro  (already * IMU_GYRO_SCALE)
    + [3.9] * 3             # accel (already / IMU_ACC_SCALE)
    + [0.4] * 2,            # pitch, roll (rad)
    dtype=np.float32,
)

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
        'foot_contact': 0.75,
        'foot_airtime': 0.5,
        'orientation':  0.1,
        'heading':      0.3,
        'yoke_joint':   0.1,
        'action_mag':   0.005,   # penalty: ||a_t||²  (commanded target velocity)
        'action_rate':  0.01,    # penalty: ||a_t - a_{t-1}||²  (chatter)
        'joint_pos':    0.5,   # reward: tolerance() bonus for joints near zero pose
        'alive':        0.5,     # bonus: added every surviving step
        'termination':  5.0,     # penalty: subtracted once when the robot falls
    }

    def __init__(
            self,
            camera='track',
            render_mode='rgb_array',
            control_hz=CONTROL_HZ,
            initial_randomness=0.0,
            inertial_mass_range=(0.00, 0.00),
            inertial_mass_noise=0.00,
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
            fix_root_body=False,
        ):
        self.camera = camera
        self.render_mode = render_mode
        self.task = task
        self.reward_coefs = {**self.DEFAULT_REWARD_COEFS, **(reward_coefs or {})}
        self.done = False
        self.control_hz = control_hz
        self.n_substeps = int(round(1.0 / (control_hz * PHYSICS_DT)))
        # Advance the servo controllers at the hardware rate (FREQ, e.g. 100Hz)
        # while physics steps at 1/PHYSICS_DT (e.g. 500Hz): one servo update every
        # `servo_update_every` substeps, matching the real 100Hz servo loop.
        self.servo_update_every = max(1, int(round(1.0 / (FREQ * PHYSICS_DT))))
        # Simulated time between servo updates — passed to the PID so its dt is
        # sim time (not wall-clock), keeping it correct for nonzero Ki/Kd.
        self.servo_dt = self.servo_update_every * PHYSICS_DT
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
        self.initialize_model()
        self._set_joint_positions()
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self._sync_action_filters()
        mujoco.mj_forward(self.model, self.data)

        self._push_step = 0
        self._push_interval = self._sample_push_interval()
        self._action_delay = 0
        self._action_buffer = deque([np.zeros(_N_JOINTS, dtype=np.float32)], maxlen=1)
        self._prev_action = np.zeros(_N_JOINTS, dtype=np.float32)
        self._last_action = np.zeros(_N_JOINTS, dtype=np.float32)

    def initialize_model(self):
        xml_content = _load_and_perturb_basic_xml(
            'scene',
            inertial_mass_range=self.inertial_mass_range,
            inertial_mass_noise=self.inertial_mass_noise,
            floor_tilt_range=self.floor_tilt_range,
            floor_friction_range=self.floor_friction_range,
            fix_root_body=self.fix_root_body,
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
        try:
            self.floor_geom_id = self.model.geom("floor").id
            self.floor_z = self.model.geom_pos[self.floor_geom_id][2]
        except KeyError:
            self.floor_geom_id = -1
            self.floor_z = 0.0
        _ground_term_bodies = ["head_base", "left_hip_back", "right_hip_back"]
        self._ground_termination_geoms = set()
        for bname in _ground_term_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            gs = self.model.body_geomadr[bid]
            gc = self.model.body_geomnum[bid]
            self._ground_termination_geoms.update(range(gs, gs + gc))
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
        self._build_servos()
        grace_steps = max(1, int(_CONTACT_GRACE_PERIOD * self.control_hz))
        self._contact_buffer = deque([False] * grace_steps, maxlen=grace_steps)
        self._foot_airtime = [0.0, 0.0]
        self._foot_was_contact = [False, False]

        # Add at the end:
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model)

    def _set_joint_positions(self):
        for jnt_name in _JOINT_NAMES:
            qpos_addr = self.model.jnt_qposadr[self.model.joint(jnt_name).id]
            self.data.qpos[qpos_addr] = 0.0
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.ctrl[i] = self.data.qpos[qpos_addr]
            # The general actuators use dyntype="filter": each has an internal
            # activation (data.act) that low-passes ctrl and drives the force.
            # mj_resetData zeros it, so seed it to the target too — else the
            # servo lags (tau=1s) and the joint sags toward zero for ~a second.
            adr = self.model.actuator_actadr[i]
            if adr >= 0:
                self.data.act[adr] = self.data.ctrl[i]

    def _build_servos(self):
        # One controller per joint, mirroring the real hardware servo model.
        # The servo works in normalised units (value in [-1, 1] -> PWM range),
        # which maps to MuJoCo joint radians via `* np.pi`. Joint limits are
        # converted from radians (jnt_range) into that normalised space.
        self.servos = []
        for i, name in enumerate(_JOINT_NAMES):
            lo, hi = self.joint_ranges[i]
            servo = Servo(
                name=name,
                pin_limits=(lo / np.pi, hi / np.pi),
                init_value=0.0,
                action_filter_alpha=self.action_filter_alpha,
            )
            # We advance the PID once per `servo_update_every` substeps, so the
            # simple_pid wall-clock rate gate must be disabled — sim time is not
            # real time. With Ki=Kd=0 the controller is a pure slew-rate limiter.
            servo.pid_controller.sample_time = None
            self.servos.append(servo)
        self._sync_action_filters()

    def _sync_action_filters(self):
        # Align each servo's internal state to the current joint position so the
        # command starts from where the robot actually is (no step-0 jump).
        for i, servo in enumerate(self.servos):
            val = float(self.data.qpos[self.joint_qpos_addrs[i]] / np.pi)
            servo._value = val
            servo.pid_controller.reset()
            servo.pid_controller.sample_time = None
            servo.pid_controller.setpoint = val
            servo.low_pass_filter.reset()
            self.data.ctrl[i] = servo.value * np.pi

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
        self._set_joint_positions()
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

        self._prev_action = np.zeros(_N_JOINTS, dtype=np.float32)
        self._last_action = np.zeros(_N_JOINTS, dtype=np.float32)

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
        joint_pos = self._get_joint_positions() / np.pi
        joint_vel = self._get_joint_velocities()
        obs = np.array([
            *joint_pos,
            *joint_vel,
            *self._get_contact_forces(),
            *gyro,
            *acc,
            *self._get_pitch_and_roll(gyro * 250, acc),
        ])
        obs = obs / _OBS_NORM
        if self.obs_noise_scale > 0:
            noisey_obs = obs + np.random.normal(0, self.obs_noise_scale, obs.shape)
        else:
            noisey_obs = obs
        return noisey_obs.astype(np.float32), obs.astype(np.float32)

    def _get_info(self):
        return {}

    def overturned(self):
        return self._get_root_upright() < 0.3

    def _root_body_on_ground(self):
        if self.floor_geom_id == -1:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == self.floor_geom_id and c.geom2 in self._ground_termination_geoms) or \
               (c.geom2 == self.floor_geom_id and c.geom1 in self._ground_termination_geoms):
                return True
        return False

    def _too_low(self):
        return self._get_root_height() < _MIN_STANDING_HEIGHT

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
                reward += self._foot_airtime[i] - 0.075
                self._foot_airtime[i] = 0.0
        self._foot_was_contact = current
        return reward

    def _get_stand_reward(self):
        upright = self._get_root_upright()
        height  = self._get_root_height()
        standing = tolerance(
            height,
            bounds=(_STANDING_HEIGHT, float('inf')),
            margin=_STANDING_HEIGHT / 4,
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

    def _get_joint_pos_reward(self):
        # Encourage all joints to stay near their default position of 0.0.
        positions = self._get_joint_positions()
        return float(np.mean([
            tolerance(float(p), bounds=(0.0, 0.0), margin=0.2)
            for p in positions
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

    def _get_smoothness_components(self):
        c = self.reward_coefs
        action_mag  = float(np.sum(np.square(self._last_action)))
        action_rate = float(np.sum(np.square(self._last_action - self._prev_action)))
        # Penalties, returned as negative contributions so the components sum
        # to the total reward.
        return {
            'action_mag':  -c['action_mag']  * action_mag,
            'action_rate': -c['action_rate'] * action_rate,
        }

    def _get_reward(self):
        """Return (total_reward, components) where the component values sum to
        total_reward."""
        c = self.reward_coefs
        stand_reward = self._get_stand_reward()
        components = self._get_smoothness_components()
        components['joint_pos'] = c['joint_pos'] * self._get_joint_pos_reward()

        if self.task == 'walk':
            velocity_reward     = self._get_velocity_reward()
            foot_contact_reward = self._get_foot_contact_reward()
            foot_airtime_reward = self._get_foot_airtime_reward()
            orientation_reward  = self._get_orientation_reward()
            heading_reward      = self._get_heading_reward()
            yoke_joint_reward   = self._get_yoke_joint_reward()
            # stand and velocity are multiplicatively coupled; split them so
            # that stand + velocity == c['stand'] * stand_reward * (1 + ...).
            stand_base = c['stand'] * stand_reward
            components.update({
                'stand':        stand_base,
                'velocity':     stand_base * c['velocity'] * velocity_reward,
                'foot_contact': c['foot_contact'] * foot_contact_reward,
                'foot_airtime': c['foot_airtime'] * foot_airtime_reward,
                'orientation':  c['orientation']  * orientation_reward,
                'heading':      c['heading']      * heading_reward,
                'yoke_joint':   c['yoke_joint']   * yoke_joint_reward,
            })
        else:
            components['stand'] = c['stand'] * stand_reward

        # Constant survival bonus: raises the value of not terminating so the
        # policy is not tempted to trade a short forward-velocity burst for an
        # early fall. The matching one-off fall penalty is applied in step().
        components['alive'] = c['alive']

        total = float(sum(components.values()))
        return total, components

    def step(self, action):
        action = action.clip(-1, 1)

        if self.max_action_delay > 0:
            self._action_buffer.append(action.copy())
            action = self._action_buffer[0]

        self._prev_action = self._last_action
        self._last_action = action.copy().astype(np.float32)

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

        # Set the 50Hz setpoint from the policy action (servo internally scales
        # by action_scale and low-pass filters the delta, as on hardware).
        for i in range(self.model.nu):
            self.servos[i].update_setpoint_delta(action[i])
        # Step physics, advancing the slew-limited servo command toward the
        # setpoint at the hardware rate (every `servo_update_every` substeps).
        for k in range(self.n_substeps):
            if k % self.servo_update_every == 0:
                for i in range(self.model.nu):
                    self.servos[i].get_pwm(dt=self.servo_dt)  # advances the PID/slew state
                    self.data.ctrl[i] = self.servos[i].value * np.pi
            mujoco.mj_step(self.model, self.data)
        noisey_state, state = self._get_obs()
        reward, reward_components = self._get_reward()
        if self.overturned() or self._root_body_on_ground() or self._too_low():
            self.done = True
            # One-off fall penalty. Because this lands on the terminal step the
            # critic does not bootstrap past it (done=1), so it directly lowers
            # the value of states leading into a fall.
            penalty = self.reward_coefs['termination']
            reward -= penalty
            reward_components['termination'] = -penalty
        # `self.done` is a true termination (the robot fell), never a time-limit
        # truncation, so report truncated=False.
        return (noisey_state, reward, self.done, False,
                {'state': state, 'reward_components': reward_components})

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        raise ValueError(f"Invalid render mode: {mode}")

    def close(self):
        if hasattr(self, '_renderer') and self._renderer is not None:
            self._renderer.close()
            self._renderer = None
