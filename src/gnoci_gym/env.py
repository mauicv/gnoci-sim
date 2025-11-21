import mujoco
import gymnasium as gym
import numpy as np
import os
from .utils import tolerance
from .load_xml import _load_and_perturb_basic_xml

_STANDING_HEIGHT = 0.157

class GnociGymEnv(gym.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 30,
    }
    
    def __init__(
            self,
            camera='track',
            render_mode='rgb_array',
            env_rate=0.005,
            initial_randomness=0.6,
            motor_gear_range=(-0.5, 1.5),
            motor_gear_noise=0.01,
            inertial_mass_range=(0.04, 0.06),
            inertial_mass_noise=0.01,
        ):
        self.camera = camera
        self.render_mode = render_mode
        self.done = False
        self.env_rate = env_rate
        self.initial_randomness = initial_randomness
        self.motor_gear_range = motor_gear_range
        self.motor_gear_noise = motor_gear_noise
        self.inertial_mass_range = inertial_mass_range
        self.inertial_mass_noise = inertial_mass_noise
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(4*3 + 4*3 + 4 + 6 + 1,), # motor positions, motor velocities, contact forces, imu data, root height
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            -1, 1, shape=(16,), dtype=np.float32
        )
        self.initialize_model()

    def initialize_model(self):
        xml_content = _load_and_perturb_basic_xml(
            'gnoci',
            motor_gear_range=self.motor_gear_range,
            motor_gear_noise=self.motor_gear_noise,
            inertial_mass_range=self.inertial_mass_range,
            inertial_mass_noise=self.inertial_mass_noise,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.model.opt.timestep = self.env_rate
        self.data = mujoco.MjData(self.model)
        self.gyro_sensor_id = self.model.sensor('gyro').id
        self.accel_sensor_id = self.model.sensor('accel').id

        self.motor_positions_sensor_ids = [
            self.model.sensor('hip-front-left-servo-pos').id,
            self.model.sensor('hip-front-right-servo-pos').id,
            self.model.sensor('hip-back-left-servo-pos').id,
            self.model.sensor('hip-back-right-servo-pos').id,
            self.model.sensor('thigh-front-left-servo-pos').id,
            self.model.sensor('thigh-front-right-servo-pos').id,
            self.model.sensor('thigh-back-left-servo-pos').id,
            self.model.sensor('thigh-back-right-servo-pos').id,
            self.model.sensor('lower-leg-front-left-servo-pos').id,
            self.model.sensor('lower-leg-front-right-servo-pos').id,
            self.model.sensor('lower-leg-back-left-servo-pos').id,
            self.model.sensor('lower-leg-back-right-servo-pos').id,
        ]

        self.motor_velocities_sensor_ids = [
            self.model.sensor('hip-front-left-servo-vel').id,
            self.model.sensor('hip-front-right-servo-vel').id,
            self.model.sensor('hip-back-left-servo-vel').id,
            self.model.sensor('hip-back-right-servo-vel').id,
            self.model.sensor('thigh-front-left-servo-vel').id,
            self.model.sensor('thigh-front-right-servo-vel').id,
            self.model.sensor('thigh-back-left-servo-vel').id,
            self.model.sensor('thigh-back-right-servo-vel').id,
            self.model.sensor('lower-leg-front-left-servo-vel').id,
            self.model.sensor('lower-leg-front-right-servo-vel').id,
            self.model.sensor('lower-leg-back-left-servo-vel').id,
            self.model.sensor('lower-leg-back-right-servo-vel').id,
        ]

        self.contact_forces_sensor_ids = [
            self.model.sensor('front-left-foot-contact').id,
            self.model.sensor('front-right-foot-contact').id,
            self.model.sensor('back-left-foot-contact').id,
            self.model.sensor('back-right-foot-contact').id,
        ]

        self.velocity_sensor_id = self.model.sensor('velocity').id

        self.servos = []
        for servo_id in range(self.model.nu):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, servo_id) == 'root':
                continue
            self.servos.append(servo_id)

        self.body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "root"
        )

    def _randomize_joint_positions(self, randomness):
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name == 'root':
                continue
            range_min, range_max = self.model.jnt_range[joint_id]
            adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[adr] = np.clip(np.random.normal(0, randomness), range_min, range_max)

    def reset(self, seed=None, **kwargs):
        self.initialize_model()
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self.done = False
        return self._get_obs(), {}

    def _get_gyro_data(self):
        return self.data.sensordata[self.gyro_sensor_id:self.gyro_sensor_id+3]

    def _get_accel_data(self):
        return self.data.sensordata[self.accel_sensor_id:self.accel_sensor_id+3]

    def _get_motor_positions(self):
        return self.data.sensordata[self.motor_positions_sensor_ids]

    def _get_motor_velocities(self):
        return self.data.sensordata[self.motor_velocities_sensor_ids]

    def _get_contact_forces(self):
        return self.data.sensordata[self.contact_forces_sensor_ids]

    def _get_obs(self):
        raw_imu_data = [
            *self._get_accel_data(),
            *self._get_gyro_data()
        ]

        obs = np.array([
            *self._get_motor_positions(),
            *self._get_motor_velocities(),
            *self._get_contact_forces(),
            *raw_imu_data,
            self._get_root_height(),
        ])

        return obs.astype(np.float32)

    def _get_info(self):
        return {}

    def overturned(self):
        return self._get_root_upright() < 0

    def _get_root_upright(self):
        xmat = self.data.xmat[self.body_id]
        z_axis = np.array([xmat[6], xmat[7], xmat[8]])
        dot = np.dot(z_axis, [0, 0, 1])
        return dot

    def _get_root_height(self):
        _, _, height = self.data.xpos[self.body_id]
        return height

    def _get_velocity(self):
        return self.data.sensordata[self.velocity_sensor_id:self.velocity_sensor_id+3]

    def _get_reward(self):
        upright, height = self._get_root_upright(), self._get_root_height()
        standing = tolerance(
            height,
            bounds=(_STANDING_HEIGHT, float('inf')),
            margin=_STANDING_HEIGHT/2
        )
        upright = (1 + upright) / 2
        stand_reward = (3*standing + upright) / 4
        forward_velocity = self._get_velocity()[0]
        velocity_reward = tolerance(
            forward_velocity,
            bounds=(1, float('inf')),
            margin=0.5
        )
        return stand_reward * (5*velocity_reward + 1) / 6

    def step(self, action):
        action = action.clip(-1, 1)

        for i, servo_id in enumerate(self.servos):
            self.data.ctrl[servo_id] = action[i]

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
                pixels = renderer.render()
                return pixels
        else:
            raise ValueError(f"Invalid render mode: {mode}")