import mujoco
import gymnasium as gym
import numpy as np
import os
from .filters import ComplementaryFilter, KalmanMPU6050Filter
from .servo import Servo
from .utils import tolerance

_STANDING_HEIGHT = 145

generic_values = {
    "kp": 0.35,
    "ki": 0.0,
    "kd": 0.05,
}


class ComplexGnociGymEnv(gym.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 30,
    }
    
    def __init__(
            self,
            env_rate=0.005,
            system_rate=0.01,
            control_rate=0.05,
            initial_randomness=0.6,
            camera='track',
            render_mode='rgb_array'
        ):
        self.camera = camera
        self.render_mode = render_mode
        self.env_rate = env_rate
        self.system_rate = system_rate
        self.control_rate = control_rate
        self.done = False
        self.initial_randomness = initial_randomness
        
        package_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(package_dir, 'desc', 'gnoci_complex.xml')
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.env_rate
        self.cf = ComplementaryFilter()
        self.kalman_filter = KalmanMPU6050Filter()

        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(20,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            -1, 1, shape=(6,), dtype=np.float32
        )

        self.gyro_sensor_id = self.model.sensor('gyro').id
        self.accel_sensor_id = self.model.sensor('accel').id

        self.motor_positions_sensor_ids = [
            self.model.sensor('hip-left-servo-pos').id,
            self.model.sensor('thigh-left-servo-pos').id,
            self.model.sensor('lower-leg-left-servo-pos').id,
            self.model.sensor('hip-right-servo-pos').id,
            self.model.sensor('thigh-right-servo-pos').id,
            self.model.sensor('lower-leg-right-servo-pos').id,
        ]

        self.motor_velocities_sensor_ids = [
            self.model.sensor('hip-left-servo-vel').id,
            self.model.sensor('thigh-left-servo-vel').id,
            self.model.sensor('lower-leg-left-servo-vel').id,
            self.model.sensor('hip-right-servo-vel').id,
            self.model.sensor('thigh-right-servo-vel').id,
            self.model.sensor('lower-leg-right-servo-vel').id,
        ]

        self.contact_forces_sensor_ids = [
            self.model.sensor('left-foot-contact').id,
            self.model.sensor('right-foot-contact').id,
        ]

        self.servos = []
        for i in range(self.model.nu):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == 'root':
                continue
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            servo_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            pin_limits=(self.model.actuator_ctrlrange[servo_id][0], self.model.actuator_ctrlrange[servo_id][1])
            servo = Servo(
                name=actuator_name,
                pin_id=0,
                pin=servo_id,
                pin_limits=pin_limits,
                init_value=0.0,
                offset=0.0,
                **generic_values
            )
            self.servos.append(servo)

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
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_joint_positions(randomness=self.initial_randomness)
        self.cf.reset()
        self.kalman_filter.reset()
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

        self.cf.update(raw_imu_data[0:3], raw_imu_data[3:6])
        filtered_imu_data = self.kalman_filter(raw_imu_data)

        obs = np.array([
            *self._get_motor_positions(),
            *self._get_motor_velocities(),
            np.float32(self.cf.roll),
            np.float32(self.cf.pitch),
            *filtered_imu_data
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

    def _get_reward(self):
        upright, height = self._get_root_upright(), self._get_root_height()
        standing = tolerance(
            height,
            bounds=(_STANDING_HEIGHT, float('inf')),
            margin=_STANDING_HEIGHT/2
        )
        upright = (1 + upright) / 2
        stand_reward = (3*standing + upright) / 4
        return stand_reward

    def step(self, action):
        action = action.clip(-1, 1)

        for i, servo in enumerate(self.servos):
            servo.update_setpoint_delta(action[i])

        for step in range(int(self.control_rate/self.env_rate)):
            mujoco.mj_step(self.model, self.data)

            if step % int(self.control_rate/self.system_rate) == 0:
                for servo in self.servos:
                    servo_pos = servo.update()
                    self.data.ctrl[servo.pin] = servo_pos

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