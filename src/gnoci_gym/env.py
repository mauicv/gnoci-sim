import mujoco
import gymnasium as gym
import numpy as np
import os
from .utils import tolerance
from .load_xml import _load_and_perturb_basic_xml

_STANDING_HEIGHT = 0.1  # TODO: calibrate once robot is simulating

_JOINT_NAMES = [
    'l_hip_l_lat_rot',
    'l_hip_l_vert_rot',
    'l_vert_rot_l_upper_leg',
    'l_upper_leg_l_lower_leg',
    'l_lower_leg_l_foot',
    'r_hip_r_lat_rot',
    'r_hip_r_vert_rot',
    'r_vert_rot_r_upper_leg',
    'r_upper_leg_r_lower_leg',
    'r_lower_leg_r_foot',
]

_TOUCH_SENSOR_NAMES = [
    'foot-toe-contact',
    'foot-heel-contact',
    'foot_mirrored-toe-contact',
    'foot_mirrored-heel-contact',
]

_N_JOINTS = len(_JOINT_NAMES)   # 10
_N_TOUCH  = len(_TOUCH_SENSOR_NAMES)  # 4


class GnociGymEnv(gym.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 30,
    }

    def __init__(
            self,
            camera=-1,
            render_mode='rgb_array',
            env_rate=0.005,
            initial_randomness=0.6,
            inertial_mass_range=(0.04, 0.06),
            inertial_mass_noise=0.01,
        ):
        self.camera = camera
        self.render_mode = render_mode
        self.done = False
        self.env_rate = env_rate
        self.initial_randomness = initial_randomness
        self.inertial_mass_range = inertial_mass_range
        self.inertial_mass_noise = inertial_mass_noise

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(_N_JOINTS + _N_JOINTS + _N_TOUCH + 1,),  # joint pos, joint vel, contact forces, root height
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(_N_JOINTS,), dtype=np.float32
        )
        self.initialize_model()

    def initialize_model(self):
        xml_content = _load_and_perturb_basic_xml(
            'scene',
            inertial_mass_range=self.inertial_mass_range,
            inertial_mass_noise=self.inertial_mass_noise,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.model.opt.timestep = self.env_rate
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
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hor_rot_body_joint"
        )

    def _randomize_joint_positions(self, randomness):
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
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

    def _get_joint_positions(self):
        return self.data.sensordata[self.joint_pos_sensor_addrs]

    def _get_joint_velocities(self):
        return self.data.qvel[self.joint_dof_addrs]

    def _get_contact_forces(self):
        return self.data.sensordata[self.touch_sensor_addrs]

    def _get_obs(self):
        obs = np.array([
            *self._get_joint_positions(),
            *self._get_joint_velocities(),
            *self._get_contact_forces(),
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
        return np.dot(z_axis, [0, 0, 1])

    def _get_root_height(self):
        _, _, height = self.data.xpos[self.body_id]
        return height

    def _get_velocity(self):
        # Linear velocity of root body in world frame (freejoint qvel[0:3])
        return self.data.qvel[0:3]

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

        velocity = self._get_velocity()
        side_v = abs(velocity[1])
        lateral_penalty = max(1.0 - 0.5 * side_v, 0.0)

        velocity_reward = tolerance(
            -velocity[0],
            bounds=(1, 2),
            margin=1
        )
        total_reward = stand_reward * (5 * velocity_reward + 1) / 6
        return total_reward * lateral_penalty

    def step(self, action):
        action = action.clip(-1, 1)
        for i in range(self.model.nu):
            self.data.ctrl[i] = action[i]
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
