from gnoci_gym.env import GnociGymEnv
import numpy as np
import json
from tqdm import tqdm
import os

ACTION_DIM = 10


def raw_to_mjx_actions(actions: np.ndarray):
    # Converts a entire rollout of raw hardware compatible actions to a set of mjx actions.
    env = GnociGymEnv()
    env.reset()
    action_set = []
    for action in actions:
        action_set.append(_raw_to_mjx_action(env, action))
    env.close()
    return np.concatenate(action_set, axis=0)


def _raw_to_mjx_action(env: GnociGymEnv, action: np.ndarray):
    # Computes action set every 5 env steps at 0.002 physics dt - i.e. applied the various 
    # mappings and filters to the raw actions. 
    action = action.clip(-1, 1)
    for i in range(env.model.nu):
        env.servos[i].update_setpoint_delta(action[i])
    action_set = []
    for k in range(env.n_substeps):
        if k % env.servo_update_every == 0:
            servo_substep_actions = []
            for i in range(env.model.nu):
                env.servos[i].get_pwm(dt=env.servo_dt)
                true_action = env.servos[i].value
                servo_substep_actions.append(true_action)
            action_set.append(servo_substep_actions)
    return action_set


def _run_tests():
    actions = np.zeros((1, ACTION_DIM))
    action_set = raw_to_mjx_actions(actions)
    assert action_set.shape == (2, 10), f"Action set shape is {action_set.shape} but should be (2, 10)"

def get_config():
    env = GnociGymEnv()
    env.reset()
    config = {
        "hardware_update_every": env.servo_update_every,    # number substeps between hardware updates    
        "servo_dt": env.servo_dt,                           # servo dt
        "physics_dt": env.model.opt.timestep,               # physics dt
        "n_substeps": env.n_substeps,                       # n_substeps per action set
        "action_hz": 1.0 / (env.model.opt.timestep * env.n_substeps),
        "joint_limits": [jr.tolist() for jr in env.joint_ranges],
        "action_dim": ACTION_DIM,
    }

    print('control hz:', 1.0 / (config['physics_dt'] * config['n_substeps']), 'Hz')
    print('hardware hz:', 1.0 / (config['physics_dt'] * config['hardware_update_every']), 'Hz')

    env.close()
    return config


def generate_chirp(config, motor_idx, seconds=3, freq_low=0.5, freq_high=5):
    joint_limits = config['joint_limits'][motor_idx]
    joint_range = joint_limits[1] - joint_limits[0]
    amplitude = joint_range / 2
    action_hz = config['action_hz']
    action_dim = config['action_dim']
    t = np.linspace(0, seconds, int(seconds * action_hz), endpoint=False)
    k = (freq_high - freq_low) / seconds
    phase = 2 * np.pi * (freq_low * t + 0.5 * k * t**2)
    action = np.zeros((len(t), action_dim))
    amplitude_signal = np.ones((len(t), action_dim)) * amplitude * t[:, np.newaxis] * 0.4
    action[:, motor_idx] = np.sin(phase)
    action *= amplitude_signal
    return action


def generate_step(config, motor_idx, seconds=1, amplitude=1.0):
    action_hz = config['action_hz']
    action_dim = config['action_dim']
    t = np.linspace(0, seconds, int(seconds * action_hz), endpoint=False)
    action = np.zeros((len(t), action_dim))
    action[:, motor_idx] = amplitude
    return action


def generate_prbs(config, motor_idx, seconds=3, amplitude=0.2, min_hold=0.05, seed=0):
    action_hz = config['action_hz']
    action_dim = config['action_dim']
    rng = np.random.default_rng(seed)
    n = int(seconds * action_hz)
    hold_steps = int(min_hold * action_hz)

    signal = np.zeros(n)
    i = 0
    val = amplitude
    while i < n:
        val = amplitude if rng.random() > 0.5 else -amplitude
        run_len = hold_steps + rng.integers(0, hold_steps)  # jittered hold
        signal[i:i+run_len] = val
        i += run_len

    action = np.zeros((n, action_dim))
    action[:, motor_idx] = signal[:n]
    return action


def generate_ramp(config, motor_idx, seconds=2, amplitude=1.0):
    action_hz = config['action_hz']
    action_dim = config['action_dim']
    n = int(seconds * action_hz)
    ramp = np.linspace(0, amplitude, n)
    action = np.zeros((n, action_dim))
    action[:, motor_idx] = ramp
    return action


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _run_tests()
    config = get_config()
    dataset={
        'config': config,
        'data': [],
    }
    print("Generating chirp rollouts...")
    for motor_idx in tqdm(range(10)):
        actions_hardware = generate_chirp(config, motor_idx, 3)
        action_mjx = raw_to_mjx_actions(actions_hardware)
        rollout = {
            'motor_idx': motor_idx,
            'type': 'chirp',
            'actions_hw': actions_hardware.tolist(),
            'actions_mjx': action_mjx.tolist(),
        }
        dataset['data'].append(rollout)


    for motor_idx in tqdm(range(10)):
        for amplitude in [-1, -0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 1]:
            actions_hardware = generate_step(config, motor_idx, 1, amplitude)
            action_mjx = raw_to_mjx_actions(actions_hardware)
            rollout = {
                'motor_idx': motor_idx,
                'type': 'step',
                'actions_hw': actions_hardware.tolist(),
                'actions_mjx': action_mjx.tolist(),
            }
            dataset['data'].append(rollout)

    for motor_idx in tqdm(range(10)):
        for _ in range(3):
            actions_hardware = generate_prbs(config, motor_idx, seed=np.random.randint(0, 1000000))
            action_mjx = raw_to_mjx_actions(actions_hardware)
            rollout = {
                'motor_idx': motor_idx,
                'type': 'prbs',
                'actions_hw': actions_hardware.tolist(),
                'actions_mjx': action_mjx.tolist(),
            }
            dataset['data'].append(rollout)

    for motor_idx in tqdm(range(10)):
        actions_hardware = generate_ramp(config, motor_idx, amplitude=0.4)
        action_mjx = raw_to_mjx_actions(actions_hardware)
        rollout = {
            'motor_idx': motor_idx,
            'type': 'ramp',
            'actions_hw': actions_hardware.tolist(),
            'actions_mjx': action_mjx.tolist(),
        }
        dataset['data'].append(rollout)

    filename = os.path.dirname(__file__) + '/dataset/dataset.json'
    print('saving dataset to', filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f)

