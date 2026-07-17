import json
import os
import numpy as np
from scipy.signal import savgol_filter

def compute_velocities(rollout):
    times = np.array(rollout['times'])
    states = np.array(rollout['measured_states'])[:, :10]
    dt = np.mean(times[1:] - times[:-1])
    v = savgol_filter(states, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)
    return v


class SysidDSInterface:
    def __init__(self, compute_velocities=True):
        self.data_dir = os.path.dirname(__file__) + '/dataset/sysid_data.json'
        self.data = json.load(open(self.data_dir))
        self.config = self.data['config']
        self.num_rollouts = len(self.data['data'])
        self.index_weights = None
        self.compute_weights()
        if compute_velocities:
            self.esimate_velocities()

    def esimate_velocities(self):
        velocities = []
        for i, rollout in enumerate(self.data['data']):
            v = compute_velocities(rollout)
            self.data['data'][i]['velocities'] = v

    def compute_weights(self):
        type_weights = {'chirp': 0, 'step': 0, 'ramp': 0, 'prbs': 0}
        for rollout in self.data['data']:
            type_weights[rollout['type']] += len(rollout['measured_states'])
        for rtype in type_weights:
            type_weights[rtype] = 1 / type_weights[rtype]

        index_weights = [
            type_weights[rollout['type']] * len(rollout['measured_states'])
            for rollout in self.data['data']
        ]
        total = sum(index_weights)
        self.index_weights = [w / total for w in index_weights]

    def get_rollout(self, index):
        return self.data['data'][index]

    def sample_index(self, count):
        return np.random.choice(self.num_rollouts, p=self.index_weights, size=count)

    def sample_subset(self, rollout, length=50):
        states = np.array(rollout['measured_states'])
        actions = np.array(rollout['actions_mjx'])
        velocities = np.array(rollout['velocities'])
        if len(states) == length:
            return states, actions, velocities
        elif len(states) < length:
            raise ValueError(f"Rollout length {len(states)} is greater than requested length {length}")
        start = np.random.randint(0, len(states) - length)
        end = start + length
        return states[start:end], actions[2*start:2*end], velocities[start:end]

    def sample(self, count, length):
        indices = self.sample_index(count)
        states = []
        actions = []
        types = []
        motor_idxs = []
        velocities = []
        # initial state and velocity conditions need to be mapped back to correct q0, qd0 values 
        for index in indices:
            rollout = self.get_rollout(index)
            s, a, v = self.sample_subset(rollout, length)
            states.append(s)
            velocities.append(v)
            actions.append(a)
            types.append(rollout['type'])
            motor_idxs.append(rollout['motor_idx'])
        return {
            'initial_states': np.array(states)[:, 0, :10],
            'initial_velocities': np.array(velocities)[:, 0, :10],
            'states': np.array(states)[:, :, :10],
            'velocities': np.array(velocities),
            'actions': np.array(actions),
            'types': np.array(types),
            'motor_idxs': np.array(motor_idxs),
        }


if __name__ == '__main__':
    ds = SysidDSInterface()
    data = ds.sample(10, 50)
    print('initial_states       ', data['initial_states'].shape)
    print('initial_velocities   ', data['initial_velocities'].shape)
    print('states               ', data['states'].shape)
    print('velocities           ', data['velocities'].shape)
    print('actions              ', data['actions'].shape)
    print('types                ', data['types'])
    print('motor_idxs           ', data['motor_idxs'])
    


