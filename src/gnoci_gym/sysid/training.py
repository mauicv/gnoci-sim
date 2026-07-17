import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import xml.etree.ElementTree as ET
import os
from gnoci_gym.load_xml import _load_and_perturb_basic_xml
from gnoci_gym.sysid.sysid_ds_interface import SysidDSInterface
import optax
import warnings
import time

warnings.filterwarnings("ignore", module="mujoco.mjx._src.mesh")


def build_model(mx, params):
    # learnable_params: frictionloss, damping, kv, kp, armature, actuator_dynprm, filter time constant
    # Note: mx.actuator_biasprm[:, 2] duplicates mx.actuator_biasprm[:, 1], we should pass in a single kv param and then replace both in the model
    # damping, frictionloss, armature, kp should be positive so should learn as log_...
    # kv should be negative so should learn as log_... and then negate when applying.

    log_kp = params['log_kp']
    log_kv = params['log_kv']
    log_tau = params['log_tau']
    log_damping = params['log_damping']
    log_frictionloss = params['log_frictionloss']
    log_armature = params['log_armature']

    kp = jax.nn.softplus(log_kp)
    kv = jax.nn.softplus(log_kv)
    tau = jax.nn.softplus(log_tau)
    damping = jax.nn.softplus(log_damping)
    frictionloss = jax.nn.softplus(log_frictionloss)
    armature = jax.nn.softplus(log_armature)

    mx = mx.tree_replace({
        'actuator_gainprm': mx.actuator_gainprm.at[:, 0].set(kp),
        'actuator_biasprm': mx.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kv),
        'actuator_dynprm':  mx.actuator_dynprm.at[:, 0].set(tau),
        'dof_damping':      damping,
        'dof_frictionloss': frictionloss,
        'dof_armature':     armature,
    })
    return mx

def extract_joint_obs():
    joint_pos = self.data.sensordata[self.joint_pos_sensor_addrs] / np.pi


def rollout(params, mx, q0, qd0, actions, n_substeps):
    mx_ = build_model(mx, params)
    data = mjx.make_data(mx_)
    data = jax.vmap(lambda q, qd: data.tree_replace({"qpos": q, "qvel": qd}), (0, 0))(q0, qd0)

    def substep(data, _):
        data = jax.vmap(mjx.step, (None, 0))(mx_, data)
        return data, None

    def step_fn(data, action):
        data = data.tree_replace({"ctrl": action})
        # n_substeps physics steps per control action
        data, _ = jax.lax.scan(substep, data, xs=None, length=n_substeps)
        return data, (data.qpos, data.qvel)

    actions_t = jnp.moveaxis(actions, 1, 0) # (B, T, A) -> (T, B, A)
    final_data, (q_traj, qd_traj) = jax.lax.scan(step_fn, data, actions_t)
    # Note that becuase there are 2n actions per n states the resulting trajectories are 
    # 2n + 1 long, we need to remove the first state and take every other state after that.
    q_traj = jnp.moveaxis(q_traj[::2], 0, 1)
    qd_traj = jnp.moveaxis(qd_traj[::2], 0, 1)
    return q_traj, qd_traj


def loss(params, mx, q_targ, q0, qd0, actions, motor_idxs):
    q_traj, _ = rollout(params, mx, q0, qd0, actions, 5)
    return jnp.mean((q_targ[:, :, motor_idxs] - q_traj[:, :, motor_idxs])**2)


grad_fn = jax.value_and_grad(loss)


def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, mx, q_targ, q0, qd0, actions, motor_idxs):
        loss_val, grads = grad_fn(params, mx, q_targ, q0, qd0, actions, motor_idxs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return train_step


if __name__ == '__main__':
    ds = SysidDSInterface()

    xml = _load_and_perturb_basic_xml('scene', fix_root_body=True, strip_contact_sensors=True)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # MJX's constraint solver uses jax.lax.while_loop (dynamic convergence stop)
    # when opt.iterations > 1, which reverse-mode autodiff can't differentiate
    # through. iterations == 1 takes the single-body path (no while_loop) — fine
    # here since contacts are disabled and the fixed-root robot stays within its
    # joint limits, so there are no active constraints to iterate on.
    model.opt.iterations = 1
    data = mujoco.MjData(model)
    mx = mjx.put_model(model)

    # learnable params (unconstrained)
    params = {
        'log_kp': jnp.log(jnp.array([500]*10)),  # single scalar/array per actuator
        'log_kv': jnp.log(jnp.array([40]*10)),
        'log_tau': jnp.log(jnp.array([0.1]*10)),
        'log_damping': jnp.log(jnp.array([0.001]*10)),
        'log_frictionloss': jnp.log(jnp.array([0.1]*10)),
        'log_armature': jnp.log(jnp.array([0.005]*10))
    }

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-2),
    )
    params = jax.device_put(params)
    opt_state = optimizer.init(params)
    opt_state = jax.device_put(opt_state)
    train_step = make_train_step(optimizer)

    n_epochs = 500
    batch = ds.sample(100, 50)  # re-sample each step, or fix a batch if you want overfitting sanity check first

    history = []

    for epoch in range(n_epochs):
        # Dataset positions/velocities/actions are in the robot's normalised
        # units (angle / π); MJX qpos/qvel/ctrl are in radians (rad/s). Scale by
        # π. NB: actions_mjx are normalised too despite the name (e.g. 0.675 ==
        # the 2.12 rad joint limit), so they need it as well. motor_idxs are
        # indices — leave unscaled.
        q0 = jnp.array(batch['initial_states']) * jnp.pi
        qd0 = jnp.array(batch['initial_velocities']) * jnp.pi
        actions = jnp.array(batch['actions']) * jnp.pi
        motor_idxs = jnp.array(batch['motor_idxs'])
        q_targ = jnp.array(batch['states']) * jnp.pi

        time_start = time.perf_counter()

        params, opt_state, loss_val = train_step(
            params, opt_state, mx, q_targ, q0, qd0, actions, motor_idxs
        )
        loss_val = jax.block_until_ready(loss_val)

        time_end = time.perf_counter()
        print(f'time taken for epoch {epoch}: {time_end - time_start:.6f} seconds')

        history.append({
            'epoch': epoch,
            'loss': float(loss_val),
            'time': time_end - time_start,
            'params': jax.tree_util.tree_map(lambda x: jax.device_get(x), params)
        })

        if epoch % 10 == 0:
            print(f'epoch {epoch}, loss {loss_val:.6f}')


