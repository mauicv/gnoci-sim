import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import xml.etree.ElementTree as ET
import os
from gnoci_gym.load_xml import _load_and_perturb_basic_xml


xml = _load_and_perturb_basic_xml('scene')
model = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(model)

# def rollout(params, q0, qd0, actions):
#     # patch model params
#     mx_ = mx.replace(dof_damping=params['damping'], ...)
#     data = mjx.make_data(mx_)
#     # ... step through mjx.step, collect q
#     return q_traj

# def loss(params, real_traj, q0, qd0, actions):
#     sim_traj = rollout(params, q0, qd0, actions)
#     return jnp.mean((real_traj - sim_traj)**2)

# grad_fn = jax.jit(jax.value_and_grad(loss))
# # then standard optax gradient descent loop

print(model)