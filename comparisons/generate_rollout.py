"""
Roll out the trained actor (model/actor-1.onnx) in sim and record the
per-step policy observation (with noise) and action to comparisons/rollout.json,
in the same {"actions": [...], "states": [...]} format replay_rollout.py reads.

The actor was trained on gnoci_gym.util.make_env's wrapped observation: an
8-frame stack of the full env observation (FrameStackObservation +
FlattenObservation, oldest frame first) concatenated with an 8-step action
history (ActionStackWrapper, oldest action first) -- see
src/gnoci_gym/util/__init__.py. Its ONNX input is 336-wide, which is exactly
8 stacked *noisy policy* observations (8 x 32, env.policy_obs_idx sliced out
of each stacked frame) plus the 8 stacked actions (8 x 10), so that's the
slice fed to it here. Only the current (most recent) frame's policy
observation and action are written out per step -- the stacks themselves are
just actor bookkeeping.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from gnoci_gym.config import CONTROL_HZ
from gnoci_gym.util import make_env, FRAME_STACK

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'actor-1.onnx')
_OUT_PATH = os.path.join(os.path.dirname(__file__), 'env-generated-rollout.json')
_VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'rendered_rollout', 'generated_rollout.mp4')
RENDER_W, RENDER_H = 640, 480


def _actor_input(wrapped_obs, base_obs_dim, policy_obs_idx):
    """wrapped_obs is [8 stacked full obs (oldest..newest) | 8 stacked actions
    (oldest..newest)]. Slice each stacked frame down to its policy_obs_idx
    portion and keep the action stack as-is."""
    obs_stack_size = base_obs_dim * FRAME_STACK
    frames = wrapped_obs[:obs_stack_size].reshape(FRAME_STACK, base_obs_dim)
    policy_frames = frames[:, policy_obs_idx].reshape(-1)
    action_stack = wrapped_obs[obs_stack_size:]
    return np.concatenate([policy_frames, action_stack]).astype(np.float32)


def _latest_policy_obs(wrapped_obs, base_obs_dim, policy_obs_idx):
    latest_frame = wrapped_obs[base_obs_dim * (FRAME_STACK - 1):base_obs_dim * FRAME_STACK]
    return latest_frame[policy_obs_idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--domain-rand', action='store_true',
                         help='use the training-distribution env config (dom_rnd_cfg) '
                              'instead of the deterministic one (test_cfg) used by default')
    parser.add_argument('--no-video', action='store_true',
                         help='skip rendering/saving the rollout video')
    args = parser.parse_args()

    session = ort.InferenceSession(_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    env = make_env(seed=args.seed, test=not args.domain_rand)
    base_obs_dim = env.unwrapped.observation_space.shape[0]
    policy_obs_idx = env.unwrapped.policy_obs_idx

    obs, _ = env.reset(seed=args.seed)

    states = [_latest_policy_obs(obs, base_obs_dim, policy_obs_idx).tolist()]
    actions = []
    target_actions = []

    renderer = None
    frames = []
    if not args.no_video:
        renderer = mujoco.Renderer(env.unwrapped.model, height=RENDER_H, width=RENDER_W)
        renderer.update_scene(env.unwrapped.data, camera="track")
        frames.append(renderer.render())

    for _ in tqdm(range(args.steps), desc="Rolling out"):
        model_input = _actor_input(obs, base_obs_dim, policy_obs_idx)
        action = session.run(None, {input_name: model_input})[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # SB3 PPO's actor head is an unsquashed Gaussian mean, so predict()
        # clips to the action space before stepping -- match that here.
        # action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.tolist())
        target_actions.append(info['target_action'])
        states.append(_latest_policy_obs(obs, base_obs_dim, policy_obs_idx).tolist())

        if renderer is not None:
            renderer.update_scene(env.unwrapped.data, camera="track")
            frames.append(renderer.render())

        if terminated or truncated:
            break

    env.close()

    with open(_OUT_PATH, 'w') as f:
        json.dump({'actions': actions, 'states': states, 'target_actions': target_actions}, f)
    print(f"Saved rollout ({len(actions)} steps) to {_OUT_PATH}")

    if renderer is not None:
        renderer.close()
        os.makedirs(os.path.dirname(_VIDEO_PATH), exist_ok=True)
        imageio.mimwrite(_VIDEO_PATH, frames, fps=CONTROL_HZ)
        print(f"Saved video to {_VIDEO_PATH}")


if __name__ == '__main__':
    main()
