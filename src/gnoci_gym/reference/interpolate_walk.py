#!/usr/bin/env python3
"""
Parse walk.xml keyframes and produce a dense interpolated trajectory.

Keyframes have no meaningful timestamps so they are treated as evenly-spaced
phases of one gait cycle [0, 1).  Hinge joints and freejoint xyz use a
periodic cubic spline; the freejoint quaternion uses SLERP per segment.

Output: walk_traj.npy — shape (n_frames, 17) qpos
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).parent

# qpos layout: freejoint xyz (0:3), quaternion (3:7), hinge joints (7:)
_QUAT_SLICE  = slice(3, 7)
_HINGE_START = 7


def _load_keyframes(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    frames = []
    for key in root.findall("key"):
        vals = [float(v) for v in key.get("qpos").split()]
        frames.append(vals)
    return np.array(frames)  # (N, D)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / np.sin(theta_0)
    s1 = np.sin(theta) / np.sin(theta_0)
    return s0 * q0 + s1 * q1


def interpolate(qpos_keys: np.ndarray, n_frames: int = 200) -> np.ndarray:
    N, D = qpos_keys.shape
    phase_keys = np.linspace(0.0, 1.0, N, endpoint=False)
    phase_out  = np.linspace(0.0, 1.0, n_frames, endpoint=False)

    out = np.zeros((n_frames, D))

    # Periodic cubic spline for freejoint xyz (0:3) and hinge joints (7:)
    scalar_idx = list(range(0, 3)) + list(range(_HINGE_START, D))
    # Append first frame at phase=1.0 so the endpoints match for periodic BC
    phases_closed = np.append(phase_keys, 1.0)
    values_closed = np.vstack([qpos_keys[:, scalar_idx], qpos_keys[[0], scalar_idx]])
    cs = CubicSpline(phases_closed, values_closed, bc_type="periodic")
    out[:, scalar_idx] = cs(phase_out)

    # SLERP per segment for quaternion (3:7)
    quats = qpos_keys[:, _QUAT_SLICE].copy()
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    for i, ph in enumerate(phase_out):
        idx = int(np.searchsorted(phase_keys, ph, side="right")) - 1
        idx_next = (idx + 1) % N
        ph0 = phase_keys[idx]
        ph1 = 1.0 if idx_next == 0 else phase_keys[idx_next]
        t = (ph - ph0) / (ph1 - ph0) if ph1 > ph0 else 0.0
        out[i, _QUAT_SLICE] = _slerp(quats[idx], quats[idx_next], t)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src",    default=str(HERE / "xml" / "walk.xml"),      help="Keyframe XML")
    parser.add_argument("--dst",    default=str(HERE / "npy" / "walk_traj.npy"), help="Output .npy")
    parser.add_argument("--frames", type=int, default=200,               help="Output frame count")
    args = parser.parse_args()

    keys = _load_keyframes(Path(args.src))
    print(f"Loaded {len(keys)} keyframes  (qpos dim={keys.shape[1]})")

    traj = interpolate(keys, n_frames=args.frames)
    np.save(args.dst, traj)
    print(f"Saved {args.dst}  shape={traj.shape}")
