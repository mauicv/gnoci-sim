# Gnoci Sim

use like so:

```py

from src import GnociGymEnv
import gymnasium as gym
import numpy as np
from tqdm import tqdm


gym.register(
    id="gnoci_gym/Gnoci-v0",
    entry_point=GnociGymEnv,
)

env = gym.make("gnoci_gym/Gnoci-v0")
state, *_ = env.reset(seed=0)

for i in tqdm(range(100)):
    action = np.random.uniform(-1, 1, 6)
    state, reward, done, truncated, _ = env.step(action)

env.reset()
```


For colab notebook example see [this](https://colab.research.google.com/drive/13MDYzpxYmWT-9RKIIiF1CMuo6QNOUCGv?authuser=1).

## Walk task with curriculum

The `walk` task removes the standing reward floor (no survival reward just for
staying upright) and exposes curriculum knobs so the policy is forced to commit
to stepping. The trainer anneals these between phases via `set_curriculum()`:

- `survival_bonus` — the only reward payable while standing still. Start small
  and decay to `0` so standing stops being competitive with walking.
- `target_velocity` / `target_velocity_band` — forward-speed target. The
  velocity reward is exactly `0` at zero speed and ramps linearly to `1` at the
  target, so even a low target forces the robot to shuffle. Ramp the target up.
- `foot_clearance_height` — height a foot must clear (relative to the other
  foot) to earn the foot-clearance reward, forcing commitment to a step.

```py
import numpy as np
from src import GnociGymEnv

# Phase 0: low target, small survival bonus to bootstrap.
env = GnociGymEnv(
    task="walk",
    survival_bonus=0.2,        # decaying standing floor
    target_velocity=0.1,       # m/s forward target (reward is 0 at v=0)
    target_velocity_band=0.1,  # full-credit half-width above the target
    foot_clearance_height=0.02,
)
state, *_ = env.reset(seed=0)

n_phases = 4
for phase in range(n_phases):
    # Anneal: survival_bonus -> 0, target_velocity 0.1 -> 0.5.
    frac = phase / (n_phases - 1)
    env.set_curriculum(
        survival_bonus=0.2 * (1.0 - frac),
        target_velocity=0.1 + 0.4 * frac,
    )

    for _ in range(2500):
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        state, reward, done, truncated, info = env.step(action)
        if done or truncated:
            state, *_ = env.reset()
```

When using a vectorized trainer (e.g. SB3), call `set_curriculum` on each worker
via `env_method("set_curriculum", survival_bonus=..., target_velocity=...)`.

The default reward coefficients can be overridden per-env with `reward_coefs`:

```py
env = GnociGymEnv(
    task="walk",
    reward_coefs={"velocity": 3.0, "foot_clearance": 0.75, "fall": 0.5},
)
```

## For development:

1. Use onshape-to-robot to pull down onshape data and convert to meshes and xml: `onshape-to-robot onshape_export`
2. Use python script to process raw onshape xml: `python process_desc.py`

Note: the src/desc/gnoci.xml and src/desc/assets files are computer generated and shouldn't be edited manually.