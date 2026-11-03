"""Synthetic accelerometer helpers shared by the acc_* tests."""

import numpy as np
import pandas as pd

HOUR = 3_600_000
MINUTE = 60_000
SECOND = 1000


def samples(start_ms, end_ms, *, moving, seed=0, hz=1.0, scale=1.0):
    """1 Hz samples; moving = vigorous noise, else still with gravity on z."""

    rng = np.random.default_rng(seed)
    step = int(SECOND / hz)
    t = np.arange(start_ms, end_ms, step, dtype=np.int64)
    n = t.size

    if moving:
        x = rng.normal(0.0, 0.3, n)
        y = rng.normal(0.0, 0.3, n)
        z = 1.0 + rng.normal(0.0, 0.3, n)
    else:
        x = rng.normal(0.0, 0.001, n)
        y = rng.normal(0.0, 0.001, n)
        z = 1.0 + rng.normal(0.0, 0.001, n)

    return pd.DataFrame({
        "timestamp_ms": t,
        "x": x * scale,
        "y": y * scale,
        "z": z * scale,
    })


def day_with_still_block(day_start, day_end, still_start, still_end, seed=0):
    return pd.concat(
        [
            samples(day_start, still_start, moving=True, seed=seed),
            samples(still_start, still_end, moving=False, seed=seed + 1),
            samples(still_end, day_end, moving=True, seed=seed + 2),
        ],
        ignore_index=True,
    )
