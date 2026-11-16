"""Controlled raw-GPS degradation for Cortex 2.0 reliability experiments.

Missingness is imposed in time, not by randomly deleting individual GPS rows.

Two patterns are supported:

distributed
    Missing time bins are scattered across the day.

contiguous
    The same amount of missing time occurs as one continuous block.

The temporal bin size is an experimental simulation parameter, not a
Cortex 2.0 data-quality threshold.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def make_missing_bin_mask(
    *,
    window_start_ms: int,
    window_end_ms: int,
    target_coverage: float,
    pattern: str,
    bin_size_ms: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Create a deterministic temporal missingness mask."""

    window_start_ms = int(window_start_ms)
    window_end_ms = int(window_end_ms)
    bin_size_ms = int(bin_size_ms)

    if window_end_ms <= window_start_ms:
        raise ValueError(
            "window_end_ms must be greater than window_start_ms"
        )

    if (
        not math.isfinite(target_coverage)
        or target_coverage <= 0
        or target_coverage > 1
    ):
        raise ValueError(
            "target_coverage must be in (0, 1]"
        )

    if bin_size_ms <= 0:
        raise ValueError(
            "bin_size_ms must be > 0"
        )

    if pattern not in {
        "distributed",
        "contiguous",
    }:
        raise ValueError(
            "pattern must be 'distributed' or 'contiguous'"
        )

    duration_ms = (
        window_end_ms
        - window_start_ms
    )

    n_bins = int(
        math.ceil(
            duration_ms
            / bin_size_ms
        )
    )

    n_missing = int(
        round(
            (1.0 - target_coverage)
            * n_bins
        )
    )

    n_missing = min(
        max(n_missing, 0),
        n_bins,
    )

    mask = np.zeros(
        n_bins,
        dtype=bool,
    )

    rng = np.random.default_rng(
        int(seed)
    )

    if n_missing:

        if pattern == "distributed":

            indices = rng.choice(
                n_bins,
                size=n_missing,
                replace=False,
            )

            mask[
                indices
            ] = True

        else:

            max_start = (
                n_bins
                - n_missing
            )

            start = int(
                rng.integers(
                    0,
                    max_start + 1,
                )
            )

            mask[
                start:
                start + n_missing
            ] = True

    return (
        mask,
        {
            "n_bins":
                n_bins,

            "n_missing_bins":
                int(
                    mask.sum()
                ),

            "planned_coverage":
                float(
                    1.0
                    - mask.mean()
                ),

            "pattern":
                pattern,

            "bin_size_ms":
                bin_size_ms,
        },
    )


def apply_missing_bin_mask_to_points(
    points: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    bin_size_ms: int,
    missing_mask: np.ndarray,
) -> pd.DataFrame:
    """Remove raw GPS observations occurring inside masked time bins."""

    if "timestamp" not in points.columns:
        raise ValueError(
            "points must contain timestamp"
        )

    if len(missing_mask) == 0:
        raise ValueError(
            "missing_mask must not be empty"
        )

    if points.empty:
        return points.copy()

    timestamps = pd.to_numeric(
        points[
            "timestamp"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(
            timestamps
        )
        & (
            timestamps
            >= window_start_ms
        )
        & (
            timestamps
            < window_end_ms
        )
    )

    retain = np.zeros(
        len(points),
        dtype=bool,
    )

    if valid.any():

        positions = np.flatnonzero(
            valid
        )

        bins = (
            (
                timestamps[
                    valid
                ].astype(
                    np.int64
                )
                - int(
                    window_start_ms
                )
            )
            // int(
                bin_size_ms
            )
        )

        bins = np.clip(
            bins,
            0,
            len(
                missing_mask
            )
            - 1,
        )

        retain[
            positions
        ] = ~missing_mask[
            bins
        ]

    return (
        points.loc[
            retain
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def downsample_points_to_time_bins(
    points: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    bin_size_ms: int,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Downsample raw GPS observations to at most one observed point per
    temporal bin.

    This is a temporal-resolution experiment, not a Cortex 2.0 quality
    threshold or missingness simulation.

    For each non-empty time bin, one observed GPS row is selected
    uniformly at random. Selection is reproducible for a fixed seed.

    Empty bins remain empty; no interpolation or synthetic GPS values
    are created.

    Parameters
    ----------
    points
        Raw GPS observations containing a ``timestamp`` column in
        milliseconds.

    window_start_ms
        Inclusive start of the analysis window.

    window_end_ms
        Exclusive end of the analysis window.

    bin_size_ms
        Temporal sampling interval in milliseconds.

    seed
        Random seed controlling which observed row is retained inside
        each non-empty bin.

    Returns
    -------
    pd.DataFrame
        Subset of the original rows, ordered by timestamp, containing
        at most one observation per temporal bin.
    """

    window_start_ms = int(window_start_ms)
    window_end_ms = int(window_end_ms)
    bin_size_ms = int(bin_size_ms)

    if window_end_ms <= window_start_ms:
        raise ValueError(
            "window_end_ms must be greater than window_start_ms"
        )

    if bin_size_ms <= 0:
        raise ValueError(
            "bin_size_ms must be > 0"
        )

    if "timestamp" not in points.columns:
        raise ValueError(
            "points must contain timestamp"
        )

    if points.empty:
        return points.copy()

    work = points.copy()

    work["_timestamp_numeric"] = pd.to_numeric(
        work["timestamp"],
        errors="coerce",
    )

    work = work.loc[
        work["_timestamp_numeric"].notna()
        & (
            work["_timestamp_numeric"]
            >= window_start_ms
        )
        & (
            work["_timestamp_numeric"]
            < window_end_ms
        )
    ].copy()

    if work.empty:
        return points.iloc[0:0].copy()

    work["_time_bin"] = (
        (
            work["_timestamp_numeric"].astype(np.int64)
            - window_start_ms
        )
        // bin_size_ms
    )

    rng = np.random.default_rng(int(seed))

    selected_indices = []

    for _, group in work.groupby(
        "_time_bin",
        sort=True,
    ):
        positions = group.index.to_numpy()

        selected_indices.append(
            rng.choice(positions)
        )

    result = (
        work.loc[selected_indices]
        .sort_values("_timestamp_numeric")
        .drop(
            columns=[
                "_timestamp_numeric",
                "_time_bin",
            ]
        )
    )

    return result.reset_index(drop=True)

