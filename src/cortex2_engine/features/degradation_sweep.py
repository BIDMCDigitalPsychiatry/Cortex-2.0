"""Degradation sweeps: at what coverage does a daily feature stop being reliable?

Construct
---------
For a daily feature, the lowest data coverage at which its error from
missing data stays smaller than the day-to-day variation it is meant to
detect. Below that coverage, a change in the feature is more likely to be
missingness than behaviour.

Procedure
---------
1. **Reference days** are well-observed participant-days supplied by the
   caller (:class:`ReferenceDay`: canonical rows per stream plus the window).
   This module never chooses them; a threshold is only as good as the
   reference set.
2. For each coverage level, run-count and replicate, a mask from
   ``missingness_simulation.make_balanced_k_run_missing_bin_mask`` (exact
   coverage, exactly ``n_runs`` missing runs, random placement, seeded) is
   applied to every degraded stream with
   ``gps_degradation.apply_missing_bin_mask_to_points``. The same mask is
   used for every stream of the day, so a degraded day loses the same hours
   in every stream.
3. The caller's ``feature_fn`` is evaluated on the intact and the degraded
   rows, and ``abs_error = |degraded - reference|`` is recorded.
4. :func:`summarise_sweep` gives, per feature × coverage (× runs), the median
   absolute error over pairs where both values exist (``median_abs_error``),
   and the share of degraded days without a value
   (``fraction_unavailable``). A degraded day that becomes
   ``insufficient_observability`` is the status system working correctly,
   not an error, so it is reported separately rather than counted as error.
5. ``within_person_mad`` is, per participant, the MAD of the reference values
   across their reference days (at least ``min_days_per_participant``), then
   the median across participants.
6. :func:`reliability_thresholds` reports, per feature, the highest tested
   coverage at which ``median_abs_error > within_person_mad``
   (``failure_coverage``), and the lowest coverage above it at which every
   tested level passes (``min_reliable_coverage``). Run-count designs are
   taken at their worst case, the largest error at each coverage.

Registry
--------
:func:`write_reliability_thresholds` writes ``min_reliable_coverage`` into
``reliability_threshold`` in ``feature_registry/features.yaml``, and records
the reference set as ``reliability_reference_set``. It refuses to run
without a named reference set. No reference set ships with the repository,
so every ``reliability_threshold`` stays ``null``.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_degradation import apply_missing_bin_mask_to_points
from cortex2_engine.features.missingness_simulation import (
    make_balanced_k_run_missing_bin_mask,
)
from cortex2_engine.features.status import VALUE_BEARING_STATUSES

FeatureFn = Callable[[Mapping[str, pd.DataFrame], int, int], Mapping[str, float]]

_BEARING = {s.value for s in VALUE_BEARING_STATUSES}


@dataclasses.dataclass
class ReferenceDay:
    """One well-observed participant-day used as ground truth.

    Attributes
    ----------
    participant_id
        Participant identifier (for the within-person MAD).
    streams
        Stream name → canonical rows with ``timestamp_ms``.
    window_start_ms, window_end_ms
        Half-open day bounds.
    """

    participant_id: str
    streams: dict[str, pd.DataFrame]
    window_start_ms: int
    window_end_ms: int


def _feature_value(result: Mapping[str, float], feature: str) -> float:
    """Value when its status is value-bearing (or has no status), else NaN."""

    status = result.get(f"{feature}_status")
    value = result.get(feature, np.nan)

    if status is not None and status not in _BEARING:
        return float("nan")

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def degrade_streams(
    streams: Mapping[str, pd.DataFrame],
    mask: np.ndarray,
    *,
    window_start_ms: int,
    window_end_ms: int,
    bin_size_ms: int,
    degrade: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Drop rows inside masked bins from the selected streams.

    Parameters
    ----------
    streams
        Stream name → canonical rows.
    mask
        Boolean missing-bin mask over the window.
    window_start_ms, window_end_ms, bin_size_ms
        Mask geometry.
    degrade
        Streams to degrade (default: all).

    Returns
    -------
    dict
        New stream mapping; streams not degraded are passed through.
    """

    names = set(streams) if degrade is None else set(degrade)
    out = {}

    for name, rows in streams.items():
        if name in names and len(rows):
            out[name] = apply_missing_bin_mask_to_points(
                rows,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                bin_size_ms=bin_size_ms,
                missing_mask=mask,
            )
        else:
            out[name] = rows

    return out


def run_degradation_sweep(
    reference_days: Sequence[ReferenceDay],
    feature_fn: FeatureFn,
    features: Sequence[str],
    *,
    coverages: Sequence[float] = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
    n_runs: Sequence[int] = (1, 4),
    n_replicates: int = 5,
    bin_size_ms: int = 300_000,
    degrade: Iterable[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate ``feature_fn`` on intact and masked copies of reference days.

    Parameters
    ----------
    reference_days
        Ground-truth days.
    feature_fn
        ``feature_fn(streams, window_start_ms, window_end_ms) -> dict``, for
        example a wrapper around ``compute_daily_activity_features``.
    features
        Keys of the returned dict to evaluate.
    coverages
        Target coverages in (0, 1).
    n_runs
        Missing-run counts (1 = one contiguous block; more = scattered).
    n_replicates
        Random placements per design cell.
    bin_size_ms
        Mask bin width (5 min).
    degrade
        Streams to mask (default: all supplied streams).
    seed
        Base seed; each cell's seed is derived deterministically.

    Returns
    -------
    pandas.DataFrame
        One row per day × feature × coverage × runs × replicate:
        ``participant_id``, ``day_index``, ``feature``, ``coverage``,
        ``n_runs``, ``replicate``, ``reference_value``, ``degraded_value``
        and ``abs_error`` (NaN when either value is missing).
        Infeasible run counts for a coverage (more runs than bins allow)
        are skipped.
    """

    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    if not features:
        raise ValueError("features must not be empty")

    rows = []

    for day_index, day in enumerate(reference_days):
        reference = feature_fn(day.streams, day.window_start_ms, day.window_end_ms)
        ref_values = {f: _feature_value(reference, f) for f in features}

        for ci, coverage in enumerate(coverages):
            for runs in n_runs:
                for replicate in range(n_replicates):
                    cell_seed = int(
                        seed * 1_000_003 + day_index * 10_007 + ci * 1_009 + runs * 101 + replicate
                    )

                    try:
                        mask, _ = make_balanced_k_run_missing_bin_mask(
                            window_start_ms=day.window_start_ms,
                            window_end_ms=day.window_end_ms,
                            target_coverage=float(coverage),
                            n_runs=int(runs),
                            bin_size_ms=bin_size_ms,
                            seed=cell_seed,
                        )
                    except ValueError:
                        continue

                    degraded = feature_fn(
                        degrade_streams(
                            day.streams,
                            mask,
                            window_start_ms=day.window_start_ms,
                            window_end_ms=day.window_end_ms,
                            bin_size_ms=bin_size_ms,
                            degrade=degrade,
                        ),
                        day.window_start_ms,
                        day.window_end_ms,
                    )

                    for feature in features:
                        ref = ref_values[feature]
                        deg = _feature_value(degraded, feature)
                        rows.append({
                            "participant_id": day.participant_id,
                            "day_index": day_index,
                            "feature": feature,
                            "coverage": float(coverage),
                            "n_runs": int(runs),
                            "replicate": replicate,
                            "reference_value": ref,
                            "degraded_value": deg,
                            "abs_error": abs(deg - ref)
                            if np.isfinite(ref) and np.isfinite(deg)
                            else np.nan,
                        })

    return pd.DataFrame(rows)


def within_person_mad(
    sweep: pd.DataFrame,
    *,
    min_days_per_participant: int = 3,
) -> pd.Series:
    """Median over participants of each participant's reference-value MAD.

    Returns
    -------
    pandas.Series
        Indexed by feature; NaN when no participant has enough days.
    """

    ref = sweep.drop_duplicates(["participant_id", "day_index", "feature"])
    ref = ref[np.isfinite(ref["reference_value"])]

    out = {}

    for feature, group in ref.groupby("feature"):
        mads = []
        for _, person in group.groupby("participant_id"):
            v = person["reference_value"].to_numpy(float)
            if v.size >= min_days_per_participant:
                mads.append(float(np.median(np.abs(v - np.median(v)))))
        out[feature] = float(np.median(mads)) if mads else np.nan

    return pd.Series(out, name="within_person_mad", dtype=float)


def summarise_sweep(
    sweep: pd.DataFrame,
    *,
    min_days_per_participant: int = 3,
) -> pd.DataFrame:
    """Median absolute error per feature × coverage × runs, against the MAD.

    Returns
    -------
    pandas.DataFrame
        ``feature``, ``coverage``, ``n_runs``, ``n_pairs``,
        ``median_abs_error``, ``fraction_unavailable``,
        ``within_person_mad`` and ``exceeds_mad``.
    """

    mad = within_person_mad(sweep, min_days_per_participant=min_days_per_participant)

    rows = []

    for (feature, coverage, runs), g in sweep.groupby(
        ["feature", "coverage", "n_runs"], sort=True
    ):
        has_ref = np.isfinite(g["reference_value"])
        errors = g.loc[has_ref, "abs_error"].dropna()
        n_ref = int(has_ref.sum())

        median_error = float(errors.median()) if len(errors) else np.nan
        threshold = mad.get(feature, np.nan)

        rows.append({
            "feature": feature,
            "coverage": coverage,
            "n_runs": runs,
            "n_pairs": len(errors),
            "median_abs_error": median_error,
            "fraction_unavailable": 1.0 - len(errors) / n_ref if n_ref else np.nan,
            "within_person_mad": threshold,
            "exceeds_mad": bool(
                np.isfinite(median_error)
                and np.isfinite(threshold)
                and median_error > threshold
            ),
        })

    return pd.DataFrame(rows)


def reliability_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    """Coverage at which each feature's error first exceeds within-person MAD.

    Returns
    -------
    pandas.DataFrame
        One row per feature: ``failure_coverage`` (highest tested coverage
        where the worst run design exceeds the MAD; NaN if none),
        ``min_reliable_coverage`` (the lowest tested coverage above every
        failure; NaN if even the highest level fails or the MAD is
        undefined) and ``within_person_mad``.
    """

    rows = []

    for feature, g in summary.groupby("feature", sort=True):
        worst = g.groupby("coverage")["exceeds_mad"].any().sort_index(ascending=False)
        mad = float(g["within_person_mad"].iloc[0])
        failing = worst[worst].index

        failure = float(failing.max()) if len(failing) else np.nan
        passing_above = worst.index[worst.index > failure] if len(failing) else worst.index

        rows.append({
            "feature": feature,
            "failure_coverage": failure,
            "min_reliable_coverage": (
                float(passing_above.min())
                if len(passing_above) and np.isfinite(mad)
                else np.nan
            ),
            "within_person_mad": mad,
        })

    return pd.DataFrame(rows)


def write_reliability_thresholds(
    thresholds: Mapping[str, float],
    registry_path: str | Path,
    *,
    reference_set: str | None,
) -> list[str]:
    """Write ``reliability_threshold`` values into the registry YAML in place.

    Only the ``reliability_threshold`` line of each named entry is changed
    (comments and layout are kept), and ``reliability_reference_set`` is
    added beneath it.

    Parameters
    ----------
    thresholds
        Feature → ``min_reliable_coverage``. NaN values are skipped.
    registry_path
        ``feature_registry/features.yaml``.
    reference_set
        Name of the reference set the thresholds came from. Required.

    Returns
    -------
    list of str
        Features updated.
    """

    if not reference_set:
        raise ValueError(
            "reference_set is required: thresholds are only written for a "
            "named reference set; otherwise they stay null."
        )

    path = Path(registry_path)
    lines = path.read_text().splitlines(keepends=True)
    updated = []

    for feature, value in thresholds.items():
        if value is None or not np.isfinite(value):
            continue

        header = re.compile(rf"^  {re.escape(feature)}:\s*$")
        start = next((i for i, line in enumerate(lines) if header.match(line)), None)

        if start is None:
            raise ValueError(f"feature {feature!r} not in registry")

        end = next(
            (i for i in range(start + 1, len(lines)) if re.match(r"^  \S", lines[i])),
            len(lines),
        )

        block = range(start + 1, end)
        target = next(
            (i for i in block if lines[i].startswith("    reliability_threshold:")), None
        )

        if target is None:
            raise ValueError(f"feature {feature!r} has no reliability_threshold field")

        new = [
            f"    reliability_threshold: {float(value):.4g}\n",
            f"    reliability_reference_set: {reference_set}\n",
        ]

        # Drop an earlier reference-set line, then replace the threshold.
        if target + 1 < end and lines[target + 1].startswith("    reliability_reference_set:"):
            del lines[target + 1]

        lines[target:target + 1] = new
        updated.append(feature)

    path.write_text("".join(lines))

    return updated
