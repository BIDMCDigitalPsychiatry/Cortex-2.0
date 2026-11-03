"""Participant-day completeness weights.

Construct
---------
A number in [0, 1] saying how much a participant-day's feature value can be
trusted, given how much of the day the source observed and how the
missing time was arranged. It is meant for weighting days in downstream
models (personal baselines, clinical alignment). It is *not* a feature and
never changes a feature's value or status.

Formula
-------
For one stream on one day, split into ``n_bins`` equal bins of
``bin_size_ms``::

    w = coverage_fraction ** alpha * (1 - longest_gap_fraction) ** beta

    longest_gap_fraction = longest_gap_ms / (n_bins * bin_size_ms)

- ``coverage_fraction`` penalises missing time (rule 1: observed time is the
  denominator of every feature, so less of it means a noisier estimate).
- ``longest_gap_fraction`` penalises concentration. With the same coverage,
  one 8 h hole can hide a whole behaviour (an evening out), while scattered
  5 min holes cannot. ``n_missing_runs`` only checks consistency
  (``0`` runs ⇔ full coverage), because the longest gap already captures
  concentration.
- The day span is ``n_bins * bin_size_ms``, so 23 h and 25 h DST days
  normalise by their real length.

``alpha = beta = 1`` by default. ``alpha = 1, beta = 0`` reduces to plain
coverage.

Several streams combine by the geometric mean (:func:`multi_stream_weight`).
One unusable stream (weight 0) then zeroes a multi-stream feature, which a
mean would hide.

Helpers build the inputs from a missing-bin mask
(:func:`completeness_weight_from_mask`, reusing
``features.missingness_structure``), from observed intervals
(:func:`completeness_weight_from_intervals`), or, when only an observed total
is known, from a conservative bound (:func:`completeness_weight_from_coverage`,
which assumes all missing time is one gap).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from cortex2_engine.features.missingness_structure import (
    compute_missingness_structure,
)


def participant_day_completeness_weight(
    coverage_fraction: float,
    longest_gap_ms: float,
    n_missing_runs: int,
    n_bins: int,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    bin_size_ms: float = 3_600_000,
) -> float:
    """Completeness weight of one stream on one participant-day.

    Parameters
    ----------
    coverage_fraction
        Observed share of the day's bins, in [0, 1].
    longest_gap_ms
        Length of the longest missing run.
    n_missing_runs
        Number of missing runs (consistency check only).
    n_bins
        Number of bins in the day.
    alpha
        Exponent on coverage (>= 0).
    beta
        Exponent on ``1 - longest_gap_fraction`` (>= 0).
    bin_size_ms
        Bin width; ``n_bins * bin_size_ms`` is the day span.

    Returns
    -------
    float
        Weight in [0, 1]; NaN when ``n_bins`` is 0 or an input is NaN.
    """

    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be >= 0")

    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be > 0")

    values = (coverage_fraction, longest_gap_ms, n_missing_runs, n_bins)

    if any(v is None or not np.isfinite(float(v)) for v in values) or n_bins <= 0:
        return float("nan")

    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError("coverage_fraction must be in [0, 1]")

    span_ms = float(n_bins) * float(bin_size_ms)

    if longest_gap_ms < 0 or longest_gap_ms > span_ms + 1e-9:
        raise ValueError("longest_gap_ms must be in [0, n_bins * bin_size_ms]")

    if n_missing_runs < 0 or n_missing_runs > n_bins:
        raise ValueError("n_missing_runs must be in [0, n_bins]")

    if (n_missing_runs == 0) != (longest_gap_ms == 0):
        raise ValueError("n_missing_runs and longest_gap_ms disagree")

    gap_fraction = min(float(longest_gap_ms) / span_ms, 1.0)

    weight = float(coverage_fraction) ** alpha * (1.0 - gap_fraction) ** beta

    return float(np.clip(weight, 0.0, 1.0))


def multi_stream_weight(weights: Mapping[str, float]) -> float:
    """Geometric mean of per-stream weights.

    Parameters
    ----------
    weights
        Stream name → weight in [0, 1]. NaN entries (stream not used) are
        ignored.

    Returns
    -------
    float
        Geometric mean; 0 if any finite weight is 0; NaN when no finite
        weight remains.
    """

    values = np.array(
        [float(v) for v in weights.values() if v is not None and np.isfinite(float(v))],
        dtype=float,
    )

    if values.size == 0:
        return float("nan")

    if np.any((values < 0) | (values > 1)):
        raise ValueError("weights must be in [0, 1]")

    if np.any(values == 0):
        return 0.0

    return float(np.exp(np.mean(np.log(values))))


def completeness_weight_from_mask(
    missing_mask: Iterable[bool],
    *,
    bin_size_ms: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Weight from a Boolean missing-bin mask (True = missing)."""

    mask = np.asarray(
        missing_mask if isinstance(missing_mask, np.ndarray) else list(missing_mask),
        dtype=bool,
    )

    if mask.size == 0:
        return float("nan")

    s = compute_missingness_structure(mask, bin_size_ms=bin_size_ms)

    return participant_day_completeness_weight(
        s["coverage"],
        s["longest_missing_ms"],
        s["missing_run_count"],
        s["n_bins"],
        alpha=alpha,
        beta=beta,
        bin_size_ms=bin_size_ms,
    )


def missing_mask_from_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    window_start_ms: int,
    window_end_ms: int,
    bin_size_ms: int = 300_000,
    min_bin_observed_fraction: float = 0.5,
) -> np.ndarray:
    """Missing-bin mask from observed half-open intervals.

    Parameters
    ----------
    intervals
        Observed ``(start_ms, end_ms)`` pairs (may overlap).
    window_start_ms, window_end_ms
        Half-open window; the last bin may be shorter.
    bin_size_ms
        Bin width (5 min).
    min_bin_observed_fraction
        Observed share of a bin for it to count as observed.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True = missing.
    """

    if window_end_ms <= window_start_ms:
        raise ValueError("window_end_ms must be after window_start_ms")

    starts = np.arange(window_start_ms, window_end_ms, bin_size_ms, dtype=float)
    ends = np.minimum(starts + bin_size_ms, window_end_ms)
    observed = np.zeros(starts.size, dtype=float)

    pairs = sorted(
        (max(float(a), window_start_ms), min(float(b), window_end_ms))
        for a, b in intervals
        if np.isfinite(a) and np.isfinite(b)
    )

    # Merge overlaps so shared time is not counted twice.
    merged: list[list[float]] = []
    for a, b in pairs:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    for a, b in merged:
        observed += np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0, None)

    return observed / (ends - starts) < min_bin_observed_fraction


def completeness_weight_from_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    window_start_ms: int,
    window_end_ms: int,
    bin_size_ms: int = 300_000,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Weight from observed intervals inside a window."""

    mask = missing_mask_from_intervals(
        intervals,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        bin_size_ms=bin_size_ms,
    )

    return completeness_weight_from_mask(
        mask, bin_size_ms=bin_size_ms, alpha=alpha, beta=beta
    )


def completeness_weight_from_coverage(
    observed_ms: float | None,
    window_ms: float,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Conservative weight when only the observed total is known.

    Missing time is assumed to form one gap (the worst arrangement), so this
    is a lower bound on :func:`participant_day_completeness_weight`.
    """

    if observed_ms is None or window_ms is None or not np.isfinite(float(observed_ms)):
        return float("nan")

    if window_ms <= 0:
        return float("nan")

    coverage = float(np.clip(float(observed_ms) / float(window_ms), 0.0, 1.0))
    gap = (1.0 - coverage) * float(window_ms)

    return participant_day_completeness_weight(
        coverage,
        gap,
        1 if gap > 0 else 0,
        1,
        alpha=alpha,
        beta=beta,
        bin_size_ms=float(window_ms),
    )


def window_weight(
    *,
    window_start_ms: int | None,
    window_end_ms: int | None,
    intervals: Iterable[tuple[float, float]] | None = None,
    observed_ms: float | None = None,
    bin_size_ms: int = 300_000,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Weight for a daily feature function's window.

    Uses ``intervals`` when given (exact arrangement), otherwise the
    conservative ``observed_ms`` bound. NaN when the window is unknown.
    """

    if window_start_ms is None or window_end_ms is None:
        return float("nan")

    if intervals is not None:
        return completeness_weight_from_intervals(
            intervals,
            window_start_ms=int(window_start_ms),
            window_end_ms=int(window_end_ms),
            bin_size_ms=bin_size_ms,
            alpha=alpha,
            beta=beta,
        )

    return completeness_weight_from_coverage(
        observed_ms,
        float(window_end_ms) - float(window_start_ms),
        alpha=alpha,
        beta=beta,
    )
