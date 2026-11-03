"""Wearable / platform-reported sleep (for example HealthKit sleep analysis).

Construct
---------
Sleep as classified by a wearable or the phone platform, summarised per night
and attributed to the local date on which each segment *ends* (the same rule as
``sleep_window``). Segments come from the canonical ``sleep`` stream:
``representation`` in {in_bed, in_sleep, in_awake}, ``duration_ms``.

What this is NOT
----------------
Not the phone-derived ``sleep_window``; the two are separate features so they
can be compared. Not validated sleep staging: in_sleep merges every asleep
stage. Overlapping segments from several sources are not de-duplicated, so the
caller should pass one source's segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.time import coerce_timestamp_ms

WEARABLE_SLEEP_KEYS = [
    "healthkit_sleep_duration_ms",
    "healthkit_in_bed_ms",
    "healthkit_awake_ms",
]


def _merged_length_ms(
    starts: np.ndarray,
    ends: np.ndarray,
) -> float:
    """Total length of the union of intervals (overlaps counted once)."""

    if starts.size == 0:
        return 0.0

    order = np.argsort(starts, kind="mergesort")
    total = 0.0
    cur_start, cur_end = starts[order[0]], ends[order[0]]

    for i in order[1:]:
        if starts[i] <= cur_end:
            cur_end = max(cur_end, ends[i])
        else:
            total += cur_end - cur_start
            cur_start, cur_end = starts[i], ends[i]

    return float(total + cur_end - cur_start)


def compute_wearable_sleep(
    sleep: pd.DataFrame,
    *,
    night_start_ms: int,
    night_end_ms: int,
    source_available: bool = True,
) -> dict:
    """Summarise reported sleep segments ending within one night window.

    Parameters
    ----------
    sleep
        Canonical sleep rows (``timestamp_ms`` = segment start,
        ``representation``, ``duration_ms``).
    night_start_ms, night_end_ms
        Night window (``sleep_window.night_window_ms``); segments whose end
        falls in ``[night_start_ms, night_end_ms)`` are attributed to it.
    source_available
        False when the episode has no sleep source at all (no wearable / not
        exported), which yields ``source_absent`` rather than
        ``no_raw_data``.

    Returns
    -------
    dict
        ``WEARABLE_SLEEP_KEYS`` with ``_status``, plus
        ``healthkit_n_segments`` and ``healthkit_n_unknown_segments``.
        Each total is the union length of its segments (overlaps once). A
        category with no segments that night is ``not_applicable``.
    """

    if night_end_ms <= night_start_ms:
        raise ValueError("night_end_ms must be greater than night_start_ms")

    if not source_available:
        out = {}

        for key in WEARABLE_SLEEP_KEYS:
            out[key] = np.nan
            out[f"{key}_status"] = FeatureStatus.SOURCE_ABSENT.value

        out["healthkit_n_segments"] = 0
        out["healthkit_n_unknown_segments"] = 0

        return out

    sleep = coerce_timestamp_ms(sleep, caller="compute_wearable_sleep")

    missing = {"representation", "duration_ms"} - set(sleep.columns)

    if missing:
        raise ValueError(
            "Missing required sleep columns: " + ", ".join(sorted(missing))
        )

    start = pd.to_numeric(sleep["timestamp_ms"], errors="coerce")
    end = start + pd.to_numeric(sleep["duration_ms"], errors="coerce")

    inside = end.notna() & (end > night_start_ms) & (end <= night_end_ms)
    segments = sleep.loc[inside].assign(_start=start[inside], _end=end[inside])

    n_segments = len(segments)

    status = (
        FeatureStatus.NO_RAW_DATA
        if n_segments == 0
        else FeatureStatus.OBSERVED
    )

    out = {}

    for key, representation in zip(
        WEARABLE_SLEEP_KEYS,
        ("in_sleep", "in_bed", "in_awake"),
    ):
        rows = segments[segments["representation"].eq(representation)]

        value = _merged_length_ms(
            rows["_start"].to_numpy(dtype=float),
            rows["_end"].to_numpy(dtype=float),
        )

        key_status = status

        # Many devices never report some categories (e.g. only "asleep"), so
        # a category absent from the night is undefined, not a zero.
        if status is FeatureStatus.OBSERVED and rows.empty:
            key_status = FeatureStatus.NOT_APPLICABLE

        out[key] = (
            value if key_status is FeatureStatus.OBSERVED else np.nan
        )
        out[f"{key}_status"] = key_status.value

    out["healthkit_n_segments"] = n_segments
    out["healthkit_n_unknown_segments"] = int(
        segments["representation"].eq("unknown").sum()
    )

    return out
