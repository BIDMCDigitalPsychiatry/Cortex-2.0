"""Phone-derived main sleep window: screen-dark run with accelerometer trimming.

Construct
---------
The participant's main nightly rest period as seen by the phone: the longest
run of screen darkness in a night window (Borger et al. 2019 used the longest
gap in touchscreen interactions). Brief screen checks inside the run are
*awakenings*, not the end of the night, and accelerometer movement at the edges
of the run trims it, so time the phone was dark but the person was moving
(for example walking around before bed) is not counted as sleep.

Algorithm (``docs/features/sleep.md``)
--------------------------------------
1. Night window: ``night_start_local`` on date D-1 to ``night_end_local`` on
   date D. The night is attributed to D, the date on which it ends.
2. Grid of ``grid_ms`` cells. A cell is *light* if any screen-on bout (or a
   SensorKit device-usage interval with unlock time) overlaps it, otherwise
   *dark*. A cell is *covered* if the accelerometer observed at least one
   epoch in it or a screen event falls in it.
3. Dark runs separated by light runs no longer than ``wake_tolerance_ms``
   are merged; each merged light run is one awakening. The longest merged run
   is the candidate sleep period.
4. Trimming: the onset moves forward past leading cells whose accelerometer
   ENMO is observed and >= ``still_threshold_mg``; the offset moves back past
   trailing moving cells. Unobserved accelerometer cells do not trim.
5. Gating: ``night_coverage_fraction`` (covered cells / night cells) must be
   >= ``min_coverage_fraction``. A candidate touching the night-window edge is
   ``censored``.

What this is NOT
----------------
Not polysomnographic or actigraphic sleep. It estimates a *phone-rest
window*: phone dark and body still. A participant who sleeps with the screen
on, or whose phone is elsewhere, is misestimated. Wearable sleep is a separate
feature (``wearable_sleep``).
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.quality.completeness import completeness_weight_from_coverage
from cortex2_engine.time import (
    parse_clock,
    to_local_clock_hours,
)

SLEEP_KEYS = [
    "sleep_onset_local_h",
    "sleep_offset_local_h",
    "sleep_duration_ms",
    "sleep_midpoint_local_h",
    "n_awakenings",
]


def night_window_ms(
    night_date: dt.date | str,
    tz: str,
    *,
    night_start_local: str = "18:00",
    night_end_local: str = "12:00",
) -> tuple[int, int]:
    """Bounds of the night that ends on ``night_date``.

    Parameters
    ----------
    night_date
        Local date on which the night ends (attribution date).
    tz
        IANA timezone.
    night_start_local, night_end_local
        Local clock bounds; the start is on the previous day.

    Returns
    -------
    tuple of int
        ``(start_ms, end_ms)``.
    """

    if isinstance(night_date, str):
        night_date = dt.date.fromisoformat(night_date)

    zone = ZoneInfo(tz)

    start = dt.datetime.combine(
        night_date - dt.timedelta(days=1),
        parse_clock(night_start_local),
        tzinfo=zone,
    )

    end = dt.datetime.combine(
        night_date,
        parse_clock(night_end_local),
        tzinfo=zone,
    )

    return round(start.timestamp() * 1000), round(end.timestamp() * 1000)


def _cells_overlapping(
    starts: np.ndarray,
    ends: np.ndarray,
    origin: int,
    grid_ms: int,
    n_cells: int,
) -> np.ndarray:

    flags = np.zeros(n_cells, dtype=bool)

    for a, b in zip(starts, ends):
        if b <= origin or a >= origin + n_cells * grid_ms:
            continue

        lo = max(0, (int(a) - origin) // grid_ms)
        hi = min(n_cells, -(-(int(b) - origin) // grid_ms))

        flags[lo:hi] = True

    return flags


def _runs(
    flags: np.ndarray,
) -> list[tuple[int, int]]:
    """Half-open [start, end) index runs where ``flags`` is True."""

    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))

    return list(
        zip(
            np.flatnonzero(edges == 1).tolist(),
            np.flatnonzero(edges == -1).tolist(),
        )
    )


def _empty_result(
    status: FeatureStatus,
    coverage: float,
    night_start_ms: int,
    night_end_ms: int,
) -> dict:

    out = {}

    for key in SLEEP_KEYS:
        out[key] = np.nan
        out[f"{key}_status"] = status.value

    if status is FeatureStatus.OBSERVED_ZERO:
        # Well-observed night with no dark period: no sleep window at all.
        out["sleep_duration_ms"] = 0.0
        out["n_awakenings"] = 0

        for key in (
            "sleep_onset_local_h",
            "sleep_offset_local_h",
            "sleep_midpoint_local_h",
        ):
            out[f"{key}_status"] = FeatureStatus.NOT_APPLICABLE.value

    out.update({
        "sleep_onset_ms": np.nan,
        "sleep_offset_ms": np.nan,
        "sleep_status": status.value,
        "night_coverage_fraction": coverage,
        "night_start_ms": night_start_ms,
        "night_end_ms": night_end_ms,
        "completeness_weight": completeness_weight_from_coverage(
            coverage * (night_end_ms - night_start_ms),
            night_end_ms - night_start_ms,
        ),
    })

    return out


def detect_sleep_window(
    screen_bouts: pd.DataFrame,
    acc_epochs: pd.DataFrame,
    *,
    night_date: dt.date | str,
    tz: str,
    night_start_local: str = "18:00",
    night_end_local: str = "12:00",
    grid_ms: int = 300_000,
    wake_tolerance_ms: int = 1_200_000,
    still_threshold_mg: float = 10.0,
    min_coverage_fraction: float = 0.7,
    screen_events: pd.DataFrame | None = None,
    device_usage: pd.DataFrame | None = None,
) -> dict:
    """Estimate the main sleep window for the night ending on ``night_date``.

    Parameters
    ----------
    screen_bouts
        Output of ``screen_bouts.build_screen_bouts`` covering the night.
    acc_epochs
        Accelerometer epochs with ``epoch_start_ms``, ``epoch_end_ms``,
        ``observed`` and ``enmo_mg`` (``resample_to_epochs`` or
        ``aggregate_epochs`` output; non-wear epochs count as still).
    night_date
        Local date the night ends on.
    tz
        IANA timezone.
    night_start_local, night_end_local
        Night search window (18:00 previous day to 12:00).
    grid_ms
        Grid resolution (5 min).
    wake_tolerance_ms
        Longest light interruption treated as an awakening rather than the
        end of the night (20 min).
    still_threshold_mg
        ENMO below which an accelerometer cell is still (10 mg).
    min_coverage_fraction
        Minimum share of covered night cells (0.7).
    screen_events
        Optional raw screen rows; any event marks its cell as covered.
    device_usage
        Optional SensorKit device-usage rows (``timestamp_ms``,
        ``interval_duration_ms``, ``total_unlock_duration_ms``); intervals
        with unlock time are light.

    Returns
    -------
    dict
        ``sleep_onset_local_h``, ``sleep_offset_local_h`` and
        ``sleep_midpoint_local_h`` (clock hours, [0, 24)),
        ``sleep_duration_ms``, ``n_awakenings``, ``sleep_onset_ms``,
        ``sleep_offset_ms``, ``sleep_status``, ``night_coverage_fraction``
        and the night bounds. Every value in ``SLEEP_KEYS`` also has a
        ``_status`` (equal to ``sleep_status`` except on a night with no dark
        period, where duration is ``observed_zero`` and clock times are
        ``not_applicable``).
    """

    if grid_ms <= 0 or wake_tolerance_ms < 0:
        raise ValueError("grid_ms must be > 0 and wake_tolerance_ms >= 0")

    if not 0.0 <= min_coverage_fraction <= 1.0:
        raise ValueError("min_coverage_fraction must be in [0, 1]")

    for name, frame, columns in (
        ("screen_bouts", screen_bouts, {"bout_start_ms", "bout_end_ms"}),
        (
            "acc_epochs",
            acc_epochs,
            {"epoch_start_ms", "epoch_end_ms", "observed", "enmo_mg"},
        ),
    ):
        missing = columns - set(frame.columns)

        if missing:
            raise ValueError(
                f"Missing required {name} columns: "
                + ", ".join(sorted(missing))
            )

    night_start_ms, night_end_ms = night_window_ms(
        night_date,
        tz,
        night_start_local=night_start_local,
        night_end_local=night_end_local,
    )

    n_cells = -(-(night_end_ms - night_start_ms) // grid_ms)

    light = _cells_overlapping(
        screen_bouts["bout_start_ms"].to_numpy(dtype=np.int64),
        screen_bouts["bout_end_ms"].to_numpy(dtype=np.int64),
        night_start_ms,
        grid_ms,
        n_cells,
    )

    if device_usage is not None and not device_usage.empty:
        used = device_usage[
            pd.to_numeric(
                device_usage.get("total_unlock_duration_ms"),
                errors="coerce",
            ).fillna(0)
            > 0
        ]

        starts = pd.to_numeric(used["timestamp_ms"], errors="coerce")
        lengths = pd.to_numeric(used["interval_duration_ms"], errors="coerce")

        light |= _cells_overlapping(
            starts.to_numpy(dtype=np.int64),
            (starts + lengths.fillna(0)).to_numpy(dtype=np.int64),
            night_start_ms,
            grid_ms,
            n_cells,
        )

    # Accelerometer per cell: covered, and moving (observed & ENMO >= thr).
    e_start = acc_epochs["epoch_start_ms"].to_numpy(dtype=np.int64)
    in_night = (e_start >= night_start_ms) & (e_start < night_end_ms)
    cell = (e_start[in_night] - night_start_ms) // grid_ms
    observed = acc_epochs["observed"].to_numpy(dtype=bool)[in_night]
    enmo = acc_epochs["enmo_mg"].to_numpy(dtype=float)[in_night]

    covered = np.zeros(n_cells, dtype=bool)
    covered[cell[observed]] = True

    enmo_sum = np.bincount(
        cell[observed],
        weights=np.nan_to_num(enmo[observed]),
        minlength=n_cells,
    )
    enmo_n = np.bincount(cell[observed], minlength=n_cells)

    with np.errstate(invalid="ignore", divide="ignore"):
        cell_enmo = np.where(enmo_n > 0, enmo_sum / enmo_n, np.nan)

    moving = np.isfinite(cell_enmo) & (cell_enmo >= still_threshold_mg)

    covered |= light

    if screen_events is not None and not screen_events.empty:
        t = pd.to_numeric(
            screen_events["timestamp_ms"],
            errors="coerce",
        ).dropna().to_numpy(dtype=np.int64)

        t = t[(t >= night_start_ms) & (t < night_end_ms)]
        covered[(t - night_start_ms) // grid_ms] = True

    coverage = float(covered.mean()) if n_cells else 0.0

    if not covered.any():
        return _empty_result(
            FeatureStatus.NO_RAW_DATA,
            coverage,
            night_start_ms,
            night_end_ms,
        )

    if coverage < min_coverage_fraction:
        return _empty_result(
            FeatureStatus.INSUFFICIENT_OBSERVABILITY,
            coverage,
            night_start_ms,
            night_end_ms,
        )

    dark_runs = _runs(~light)

    if not dark_runs:
        return _empty_result(
            FeatureStatus.OBSERVED_ZERO,
            coverage,
            night_start_ms,
            night_end_ms,
        )

    tolerance_cells = wake_tolerance_ms // grid_ms

    merged = [[*dark_runs[0], 0]]

    for start, end in dark_runs[1:]:
        if start - merged[-1][1] <= tolerance_cells:
            merged[-1][1] = end
            merged[-1][2] += 1
        else:
            merged.append([start, end, 0])

    lengths = [end - start for start, end, _ in merged]
    best = int(np.argmax(lengths))
    start, end, n_awakenings = merged[best]

    while start < end and moving[start]:
        start += 1

    while end > start and moving[end - 1]:
        end -= 1

    if end <= start:
        return _empty_result(
            FeatureStatus.OBSERVED_ZERO,
            coverage,
            night_start_ms,
            night_end_ms,
        )

    # Awakenings that the trimming cut off no longer count.
    inside_light = _runs(light[start:end])
    n_awakenings = min(n_awakenings, len(inside_light))

    onset_ms = night_start_ms + start * grid_ms
    offset_ms = min(night_start_ms + end * grid_ms, night_end_ms)

    status = (
        FeatureStatus.CENSORED
        if start == 0 or end == n_cells
        else FeatureStatus.OBSERVED
    )

    onset_h, offset_h = to_local_clock_hours([onset_ms, offset_ms], tz)
    midpoint_h = to_local_clock_hours([(onset_ms + offset_ms) // 2], tz)[0]

    out = {
        "sleep_onset_local_h": float(onset_h),
        "sleep_offset_local_h": float(offset_h),
        "sleep_duration_ms": float(offset_ms - onset_ms),
        "sleep_midpoint_local_h": float(midpoint_h),
        "n_awakenings": int(n_awakenings),
        "sleep_onset_ms": int(onset_ms),
        "sleep_offset_ms": int(offset_ms),
        "sleep_status": status.value,
        "night_coverage_fraction": coverage,
        "night_start_ms": night_start_ms,
        "night_end_ms": night_end_ms,
        "completeness_weight": completeness_weight_from_coverage(
            coverage * (night_end_ms - night_start_ms),
            night_end_ms - night_start_ms,
        ),
    }

    for key in SLEEP_KEYS:
        out[f"{key}_status"] = status.value

    return out
