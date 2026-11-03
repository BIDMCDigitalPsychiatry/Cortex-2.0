"""Screen-on bouts and daily screen-use features.

Construct
---------
When, and in how many separate episodes, the phone's screen was on. A *bout*
runs from an on event (``screen_on`` or ``unlock``) to the next off event
(``screen_off`` or ``lock``). The encoding is the canonical ``event`` column
that the adapter normalised; it is never re-inferred here.

Censoring, never dropping
-------------------------
- A bout already on at the window start (the last earlier event was "on",
  or the first event in the window is "off") starts at the window start and
  is ``censored`` (``left_window``).
- A bout still on at the window end is clipped and ``censored``
  (``right_window``).
- A bout longer than ``max_bout_ms`` is kept and ``censored``
  (``exceeds_max_bout``). Such bouts usually mean a missed off event, so they
  contribute at most ``max_bout_ms`` to totals. Legacy Cortex silently dropped
  screen bouts over 2 h; that biased heavy-use days downwards and is not
  reproduced.

What this is NOT
----------------
Not app usage or attention: a screen can be on with nobody looking at it.
Not evidence of *no* use when events are absent: the caller supplies
``supported_duration_ms``, the time over which the screen stream was
observably recording (see :func:`screen_supported_duration_ms`).
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cortex2_engine.features.status import (
    FeatureStatus,
    gated_value,
    status_from_counts,
)
from cortex2_engine.time import HOUR_MS, coerce_timestamp_ms, parse_clock

ON_EVENTS = frozenset({"screen_on", "unlock"})
OFF_EVENTS = frozenset({"screen_off", "lock"})

BOUT_OBSERVED = FeatureStatus.OBSERVED.value
BOUT_CENSORED = FeatureStatus.CENSORED.value

BOUT_COLUMNS = [
    "bout_start_ms",
    "bout_end_ms",
    "duration_ms",
    "counted_duration_ms",
    "bout_status",
    "censor_reason",
]


def _empty_bouts() -> pd.DataFrame:

    return pd.DataFrame({
        "bout_start_ms": pd.Series(dtype="int64"),
        "bout_end_ms": pd.Series(dtype="int64"),
        "duration_ms": pd.Series(dtype=float),
        "counted_duration_ms": pd.Series(dtype=float),
        "bout_status": pd.Series(dtype=object),
        "censor_reason": pd.Series(dtype=object),
    })


def build_screen_bouts(
    screen: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    max_bout_ms: int = 14_400_000,
    debounce_ms: int = 1000,
) -> pd.DataFrame:
    """Turn canonical screen events into screen-on bouts within a window.

    Parameters
    ----------
    screen
        Canonical screen rows (``timestamp_ms``, ``event``). Events before the
        window are used only to establish the state at the window start.
    window_start_ms, window_end_ms
        Half-open window.
    max_bout_ms
        Bouts longer than this are censored and contribute at most this much
        (default 4 h).
    debounce_ms
        Off-to-on gaps shorter than this merge two bouts into one (screen
        flicker, lock/unlock jitter).

    Returns
    -------
    pandas.DataFrame
        ``BOUT_COLUMNS``. ``duration_ms`` is the clipped bout length;
        ``counted_duration_ms`` is ``min(duration_ms, max_bout_ms)``.
    """

    window_start_ms = int(window_start_ms)
    window_end_ms = int(window_end_ms)

    if window_end_ms <= window_start_ms:
        raise ValueError(
            "window_end_ms must be greater than window_start_ms"
        )

    if max_bout_ms <= 0 or debounce_ms < 0:
        raise ValueError(
            "max_bout_ms must be > 0 and debounce_ms >= 0"
        )

    screen = coerce_timestamp_ms(
        screen,
        caller="build_screen_bouts",
    )

    if "event" not in screen.columns:
        raise ValueError("Missing required screen columns: event")

    x = pd.DataFrame({
        "t": pd.to_numeric(screen["timestamp_ms"], errors="coerce"),
        "event": screen["event"].astype("string"),
    }).dropna()

    x = x[x["event"].isin(ON_EVENTS | OFF_EVENTS)]
    x = x[x["t"] < window_end_ms]
    x = x.sort_values("t", kind="mergesort")

    before = x[x["t"] < window_start_ms]
    inside = x[x["t"] >= window_start_ms]

    on_at_start = (
        not before.empty
        and before["event"].iloc[-1] in ON_EVENTS
    )

    if (
        not on_at_start
        and before.empty
        and not inside.empty
        and inside["event"].iloc[0] in OFF_EVENTS
    ):
        # An off event with no earlier on: the screen was on before the
        # window began.
        on_at_start = True

    raw: list[list] = []
    current_start = window_start_ms if on_at_start else None
    left_censored = on_at_start

    for t, event in zip(
        inside["t"].to_numpy(dtype=np.int64),
        inside["event"].to_numpy(dtype=object),
    ):
        if event in ON_EVENTS:
            if current_start is None:
                current_start = int(t)
                left_censored = False

        elif current_start is not None:
            raw.append([current_start, int(t), left_censored, False])
            current_start = None
            left_censored = False

    if current_start is not None:
        raw.append([current_start, window_end_ms, left_censored, True])

    if not raw:
        return _empty_bouts()

    # Debounce: merge bouts separated by off-gaps shorter than debounce_ms.
    merged = [raw[0]]

    for start, end, left, right in raw[1:]:
        if start - merged[-1][1] < debounce_ms:
            merged[-1][1] = end
            merged[-1][3] = right
        else:
            merged.append([start, end, left, right])

    out = pd.DataFrame(
        merged,
        columns=["bout_start_ms", "bout_end_ms", "left", "right"],
    )

    out["duration_ms"] = (
        out["bout_end_ms"] - out["bout_start_ms"]
    ).astype(float)

    too_long = out["duration_ms"] > max_bout_ms

    reason = np.select(
        [out["left"], out["right"], too_long],
        ["left_window", "right_window", "exceeds_max_bout"],
        default=None,
    )

    out["censor_reason"] = pd.Series(reason, dtype=object).where(
        pd.Series(reason).notna(),
        None,
    )

    out["bout_status"] = np.where(
        out["censor_reason"].notna(),
        BOUT_CENSORED,
        BOUT_OBSERVED,
    )

    out["counted_duration_ms"] = np.minimum(
        out["duration_ms"],
        float(max_bout_ms),
    )

    return out[BOUT_COLUMNS].reset_index(drop=True)


def screen_supported_duration_ms(
    screen: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    max_silence_ms: int = 43_200_000,
) -> float:
    """Time in a window over which the screen stream was evidently recording.

    Screen events are event-driven, so silence is ambiguous. A stretch of
    time is *supported* when it lies between two observed screen or battery
    events (or an event and a window edge) no more than ``max_silence_ms``
    apart. The 12 h default accepts a night without events and rejects a day
    with none.

    Parameters
    ----------
    screen
        Canonical screen rows (any event, including ``unknown`` battery
        updates).
    window_start_ms, window_end_ms
        Half-open window.
    max_silence_ms
        Longest event-free gap still counted as recording.

    Returns
    -------
    float
        Supported milliseconds within the window (0 with no events).
    """

    screen = coerce_timestamp_ms(
        screen,
        caller="screen_supported_duration_ms",
    )

    t = pd.to_numeric(screen["timestamp_ms"], errors="coerce").dropna()
    t = np.sort(t[(t >= window_start_ms) & (t < window_end_ms)].to_numpy())

    if t.size == 0:
        return 0.0

    points = np.concatenate([[window_start_ms], t, [window_end_ms]])
    gaps = np.diff(points).astype(float)

    return float(gaps[gaps <= max_silence_ms].sum())


def local_clock_intervals(
    window_start_ms: int,
    window_end_ms: int,
    *,
    tz: str,
    start_local: str,
    end_local: str,
) -> list[tuple[int, int]]:
    """Recurring local clock intervals (e.g. 22:00–06:00) inside a window.

    Parameters
    ----------
    window_start_ms, window_end_ms
        Half-open window.
    tz
        IANA timezone.
    start_local, end_local
        Local clock bounds. An end earlier than the start wraps midnight.

    Returns
    -------
    list of tuple
        Half-open ``(start_ms, end_ms)`` pieces clipped to the window.
    """

    zone = ZoneInfo(tz)
    start_clock = parse_clock(start_local)
    end_clock = parse_clock(end_local)
    wraps = end_clock <= start_clock

    first = dt.datetime.fromtimestamp(window_start_ms / 1000, zone).date()
    last = dt.datetime.fromtimestamp(window_end_ms / 1000, zone).date()

    out = []
    date = first - dt.timedelta(days=1)

    while date <= last:
        lo = dt.datetime.combine(date, start_clock, tzinfo=zone)
        hi_date = date + dt.timedelta(days=1) if wraps else date
        hi = dt.datetime.combine(hi_date, end_clock, tzinfo=zone)

        a = max(round(lo.timestamp() * 1000), window_start_ms)
        b = min(round(hi.timestamp() * 1000), window_end_ms)

        if b > a:
            out.append((a, b))

        date += dt.timedelta(days=1)

    return out


def _overlap_ms(
    starts: np.ndarray,
    ends: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray:

    total = np.zeros(starts.size, dtype=float)

    for a, b in intervals:
        total += np.clip(
            np.minimum(ends, b) - np.maximum(starts, a),
            0,
            None,
        )

    return total


def compute_daily_screen_features(
    bouts: pd.DataFrame,
    *,
    supported_duration_ms: float,
    tz: str = "UTC",
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
    nocturnal_start_local: str = "22:00",
    nocturnal_end_local: str = "06:00",
    micro_bout_ms: int = 30_000,
    min_supported_ms: float = 43_200_000,
) -> dict:
    """Daily screen-use summaries for one participant-day.

    Parameters
    ----------
    bouts
        Output of :func:`build_screen_bouts` for the day.
    supported_duration_ms
        Time the screen stream was observably recording (denominator).
    tz
        IANA timezone for the nocturnal window.
    window_start_ms, window_end_ms
        Day bounds; required for nocturnal features (otherwise those are
        ``not_applicable``).
    nocturnal_start_local, nocturnal_end_local
        Local nocturnal window (22:00–06:00).
    micro_bout_ms
        Bouts shorter than this are micro-bouts (30 s), i.e. glances.
    min_supported_ms
        Minimum supported time for values to be reported (12 h).

    Returns
    -------
    dict
        ``screen_on_ms``, ``n_bouts``, ``median_bout_ms``,
        ``micro_bout_rate_per_hour`` (micro-bouts per supported hour),
        ``screen_fragmentation_index`` (bouts per screen-on hour),
        ``nocturnal_screen_on_ms``, ``nocturnal_n_bouts``, each with a
        ``_status``, plus ``n_censored_bouts`` and ``supported_duration_ms``.
        A total that includes a censored bout gets status ``censored``.
    """

    missing = set(BOUT_COLUMNS) - set(bouts.columns)

    if missing:
        raise ValueError(
            "Missing required bout columns: " + ", ".join(sorted(missing))
        )

    supported_duration_ms = float(supported_duration_ms)

    if not np.isfinite(supported_duration_ms) or supported_duration_ms < 0:
        raise ValueError("supported_duration_ms must be finite and >= 0")

    counted = bouts["counted_duration_ms"].to_numpy(dtype=float)
    duration = bouts["duration_ms"].to_numpy(dtype=float)
    censored = bouts["bout_status"].eq(BOUT_CENSORED).to_numpy()
    n_bouts = len(bouts)

    screen_on_ms = float(counted.sum())
    screen_on_hours = screen_on_ms / HOUR_MS

    raw = {
        "screen_on_ms": screen_on_ms,
        "n_bouts": float(n_bouts),
        "median_bout_ms": (
            float(np.median(duration[~censored]))
            if (~censored).any()
            else np.nan
        ),
        "micro_bout_rate_per_hour": (
            float((duration < micro_bout_ms).sum())
            / (supported_duration_ms / HOUR_MS)
            if supported_duration_ms > 0
            else np.nan
        ),
        "screen_fragmentation_index": (
            n_bouts / screen_on_hours if screen_on_hours > 0 else np.nan
        ),
    }

    has_window = window_start_ms is not None and window_end_ms is not None

    if has_window:
        night = local_clock_intervals(
            int(window_start_ms),
            int(window_end_ms),
            tz=tz,
            start_local=nocturnal_start_local,
            end_local=nocturnal_end_local,
        )

        starts = bouts["bout_start_ms"].to_numpy(dtype=np.int64)
        ends = np.minimum(
            bouts["bout_end_ms"].to_numpy(dtype=np.int64),
            starts + counted.astype(np.int64),
        )

        nocturnal = _overlap_ms(starts, ends, night)

        raw["nocturnal_screen_on_ms"] = float(nocturnal.sum())
        raw["nocturnal_n_bouts"] = float((nocturnal > 0).sum())

    else:
        raw["nocturnal_screen_on_ms"] = np.nan
        raw["nocturnal_n_bouts"] = np.nan

    any_censored = bool(censored.any())

    out: dict = {}

    for key, value in raw.items():

        status = status_from_counts(
            supported_duration_ms,
            min_supported_ms,
            value,
        )

        if key.startswith("nocturnal") and not has_window:
            status = (
                FeatureStatus.NOT_APPLICABLE
                if status
                not in {
                    FeatureStatus.INSUFFICIENT_OBSERVABILITY,
                    FeatureStatus.NO_RAW_DATA,
                }
                else status
            )

        if (
            any_censored
            and key in {"screen_on_ms", "nocturnal_screen_on_ms"}
            and status is FeatureStatus.OBSERVED
        ):
            status = FeatureStatus.CENSORED

        out[key] = gated_value(value, status)
        out[f"{key}_status"] = status.value

    out["n_censored_bouts"] = int(censored.sum())
    out["supported_duration_ms"] = supported_duration_ms

    return out
