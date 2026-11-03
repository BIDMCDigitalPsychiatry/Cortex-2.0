"""Daily behavioural anchors: clock times of recurring daily events.

Construct
---------
An *anchor* is the local clock time of an event that recurs once a day and
structures it (Monk et al. 1990, Social Rhythm Metric). Five passive anchors
are built:

| Anchor | Source | Event |
|---|---|---|
| ``sleep_offset`` | ``sleep_window.detect_sleep_window`` | end of the phone sleep window |
| ``first_unlock`` | screen events | first ``unlock`` (else ``screen_on``) |
| ``first_home_departure`` | GPS edges + home record | first supported edge leaving home |
| ``first_communication`` | telephony | first call or SMS |
| ``last_communication`` | telephony | last call or SMS |

All except ``sleep_offset`` are searched in the *anchor day*
``[date + earliest_local, date + 1 day + earliest_local)``, by default
04:00–04:00. A 01:00 unlock therefore belongs to the previous evening rather
than becoming the "first unlock" of the new day. ``sleep_offset`` comes from
the night attributed to ``date`` (its end date).

Output is long format, one row per anchor that occurred, with
``clock_hours`` in [0, 24). Anchors that did not occur or could not be
observed are simply absent. Regularity is computed from the rows that exist
(``social_rhythm``), never by imputing a time.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_geometry import haversine_m
from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.time import local_day_bounds_ms, parse_clock, to_local_clock_hours

ANCHOR_NAMES = [
    "sleep_offset",
    "first_unlock",
    "first_home_departure",
    "first_communication",
    "last_communication",
]

ANCHOR_COLUMNS = ["date_local", "anchor", "timestamp_ms", "clock_hours"]


def anchor_day_bounds_ms(
    date: dt.date | str,
    tz: str,
    *,
    earliest_local: str = "04:00",
) -> tuple[int, int]:
    """Half-open anchor-day bounds ``[date + earliest, date + 1 + earliest)``."""

    return local_day_bounds_ms(date, tz, earliest_local)


def _in(ts: pd.Series, start: int, end: int) -> pd.Series:

    t = pd.to_numeric(ts, errors="coerce")
    return (t >= start) & (t < end)


def build_daily_anchors(
    *,
    date: dt.date | str,
    tz: str,
    sleep: dict | None = None,
    screen_events: pd.DataFrame | None = None,
    gps_edges: pd.DataFrame | None = None,
    home: dict | None = None,
    telephony: pd.DataFrame | None = None,
    earliest_local: str = "04:00",
) -> pd.DataFrame:
    """Build the anchors of one local day.

    Parameters
    ----------
    date
        Local date.
    tz
        IANA timezone.
    sleep
        ``detect_sleep_window`` output for the night ending on ``date``.
    screen_events
        Canonical screen rows (``timestamp_ms``, ``event``).
    gps_edges
        ``build_gps_intervals`` edges covering the anchor day.
    home
        ``infer_home_location`` record; departures need ``observed`` status.
    telephony
        Canonical telephony rows.
    earliest_local
        Start of the anchor day (04:00).

    Returns
    -------
    pandas.DataFrame
        ``ANCHOR_COLUMNS``, one row per anchor that occurred.
    """

    parse_clock(earliest_local)
    day = date if isinstance(date, dt.date) else dt.date.fromisoformat(date)
    start, end = anchor_day_bounds_ms(day, tz, earliest_local=earliest_local)

    found: dict[str, int] = {}

    if sleep is not None:
        offset = sleep.get("sleep_offset_ms", np.nan)
        offset_status = sleep.get("sleep_offset_local_h_status")

        if (
            offset is not None
            and np.isfinite(offset)
            and offset_status
            in (None, FeatureStatus.OBSERVED.value, FeatureStatus.CENSORED.value)
        ):
            found["sleep_offset"] = int(offset)

    if screen_events is not None and not screen_events.empty:
        s = screen_events[_in(screen_events["timestamp_ms"], start, end)]
        events = s["event"].astype("string").str.lower()

        for name in ("unlock", "screen_on"):
            hits = pd.to_numeric(s.loc[(events == name).fillna(False).to_numpy(), "timestamp_ms"])
            if len(hits):
                found["first_unlock"] = int(hits.min())
                break

    if (
        gps_edges is not None
        and not gps_edges.empty
        and home is not None
        and home.get("home_status") == FeatureStatus.OBSERVED.value
    ):
        e = gps_edges[
            gps_edges["supported"].fillna(False).astype(bool)
            & _in(gps_edges["start_ms"], start, end)
        ]

        if len(e):
            radius = float(home["home_radius_m"])
            inside_start = haversine_m(
                home["home_latitude"], home["home_longitude"],
                e["start_latitude"].to_numpy(float), e["start_longitude"].to_numpy(float),
            ) <= radius
            inside_end = haversine_m(
                home["home_latitude"], home["home_longitude"],
                e["end_latitude"].to_numpy(float), e["end_longitude"].to_numpy(float),
            ) <= radius

            leaving = inside_start & ~inside_end

            if leaving.any():
                found["first_home_departure"] = int(
                    pd.to_numeric(e["start_ms"]).to_numpy()[leaving].min()
                )

    if telephony is not None and not telephony.empty:
        c = telephony[_in(telephony["timestamp_ms"], start, end)]

        if "channel" in c.columns:
            c = c[c["channel"].astype("string").isin(["call", "sms"]).fillna(False).to_numpy()]

        if len(c):
            ts = pd.to_numeric(c["timestamp_ms"])
            found["first_communication"] = int(ts.min())
            found["last_communication"] = int(ts.max())

    rows = [
        {"date_local": day.isoformat(), "anchor": name, "timestamp_ms": found[name]}
        for name in ANCHOR_NAMES
        if name in found
    ]

    out = pd.DataFrame(rows, columns=ANCHOR_COLUMNS[:3])
    out["clock_hours"] = (
        to_local_clock_hours(out["timestamp_ms"].to_numpy(), tz)
        if len(out)
        else pd.Series(dtype=float)
    )

    return out[ANCHOR_COLUMNS]
