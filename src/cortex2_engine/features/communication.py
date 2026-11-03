"""Call and SMS communication features from the canonical telephony stream.

Construct
---------
How much, with how many people, and how evenly the participant communicated
by phone call and SMS over a window (one local day, or seven).

Counterparts are identified only by ``contact_hash`` (study-scoped
pseudonymous hash). Raw numbers never reach this module. Rows without a hash
count towards totals but not towards contact-based measures.

Features
--------
For each window:

- ``call_count``, ``sms_count`` and their by-direction variants
  (``call_count_incoming`` / ``_outgoing`` / ``_missed``,
  ``sms_count_incoming`` / ``_outgoing``);
- ``call_duration_ms`` (and ``_incoming`` / ``_outgoing``), summed over
  connected calls;
- ``n_unique_contacts`` — distinct hashed counterparts;
- ``communication_reciprocity`` — share of contacts with both incoming
  (including missed) and outgoing events;
- ``communication_entropy_nats`` — Shannon entropy (natural log) of events
  per contact; 0 when all communication is with one person;
- ``outgoing_fraction`` — outgoing / (incoming + outgoing) events, missed and
  unknown excluded.

Observability
-------------
Telephony is event-driven: a day without rows is ambiguous. Counts are
``observed_zero`` only when the caller shows the logger was running, that is
``stream_active_in_episode=True`` (the participant has telephony rows
elsewhere in the episode) **and** ``source_observed_ms`` (for example the
day's screen-supported time) reaches ``min_observed_ms``. Otherwise an empty
window is ``no_raw_data``. iOS does not expose call or SMS logs, so with
``device_os="ios"`` every value is ``source_absent``. Ratios with an empty
denominator are ``not_applicable``.

What this is NOT
----------------
Not social contact in general: messaging apps, email and in-person contact
are invisible here.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.time import local_day_bounds_ms

COUNT_KEYS = [
    "call_count",
    "call_count_incoming",
    "call_count_outgoing",
    "call_count_missed",
    "call_duration_ms",
    "call_duration_ms_incoming",
    "call_duration_ms_outgoing",
    "sms_count",
    "sms_count_incoming",
    "sms_count_outgoing",
    "n_unique_contacts",
]

RATIO_KEYS = [
    "communication_reciprocity",
    "communication_entropy_nats",
    "outgoing_fraction",
]

COMMUNICATION_KEYS = COUNT_KEYS + RATIO_KEYS


def _fill(out: dict, keys: list[str], status: FeatureStatus, suffix: str) -> None:

    for key in keys:
        out[f"{key}{suffix}"] = np.nan
        out[f"{key}{suffix}_status"] = status.value


def shannon_entropy_nats(counts: np.ndarray) -> float:
    """Shannon entropy in nats of a count vector (NaN when empty)."""

    c = np.asarray(counts, dtype=float)
    c = c[c > 0]

    if c.size == 0:
        return np.nan

    p = c / c.sum()

    return float(-(p * np.log(p)).sum())


def compute_communication_features(
    telephony: pd.DataFrame,
    *,
    window_start_ms: int,
    window_end_ms: int,
    device_os: str | None = None,
    source_observed_ms: float | None = None,
    stream_active_in_episode: bool = False,
    min_observed_ms: float = 43_200_000,
    suffix: str = "",
) -> dict:
    """Communication features over one half-open window.

    Parameters
    ----------
    telephony
        Canonical telephony rows (``timestamp_ms``, ``channel``,
        ``direction``, optional ``duration_ms`` and ``contact_hash``).
    window_start_ms, window_end_ms
        Half-open window.
    device_os
        ``"ios"`` makes every value ``source_absent``.
    source_observed_ms
        Time the device is known to have been collecting during the window
        (for example screen-supported time). ``None`` means unknown.
    stream_active_in_episode
        Whether the participant has telephony rows anywhere in the episode.
    min_observed_ms
        Observed time needed for zero counts to be ``observed_zero`` (12 h).
    suffix
        Appended to every key (``"_7d"`` for weekly windows).

    Returns
    -------
    dict
        ``COMMUNICATION_KEYS`` (with ``suffix``) and their ``_status``, plus
        ``telephony_n_records<suffix>`` and ``telephony_n_unknown<suffix>``.
    """

    if window_end_ms <= window_start_ms:
        raise ValueError("window_end_ms must be after window_start_ms")

    required = {"timestamp_ms", "channel", "direction"}
    missing = required - set(telephony.columns)

    if missing:
        raise ValueError(
            "Missing required telephony columns: " + ", ".join(sorted(missing))
        )

    out: dict = {}

    t = pd.to_numeric(telephony["timestamp_ms"], errors="coerce")
    x = telephony[(t >= window_start_ms) & (t < window_end_ms)]

    channel = x["channel"].astype("string").str.lower()
    direction = x["direction"].astype("string").str.lower()

    unknown = (
        ~channel.isin(["call", "sms"]) | ~direction.isin(["incoming", "outgoing", "missed"])
    ).fillna(True)

    out[f"telephony_n_records{suffix}"] = len(x)
    out[f"telephony_n_unknown{suffix}"] = int(unknown.sum())

    if device_os is not None and str(device_os).lower() == "ios":
        _fill(out, COMMUNICATION_KEYS, FeatureStatus.SOURCE_ABSENT, suffix)
        return out

    known = x[~unknown.to_numpy()]
    channel = channel[~unknown.to_numpy()]
    direction = direction[~unknown.to_numpy()]

    if known.empty:
        logger_running = (
            stream_active_in_episode
            and source_observed_ms is not None
            and float(source_observed_ms) >= min_observed_ms
        )

        if not logger_running:
            status = (
                FeatureStatus.NO_RAW_DATA
                if len(x) == 0
                else FeatureStatus.INSUFFICIENT_OBSERVABILITY
            )
            _fill(out, COMMUNICATION_KEYS, status, suffix)
            return out

        for key in COUNT_KEYS:
            out[f"{key}{suffix}"] = 0.0
            out[f"{key}{suffix}_status"] = FeatureStatus.OBSERVED_ZERO.value

        _fill(out, RATIO_KEYS, FeatureStatus.NOT_APPLICABLE, suffix)
        return out

    is_call = (channel == "call").to_numpy()
    is_sms = (channel == "sms").to_numpy()
    d = direction.to_numpy()

    duration = (
        pd.to_numeric(known["duration_ms"], errors="coerce").fillna(0).to_numpy()
        if "duration_ms" in known.columns
        else np.zeros(len(known))
    )

    values = {
        "call_count": is_call.sum(),
        "call_count_incoming": (is_call & (d == "incoming")).sum(),
        "call_count_outgoing": (is_call & (d == "outgoing")).sum(),
        "call_count_missed": (is_call & (d == "missed")).sum(),
        "call_duration_ms": duration[is_call & (d != "missed")].sum(),
        "call_duration_ms_incoming": duration[is_call & (d == "incoming")].sum(),
        "call_duration_ms_outgoing": duration[is_call & (d == "outgoing")].sum(),
        "sms_count": is_sms.sum(),
        "sms_count_incoming": (is_sms & (d == "incoming")).sum(),
        "sms_count_outgoing": (is_sms & (d == "outgoing")).sum(),
    }

    contacts = (
        known["contact_hash"].astype("string")
        if "contact_hash" in known.columns
        else pd.Series(pd.NA, index=known.index, dtype="string")
    )

    has_contact = contacts.notna().to_numpy()
    values["n_unique_contacts"] = int(contacts[has_contact].nunique())

    for key, value in values.items():
        value = float(value)
        out[f"{key}{suffix}"] = value
        out[f"{key}{suffix}_status"] = (
            FeatureStatus.OBSERVED_ZERO if value == 0 else FeatureStatus.OBSERVED
        ).value

    # Contact-based ratios.
    if has_contact.any():
        c = pd.DataFrame({
            "contact": contacts[has_contact].to_numpy(),
            "outgoing": d[has_contact] == "outgoing",
        })

        per_contact = c.groupby("contact")["outgoing"].agg(["any", "all", "size"])
        reciprocal = per_contact["any"] & ~per_contact["all"]

        ratios = {
            "communication_reciprocity": float(reciprocal.mean()),
            "communication_entropy_nats": shannon_entropy_nats(
                per_contact["size"].to_numpy()
            ),
        }

    else:
        ratios = {
            "communication_reciprocity": np.nan,
            "communication_entropy_nats": np.nan,
        }

    n_out = float((d == "outgoing").sum())
    n_in = float((d == "incoming").sum())

    ratios["outgoing_fraction"] = n_out / (n_out + n_in) if n_out + n_in else np.nan

    for key, value in ratios.items():
        if np.isnan(value):
            status = FeatureStatus.NOT_APPLICABLE
        elif value == 0:
            status = FeatureStatus.OBSERVED_ZERO
        else:
            status = FeatureStatus.OBSERVED

        out[f"{key}{suffix}"] = value
        out[f"{key}{suffix}_status"] = status.value

    return out


def compute_daily_communication(
    telephony: pd.DataFrame,
    *,
    date: dt.date | str,
    tz: str,
    day_start: str = "00:00",
    **kwargs,
) -> dict:
    """Communication features for one local day.

    Parameters
    ----------
    telephony
        Canonical telephony rows.
    date
        Local date.
    tz
        IANA timezone.
    day_start
        Local clock time at which the day starts.
    **kwargs
        Passed to :func:`compute_communication_features`.

    Returns
    -------
    dict
        See :func:`compute_communication_features`.
    """

    start, end = local_day_bounds_ms(date, tz, day_start)

    return compute_communication_features(
        telephony,
        window_start_ms=start,
        window_end_ms=end,
        **kwargs,
    )


def compute_weekly_communication(
    telephony: pd.DataFrame,
    *,
    date: dt.date | str,
    tz: str,
    window_days: int = 7,
    day_start: str = "00:00",
    min_observed_ms: float = 4 * 43_200_000,
    **kwargs,
) -> dict:
    """Communication features over the ``window_days`` local days ending on ``date``.

    Parameters
    ----------
    telephony
        Canonical telephony rows.
    date
        Last local date of the window (inclusive).
    tz
        IANA timezone.
    window_days
        Window length in local days (7).
    day_start
        Local clock time at which a day starts.
    min_observed_ms
        Observed time needed for zero counts (4 × 12 h).
    **kwargs
        Passed to :func:`compute_communication_features`
        (``source_observed_ms`` should be the window total).

    Returns
    -------
    dict
        Keys carry the suffix ``_7d`` (``_<n>d`` for other lengths).
    """

    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    last = date if isinstance(date, dt.date) else dt.date.fromisoformat(date)
    _, end = local_day_bounds_ms(last, tz, day_start)
    start, _ = local_day_bounds_ms(
        last - dt.timedelta(days=window_days - 1), tz, day_start
    )

    return compute_communication_features(
        telephony,
        window_start_ms=start,
        window_end_ms=end,
        min_observed_ms=min_observed_ms,
        suffix=f"_{window_days}d",
        **kwargs,
    )
