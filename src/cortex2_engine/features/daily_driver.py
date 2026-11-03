"""Daily feature driver: canonical streams → one long participant-day table.

For every episode and local day, the driver runs each feature family on the
canonical streams read through :class:`canonical.reader.EpisodeReader` and
emits one row per feature:

``study, participant_id, episode_id, date_local, feature, value, unit,
feature_status, observability, observed_duration_ms, n_observations,
completeness_weight, algorithm, algorithm_version, parameter_set_hash``

- ``observability`` is ``observed_duration_ms`` divided by the local day
  length (23 h or 25 h on DST days); the denominator is the family's own
  observation window.
- ``value`` is NaN whenever ``feature_status`` is not value-bearing (rule 2).
- ``parameter_set_hash`` is a blake2b hash of :class:`DriverParameters`, so
  outputs made with different thresholds never mix silently.

Families: accelerometer, screen, phone sleep, wearable sleep, GPS (mobility,
activity space, location, home time, novelty, routine) and communication.
Multi-day constructs (IS/IV/RA, circadian movement, social rhythm,
baselines, clinical alignment) are computed from this table or from their
own functions and are not emitted per day here.

A stream the adapter cannot provide (no files) is read as empty, so that
family reports ``no_raw_data``, not an error. Each stream is read once per
episode and sliced per day. Very large accelerometer episodes should be
run one episode at a time (the CLI does).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.acc_daily import (
    ACC_ALGORITHM,
    ACC_ALGORITHM_VERSION,
    compute_daily_activity_features,
)
from cortex2_engine.features.acc_preprocess import resample_to_epochs
from cortex2_engine.features.communication import compute_communication_features
from cortex2_engine.features.gps_activity_space import compute_daily_activity_space
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)
from cortex2_engine.features.gps_home import (
    compute_daily_home_time,
    infer_home_location,
)
from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_jitter import filter_gps_jitter
from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
)
from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)
from cortex2_engine.features.gps_place_candidates import build_place_candidates
from cortex2_engine.features.gps_routine import location_novelty_fraction, routine_index
from cortex2_engine.features.gps_spatial_semantics import add_gps_spatial_semantics
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)
from cortex2_engine.features.screen_bouts import (
    build_screen_bouts,
    compute_daily_screen_features,
    screen_supported_duration_ms,
)
from cortex2_engine.features.sleep_window import (
    detect_sleep_window,
    night_window_ms,
)
from cortex2_engine.features.status import VALUE_BEARING_STATUSES, FeatureStatus
from cortex2_engine.features.wearable_sleep import (
    WEARABLE_SLEEP_KEYS,
    compute_wearable_sleep,
)
from cortex2_engine.time import iter_local_days

DRIVER_VERSION = "1.0"

LONG_COLUMNS = [
    "study",
    "participant_id",
    "episode_id",
    "date_local",
    "feature",
    "value",
    "unit",
    "feature_status",
    "observability",
    "observed_duration_ms",
    "n_observations",
    "completeness_weight",
    "algorithm",
    "algorithm_version",
    "parameter_set_hash",
]

# feature -> (unit, status key). Units must match feature_registry/features.yaml
# (checked by tests/test_daily_driver.py).
FEATURE_SPECS: dict[str, dict[str, tuple[str, str]]] = {
    "accelerometer": {
        "mean_enmo_mg": ("mg", "mean_enmo_mg_status"),
        "acc_nonwear_fraction": ("fraction", "acc_nonwear_fraction_status"),
        "active_minutes": ("minutes_per_day", "active_minutes_status"),
        "sedentary_minutes": ("minutes_per_day", "sedentary_minutes_status"),
        "astp_per_min": ("per_minute", "astp_per_min_status"),
        "satp_per_min": ("per_minute", "satp_per_min_status"),
        "mean_active_bout_min": ("minutes", "mean_active_bout_min_status"),
        "active_bout_gini": ("index", "active_bout_gini_status"),
    },
    "screen": {
        "screen_on_ms": ("milliseconds_per_day", "screen_on_ms_status"),
        "n_bouts": ("count_per_day", "n_bouts_status"),
        "median_bout_ms": ("milliseconds", "median_bout_ms_status"),
        "micro_bout_rate_per_hour": ("per_supported_hour", "micro_bout_rate_per_hour_status"),
        "screen_fragmentation_index": (
            "bouts_per_screen_on_hour",
            "screen_fragmentation_index_status",
        ),
        "nocturnal_screen_on_ms": ("milliseconds_per_day", "nocturnal_screen_on_ms_status"),
        "nocturnal_n_bouts": ("count_per_day", "nocturnal_n_bouts_status"),
    },
    "sleep": {
        "sleep_onset_local_h": ("local_clock_hour", "sleep_onset_local_h_status"),
        "sleep_offset_local_h": ("local_clock_hour", "sleep_offset_local_h_status"),
        "sleep_duration_ms": ("milliseconds", "sleep_duration_ms_status"),
        "sleep_midpoint_local_h": ("local_clock_hour", "sleep_midpoint_local_h_status"),
        "n_awakenings": ("count_per_night", "n_awakenings_status"),
    },
    "wearable_sleep": {
        key: ("milliseconds", f"{key}_status") for key in WEARABLE_SLEEP_KEYS
    },
    "gps": {
        "supported_displacement_distance_m": (
            "meters",
            "supported_displacement_distance_m_status",
        ),
        "supported_radius_of_gyration_m": ("meters", "supported_radius_of_gyration_m_status"),
        "implausible_transition_fraction": (
            "fraction",
            "implausible_transition_fraction_status",
        ),
        "convex_hull_area_km2": ("square_kilometers", "convex_hull_area_km2_status"),
        "sde_area_km2": ("square_kilometers", "sde_area_km2_status"),
        "anisotropy": ("fraction", "anisotropy_status"),
        "major_axis_bearing_deg": ("degrees", "major_axis_bearing_deg_status"),
        "location_entropy_nats": ("nats", "location_entropy_status"),
        "dominant_place_fraction": ("fraction", "dominant_place_status"),
        "home_time_ms": ("milliseconds", "home_time_ms_status"),
        "home_time_fraction": ("fraction", "home_time_fraction_status"),
        "location_novelty_fraction": ("fraction", "location_novelty_fraction_status"),
        "routine_index": ("fraction", "routine_index_status"),
    },
    "communication": {
        "call_count": ("count_per_day", "call_count_status"),
        "call_count_incoming": ("count_per_day", "call_count_incoming_status"),
        "call_count_outgoing": ("count_per_day", "call_count_outgoing_status"),
        "call_count_missed": ("count_per_day", "call_count_missed_status"),
        "call_duration_ms": ("milliseconds", "call_duration_ms_status"),
        "call_duration_ms_incoming": ("milliseconds", "call_duration_ms_incoming_status"),
        "call_duration_ms_outgoing": ("milliseconds", "call_duration_ms_outgoing_status"),
        "sms_count": ("count_per_day", "sms_count_status"),
        "sms_count_incoming": ("count_per_day", "sms_count_incoming_status"),
        "sms_count_outgoing": ("count_per_day", "sms_count_outgoing_status"),
        "n_unique_contacts": ("count_per_day", "n_unique_contacts_status"),
        "communication_reciprocity": ("fraction", "communication_reciprocity_status"),
        "communication_entropy_nats": ("nats", "communication_entropy_nats_status"),
        "outgoing_fraction": ("fraction", "outgoing_fraction_status"),
    },
}


@dataclasses.dataclass(frozen=True)
class DriverParameters:
    """Thresholds the driver passes to the feature functions.

    Defaults are the per-module defaults; ``gps_max_gap_ms`` matches the
    repository's GPS workflows (60 s).
    """

    gps_max_gap_ms: int = 60_000
    gps_accuracy_max_m: float = 100.0
    gps_max_speed_kmh: float = 250.0
    gps_max_plausible_speed_kmh: float = 1000.0
    place_compatibility_multiplier: float = 1.0
    acc_active_threshold_mg: float = 30.0
    acc_min_valid_minutes: int = 600
    screen_max_bout_ms: int = 14_400_000
    screen_min_supported_ms: float = 43_200_000
    communication_min_observed_ms: float = 43_200_000

    def hash(self) -> str:
        """Stable 16-hex-digit blake2b hash of the parameter values."""

        payload = json.dumps(
            {"driver_version": DRIVER_VERSION, **dataclasses.asdict(self)},
            sort_keys=True,
        )
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def _read(reader, stream: str, episode_id: str) -> pd.DataFrame:
    """All canonical rows of one stream for an episode (empty if unavailable)."""

    if stream not in getattr(reader.adapter, "STREAM_PREFIX", {stream: stream}):
        return pd.DataFrame()

    try:
        chunks = list(reader.iter_stream(stream, episode_id=episode_id))
    except FileNotFoundError:
        return pd.DataFrame()

    chunks = [c for c in chunks if len(c)]

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True).sort_values(
        "timestamp_ms", kind="mergesort"
    )


def _slice(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:

    if df.empty:
        return df

    t = df["timestamp_ms"].to_numpy()
    return df[(t >= start_ms) & (t < end_ms)]


def _rows(
    family: str,
    result: dict,
    *,
    context: dict,
    observed_ms: float,
    n_observations: int,
    day_ms: float,
    algorithm: str,
    algorithm_version: str,
) -> list[dict]:

    rows = []
    weight = result.get("completeness_weight", np.nan)

    for feature, (unit, status_key) in FEATURE_SPECS[family].items():
        status = result.get(status_key, FeatureStatus.NO_RAW_DATA.value)
        value = result.get(feature, np.nan)

        if status not in {s.value for s in VALUE_BEARING_STATUSES}:
            value = np.nan

        rows.append({
            **context,
            "feature": feature,
            "value": float(value) if value is not None and np.isfinite(value) else np.nan,
            "unit": unit,
            "feature_status": status,
            "observability": observed_ms / day_ms if day_ms > 0 else np.nan,
            "observed_duration_ms": float(observed_ms),
            "n_observations": int(n_observations),
            "completeness_weight": weight,
            "algorithm": algorithm,
            "algorithm_version": algorithm_version,
        })

    return rows


def compute_episode_daily_features(
    reader,
    episode: pd.Series,
    *,
    study: str,
    tz: str,
    day_start: str = "00:00",
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    params: DriverParameters | None = None,
) -> pd.DataFrame:
    """Long daily feature table for one episode.

    Parameters
    ----------
    reader
        ``EpisodeReader`` over the study adapter.
    episode
        One episode row (``episode_id``, ``participant_id``, ``start_ms``,
        ``end_ms``; optional ``device_os``).
    study
        Study name.
    tz, day_start
        Local-day definition (``StudyConfig``).
    start_date, end_date
        Optional inclusive local-date limits.
    params
        Driver thresholds.

    Returns
    -------
    pandas.DataFrame
        ``LONG_COLUMNS``.
    """

    p = params or DriverParameters()
    episode_id = str(episode["episode_id"])
    start_ms, end_ms = int(episode["start_ms"]), int(episode["end_ms"])
    device_os = episode.get("device_os") if hasattr(episode, "get") else None
    device_os = None if device_os is None or pd.isna(device_os) else str(device_os)

    acc = _read(reader, "accelerometer", episode_id)
    screen = _read(reader, "screen", episode_id)
    gps_raw = _read(reader, "gps", episode_id)
    telephony = _read(reader, "telephony", episode_id)
    sleep = _read(reader, "sleep", episode_id)

    gps, _ = filter_gps_jitter(
        gps_raw if not gps_raw.empty else pd.DataFrame(columns=["timestamp_ms", "latitude", "longitude"]),
        accuracy_max_m=p.gps_accuracy_max_m,
        max_speed_kmh=p.gps_max_speed_kmh,
    )

    days = [
        d for d in iter_local_days(start_ms, end_ms, tz, day_start=day_start)
        if (start_date is None or d.date >= start_date)
        and (end_date is None or d.date <= end_date)
    ]

    # GPS: per-day edges, then episode-wide place candidates for location,
    # home and routine (place ids must be shared across days).
    day_edges: dict[dt.date, pd.DataFrame] = {}
    segments = []

    for day in days:
        g = _slice(gps, day.start_ms, day.end_ms)

        if len(g) < 2:
            day_edges[day.date] = pd.DataFrame()
            continue

        edges = build_gps_intervals(
            g, window_start_ms=day.start_ms, window_end_ms=day.end_ms, max_gap_ms=p.gps_max_gap_ms
        )
        spatial = add_gps_spatial_semantics(edges)
        day_edges[day.date] = add_gps_trajectory_plausibility(
            spatial, max_plausible_speed_kmh=p.gps_max_plausible_speed_kmh
        )

        micro = build_anchor_compatible_location_episodes(spatial)
        if not micro.empty:
            segments.append(micro.assign(date=day.date.isoformat()))

    if segments:
        assignments, places = build_place_candidates(
            pd.concat(segments, ignore_index=True),
            compatibility_multiplier=p.place_compatibility_multiplier,
        )
    else:
        assignments = pd.DataFrame(
            columns=["date", "episode_start_ms", "episode_end_ms", "duration_ms", "place_candidate_id"]
        )
        places = pd.DataFrame(
            columns=["place_candidate_id", "representative_latitude", "representative_longitude"]
        )

    home = infer_home_location(assignments, places, tz=tz)
    stream_active = not telephony.empty
    sleep_available = not sleep.empty
    phash = p.hash()

    rows: list[dict] = []

    for day in days:
        context = {
            "study": study,
            "participant_id": str(episode["participant_id"]),
            "episode_id": episode_id,
            "date_local": day.date.isoformat(),
        }
        day_ms = float(day.duration_ms)
        win = {"window_start_ms": day.start_ms, "window_end_ms": day.end_ms}

        # Accelerometer.
        a = compute_daily_activity_features(
            _slice(acc, day.start_ms, day.end_ms) if not acc.empty
            else pd.DataFrame(columns=["timestamp_ms", "x", "y", "z"]),
            **win,
            active_threshold_mg=p.acc_active_threshold_mg,
            min_valid_minutes=p.acc_min_valid_minutes,
        )
        rows += _rows(
            "accelerometer", a, context=context,
            observed_ms=a.get("acc_valid_minutes", 0) * 60_000.0,
            n_observations=a.get("acc_n_samples", 0), day_ms=day_ms,
            algorithm=ACC_ALGORITHM, algorithm_version=ACC_ALGORITHM_VERSION,
        )

        # Screen.
        screen_day = _slice(screen, day.start_ms, day.end_ms)
        screen_context = (
            screen[screen["timestamp_ms"] < day.end_ms] if not screen.empty else screen
        )
        if screen_context.empty:
            screen_context = pd.DataFrame(columns=["timestamp_ms", "event"])
        bouts = build_screen_bouts(screen_context, **win, max_bout_ms=p.screen_max_bout_ms)
        supported = (
            screen_supported_duration_ms(screen_context, **win) if not screen.empty else 0.0
        )
        s = compute_daily_screen_features(
            bouts, supported_duration_ms=supported, tz=tz, **win,
            min_supported_ms=p.screen_min_supported_ms,
        )
        if screen_day.empty:
            s = _force_status(s, "screen", FeatureStatus.NO_RAW_DATA)
        rows += _rows(
            "screen", s, context=context, observed_ms=supported,
            n_observations=len(screen_day), day_ms=day_ms,
            algorithm="cortex2.screen_bouts", algorithm_version="1.0",
        )

        # Phone sleep: the night ending on this date.
        n_start, n_end = night_window_ms(day.date, tz)
        night_screen = screen[screen["timestamp_ms"] < n_end] if not screen.empty else pd.DataFrame(
            columns=["timestamp_ms", "event"]
        )
        night_bouts = build_screen_bouts(
            night_screen, window_start_ms=n_start, window_end_ms=n_end, max_bout_ms=p.screen_max_bout_ms
        )
        night_epochs = resample_to_epochs(
            _slice(acc, n_start, n_end) if not acc.empty
            else pd.DataFrame(columns=["timestamp_ms", "x", "y", "z"]),
            window_start_ms=n_start, window_end_ms=n_end, epoch_ms=60_000,
        )
        sl = detect_sleep_window(
            night_bouts, night_epochs, night_date=day.date, tz=tz,
            screen_events=_slice(screen, n_start, n_end) if not screen.empty else None,
        )
        rows += _rows(
            "sleep", sl, context=context,
            observed_ms=sl.get("night_coverage_fraction", 0.0) * (n_end - n_start),
            n_observations=len(_slice(screen, n_start, n_end)), day_ms=float(n_end - n_start),
            algorithm="cortex2.sleep_window", algorithm_version="1.0",
        )

        # Wearable sleep.
        sleep_night = _slice(sleep, n_start - 86_400_000, n_end)
        ws = compute_wearable_sleep(
            sleep_night if not sleep_night.empty
            else pd.DataFrame(columns=["timestamp_ms", "representation", "duration_ms"]),
            night_start_ms=n_start, night_end_ms=n_end, source_available=sleep_available,
        )
        rows += _rows(
            "wearable_sleep", ws, context=context, observed_ms=0.0,
            n_observations=ws.get("healthkit_n_segments", 0), day_ms=float(n_end - n_start),
            algorithm="cortex2.wearable_sleep", algorithm_version="1.0",
        )

        # GPS.
        rows += _gps_rows(
            day_edges.get(day.date, pd.DataFrame()), assignments, home,
            day=day, tz=tz, context=context, win=win,
            n_observations=len(_slice(gps, day.start_ms, day.end_ms)),
        )

        # Communication.
        c = compute_communication_features(
            telephony if not telephony.empty
            else pd.DataFrame(columns=["timestamp_ms", "channel", "direction"]),
            **win, device_os=device_os, source_observed_ms=supported,
            stream_active_in_episode=stream_active,
            min_observed_ms=p.communication_min_observed_ms,
        )
        rows += _rows(
            "communication", c, context=context, observed_ms=supported,
            n_observations=c.get("telephony_n_records", 0), day_ms=day_ms,
            algorithm="cortex2.communication", algorithm_version="1.0",
        )

    out = pd.DataFrame(rows, columns=[c for c in LONG_COLUMNS if c != "parameter_set_hash"])
    out["parameter_set_hash"] = phash

    return out[LONG_COLUMNS]


def _force_status(result: dict, family: str, status: FeatureStatus) -> dict:
    """Override every status of a family (e.g. a day without any raw row)."""

    out = dict(result)
    for feature, (_, status_key) in FEATURE_SPECS[family].items():
        out[feature] = np.nan
        out[status_key] = status.value
    return out


def _gps_rows(
    edges: pd.DataFrame,
    assignments: pd.DataFrame,
    home: dict,
    *,
    day,
    tz: str,
    context: dict,
    win: dict,
    n_observations: int,
) -> list[dict]:

    result: dict = {}

    if edges.empty:
        result = _force_status({}, "gps", FeatureStatus.NO_RAW_DATA)
        supported = 0.0
        weight = 0.0
    else:
        supported = float(edges.loc[edges["supported"].astype(bool), "duration_ms"].sum())
        mobility = compute_daily_mobility_features(edges, **win)
        result.update(mobility)
        result.update(compute_daily_activity_space(edges, **win))
        result.update(compute_daily_home_time(edges, home, **win))

        day_assign = (
            assignments[assignments["date"].astype(str) == day.date.isoformat()]
            if "date" in assignments.columns
            else assignments.iloc[0:0]
        )
        result.update(compute_daily_location_features(day_assign, supported_duration_ms=supported, **win))
        weight = mobility["completeness_weight"]

    if not assignments.empty:
        result.update(location_novelty_fraction(assignments, date=day.date, tz=tz))
        result.update(routine_index(assignments, date=day.date, tz=tz))
    else:
        for key in ("location_novelty_fraction", "routine_index"):
            result[key] = np.nan
            result[f"{key}_status"] = FeatureStatus.NO_RAW_DATA.value

    # One weight for the family: the supported-edge weight of the day.
    result["completeness_weight"] = weight

    return _rows(
        "gps", result, context=context, observed_ms=supported,
        n_observations=n_observations, day_ms=float(day.duration_ms),
        algorithm="cortex2.gps", algorithm_version="1.0",
    )


def iter_daily_features(
    adapter,
    episodes: pd.DataFrame,
    config,
    *,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    params: DriverParameters | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield the long daily feature table one episode at a time.

    Parameters
    ----------
    adapter
        Study adapter (``BaseAdapter``).
    episodes
        Canonical episodes table.
    config
        ``StudyConfig`` (``name``, ``timezone``, ``day_start``).
    start_date, end_date
        Optional inclusive local-date limits.
    params
        Driver thresholds.

    Yields
    ------
    pandas.DataFrame
        ``LONG_COLUMNS`` rows for one episode.
    """

    from cortex2_engine.canonical.reader import EpisodeReader

    reader = EpisodeReader(adapter, episodes)

    for _, episode in reader.episodes.iterrows():
        yield compute_episode_daily_features(
            reader,
            episode,
            study=config.name,
            tz=config.timezone,
            day_start=config.day_start,
            start_date=start_date,
            end_date=end_date,
            params=params,
        )


def write_daily_features(
    adapter,
    episodes: pd.DataFrame,
    config,
    *,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    params: DriverParameters | None = None,
) -> Path:
    """Run :func:`iter_daily_features` and write one parquet file.

    Returns
    -------
    pathlib.Path
        ``<output_dir>/features/daily_features_long.parquet``.
    """

    frames = list(
        iter_daily_features(
            adapter, episodes, config,
            start_date=start_date, end_date=end_date, params=params,
        )
    )

    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=LONG_COLUMNS)
    )

    path = Path(config.output_dir) / "features" / "daily_features_long.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    return path
