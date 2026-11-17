"""Home time from Cortex place episodes using the production nighttime definition.

Home is the persistent place winning the most sufficiently observed nights
(00:00--06:00 local time) in the first 14 local days. Place IDs must be
calculated on the same timeline as all other mobility features.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HomeConfig:
    timezone: str = "America/New_York"
    reference_days: int = 14
    min_valid_nights: int = 5
    min_night_observed_ms: float = 2 * 60 * 60 * 1000
    min_home_confidence: float = 0.50
    min_daily_supported_ms: float = 6 * 60 * 60 * 1000


def calculate_home_time(assignments: pd.DataFrame, *,
                        supported_duration_ms: float,
                        start_ms: int, end_ms: int,
                        config: HomeConfig = HomeConfig()) -> dict:
    """Return home time for a window and an auditable inferred home state.

    This does not presume that the dominant place is home. Lack of reference
    nights yields missing home time, rather than a fabricated anchor.
    """
    missing = {
        "episode_start_ms", "episode_end_ms", "place_candidate_id"
    } - set(assignments.columns)
    if missing:
        raise ValueError(f"Home assignments missing {sorted(missing)}")
    a = assignments.copy()
    for col in ("episode_start_ms", "episode_end_ms", "place_candidate_id"):
        a[col] = pd.to_numeric(a[col], errors="coerce")
    a = a.dropna(subset=["episode_start_ms", "episode_end_ms", "place_candidate_id"])
    a = a.loc[a.episode_end_ms > a.episode_start_ms]
    result = {
        "home_time_ms": np.nan,
        "home_place_candidate_id": np.nan,
        "home_confidence": np.nan,
        "home_valid_nights": 0,
        "home_status": "insufficient_observability",
    }
    if a.empty:
        return result
    first_date = pd.to_datetime(a.episode_start_ms.min(), unit="ms", utc=True).tz_convert(config.timezone).date()
    starts = a.episode_start_ms.to_numpy(dtype=float)
    ends = a.episode_end_ms.to_numpy(dtype=float)
    places = a.place_candidate_id.to_numpy(dtype=int)
    winners = []
    for offset in range(config.reference_days):
        day = first_date + pd.Timedelta(days=offset)
        lo = pd.Timestamp(f"{day} 00:00:00").tz_localize(config.timezone)
        hi = pd.Timestamp(f"{day} 06:00:00").tz_localize(config.timezone)
        lo_ms = lo.tz_convert("UTC").timestamp() * 1000
        hi_ms = hi.tz_convert("UTC").timestamp() * 1000
        overlap = np.maximum(0.0, np.minimum(ends, hi_ms) - np.maximum(starts, lo_ms))
        if overlap.sum() < config.min_night_observed_ms:
            continue
        sums = pd.DataFrame({"place": places, "duration": overlap})
        sums = sums.loc[sums.duration > 0].groupby("place").duration.sum()
        if len(sums):
            winners.append(int(sums.sort_values(ascending=False).index[0]))
    result["home_valid_nights"] = len(winners)
    if len(winners) < config.min_valid_nights:
        return result
    counts = pd.Series(winners).value_counts()
    home_id = int(counts.index[0])
    confidence = float(counts.iloc[0] / len(winners))
    result["home_confidence"] = confidence
    if confidence < config.min_home_confidence:
        return result
    result["home_place_candidate_id"] = home_id
    overlap = np.maximum(0.0, np.minimum(ends, float(end_ms)) - np.maximum(starts, float(start_ms)))
    if supported_duration_ms < config.min_daily_supported_ms:
        return result
    result["home_time_ms"] = float(overlap[places == home_id].sum())
    result["home_status"] = "observed_or_reconstructed"
    return result
