"""Rolling personal baselines and robust within-person deviations.

Construct
---------
How unusual a participant-day is *for that participant*: the day's value
compared with the participant's own recent typical value, scaled by their own
recent variability.

Estimator
---------
For each participant, days are ordered by ``date_local``. The baseline for
day *t* uses the previous ``window_observations`` **valid observations**
(status ``observed`` or ``observed_zero``, finite value, positive weight)
strictly before *t*. Day *t* never enters its own baseline, so there is no
leakage. Counting observations, not calendar days, means a gap does not
shrink the reference set; the date span is reported as
``baseline_span_days``.

With weights ``w`` (``completeness_weight`` by default; missing weights
count as 1, zero weights exclude the day):

- ``baseline_value`` — weighted median;
- ``baseline_mad`` — weighted median of ``|x - baseline_value|``;
- ``baseline_iqr`` — weighted 75th minus 25th percentile;
- ``deviation = value - baseline_value``;
- ``robust_z = deviation / (1.4826 × MAD)``. If MAD is 0 it falls back to
  ``IQR / 1.349``. If both are 0 (no variation), ``robust_z`` is NaN with
  ``robust_z_status = not_applicable``;
- ``direction`` ∈ {``above``, ``below``, ``at``} when
  ``directional_deviations`` is on.

The weighted quantile is the lower weighted quantile (the smallest value
whose cumulative weight reaches the target share), averaged with the next
value when the cumulative weight hits the target exactly. With equal weights
this reproduces the ordinary median.

Fewer than ``min_observations`` valid prior observations gives
``baseline_status = insufficient_observability``. A day whose own value is
not valid keeps its baseline, but its deviation is NaN with that day's own
status, which takes precedence over a thin baseline.

Defaults follow ``configs/default.yaml::baseline`` (:class:`BaselineConfig`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from cortex2_engine.features.status import FeatureStatus

MAD_TO_SD = 1.4826
IQR_TO_SD = 1.349

STATISTICS = ("median", "mad", "iqr")

_VALID = {FeatureStatus.OBSERVED.value, FeatureStatus.OBSERVED_ZERO.value}


class BaselineConfig(BaseModel):
    """``baseline`` section of ``configs/default.yaml``."""

    enabled: bool = True
    candidate_observations: list[int] = Field(default_factory=lambda: [3, 5, 7, 10, 14])
    statistics: list[str] = Field(default_factory=lambda: list(STATISTICS))
    directional_deviations: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaselineConfig:
        """Read the ``baseline`` section of a config file."""

        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data.get("baseline", {}))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile (see module docstring); NaN when empty."""

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]

    if v.size == 0:
        return float("nan")

    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]

    cumulative = np.cumsum(w) / w.sum()
    i = int(np.searchsorted(cumulative, q - 1e-12))
    i = min(i, v.size - 1)

    if np.isclose(cumulative[i], q) and i + 1 < v.size:
        return float((v[i] + v[i + 1]) / 2.0)

    return float(v[i])


def _long_view(
    daily: pd.DataFrame,
    *,
    feature: str,
    participant_column: str,
    weight_column: str | None,
) -> pd.DataFrame:

    if "feature" in daily.columns and "value" in daily.columns:
        x = daily[daily["feature"] == feature]
        if x.empty:
            raise ValueError(f"feature {feature!r} not found (long or wide format)")
        value = x["value"]
        status = (
            x["feature_status"]
            if "feature_status" in x.columns
            else pd.Series(FeatureStatus.OBSERVED.value, index=x.index)
        )
    elif feature in daily.columns:
        x = daily
        value = x[feature]
        status = (
            x[f"{feature}_status"]
            if f"{feature}_status" in x.columns
            else pd.Series(FeatureStatus.OBSERVED.value, index=x.index)
        )
    else:
        raise ValueError(f"feature {feature!r} not found (long or wide format)")

    for column in (participant_column, "date_local"):
        if column not in x.columns:
            raise ValueError(f"Missing required column: {column}")

    weight = (
        pd.to_numeric(x[weight_column], errors="coerce")
        if weight_column and weight_column in x.columns
        else pd.Series(1.0, index=x.index)
    )

    out = pd.DataFrame({
        "participant_id": x[participant_column].to_numpy(),
        "date_local": pd.to_datetime(x["date_local"]).dt.date.to_numpy(),
        "feature": feature,
        "value": pd.to_numeric(value, errors="coerce").to_numpy(),
        "feature_status": status.astype(str).to_numpy(),
        "weight": weight.fillna(1.0).to_numpy(),
    })

    return out.sort_values(["participant_id", "date_local"], kind="mergesort")


def rolling_personal_baseline(
    daily: pd.DataFrame,
    *,
    feature: str,
    window_observations: int | Sequence[int] = 7,
    statistics: Iterable[str] = STATISTICS,
    min_observations: int = 3,
    weight_column: str | None = "completeness_weight",
    participant_column: str = "participant_id",
    directional_deviations: bool = True,
    allowed_windows: Iterable[int] | None = (3, 5, 7, 10, 14),
) -> pd.DataFrame:
    """Prior-only rolling baseline, deviation and robust z per participant-day.

    Parameters
    ----------
    daily
        Long table (``feature``, ``value``, ``feature_status``) or wide
        table (``<feature>``, ``<feature>_status``), with
        ``participant_column`` and ``date_local``.
    feature
        Feature name.
    window_observations
        Prior valid observations in the baseline; one value or several
        (results are stacked with a ``window_observations`` column).
    statistics
        Subset of ``("median", "mad", "iqr")`` to report. The median is
        always computed because deviations need it.
    min_observations
        Valid prior observations needed (3).
    weight_column
        Weight column (``completeness_weight``); ``None`` for equal weights.
    participant_column
        Participant identifier column.
    directional_deviations
        Add ``direction`` (``configs/default.yaml``: true).
    allowed_windows
        Permitted window sizes (``baseline.candidate_observations``);
        ``None`` allows any size >= ``min_observations``.

    Returns
    -------
    pandas.DataFrame
        One row per participant-day and window size.
    """

    windows = (
        [int(window_observations)]
        if np.isscalar(window_observations)
        else [int(w) for w in window_observations]
    )

    requested = list(statistics)
    unknown = set(requested) - set(STATISTICS)

    if unknown:
        raise ValueError(f"Unknown statistics: {sorted(unknown)}")

    if min_observations < 1:
        raise ValueError("min_observations must be >= 1")

    for w in windows:
        if allowed_windows is not None and w not in set(allowed_windows):
            raise ValueError(f"window_observations {w} not in {sorted(allowed_windows)}")
        if w < min_observations:
            raise ValueError("window_observations must be >= min_observations")

    x = _long_view(
        daily,
        feature=feature,
        participant_column=participant_column,
        weight_column=weight_column,
    )

    frames = []

    for window in windows:
        rows = []

        for _, g in x.groupby("participant_id", sort=True):
            values = g["value"].to_numpy(float)
            weights = g["weight"].to_numpy(float)
            valid = (
                g["feature_status"].isin(_VALID).to_numpy()
                & np.isfinite(values)
                & (weights > 0)
            )

            history_idx: list[int] = []

            for i in range(len(g)):
                prior = history_idx[-window:]
                rows.append(
                    _baseline_row(
                        g.iloc[i],
                        values[prior],
                        weights[prior],
                        [g["date_local"].iloc[j] for j in prior],
                        is_valid=bool(valid[i]),
                        window=window,
                        min_observations=min_observations,
                        requested=requested,
                        directional=directional_deviations,
                    )
                )

                if valid[i]:
                    history_idx.append(i)

        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _baseline_row(
    day: pd.Series,
    values: np.ndarray,
    weights: np.ndarray,
    dates: list,
    *,
    is_valid: bool,
    window: int,
    min_observations: int,
    requested: list[str],
    directional: bool,
) -> dict:

    row = {
        "participant_id": day["participant_id"],
        "date_local": day["date_local"].isoformat(),
        "feature": day["feature"],
        "window_observations": window,
        "value": day["value"],
        "feature_status": day["feature_status"],
        "baseline_n_observations": int(values.size),
        "baseline_span_days": (
            (dates[-1] - dates[0]).days + 1 if dates else 0
        ),
    }

    enough = values.size >= min_observations

    median = weighted_quantile(values, weights, 0.5) if enough else np.nan
    mad = (
        weighted_quantile(np.abs(values - median), weights, 0.5) if enough else np.nan
    )
    iqr = (
        weighted_quantile(values, weights, 0.75) - weighted_quantile(values, weights, 0.25)
        if enough
        else np.nan
    )

    row["baseline_value"] = median
    if "mad" in requested:
        row["baseline_mad"] = mad
    if "iqr" in requested:
        row["baseline_iqr"] = iqr

    row["baseline_status"] = (
        FeatureStatus.OBSERVED if enough else FeatureStatus.INSUFFICIENT_OBSERVABILITY
    ).value

    if not is_valid:
        # The day's own reason (e.g. no_raw_data) comes first.
        try:
            deviation_status = FeatureStatus(day["feature_status"])
        except ValueError:
            deviation_status = FeatureStatus.NOT_APPLICABLE
        if deviation_status.value in _VALID:
            # Valid status but non-finite value or zero weight.
            deviation_status = FeatureStatus.NOT_APPLICABLE
    elif not enough:
        deviation_status = FeatureStatus.INSUFFICIENT_OBSERVABILITY
    else:
        deviation_status = None

    if deviation_status is not None:
        row.update({
            "deviation": np.nan,
            "deviation_status": deviation_status.value,
            "robust_z": np.nan,
            "robust_z_status": deviation_status.value,
        })
        if directional:
            row["direction"] = None
        return row

    deviation = float(day["value"]) - median
    scale = MAD_TO_SD * mad if mad > 0 else (iqr / IQR_TO_SD if iqr > 0 else 0.0)

    row["deviation"] = deviation
    row["deviation_status"] = (
        FeatureStatus.OBSERVED_ZERO if deviation == 0 else FeatureStatus.OBSERVED
    ).value

    if scale > 0:
        z = deviation / scale
        row["robust_z"] = z
        row["robust_z_status"] = (
            FeatureStatus.OBSERVED_ZERO if z == 0 else FeatureStatus.OBSERVED
        ).value
    else:
        row["robust_z"] = np.nan
        row["robust_z_status"] = FeatureStatus.NOT_APPLICABLE.value

    if directional:
        row["direction"] = "above" if deviation > 0 else ("below" if deviation < 0 else "at")

    return row
