"""Align daily features to clinical assessments.

Construct
---------
For each clinical assessment (for example a PHQ-9 at a study visit), summaries
of each daily feature over the days that precede it, so passive features can
be related to clinical scores without look-ahead.

Windows
-------
For an assessment on local date *D* and each ``d`` in
``preceding_windows_days`` (default 1, 3, 7), the window is the ``d`` days
``[D - d, D - 1]``. The assessment day is excluded, so values cannot
include behaviour after the assessment. With ``same_day=True``, day *D* is
summarised separately as ``same_day``. Same-day features can include
post-assessment hours, so treat them as concurrent, not predictive.

Per feature and window the output has:

- ``<feature>__<window>_mean``, ``_median`` and ``_weighted_mean``
  (weights = ``completeness_weight``, missing weights count as 1);
- ``<feature>__<window>_n_days`` — contributing days, which are days whose
  status is ``observed`` or ``observed_zero`` with a finite value;
- ``<feature>__<window>_mean_weight`` — mean ``completeness_weight`` of
  the contributing days;
- ``<feature>__<window>_status`` — ``observed`` (or ``observed_zero`` when
  the mean is 0), or ``insufficient_observability`` with fewer than
  ``min_days`` contributing days.

``<window>`` is ``prev1d``, ``prev3d``, ``prev7d`` or ``same_day``.

All clinical columns are carried through unchanged, giving one output row per
assessment.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus

_VALID = {FeatureStatus.OBSERVED.value, FeatureStatus.OBSERVED_ZERO.value}


def _summaries(values: np.ndarray, weights: np.ndarray, min_days: int) -> dict:

    n = int(values.size)

    if n < min_days or n == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "weighted_mean": np.nan,
            "n_days": n,
            "mean_weight": float(weights.mean()) if n else np.nan,
            "status": FeatureStatus.INSUFFICIENT_OBSERVABILITY.value,
        }

    mean = float(values.mean())
    w = np.where(np.isfinite(weights), weights, 1.0)

    return {
        "mean": mean,
        "median": float(np.median(values)),
        "weighted_mean": float(np.average(values, weights=w)) if w.sum() > 0 else np.nan,
        "n_days": n,
        "mean_weight": float(np.nanmean(weights)) if np.isfinite(weights).any() else np.nan,
        "status": (
            FeatureStatus.OBSERVED_ZERO if mean == 0 else FeatureStatus.OBSERVED
        ).value,
    }


def align_features_to_clinical(
    daily: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    preceding_windows_days: Iterable[int] = (1, 3, 7),
    same_day: bool = True,
    features: Iterable[str] | None = None,
    participant_column: str = "participant_id",
    date_column: str = "assessment_date",
    weight_column: str = "completeness_weight",
    min_days: int = 1,
) -> pd.DataFrame:
    """One row per clinical assessment with lagged feature aggregates.

    Parameters
    ----------
    daily
        Long daily features (``participant_column``, ``date_local``,
        ``feature``, ``value``, ``feature_status``, optional
        ``completeness_weight``), as written by the daily driver.
    clinical
        Assessments with ``participant_column`` and ``date_column`` (local
        date of the assessment).
    preceding_windows_days
        Window lengths in days before the assessment (1, 3, 7).
    same_day
        Also summarise the assessment day.
    features
        Features to align (default: all in ``daily``).
    participant_column, date_column, weight_column
        Column names.
    min_days
        Contributing days needed for a value (1).

    Returns
    -------
    pandas.DataFrame
        ``clinical`` columns plus the aggregates described in the module
        docstring, in the input order of ``clinical``.
    """

    windows = sorted({int(d) for d in preceding_windows_days})

    if any(d < 1 for d in windows):
        raise ValueError("preceding_windows_days must be >= 1")

    if min_days < 1:
        raise ValueError("min_days must be >= 1")

    for name, frame, columns in (
        ("daily", daily, {participant_column, "date_local", "feature", "value"}),
        ("clinical", clinical, {participant_column, date_column}),
    ):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required {name} columns: " + ", ".join(sorted(missing)))

    d = daily.copy()
    d["_date"] = pd.to_datetime(d["date_local"]).dt.normalize()
    d["_value"] = pd.to_numeric(d["value"], errors="coerce")
    d["_weight"] = (
        pd.to_numeric(d[weight_column], errors="coerce")
        if weight_column in d.columns
        else np.nan
    )
    status = (
        d["feature_status"].astype(str)
        if "feature_status" in d.columns
        else pd.Series(FeatureStatus.OBSERVED.value, index=d.index)
    )
    d = d[status.isin(_VALID).to_numpy() & np.isfinite(d["_value"].to_numpy())]

    names = sorted(set(daily["feature"])) if features is None else list(features)

    spans = [(f"prev{n}d", n, 1) for n in windows]
    if same_day:
        spans.append(("same_day", 0, 0))

    by_participant = {pid: g for pid, g in d.groupby(participant_column, sort=False)}
    assessment_dates = pd.to_datetime(clinical[date_column]).dt.normalize()

    rows = []

    for (_, assessment), date in zip(clinical.iterrows(), assessment_dates, strict=True):
        row = assessment.to_dict()
        g = by_participant.get(assessment[participant_column])

        for feature in names:
            f = g[g["feature"] == feature] if g is not None else None

            for label, back, forward_gap in spans:
                # Window = [date - back, date - forward_gap]; same day: [date, date].
                first = date - pd.Timedelta(days=back)
                last = date - pd.Timedelta(days=forward_gap)

                if f is None or pd.isna(date):
                    sel = np.array([], dtype=float), np.array([], dtype=float)
                else:
                    inside = (f["_date"] >= first) & (f["_date"] <= last)
                    sel = (
                        f.loc[inside, "_value"].to_numpy(float),
                        f.loc[inside, "_weight"].to_numpy(float),
                    )

                for key, value in _summaries(*sel, min_days).items():
                    row[f"{feature}__{label}_{key}"] = value

        rows.append(row)

    return pd.DataFrame(rows, index=clinical.index)
