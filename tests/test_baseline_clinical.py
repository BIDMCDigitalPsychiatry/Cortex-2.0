import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cortex2_engine.baseline.personal import (
    BaselineConfig,
    rolling_personal_baseline,
    weighted_quantile,
)
from cortex2_engine.clinical.alignment import align_features_to_clinical

REPO = Path(__file__).resolve().parents[1]


def _long(values, *, feature="steps", pid="P1", start="2026-01-01", status=None, weights=None):
    d0 = dt.date.fromisoformat(start)
    n = len(values)
    return pd.DataFrame({
        "participant_id": pid,
        "date_local": [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)],
        "feature": feature,
        "value": values,
        "feature_status": status or ["observed"] * n,
        "completeness_weight": weights if weights is not None else [1.0] * n,
    })


def test_config_defaults_match_repository_yaml():
    from_file = BaselineConfig.from_yaml(REPO / "configs" / "default.yaml")
    assert from_file == BaselineConfig()
    assert from_file.candidate_observations == [3, 5, 7, 10, 14]


def test_weighted_quantile_matches_median_with_equal_weights():
    for v in ([3, 1, 2], [4, 1, 3, 2], [5.0]):
        assert weighted_quantile(np.array(v), np.ones(len(v)), 0.5) == np.median(v)
    # A heavy weight pulls the median towards its value.
    assert weighted_quantile(np.array([1, 2, 10]), np.array([1, 1, 10]), 0.5) == 10
    assert np.isnan(weighted_quantile(np.array([1.0]), np.array([0.0]), 0.5))


def test_baseline_is_prior_only_and_robust_z():
    values = [10, 12, 11, 13, 9, 30]
    out = rolling_personal_baseline(_long(values), feature="steps", window_observations=5)

    first = out.iloc[0]
    assert first["baseline_status"] == "insufficient_observability"
    assert np.isnan(first["robust_z"])

    last = out.iloc[-1]
    prior = np.array(values[:5], float)
    med = np.median(prior)
    mad = np.median(np.abs(prior - med))
    assert last["baseline_value"] == med  # 30 is not in its own baseline
    assert last["baseline_mad"] == mad
    assert last["deviation"] == 30 - med
    assert last["robust_z"] == pytest.approx((30 - med) / (1.4826 * mad))
    assert last["direction"] == "above"
    assert last["baseline_n_observations"] == 5


def test_window_counts_valid_observations_and_skips_invalid_days():
    values = [10, 11, np.nan, 12, 13, 14, 20]
    status = ["observed", "observed", "no_raw_data", "observed", "observed", "insufficient_observability", "observed"]
    out = rolling_personal_baseline(_long(values, status=status), feature="steps", window_observations=3)

    missing_day = out.iloc[2]
    assert missing_day["deviation_status"] == "no_raw_data"
    assert missing_day["baseline_status"] == "insufficient_observability"  # only 2 prior

    last = out.iloc[-1]
    assert last["baseline_value"] == 12  # prior valid: 11, 12, 13
    assert last["baseline_span_days"] == 4  # 01-02 .. 01-05


def test_zero_weight_days_are_excluded_and_flat_history_is_not_applicable():
    out = rolling_personal_baseline(
        _long([5, 5, 5, 100, 5], weights=[1, 1, 1, 0, 1]), feature="steps", window_observations=3
    )
    assert out.iloc[3]["deviation_status"] == "not_applicable"  # zero weight
    flat = out.iloc[4]
    assert flat["baseline_value"] == 5
    assert flat["deviation"] == 0 and flat["deviation_status"] == "observed_zero"
    assert flat["direction"] == "at"
    assert flat["robust_z_status"] == "not_applicable"


def test_iqr_fallback_wide_format_multiple_windows_and_participants():
    # MAD = 0 but IQR > 0: z uses IQR / 1.349.
    values = [1, 1, 5, 5, 5, 7]
    wide = pd.DataFrame({
        "participant_id": "P2",
        "date_local": pd.date_range("2026-02-01", periods=6).date,
        "steps": values,
    })
    out = rolling_personal_baseline(
        wide, feature="steps", window_observations=[5, 3], weight_column=None,
        statistics=("median", "iqr"), directional_deviations=False,
    )
    five = out[out["window_observations"] == 5].iloc[-1]
    assert "baseline_mad" not in out.columns and "direction" not in out.columns
    assert five["robust_z"] == pytest.approx((7 - 5) / (4 / 1.349))

    both = rolling_personal_baseline(
        pd.concat([_long([1, 2, 3, 4]), _long([7, 8, 9, 10], pid="P9")]),
        feature="steps", window_observations=3,
    )
    assert both.groupby("participant_id")["baseline_value"].last().to_dict() == {"P1": 2.0, "P9": 8.0}


def test_baseline_validation():
    df = _long([1, 2, 3])
    with pytest.raises(ValueError, match="not in"):
        rolling_personal_baseline(df, feature="steps", window_observations=4)
    with pytest.raises(ValueError, match=">= min_observations"):
        rolling_personal_baseline(df, feature="steps", window_observations=3, min_observations=5)
    with pytest.raises(ValueError, match="Unknown statistics"):
        rolling_personal_baseline(df, feature="steps", statistics=["mean"])
    with pytest.raises(ValueError):
        rolling_personal_baseline(df, feature="steps", min_observations=0)
    with pytest.raises(ValueError, match="not found"):
        rolling_personal_baseline(df, feature="sleep")
    with pytest.raises(ValueError, match="Missing required column"):
        rolling_personal_baseline(df.drop(columns="date_local"), feature="steps")
    # Unknown status strings are treated as not applicable.
    odd = rolling_personal_baseline(
        _long([1, 2, 3, 4], status=["observed"] * 3 + ["weird"]), feature="steps",
        window_observations=3, allowed_windows=None,
    )
    assert odd.iloc[-1]["deviation_status"] == "not_applicable"


def test_clinical_alignment_windows_exclude_assessment_day():
    daily = pd.concat([
        _long([1, 2, 3, 4, 5, 6, 7, 100], start="2026-03-01",
              weights=[1, 1, 1, 1, 0.5, 0.5, 0.5, 1]),
        _long([0, 0, 0], feature="calls", start="2026-03-06"),
    ])
    clinical = pd.DataFrame({
        "participant_id": ["P1", "P1", "P2"],
        "assessment_date": ["2026-03-08", "2026-03-02", "2026-03-08"],
        "phq9": [12, 9, 4],
    })
    out = align_features_to_clinical(daily, clinical)

    first = out.iloc[0]
    assert first["phq9"] == 12
    assert first["steps__prev1d_mean"] == 7
    assert first["steps__prev3d_mean"] == 6
    assert first["steps__prev3d_mean_weight"] == 0.5
    assert first["steps__prev7d_n_days"] == 7
    assert first["steps__prev7d_median"] == 4
    assert first["steps__same_day_mean"] == 100  # concurrent only
    assert first["steps__prev7d_weighted_mean"] == pytest.approx(
        np.average([1, 2, 3, 4, 5, 6, 7], weights=[1, 1, 1, 1, .5, .5, .5])
    )
    assert first["calls__prev1d_mean"] == 0
    assert first["calls__prev1d_status"] == "observed_zero"

    early = out.iloc[1]
    assert early["steps__prev7d_n_days"] == 1
    assert np.isnan(early["calls__prev3d_mean"])

    other = out.iloc[2]
    assert other["steps__prev1d_status"] == "insufficient_observability"
    assert other["steps__prev1d_n_days"] == 0


def test_clinical_alignment_status_filter_options_and_validation():
    daily = _long([1, np.nan, 3], status=["observed", "no_raw_data", "observed"], start="2026-04-01")
    clinical = pd.DataFrame({"participant_id": ["P1"], "assessment_date": ["2026-04-04"]})
    out = align_features_to_clinical(
        daily, clinical, preceding_windows_days=[3], same_day=False, min_days=3
    )
    assert out.iloc[0]["steps__prev3d_n_days"] == 2
    assert out.iloc[0]["steps__prev3d_status"] == "insufficient_observability"
    assert not any("same_day" in c for c in out.columns)

    with pytest.raises(ValueError):
        align_features_to_clinical(daily, clinical, preceding_windows_days=[0])
    with pytest.raises(ValueError):
        align_features_to_clinical(daily, clinical, min_days=0)
    with pytest.raises(ValueError, match="Missing required clinical columns"):
        align_features_to_clinical(daily, clinical.drop(columns="assessment_date"))
