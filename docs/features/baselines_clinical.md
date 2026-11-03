# Personal baselines and clinical alignment (Phase 8)

## Personal baseline — `baseline/personal.py`

`rolling_personal_baseline(daily, *, feature, window_observations=7, statistics=("median",
"mad", "iqr"), min_observations=3, weight_column="completeness_weight",
directional_deviations=True, allowed_windows=(3, 5, 7, 10, 14))`

Defaults mirror `configs/default.yaml::baseline`. `BaselineConfig.from_yaml(path)` reads that
section, and a test checks that the code defaults equal the file.

- **Prior-only.** Day *t*'s baseline uses the previous `window_observations` valid observations
  (`observed`/`observed_zero`, finite, weight > 0), never day *t* itself. Counting
  observations rather than calendar days keeps the reference set size fixed across gaps.
  `baseline_span_days` reports the calendar span.
- **Weighted** by `completeness_weight` (missing = 1, zero = excluded). The weighted median is
  the lower weighted quantile, averaged at exact ties, so equal weights give the ordinary
  median.
- `baseline_value` (median), `baseline_mad`, `baseline_iqr`, `deviation = value − median`,
  `robust_z = deviation / (1.4826 × MAD)`. If MAD is 0 the scale falls back to IQR / 1.349;
  if both are 0, `robust_z` is `not_applicable`. `direction` ∈ {above, below, at}.
- **Statuses.** `baseline_status` is `insufficient_observability` with fewer than
  `min_observations` prior observations. When the day itself is not valid, its deviation
  takes the day's own status (for example `no_raw_data`), which outranks a thin baseline.

Input may be long (`feature`, `value`, `feature_status`, as written by the daily driver) or
wide (`<feature>`, `<feature>_status`).

## Clinical alignment — `clinical/alignment.py`

`align_features_to_clinical(daily, clinical, *, preceding_windows_days=(1, 3, 7),
same_day=True, features=None, date_column="assessment_date", min_days=1)`

One row per assessment. For an assessment on local date *D*:

| Window | Days |
|---|---|
| `prev{d}d` | *D − d* … *D − 1* (assessment day excluded, no look-ahead) |
| `same_day` | *D* only; can include post-assessment hours, so it is **concurrent** |

Per feature and window: `_mean`, `_median`, `_weighted_mean`, `_n_days` (valid days only),
`_mean_weight` (mean `completeness_weight` of the contributing days) and `_status`. Defaults
mirror `configs/default.yaml::clinical`.

## References

- Leys C, Ley C, Klein O, Bernard P, Licata L (2013). Detecting outliers: do not use standard
  deviation around the mean, use absolute deviation around the median. *J Exp Soc Psychol*
  49:764–766. (1.4826 MAD scaling)
