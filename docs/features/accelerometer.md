# Accelerometer features

Modules: `features/acc_preprocess.py`, `acc_nonwear.py`, `acc_circadian.py`,
`acc_fragmentation.py`, `acc_daily.py`.

> **Why this file defines the formulas.** The feature-expansion brief refers to "the design
> document" for the IS/IV/RA formulas and the non-wear rule. The repository's design document
> (`legacy_cortex_vs_cortex2_gps_feature_design.md`) covers GPS only, so the definitions
> below are the specification the code is tested against (`docs/open_questions.md` #12).

## Pipeline

```text
raw x,y,z ─► detect_unit ─► 1 s epochs (ENMO, axis sums/min/max) ─► detect_nonwear
         ─► 1 min epochs (non-wear removed) ─► mean ENMO, fragmentation   (one day)
                                            └► hourly matrix ─► IS, IV, RA (≥ 5 days)
```

## Units and ENMO

- `detect_unit(acc, *, gravity_band_g=(0.7, 1.3))`: the median vector magnitude must be in
  the band (→ `g`) or in the band × 9.80665 (→ `m_s2`); otherwise `unknown` (e.g. gravity
  already removed). Days with `unknown` units get `extraction_failed` and
  `acc_error_type = "accelerometer_unit_unknown"`.
- ENMO = max(‖(x, y, z)‖ − 1 g, 0) (van Hees et al. 2013), reported in mg. No
  auto-calibration is applied.

## Non-wear (`detect_nonwear`)

van Hees et al. (2011), GGIR form. For windows of `window_ms` (60 min) stepped by `step_ms`
(15 min), an axis is **still** when SD < `sd_threshold_mg` (13 mg) **and** range <
`range_threshold_mg` (50 mg). A window is non-wear when ≥ `min_axes` (2) axes are still.
SD (ddof = 1) and range are computed exactly from per-epoch sums, sums of squares, minima
and maxima.

The original paper's wording ("SD … or value range …") reads as OR across the two criteria.
GGIR's implementation, used here, requires both per axis. That is the more conservative
choice, giving less non-wear.

**Smartphone adaptation.**

| Wrist-device rule | Phone adaptation |
|---|---|
| Non-wear = device off wrist | Non-wear = phone not on the body (desk, charger, bedside) |
| Missing samples rarely occur | Epochs with no samples are *unobserved*, never non-wear; a window needs ≥ `min_window_observed_fraction` (0.5) observed time to be judged |
| GGIR labels the centre 15-min block of each still window | Non-wear = **union** of still windows, so a still period of ≥ 60 min is recovered in full |
| — | Optional battery evidence flags intervals where the battery rose ≥ `charging_min_rise_fraction` (0.01) as `charging` (annotation only) |

`acc_nonwear_fraction` = non-wear time / time covered by judged windows, reported when
judged time ≥ `minimum_evaluable_ms` (1 h).

## Daily activity (`compute_daily_activity_features`)

| Feature | Unit | Definition | Minimum |
|---|---|---|---|
| `mean_enmo_mg` | mg | mean ENMO over valid (observed, worn) minutes | 600 valid min |
| `active_minutes`, `sedentary_minutes` | min | valid minutes with ENMO ≥ / < `active_threshold_mg` (30 mg) | 600 valid min |
| `astp_per_min` | 1/min | active bouts / active minutes (Di et al. 2017) | 600 valid min |
| `satp_per_min` | 1/min | sedentary bouts / sedentary minutes | 600 valid min |
| `mean_active_bout_min` | min | mean active bout length | 600 valid min |
| `active_bout_gini` | index | Gini of active bout lengths (≥ 2 bouts) | 600 valid min |
| `acc_nonwear_fraction` | fraction | see above | 1 h judged |

A minute is valid when ≥ `minute_min_observed_fraction` (0.5) of it is observed and worn.
Invalid minutes break bouts and are never counted as sedentary. Bouts touching an invalid
minute or the day edge are counted in `n_censored_bouts`. On a day with zero active
minutes and adequate wear, `active_minutes` is `observed_zero` and the bout metrics are
`not_applicable`.

## Circadian rest–activity (`acc_circadian`)

Hourly matrix: days × 24 local **clock** hours of mean ENMO (mg). A cell is observed when
≥ `min_hour_observed_fraction` (0.5) of the hour's real minutes are valid; on DST days the
repeated hour has 120 minutes and the skipped hour none. Missing days appear as unobserved
rows, so rows are consecutive calendar days.

A day is valid with ≥ `min_observed_hours_per_day` (18) observed hours. Metrics require
≥ `min_days` (5) valid days and at most `max_excluded_fraction` (0.2) unobserved cells within
valid days. With `x_i` the observed values in time order, `n` their count, `x̄` their mean,
and `x̄_h` the mean of clock hour h over valid days (Van Someren et al. 1999; Witting et al.
1990):

```text
IS = n · Σ_h (x̄_h − x̄)² / (p · Σ_i (x_i − x̄)²)       p = clock hours with data (24 when complete)
IV = n · Σ_pairs (x_i − x_{i−1})² / (n_pairs · Σ_i (x_i − x̄)²)
RA = (M10 − L5) / (M10 + L5)
```

`n_pairs` counts consecutive clock hours that are both observed, chaining 23:00 → 00:00
only between adjacent valid days. With complete data, `n_pairs = n − 1` and the formulas are
the published ones. M10 and L5 are the highest 10-h and lowest 5-h circular windows of the
average profile. `m10_onset_local_h` and `l5_onset_local_h` are their starting clock hours,
with ties going to the earliest.

## Statuses

`no_raw_data` (no samples), `extraction_failed` (unit unknown), `insufficient_observability`
(below the minima above), `not_applicable` (constant signal; no active bouts), `observed_zero`
(no activity on a well-observed day), `observed`.

## References

- van Hees VT, et al. (2011). Estimation of daily energy expenditure in pregnant and
  non-pregnant women using a wrist-worn tri-axial accelerometer. *PLoS ONE* 6:e22922.
- van Hees VT, et al. (2013). Separating movement and gravity components in an acceleration
  signal and implications for the assessment of human daily physical activity. *PLoS ONE*
  8:e61691.
- Van Someren EJW, et al. (1999). Bright light therapy: improved sensitivity to its effects
  on rest-activity rhythms in Alzheimer patients by application of nonparametric methods.
  *Chronobiology International* 16:505–518.
- Witting W, et al. (1990). Alterations in the circadian rest-activity rhythm in aging and
  Alzheimer's disease. *Biological Psychiatry* 27:563–572.
- Di J, et al. (2017). Patterns of sedentary and active time accumulation are associated with
  mortality in US adults: the NHANES study. *bioRxiv* 182337.

## Known limitations

- Thresholds (13 mg, 50 mg, 30 mg) come from wrist devices. For phones they are starting
  values to calibrate, not validated cut-points.
- Phones often sample accelerometry in bursts. Minutes with < 50 % observed time are
  invalid, so sparse sampling can make whole days `insufficient_observability`. That is the
  intended outcome.
- Non-wear detection is per call. For continuity across midnight, pass a window padded by
  `nonwear_window_ms`, or run it on the episode and slice per day.
