# Changelog

All notable changes to Cortex 2.0 are recorded here, one section per phase of the
feature-expansion work on `feature/cortex2-feature-expansion`. Each entry lists new
modules, parameter defaults and literature references so that a methods section can
be written from this file. Open design questions and brief-vs-repo conflicts are
tracked in [`open_questions.md`](open_questions.md).

## Phase 0 — Baseline repair

The suite was not green at `229f4cf` under current dependency resolution
(pandas 3.1, numpy 2.4, ruff 0.17): 10 of 165 tests failed and `ruff check src tests`
reported 68 findings.

### Fixed
- **Datetime → epoch-ms conversion assumed nanosecond resolution.** pandas ≥ 3 builds
  `datetime64[us, UTC]` by default, so `series.astype("int64") // 1_000_000` was wrong by a
  factor of 1000. Every canonical stream failed validation (`timestamp_ms` vs
  `timestamp_utc` mismatch) and episode bounds were mis-scaled. New
  `canonical/timestamps.py::datetime_to_epoch_ms` does the conversion with timedelta
  arithmetic (any resolution) and is used in `canonical/validation.py`,
  `canonical/episodes.py` and `adapters/generic/csv.py`.
  Tests: `tests/test_canonical_timestamps.py`.
- ruff 0.17 default rules: import ordering, redundant casts, `ClassVar` annotations on
  adapter/index class tables, `itertools.pairwise`, `typing.Annotated` CLI options.
  Behaviour unchanged; `int(round(...))` kept (with a `noqa`) because `round()` on a
  numpy < 2 float returns a float.

### Changed
- Dev extra: `pytest-cov>=5` added; `ruff>=0.17` (the version whose default rule set
  defines "ruff clean" for this work).

## Phase 1 — Schemas and adapter branches

### Added
- Canonical schemas `telephony`, `survey`, `cognitive`, `steps`, `sleep`
  (`src/cortex2_engine/schema_specs/*.yaml`), registered in `canonical/schemas.py`.
  - `telephony`: `channel` ∈ {call, sms, unknown}, `direction` ∈ {incoming, outgoing,
    missed, unknown}, `duration_ms`, `contact_hash`.
  - `survey`: `instrument`, `item`, `value_numeric`, `value_text`, `response_latency_ms`,
    `activity_id` (one row per answered item; scoring is a feature, not an adapter step).
  - `cognitive`: `task` (open snake_case set: jewels_a, jewels_b, trails_b, spatial_span…),
    `session_id`, `level`, `response_time_ms`, `correct`.
  - `steps`: `step_count`, optional `interval_duration_ms`.
  - `sleep`: `representation` ∈ {in_bed, in_sleep, in_awake, unknown}, `duration_ms`.
- **Battery decision:** no `battery` schema. Battery rides on
  `screen.battery_level_fraction` (already produced by the adapter and consumed by
  `missingness/hourly_context.py`).
- `episodes` schema: optional `device_os` (ios | android | unknown), required to emit
  `source_absent` for platform-impossible streams (telephony on iOS).
- `validation.py`: generic `allowed_values` check (any categorical column), alongside the
  existing `allowed_events`.
- `MindLAMPAdapter`: `_telephony`, `_survey`, `_cognitive`, `_steps`, `_sleep`; a
  `_builder()` dispatch table replaces the if/elif chain; `_pseudonymous_hash(kind,
  value)` is the single hashing primitive (blake2b-128 over
  `study_id|source_platform|kind|value`). `_nearby_identifier_hash` delegates to it and its
  output is byte-identical (tested). Telephony counterparts are digit-normalised and hashed
  in one `telephony_contact` namespace, so one person has one token across calls and SMS.
- Unit conversions made explicit: mindLAMP `call_duration` (seconds) → `duration_ms`.
  Direction codes follow Android `CallLog.Calls.TYPE` / `Telephony.Sms.TYPE`. Sleep codes
  follow HealthKit `HKCategoryValueSleepAnalysis`.
- Tests: `tests/test_mindlamp_adapter_new_streams.py` (13) with fixture CSVs under
  `tests/fixtures/mindlamp/`.

### Registry
- No change: this phase adds streams, not features.

## Phase 2 — Shared time utilities and status unification

### Added
- `cortex2_engine.time`: `local_day_bounds_ms(date, tz, day_start)`,
  `iter_local_days(start_ms, end_ms, tz, *, day_start)` (yields `LocalDay` records with
  clipped and full-day bounds), `local_dates`, `to_local_clock_hours`, `circular_sd_hours`,
  `circular_mean_hours`, `parse_clock`, `coerce_timestamp_ms`. All built on `zoneinfo`.
  Local days are 23 h / 24 h / 25 h; DST is tested on 2026-03-08 and 2026-11-01 in
  America/New_York.
  - Circular SD = `sqrt(-2 ln R) · 24 / 2π` hours (Mardia & Jupp 2000), where `R` is the
    mean resultant length of clock angles. `R < 1e-12` → `inf` (no mean direction).
- `StudyConfig.day_start` (default `"00:00"`, mirrors `configs/default.yaml::time.day_start`),
  with validation of `day_start` and of `timezone` as an IANA zone.
- `FeatureStatus`: new members `insufficient_observability`, `source_absent`, `censored`.
  `feature_empty` is deprecated and never emitted. Helpers `status_from_counts(n_observed_ms,
  minimum_observed_ms, value, *, n_records=None)` and `gated_value(value, status)`.
- `docs/status_vocabulary.md`; `tests/test_status_vocabulary.py` (AST scan of `features/`).

### Changed
- **GPS status values:** `observed_value` → `observed` in `gps_location_features` and
  `gps_observability` (now `FeatureStatus.<MEMBER>.value`). Outputs written earlier must be
  mapped on read.
- **GPS chain column names:** `build_gps_intervals` and
  `gps_degradation.apply_missing_bin_mask_to_points` take canonical `timestamp_ms`
  (and `accuracy_m`). The legacy `timestamp` still works with a `DeprecationWarning`, and
  legacy `accuracy` is accepted silently, so existing `workflows/` keep running unchanged.
- **`compute_daily_mobility_features`:** gains `minimum_supported_duration_ms=0.0` and
  `_status` keys for `supported_displacement_distance_m`, `supported_radius_of_gyration_m`
  and `implausible_transition_fraction`. With no spatially supported time the distance is
  now **NaN / `insufficient_observability`** (was `0.0`). With no displacement candidates the
  implausible-transition fraction is **NaN / `not_applicable`** (was `0.0`). Evidence counts
  (`displacement_edge_count`, `implausible_transition_count`) are always reported.
- `gps_location_episodes` and `gps_anchor_episodes` emit no feature statuses (only
  edge-level `spatial_status`), so they needed no change.

### Registry
- No new features. Status semantics for existing entries are now those of
  `docs/status_vocabulary.md`.

## Phase 3 — Accelerometer feature family

Specification: `docs/features/accelerometer.md`.

### Added
- `features/acc_preprocess.py`: `detect_unit(acc, *, gravity_band_g=(0.7, 1.3))`,
  `enmo_g(x, y, z)`, `resample_to_epochs(acc, *, window_start_ms, window_end_ms,
  epoch_ms=1000, unit=None)` (1 s epochs with exact SD/range sufficient statistics),
  `aggregate_epochs(epochs, *, epoch_ms=60_000, min_observed_fraction=0.5, exclude=None)`.
- `features/acc_nonwear.py`: `detect_nonwear(epochs, *, window_ms=3_600_000,
  step_ms=900_000, sd_threshold_mg=13.0, range_threshold_mg=50.0, min_axes=2,
  min_window_observed_fraction=0.5, battery=None, charging_min_rise_fraction=0.01,
  minimum_evaluable_ms=3_600_000)`, returning the epoch mask, intervals and a daily summary.
  van Hees et al. 2011 (GGIR form: SD **and** range per axis), with the phone adaptation:
  unobserved ≠ non-wear, union of still windows, charging annotation.
- `features/acc_circadian.py`: `hourly_activity_matrix` (DST-aware clock hours, contiguous
  days), `interdaily_stability`, `intradaily_variability`, `relative_amplitude`
  (M10/L5 with onsets) and `compute_circadian_features`. Defaults: `min_days=5`,
  `min_observed_hours_per_day=18`, `max_excluded_fraction=0.2`. Van Someren et al. 1999 /
  Witting et al. 1990 formulas with explicit missing-hour handling.
- `features/acc_fragmentation.py`: `activity_fragmentation(epochs_1min, *,
  active_threshold_mg=30.0, min_valid_minutes=600)` returning ASTP, SATP, mean active bout,
  active-bout Gini, active and sedentary minutes, and censored-bout count (Di et al. 2017);
  `gini_coefficient`.
- `features/acc_daily.py`: `prepare_minute_epochs` and `compute_daily_activity_features`
  (flat participant-day dict; `algorithm = cortex2.acc_daily` v1.0).
- Tests: `test_acc_preprocess.py`, `test_acc_nonwear.py`, `test_acc_circadian.py`,
  `test_acc_fragmentation.py`, `test_acc_daily.py` (43), with the synthetic generator
  `tests/acc_synthetic.py`. They cover an analytic sinusoid (IS = 1, IV → 4 sin²(π/24)),
  white noise (IV ≈ 2), 6 h of injected non-wear recovered exactly, g vs m/s² unit
  confusion, and a 23 h DST day.

### Registry
- 15 entries (`mean_enmo_mg`, `acc_nonwear_fraction`, fragmentation and circadian
  features), with `engine: cortex2`, `native: true`, `minimum_observability` and
  `reference`.

### References
van Hees 2011 (PLoS ONE 6:e22922); van Hees 2013 (PLoS ONE 8:e61691); Van Someren 1999
(Chronobiol Int 16:505); Witting 1990 (Biol Psychiatry 27:563); Di 2017 (bioRxiv 182337).

## Phase 4 — Screen and sleep

Specifications: `docs/features/screen.md`, `docs/features/sleep.md`.

### Added
- `features/screen_bouts.py`: `build_screen_bouts(screen, *, window_start_ms,
  window_end_ms, max_bout_ms=14_400_000, debounce_ms=1000)`. Bouts over the cap, open at the
  window end, or carried in from before the window are **kept** with `bout_status =
  censored` and a `censor_reason`. Also `screen_supported_duration_ms(…,
  max_silence_ms=43_200_000)`, `local_clock_intervals` (DST-aware recurring clock windows)
  and `compute_daily_screen_features(bouts, *, supported_duration_ms, tz,
  window_start_ms, window_end_ms, nocturnal_start_local="22:00",
  nocturnal_end_local="06:00", micro_bout_ms=30_000, min_supported_ms=43_200_000)`.
- `features/sleep_window.py`: `night_window_ms` and `detect_sleep_window(screen_bouts,
  acc_epochs, *, night_date, tz, night_start_local="18:00", night_end_local="12:00",
  grid_ms=300_000, wake_tolerance_ms=1_200_000, still_threshold_mg=10.0,
  min_coverage_fraction=0.7, screen_events=None, device_usage=None)`. Screen-dark run with
  awakening merging and accelerometer edge trimming (after Borger et al. 2019). The night is
  attributed to its end date.
- `features/wearable_sleep.py`: `compute_wearable_sleep(sleep, *, night_start_ms,
  night_end_ms, source_available=True)` giving `healthkit_sleep_duration_ms`,
  `healthkit_in_bed_ms` and `healthkit_awake_ms` (union lengths; unreported categories are
  `not_applicable`).
- Tests: `test_screen_bouts.py`, `test_sleep_window.py`, `test_wearable_sleep.py` (31).

### Registry
- 15 entries (screen totals and rates, nocturnal variants, phone sleep window, wearable sleep).

### References
Borger, Huber & Ghosh 2019 (npj Digit Med 2:73); Apple HealthKit
`HKCategoryValueSleepAnalysis`.

## Phase 5 — GPS upgrades

Specification: `docs/features/gps.md`. The existing chain is extended, not rewritten.

### Added
- `features/gps_geometry.py`: `haversine_m` (latitude first), `project_local_m` (local
  equirectangular, `cos(radians(lat0))`), `edge_midpoints`.
- `features/gps_jitter.py`: `filter_gps_jitter(gps, *, accuracy_max_m=100.0,
  max_speed_kmh=250.0)` → `(filtered, n_dropped_by_reason)`. Reasons:
  `missing_coordinates`, `invalid_coordinates`, `accuracy_exceeds_max`, `speed_spike`.
- `features/gps_stays.py`: `build_stays(anchor_episodes, *, sampling_interval_ms,
  r_stay_m=50.0, t_floor_ms=600_000, dwell_multiplier=3.0)`, with dynamic dwell threshold
  `max(t_floor_ms, dwell_multiplier × sampling_interval_ms)` on observed duration.
- `features/gps_home.py`: `infer_home_location(assignments, places, *, tz,
  reference_days=14, min_nights=5, night_start_local="00:00", night_end_local="06:00",
  min_night_observed_ms=7_200_000, min_confidence=0.5, r_min_m=50.0)` (episode-level
  night-majority home with `home_confidence`, `home_status` and `home_reason`), and
  `compute_daily_home_time(edges, home, *, r_min_m=50.0, min_supported_ms=21_600_000)`.
- `features/gps_activity_space.py`: `convex_hull`, `convex_hull_area_km2`,
  `standard_deviational_ellipse` and `compute_daily_activity_space(edges, *,
  min_supported_ms=21_600_000, median_accuracy_m=None)`. Areas at or below the squared
  median accuracy are `observed_zero`.
- `features/gps_routine.py`: `location_novelty_fraction` (28-day history, 7 history days
  required), `hourly_place_sequence` and `routine_index` (7-day window,
  `min_shared_hours=12`, `min_days=3`). DST days are merged by clock hour.
- `features/gps_circadian_movement.py`: `compute_circadian_movement(edges, *,
  period_band_h=(23.5, 24.5), n_periods=11, min_supported_ms=43_200_000,
  min_span_ms=86_400_000)`, after Saeb et al. 2015.
- Tests: `test_gps_jitter_stays.py`, `test_gps_home.py`, `test_gps_activity_space.py`,
  `test_gps_routine_circadian.py` (31), plus the `tests/gps_synthetic.py` helper. Coverage
  of the new modules is 100 %.

### Registry
- 9 entries: `home_time_ms`, `home_time_fraction`, `convex_hull_area_km2`,
  `sde_area_km2`, `anisotropy`, `major_axis_bearing_deg`, `location_novelty_fraction`,
  `routine_index`, `circadian_movement`. Legacy `hometime` (engine `cortex`) is unchanged.

### References
Andrew 1979; Barnett & Onnela 2020; Canzian & Musolesi 2015; Lefever 1926; Saeb et al. 2015;
Sherman et al. 2005.

## Phase 6 — Communication, cognition and surveys

Specifications: `docs/features/communication.md`, `docs/features/cognition_surveys.md`.

### Added
- `features/communication.py`: `compute_communication_features(telephony, *,
  window_start_ms, window_end_ms, device_os=None, source_observed_ms=None,
  stream_active_in_episode=False, min_observed_ms=43_200_000, suffix="")`, with daily and
  7-day wrappers. Provides counts and durations by channel and direction,
  `n_unique_contacts`, `communication_reciprocity`, `communication_entropy_nats` and
  `outgoing_fraction`. `source_absent` on iOS; zero counts need evidence that the logger
  ran.
- `features/anchors.py`: `build_daily_anchors(*, date, tz, sleep, screen_events,
  gps_edges, home, telephony, earliest_local="04:00")` giving sleep offset, first
  unlock, first home departure, and first/last communication.
- `features/social_rhythm.py`: `social_rhythm_regularity(anchors_long, *, window_days=7,
  min_days_per_anchor=4, regular_threshold_min=45.0, min_anchors=2)` using
  `circular_mean_hours` / `circular_sd_hours`.
- `features/cognitive_variability.py`: per-session `rt_cov`, `rt_tau_ms`
  (method-of-moments ex-Gaussian) and `rt_slope_ms_per_trial` (`min_trials=8`,
  `rt_min_ms=150`, `rt_max_multiple_of_median=3.0`), plus weekly medians.
- `features/survey_scoring.py`: pydantic `SurveyScoringSpec` (`category_list`,
  `questions`, `maps`, `reverse_items`, `min_items`), `score_surveys` and
  `load_builtin_spec`. Built-in `schema_specs/survey_specs/{phq9,gad7}.yaml` are added
  to package data.
- Tests: `test_communication.py`, `test_anchors_social_rhythm.py`,
  `test_cognitive_survey.py` (26).

### Registry
- 14 entries (communication 7, social rhythm 2, cognition 3, PHQ-9 and GAD-7 totals).

### References
Monk et al. 1990; Shannon 1948; MacDonald et al. 2006; Olivier & Norberg 2010; Kroenke et
al. 2001; Spitzer et al. 2006.

## Phase 7 — Completeness weight and missingness phenotype

Specification: `docs/features/missingness.md`.

### Added
- `quality/completeness.py`: `participant_day_completeness_weight(coverage_fraction,
  longest_gap_ms, n_missing_runs, n_bins, *, alpha=1.0, beta=1.0, bin_size_ms=3_600_000)`
  = coverage^α × (1 − longest-gap fraction)^β; `multi_stream_weight` (geometric mean);
  mask, interval and conservative-coverage helpers; `window_weight`.
- `completeness_weight` in every daily feature dict: accelerometer, screen, phone sleep,
  wearable sleep (NaN by design), GPS mobility, location, home time, activity space,
  novelty, routine and communication (`completeness_weight_7d` for weekly windows). GPS
  daily functions gain optional `window_start_ms` / `window_end_ms` keywords; without them
  the weight is NaN.
- `missingness/phenotype.py`: `daily_missingness_phenotype(hourly, *, tz,
  day_start="00:00", min_eligible_hours=12, max_gap_fraction_complete=0.1,
  dominance_fraction=0.6, low_battery_fraction=0.15)` with six classes (`no_signal`,
  `complete`, `battery_depletion`, `shared_interruption`, `gps_specific_dropout`,
  `mixed`), built on `add_hourly_missingness_transitions` and `battery_last`.
- Tests: `test_completeness.py`, `test_missingness_phenotype.py` (13).

### Registry
- `completeness_weight` and `missingness_phenotype` (category `quality`).

### References
Onset regimes: `docs/gps_missingness_wifi_module.md` (this repository).

## Phase 8 — Baselines and clinical alignment

Specification: `docs/features/baselines_clinical.md`.

### Added
- `baseline/personal.py`: `rolling_personal_baseline(daily, *, feature,
  window_observations ∈ {3, 5, 7, 10, 14}, statistics=("median", "mad", "iqr"),
  min_observations=3, weight_column="completeness_weight")`. Prior-only, weighted
  baselines with `baseline_value`, `deviation`, `robust_z` (MAD, with IQR fallback) and
  `direction`. `BaselineConfig` mirrors `configs/default.yaml::baseline`; `weighted_quantile`.
- `clinical/alignment.py`: `align_features_to_clinical(daily, clinical, *,
  preceding_windows_days=(1, 3, 7), same_day=True)` gives one row per assessment with
  mean/median/weighted mean, valid-day counts and mean `completeness_weight` per window.
  The assessment day is excluded from the preceding windows.
- Tests: `test_baseline_clinical.py` (9). Coverage: `baseline/` 99 %, `clinical/` 100 %.

### Registry
- Generic derived entries `robust_z` and `deviation`.

### References
Leys et al. 2013 (MAD scaling).
