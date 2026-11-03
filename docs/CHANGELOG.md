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
