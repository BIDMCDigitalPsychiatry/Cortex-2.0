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
