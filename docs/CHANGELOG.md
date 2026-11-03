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
