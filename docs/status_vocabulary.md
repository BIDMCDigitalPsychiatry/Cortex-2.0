# Feature status vocabulary

Every scalar feature `<name>` that Cortex 2.0 emits is paired with `<name>_status`. The
status says **how the value may be interpreted**. It is not a quality score; quality is
carried separately by `observability`, `observed_duration_ms` and `completeness_weight`.

Defined in `src/cortex2_engine/features/status.py` (`FeatureStatus`, a `str` Enum).
`tests/test_status_vocabulary.py` parses every module under `features/` and fails if any
emitted `*_status` literal is not a member.

| Status | Value | Use when |
|---|---|---|
| `observed` | number | Computed with adequate observability. |
| `observed_zero` | `0` | Adequate observability and the construct is genuinely absent (0 trips, 0 calls). The **only** status under which a numeric zero may be emitted. |
| `insufficient_observability` | NaN | Some data, but below the feature's documented minimum (e.g. `min_supported_ms`). |
| `no_raw_data` | NaN | Stream exists for the episode, zero records in the window. |
| `source_absent` | NaN | Stream cannot exist for this episode/platform (telephony on iOS; source interval does not cover the day). |
| `extraction_failed` | NaN | Processing error; the long table carries `error_type`. |
| `analysis_masked` | retained | Excluded by a downstream rule (e.g. a clinical-window filter); the value is kept for audit. |
| `censored` | lower bound | The value is a bound, not an estimate: a screen bout hit `max_bout_ms`, or a bout/night is truncated by the window or episode end. |
| `not_applicable` | NaN | The construct is undefined for this unit even with good data (median bout length on a day with no bouts; implausible-transition fraction with no displacement candidates). |
| `feature_empty` | — | **Deprecated, never emitted.** Kept so that older outputs still parse. |

## Precedence

`status_from_counts(n_observed_ms, minimum_observed_ms, value, *, n_records=None)` applies:

1. `n_records == 0` → `no_raw_data`
2. observed time missing, ≤ 0, or below the minimum → `insufficient_observability`
3. value missing → `not_applicable`
4. value == 0 → `observed_zero`
5. otherwise → `observed`

`source_absent`, `extraction_failed`, `analysis_masked` and `censored` are set explicitly by
the code that knows about them; they are never inferred from counts.
`gated_value(value, status)` returns NaN unless the status is value-bearing (`observed`,
`observed_zero`, `censored`, `analysis_masked`).

## Relation to the GPS design document

`docs/legacy_cortex_vs_cortex2_gps_feature_design.md` §8 lists an *initial* vocabulary
(`observed_value`, `extraction_failure`, `masked`). The code contract is the enum above:
`observed_value` → `observed`, `extraction_failure` → `extraction_failed`, `masked` →
`analysis_masked`. Outputs written before this change contain `observed_value` and should
be mapped on read.

## Not feature statuses

`spatial_status` (`gps_spatial_semantics`) and `trajectory_status`
(`gps_trajectory_plausibility`) label individual GPS **edges** (`temporal_gap`,
`accuracy_overlap`, `plausible_displacement`…). They describe evidence, not the
interpretability of a feature value, and keep their own vocabularies.
`quality/daily.py::source_status` (a `SourceBoundaryStatus`) likewise describes source
availability, not a feature.
