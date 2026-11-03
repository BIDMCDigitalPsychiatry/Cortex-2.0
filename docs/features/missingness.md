# Completeness weight and missingness phenotype (Phase 7)

This page is the design of record for both constructs. The repository's design documents do
not define them; see `open_questions.md` #12.

## Completeness weight — `quality/completeness.py`

A number in [0, 1] attached to every daily feature dict as `completeness_weight`. It weights
days in downstream models (baselines, clinical alignment). It never changes a feature's value
or status.

    w = coverage_fraction^α × (1 − longest_gap_ms / (n_bins × bin_size_ms))^β

`participant_day_completeness_weight(coverage_fraction, longest_gap_ms, n_missing_runs, n_bins,
*, alpha=1.0, beta=1.0, bin_size_ms=3_600_000)`

- Coverage penalises missing time. The longest-gap term penalises concentration: one 6 h hole
  can hide a whole behaviour, while scattered short holes cannot.
- `n_missing_runs` is a consistency check (0 runs ⇔ no gap).
- The span is `n_bins × bin_size_ms`, so 23 h and 25 h days normalise by their real length.
- `alpha = 1, beta = 0` reduces to plain coverage.

`multi_stream_weight(weights)` is the geometric mean of the finite per-stream weights. A zero
weight on any used stream gives 0; NaN entries (stream not used) are ignored.

### Per-function inputs

| Function | Weight from |
|---|---|
| `acc_daily.compute_daily_activity_features` | 1-min mask of worn, observed minutes (exact) |
| `screen_bouts.compute_daily_screen_features` | supported time over the window (conservative bound) |
| `sleep_window.detect_sleep_window` | night coverage (conservative bound) |
| `gps_mobility_features.compute_daily_mobility_features`, `gps_home.compute_daily_home_time`, `gps_activity_space.compute_daily_activity_space` | 5-min mask of supported edges (exact); needs `window_start_ms`/`window_end_ms` |
| `gps_location_features.compute_daily_location_features` | supported time (conservative bound); needs the window |
| `gps_routine.location_novelty_fraction`, `routine_index` | 5-min mask of place assignments on the local day |
| `communication.compute_communication_features` | `source_observed_ms` over the window (conservative bound); weekly key `completeness_weight_7d` |
| `wearable_sleep.compute_wearable_sleep` | NaN: device-reported segments have no observed-time denominator |
| `missingness.phenotype.daily_missingness_phenotype` | geometric mean of hourly GPS and ACC masks over eligible hours |

The **conservative bound** (`completeness_weight_from_coverage`) is used when only an
observed total is known. It assumes all missing time is one gap, so it is a lower bound on the
exact weight. Without a window the weight is NaN rather than guessed.

## Missingness phenotype — `missingness/phenotype.py`

`daily_missingness_phenotype(hourly, *, tz, day_start="00:00", min_eligible_hours=12,
max_gap_fraction_complete=0.1, dominance_fraction=0.6, low_battery_fraction=0.15)`

The input is the episode hourly table (`missingness.hourly_context`). The onset logic comes from
`missingness.transitions.add_hourly_missingness_transitions`, and battery from
`battery_last` (`screen.battery_level_fraction`). Hours with an unavailable or partial source
are not eligible.

Gap hours per day are split into: GPS-only (ACC observed), shared (both missing) and ACC-only.
The two onset regimes come from `docs/gps_missingness_wifi_module.md`.

| # | Class | Rule (first match wins) |
|---|---|---|
| 1 | `no_signal` | every eligible hour is a shared gap |
| 2 | `complete` | gap hours ≤ 10 % of eligible hours |
| 3 | `battery_depletion` | shared gap hours ≥ 60 % of gap hours **and** ≥ half of shared onsets follow an hour whose last battery reading is ≤ 0.15 |
| 4 | `shared_interruption` | shared gap hours ≥ 60 % of gap hours (no low-battery onsets) |
| 5 | `gps_specific_dropout` | GPS-only gap hours ≥ 60 % of gap hours |
| 6 | `mixed` | otherwise (including accelerometer-only gaps) |

Statuses: `observed` with ≥ 12 eligible hours; `insufficient_observability` below that;
`source_absent` when no hour is eligible. Without battery data a shared gap is never labelled
`battery_depletion`.

**Not** a causal mechanism (MCAR/MAR/MNAR), and not proof that the device switched off.
