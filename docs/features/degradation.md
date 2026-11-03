# Degradation sweeps and `reliability_threshold`

Module: `cortex2_engine.features.degradation_sweep`. It does not depend on `workflows/`.

## Question

At what data coverage does a daily feature stop being reliable? Here, a feature
is reliable at a coverage level when the error caused by missing data is smaller
than the day-to-day variation within a person. That variation is what the
feature is meant to detect.

## Procedure

1. **Reference days.** The caller supplies well-observed participant-days as
   `ReferenceDay(participant_id, streams, window_start_ms, window_end_ms)`. The
   module never selects them. A threshold is only as good as its reference set.
2. **Masks.** Masks come from the existing
   `missingness_simulation.make_balanced_k_run_missing_bin_mask`, which gives
   exact coverage, exactly `n_runs` missing runs, random placement and a fixed
   seed. Each design cell is coverage × `n_runs` × replicate, with defaults
   coverage 0.9 … 0.3, `n_runs` ∈ {1, 4}, 5 replicates and 5-minute bins.
   Rows are removed with `gps_degradation.apply_missing_bin_mask_to_points`.
   Every degraded stream of a day loses the same bins.
3. **Error.** `feature_fn(streams, window_start_ms, window_end_ms)` runs on the
   intact rows and on the degraded rows, giving `abs_error = |degraded − reference|`.
   A value counts only when its `<feature>_status` is value-bearing
   (`observed` / `observed_zero`).
4. **Summary** (`summarise_sweep`), per feature × coverage × `n_runs`:
   - `median_abs_error` over pairs where both values exist;
   - `fraction_unavailable`, the share of degraded days with no value.

   A degraded day that turns `insufficient_observability` is the status system
   doing its job. It is reported in `fraction_unavailable`, not counted as error.
5. **Within-person MAD.** For each participant, take the MAD of the reference
   values across their reference days; a participant needs at least
   `min_days_per_participant` days (default 3). The median across participants
   gives `within_person_mad`.
6. **Thresholds** (`reliability_thresholds`). Each coverage level is judged by
   its worst run design.
   - `failure_coverage` is the highest tested coverage at which
     `median_abs_error > within_person_mad`.
   - `min_reliable_coverage` is the lowest tested coverage above every failing
     level.
   - When even the highest level fails, or the MAD is undefined, the threshold
     is NaN.

## Registry

`write_reliability_thresholds(thresholds, registry_path, *, reference_set)`
writes `min_reliable_coverage` into each entry's `reliability_threshold` and adds
`reliability_reference_set: <name>`. It edits only those lines, so comments and
layout are kept. It **refuses to run without a named `reference_set`**.

**No reference set ships with this repository, so every `reliability_threshold`
in `feature_registry/features.yaml` stays `null`.** A test guards this.
Producing real thresholds needs the following:

1. Choose reference days, e.g. days with ≥ 95 % coverage on the streams the
   feature uses, ≥ 3 per participant.
2. Run the sweep with a `feature_fn` wrapping the feature's compute function.
   An example wrapper is in `tests/test_degradation_sweep.py`.
3. Call `write_reliability_thresholds` with a reference-set name that identifies
   the study, data export and selection rule.

## Example

```python
from cortex2_engine.features.acc_daily import compute_daily_activity_features
from cortex2_engine.features.degradation_sweep import (
    ReferenceDay, run_degradation_sweep, summarise_sweep,
    reliability_thresholds, write_reliability_thresholds,
)

def acc_fn(streams, start, end):
    return compute_daily_activity_features(
        streams["accelerometer"], window_start_ms=start, window_end_ms=end
    )

sweep = run_degradation_sweep(reference_days, acc_fn, ["active_minutes", "mean_enmo_mg"])
t = reliability_thresholds(summarise_sweep(sweep))
write_reliability_thresholds(
    dict(zip(t["feature"], t["min_reliable_coverage"])),
    "feature_registry/features.yaml",
    reference_set="bidmc_smp2_2026-08-28_cov95",
)
```

## Caveats

- Masks are uniform-random in placement. Real missingness clusters at night and
  at low battery (`docs/features/missingness.md`), so these thresholds are
  optimistic for features that depend on the time of day. Run a sweep with
  night-anchored masks before trusting a threshold for sleep or circadian
  features.
- Features that are unavailable at low coverage (`fraction_unavailable` near 1)
  have no error to measure there. In that case the threshold reflects the
  feature's own observability gate.
- The comparison uses the median, so it describes a typical day, not the worst one.
