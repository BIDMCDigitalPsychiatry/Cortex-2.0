# Sleep features

Modules: `features/sleep_window.py` (phone-derived) and `features/wearable_sleep.py`
(wearable/platform-reported). The two are kept separate so they can be compared.

> The brief refers to the design document for the screen-dark sleep method. That document is
> GPS-only, so this page is the specification (`open_questions.md` #12).

## Phone-derived sleep window — `detect_sleep_window(screen_bouts, acc_epochs, *, night_date, tz, …)`

Construct: the main nightly **phone-rest window**, with the phone dark and the body still.
Based on Borger et al. (2019), who used the longest gap in touchscreen interactions, with
accelerometer trimming added.

1. **Night window:** `night_start_local` (18:00) on D−1 → `night_end_local` (12:00) on D.
   The night is attributed to **D**, the date on which it ends. DST is respected; a
   spring-forward night is 17 h.
2. **Grid:** `grid_ms` (5 min) cells. *Light* = overlapped by a screen-on bout or by a
   SensorKit `device_usage` interval with unlock time. *Covered* = the accelerometer observed
   an epoch in the cell, the cell is light, or a raw screen event falls in it.
3. **Merging:** dark runs separated by light runs ≤ `wake_tolerance_ms` (20 min) merge, and
   each merged light run is one awakening. The longest merged run is the candidate.
4. **Accelerometer trimming:** leading and trailing cells whose mean ENMO is observed and
   ≥ `still_threshold_mg` (10 mg) are removed. Unobserved cells never trim, and non-wear
   (phone on a nightstand) counts as still.
5. **Gating:** `night_coverage_fraction` ≥ `min_coverage_fraction` (0.7), otherwise
   `insufficient_observability`. A candidate touching the night-window edge is `censored`.
   A covered night with no dark cell gives `sleep_duration_ms = 0` (`observed_zero`), and
   the clock times are `not_applicable`.

Outputs: `sleep_onset_local_h`, `sleep_offset_local_h`, `sleep_midpoint_local_h` (clock
hours in [0, 24), midpoint taken at the elapsed-time midpoint), `sleep_duration_ms`,
`n_awakenings`, `sleep_onset_ms`, `sleep_offset_ms`, `sleep_status`,
`night_coverage_fraction`, with a per-value `_status`.

## Wearable sleep — `compute_wearable_sleep(sleep, *, night_start_ms, night_end_ms, source_available=True)`

From the canonical `sleep` stream (HealthKit sleep analysis). Segments are attributed to the
night in which they **end**. `healthkit_sleep_duration_ms` (in_sleep),
`healthkit_in_bed_ms` and `healthkit_awake_ms` are union lengths, so overlaps count once.
A category the device did not report that night is `not_applicable`, not zero. With no
source for the episode the status is `source_absent`.

## References

- Borger JN, Huber R, Ghosh A (2019). Capturing sleep–wake cycles by using day-to-day
  smartphone touchscreen interactions. *npj Digital Medicine* 2:73.

## Limitations

A participant who sleeps with the screen on, or whose phone is in another room while they
are awake, is misestimated. Coverage depends on accelerometer sampling, because screen
events alone cannot tell darkness from not recording. Thresholds are not validated against
PSG or actigraphy in this cohort.
