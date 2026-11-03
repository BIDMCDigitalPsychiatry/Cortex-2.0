# GPS features (Phase 5)

These modules extend the existing chain
(`build_gps_intervals` → location/anchor episodes → `build_place_candidates`) without
rewriting it. Distances are great-circle (`gps_geometry.haversine_m`, latitude first). Planar
work uses a local equirectangular projection with `cos(radians(lat0))`
(`gps_geometry.project_local_m`). Legacy Cortex swapped latitude and longitude in its
Haversine call and took the cosine of a latitude in degrees; neither bug is reproduced.

## Jitter filter — `gps_jitter.filter_gps_jitter(gps, *, accuracy_max_m=100.0, max_speed_kmh=250.0)`

Applied **before** `build_gps_intervals`. Each dropped row is counted once, under the first
reason that applies:

| Reason | Rule |
|---|---|
| `missing_coordinates` | time, latitude or longitude missing |
| `invalid_coordinates` | \|lat\| > 90, \|lon\| > 180, or exactly (0, 0) ("null island") |
| `accuracy_exceeds_max` | reported accuracy radius > `accuracy_max_m` |
| `speed_spike` | the speed into **and** out of a fix both exceed `max_speed_kmh` while the speed across it does not (an out-and-back jump) |

Sustained fast travel (a train) is kept, because the speed across the point is also high.
Rows with no accuracy are kept and counted in `n_accuracy_missing`.

## Stays — `gps_stays.build_stays(anchor_episodes, *, sampling_interval_ms, r_stay_m=50.0, t_floor_ms=600_000, dwell_multiplier=3.0)`

Consecutive anchor-compatible episodes merge while each one lies within `r_stay_m` of the
running duration-weighted centre. A stay is kept when its **observed** duration reaches

    dwell_threshold_ms = max(t_floor_ms, dwell_multiplier × sampling_interval_ms)

so sparse sampling cannot manufacture stays. `span_ms` includes unobserved gaps between
merged episodes; `observed_duration_ms` does not.

## Home — `gps_home.infer_home_location(assignments, places, *, tz, reference_days=14, min_nights=5, night_start_local="00:00", night_end_local="06:00", min_night_observed_ms=7_200_000, min_confidence=0.5, r_min_m=50.0)`

One record per participant-episode (GPS design doc §4: home is latent and episode-level, not
the day's top-ranked place).

1. Reference period: the first `reference_days` local days from the first assignment.
2. A local night (00:00–06:00, DST-aware) is valid when place assignments cover ≥ 2 h of it.
3. The night's place is the one with the most observed night time.
4. Home is the place winning most valid nights; `home_confidence` = its share of valid nights.
5. `home_status` = `observed` when there are ≥ `min_nights` valid nights and confidence ≥
   `min_confidence`; otherwise `insufficient_observability` with `home_reason`
   (`no_valid_nights`, `too_few_valid_nights`, `inconsistent_night_place`).

`home_radius_m = max(r_min_m, seed_support_radius_m)`.

### Daily home time — `compute_daily_home_time(edges, home, *, r_min_m=50.0, min_supported_ms=21_600_000)`

Supported edge time within `home_radius_m` of home. Edges with both ends inside count fully;
with one end inside, half. `home_time_fraction` divides by supported time, never by 24 h.
Unobserved time is never assumed to be at home. Without an observed home record both values
are NaN with the home status.

## Activity space — `gps_activity_space.compute_daily_activity_space(edges, *, min_supported_ms=21_600_000, median_accuracy_m=None)`

Points are midpoints of supported edges, weighted by duration and projected around their
weighted centre.

| Feature | Definition |
|---|---|
| `convex_hull_area_km2` | Andrew's monotone-chain hull, shoelace area (unweighted) |
| `sde_area_km2` | π·a·b, where a and b are the square roots of the eigenvalues of the duration-weighted 2×2 covariance (closed form) |
| `anisotropy` | 1 − b/a (0 = circle) |
| `major_axis_bearing_deg` | compass bearing of the major axis, in [0, 180) |

**Zeros.** An area at or below `median_accuracy_m²` (median of the edges' accuracies unless
supplied) is reported as `0.0` with `observed_zero`: the participant did not move beyond
measurement noise. The anisotropy and bearing of such an ellipse are `not_applicable`. Below 6 h
of supported time every value is `insufficient_observability`.

*Limitation:* the squared accuracy radius is the floor the brief specifies. Scatter with SD
close to the reported accuracy yields a hull larger than accuracy², so very noisy still days
can show small positive areas. Raise the floor (for example to π·accuracy²) if that matters
for a study.

## Routine — `gps_routine`

Inputs are episode-wide place assignments, so place ids are stable across days.

- `location_novelty_fraction(assignments, *, date, tz, history_days=28, min_history_days=7,
  min_supported_ms=21_600_000, min_day_observed_ms=3_600_000)`: the share of the day's
  assigned time at places with no assigned time in the previous 28 days. It needs 6 h on the
  day and ≥ 7 history days with ≥ 1 h each.
- `routine_index(assignments, *, date, tz, window_days=7, min_shared_hours=12, min_days=3,
  min_hour_observed_ms=1_800_000)`: each local clock hour gets its dominant place when ≥ 30 min
  is assigned. For each of the previous 7 days with ≥ 12 shared observed hours, similarity =
  matching hours / shared hours. The index is the mean over those days, and needs ≥ 3 of
  them. On a 25 h day the repeated hour is merged into one label; on a 23 h day the skipped
  hour is absent. Unobserved hours are excluded, never treated as "away". A fully observed
  day with no matches is `observed_zero`.

## Circadian movement — `gps_circadian_movement.compute_circadian_movement(edges, *, period_band_h=(23.5, 24.5), n_periods=11, min_supported_ms=43_200_000, min_span_ms=86_400_000)`

After Saeb et al. (2015). For each axis (projected east and north metres, which are linear in
longitude and latitude), a duration-weighted least-squares fit of a mean plus a sinusoid is
made at 11 periods across 23.5–24.5 h. The axis energy E is the mean share of variance
explained (normalised least-squares spectral power, in [0, 1]), and
`circadian_movement = log(E_east + E_north)`.

- Computed over a multi-day window; keep the window length fixed within a study.
- Needs ≥ 12 h supported time **and** a ≥ 24 h span (one full cycle).
- No movement on either axis → `not_applicable` (rhythm undefined), not zero.

*Difference from Saeb et al.:* they report the band energy of the raw least-squares spectrum
in degrees². The normalised power used here makes the value independent of how far the
participant travels, so it measures periodicity alone. This is listed in
`open_questions.md` (#17).

## Statuses

All `*_status` values come from `FeatureStatus` (`docs/status_vocabulary.md`).

## References

- Andrew AM (1979). Another efficient algorithm for convex hulls in two dimensions. *Inf
  Process Lett* 9:216–219.
- Barnett I, Onnela J-P (2020). Inferring mobility measures from GPS traces with missing
  data. *Biostatistics* 21:e98–e112.
- Canzian L, Musolesi M (2015). Trajectories of depression. *UbiComp '15*, 1293–1304.
- Lefever DW (1926). Measuring geographic concentration by means of the standard
  deviational ellipse. *Am J Sociol* 32:88–94.
- Saeb S et al. (2015). Mobile phone sensor correlates of depressive symptom severity in
  daily-life behavior. *J Med Internet Res* 17:e175.
- Sherman JE et al. (2005). A suite of methods for representing activity space in a
  healthcare accessibility context. *Int J Health Geogr* 4:24.
