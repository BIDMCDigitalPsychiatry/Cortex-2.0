# Cortex vs Cortex 2.0: GPS Feature Design

**Status:** Design specification  
**Cortex 2.0 stage:** Feature semantics / reliability  
**Study instance used for development:** Digital Clinic  
**Purpose:** Define where Cortex 2.0 intentionally differs from Cortex before implementing Cortex 2.0-native GPS features.

## 1. Design principle

Cortex 2.0 is not intended to reproduce Cortex feature values.

Cortex is treated as:

1. an existing comparison pipeline;
2. a source of useful implementation ideas;
3. a benchmark whose assumptions and failure modes should be made explicit.

The Cortex 2.0 pipeline is:

Raw data
→ platform adapter
→ canonical sensor schema
→ source availability
→ observability / data quality
→ Cortex 2.0-native feature computation
→ feature semantics
→ degradation / reliability
→ participant-specific baselines
→ clinical alignment

Cortex may be run in parallel for comparison, but it is not the computational definition of Cortex 2.0 features.

---

# 2. Source-derived Cortex behavior

## 2.1 Home time

Cortex `hometime` depends on the primary feature `significant_locations`.

The returned home-time value is the duration of the significant location with `rank == 0`.

Therefore, operationally:

> Cortex home = highest-ranked significant location.

This is not an explicit inference of a participant's residential home.

### Concern

The most prevalent observed location can differ from a person's actual home because of:

- irregular GPS sampling;
- work or hospital stays;
- missing nighttime data;
- short analysis windows;
- changes in sampling density.

Cortex 2.0 should therefore distinguish:

- **dominant observed location**
from
- **inferred home location**.

---

## 2.2 Significant locations

The default Cortex implementation uses `method="mode"`.

For this method Cortex:

1. retrieves GPS records;
2. removes consecutive duplicate timestamps;
3. rounds latitude and longitude to 3 decimal places;
4. counts rounded coordinate pairs;
5. identifies sufficiently frequent coordinate pairs;
6. merges nearby clusters;
7. assigns cluster prevalence based on GPS-record counts.

The default parameters include:

- `min_cluster_size = 0.01`
- `max_dist = 300 m`
- `max_clusters = -1`

### Concern

Cluster prevalence is substantially driven by **number of GPS observations**.

This means that:

> observation density can influence inferred behavioral prevalence.

If one location is sampled every minute and another every ten minutes, point-count proportions do not necessarily represent proportional time spent at those locations.

Cortex 2.0 should use time-weighted occupancy rather than raw observation-count weighting.

---

# 3. Location entropy

Cortex entropy is calculated from the proportions returned by `significant_locations`.

Conceptually:

    entropy = -Σ p_i log(p_i)

where `p_i` is the significant-location proportion.

Because the underlying proportions are based on observed GPS records, Cortex entropy can inherit sampling-density effects.

## Cortex 2.0 replacement

Cortex 2.0 location entropy should be based on estimated **time occupancy**:

    p_i = estimated dwell time at location i
          ----------------------------------
          total observable location time

Therefore:

- a burst of frequent GPS sampling should not by itself increase a location's weight;
- unobservable time should not silently become time spent at any location;
- reliability should depend on temporal observability as well as location coverage.

---

# 4. Home inference

Cortex 2.0 should not define daily home as simply the most frequently sampled location on that day.

Instead, home should be a participant-level latent location estimated over a suitable longitudinal reference interval.

Candidate evidence may include:

- repeated nighttime occupancy;
- consistency across days;
- duration-weighted occupancy;
- stability across weeks.

The exact home-inference method will be implemented and validated separately.

A daily feature can then be:

> estimated observable time spent within the participant-specific home region.

Important distinction:

- `home_location` is a participant-level estimate;
- `home_time` is a day-level feature.

---

# 5. Cortex trip detection

Cortex `trips` operates directly on GPS.

The implementation:

1. reverses GPS events into chronological order;
2. computes time difference between consecutive GPS observations;
3. computes geographic displacement;
4. derives speed;
5. labels segments stationary/moving;
6. constructs trips from movement bouts.

Two implementation details require explicit validation before Cortex trip metrics are treated as a reference.

## 5.1 Time-threshold unit concern

The implementation computes:

    dt = timestamp difference / (1000 * 3600)

so `dt` is expressed in **hours**.

It then compares:

    dt > TIME_THRESHOLD

where:

    TIME_THRESHOLD = 600

This creates a unit/semantic concern that should be tested directly rather than inherited into Cortex 2.0.

Cortex 2.0 must use explicitly named units, for example:

    gap_seconds
    speed_kmh
    distance_km

No unlabeled numerical temporal threshold should be permitted.

## 5.2 Coordinate-order concern

The Cortex Haversine helper is defined with longitude/latitude arguments, while the call site appears to provide columns in the opposite order.

This should be tested using known geographic coordinates.

Cortex 2.0 should use a unit-tested distance implementation with explicit argument names.

---

# 6. Trip distance

Cortex 2.0 trip distance should be based on chronological GPS observations after explicit quality handling.

The computation must distinguish:

- plausible movement;
- GPS jitter;
- implausible jumps;
- long unobserved gaps;
- observations with poor positional accuracy.

A long temporal gap must not be interpreted as observed travel merely because the endpoints are geographically distant.

Cortex 2.0 should retain:

- observed path distance;
- number of movement segments;
- observable trip time;
- unresolved gap time;
- reliability metadata.

---

# 7. Trip duration

Trip duration should measure time classified as observable movement.

It must not count long missing intervals as observed travel.

Cortex 2.0 must distinguish:

- `observed_zero`: adequate observation and no trip detected;
- `insufficient_observability`: not enough evidence to determine whether travel occurred;
- `no_raw_data`: no GPS observations;
- `source_absent`: GPS source unavailable for the episode;
- `extraction_failure`: source should exist but processing failed.

A numerical zero must never be used as a generic substitute for missing data.

---

# 8. Missing-data semantics

This is a core Cortex 2.0 difference.

Every derived feature should have a value plus an explicit status.

Initial status vocabulary:

- `observed_value`
- `observed_zero`
- `insufficient_observability`
- `no_raw_data`
- `source_absent`
- `extraction_failure`
- `masked`

The feature value should be null whenever a numerical behavioral interpretation is not justified.

---

# 9. Feature output contract

A Cortex 2.0 feature result should contain at minimum:

- `study`
- `participant_id`
- `episode_id`
- `date`
- `feature`
- `value`
- `unit`
- `feature_status`
- `observability`
- `observed_duration`
- `n_observations`
- `algorithm`
- `algorithm_version`

Where appropriate it may also contain:

- `reliability`
- `uncertainty`
- `n_segments`
- `longest_gap`
- `source_platform`
- `parameter_set`

Feature value and feature reliability are different quantities.

For example:

> home_time = 14.2 hours

does not imply:

> home_time is reliable.

---

# 10. Reliability and degradation

Cortex 2.0 should estimate feature reliability empirically.

For high-observability reference days:

1. compute the Cortex 2.0-native reference feature;
2. impose controlled missingness;
3. recompute the feature;
4. compare degraded estimates with the reference;
5. model error as a function of observability and missingness pattern.

At minimum, simulate:

- approximately distributed loss;
- contiguous outage blocks.

Matched coverage levels should be compared.

Example:

> 50% distributed observation

may have very different reliability from:

> one contiguous 12-hour gap.

Therefore coverage alone should not define feature reliability.

---

# 11. Reference-day limitation

The initial Digital Clinic benchmark contains:

- 65 GPS DQ = 1.0 days
- from 3 participants

These days are useful for developing and internally testing degradation methods.

They are **not sufficient to establish universal reliability thresholds**.

Reliability thresholds should eventually be evaluated across:

- more participants;
- operating systems;
- sensing patterns;
- diagnoses;
- studies;
- feature engines.

---

# 12. Initial Cortex 2.0 GPS features

The first native implementation will include:

### Location

1. participant-level home-location inference;
2. daily time-weighted home time;
3. daily time-weighted location entropy.

### Mobility

4. daily trip distance;
5. daily trip duration.

Supporting quantities should include:

- GPS observability;
- number of observations;
- temporal coverage;
- longest unobserved gap;
- number of observable intervals;
- number of inferred trips;
- unresolved gap duration.

---

# 13. Cortex comparison

After Cortex 2.0-native features are implemented, Cortex and Cortex 2.0 should be compared on the same participant-days.

The comparison should ask:

- Do the feature values differ?
- Why do they differ?
- How sensitive is each feature to missingness?
- Does one implementation turn missingness into behavioral zero?
- Does sampling density affect the estimate?
- Which implementation degrades more gracefully?
- Which feature better preserves within-person behavioral change?

The goal is not to make Cortex 2.0 agree with Cortex.

The goal is to determine which assumptions produce more valid and reliable digital phenotypes.

---

# 14. Development rule

When implementing a Cortex 2.0 feature:

> Do not copy an existing feature algorithm merely to reproduce its numerical output.

Instead:

1. define the behavioral construct;
2. define the sensor evidence required to estimate it;
3. define missing/zero/failure semantics;
4. implement the estimator;
5. test controlled edge cases;
6. characterize degradation under missingness;
7. compare against existing implementations.

This specification governs the first Cortex 2.0-native GPS feature module.
