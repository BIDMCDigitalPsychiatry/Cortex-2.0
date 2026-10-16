# GPS Missingness and Wi-Fi Context Analysis

**Status:** Complete  
**RAPHE stage:** Observability / missingness characterization  
**Study instance:** Digital Clinic  
**Completed:** October 2026

## Objective

Determine whether passive-sensing missingness is systematically patterned rather than a single undifferentiated process, with particular focus on GPS dropout and preceding nearby-device context.

The analysis distinguishes:

1. **GPS-specific onset:** GPS transitions from observed to absent while accelerometer data remain available/healthy.
2. **Shared onset:** GPS transitions to absent while accelerometer data are also missing or degraded.

An onset is counted only at the observed-to-missing transition; continuation hours within a gap are not counted as additional onsets.

## Analysis cohort

The contextual hourly risk set contained:

- 228 episodes with sufficient GPS/ACC evidence
- 184,558 hourly risk-set rows
- 886 GPS-specific onsets
- 2,391 shared GPS+ACC onsets

The Digital Clinic episode window is the interval from First Clinician Meeting through Last Clinician Meeting.

## Context features

Hourly context was derived from the following streams:

- accelerometer data quality
- screen/battery evidence
- device usage
- nearby-device detections
  - Wi-Fi
  - Bluetooth

Nearby-device data represent **detected nearby devices**, not confirmed network connections.

Therefore:

- Wi-Fi evidence = at least one Wi-Fi identifier detected in the relevant hour
- absence of Wi-Fi evidence does **not** imply that Wi-Fi was disabled or disconnected
- nearby-device counts should not be interpreted as scan counts

## Statistical approach

Primary onset analyses used cause-specific logistic regression:

- GPS-specific onset vs no onset, excluding shared-onset hours
- shared onset vs no onset, excluding GPS-specific-onset hours

Models used:

- episode fixed effects
- cluster-robust standard errors by episode
- previous-hour ACC data quality
- previous-hour screen evidence
- previous-hour Wi-Fi evidence
- previous-hour Bluetooth evidence
- prior onset history
- episode day
- UTC hour sine/cosine terms
- UTC day-of-week sine/cosine terms

Continuous predictors were standardized; binary evidence variables were not.

UTC time was included as a nuisance temporal control and should not be interpreted as local behavioral time.

These are fixed-effects logistic GLMs, not conditional logistic regression.

## Primary multimodal finding

Wi-Fi, rather than Bluetooth, accounted for most of the nearby-device association.

### GPS-specific onset

Adjusted previous-hour effects:

- Wi-Fi evidence: OR = 0.180, 95% CI 0.077–0.422, p < 0.001
- Bluetooth evidence: OR = 0.673, 95% CI crossing 1, p = 0.291

The four-state nearby-device model showed the same pattern relative to no nearby-device evidence:

- Wi-Fi only: OR = 0.177, 95% CI 0.077–0.410, p < 0.001
- Bluetooth only: OR = 0.588, p = 0.254
- Wi-Fi + Bluetooth: OR = 0.155, 95% CI 0.038–0.629, p = 0.009

The association remained after removal of onset-heavy episodes and across multiple previous-hour lags.

### Shared GPS+ACC onset

The pattern was qualitatively different:

- previous ACC quality: OR ≈ 0.22
- Wi-Fi evidence: OR ≈ 2.99, p < 0.001
- Bluetooth evidence: OR ≈ 1.34, p = 0.119

Lower battery level and later episode day were also associated with shared onset.

This supports at least two distinct missingness regimes rather than one generic sensor-failure process.

## Battery and device-state findings

For GPS-specific onset:

- low battery was not supported as an explanation
- ACC generally remained healthy
- device activity could continue
- the strongest contextual signal was absence of preceding Wi-Fi detection

For shared onset:

- ACC degradation was strong and consistent
- lower battery was associated with onset
- device evidence was reduced
- the pattern was more consistent with a broader interruption

## True phone operating system ascertainment

Phone OS was obtained directly from `lamp.analytics` SensorEvents.

Relevant analytics fields included:

- `data.type == "login"`
- `data.device_type`
- `data.user_agent`

Phone classification retained only:

- `iOS`
- `Android`

Non-phone event types such as `Dashboard` and `Apple watchOS` were excluded from phone classification.

Episode-level OS results:

- iOS: 184 episodes
- Android: 41 episodes
- unknown: 3 episodes
- mixed iOS/Android: 0 episodes

SensorKit was useful only as positive evidence of iOS:

- 163 episodes had SensorKit evidence within the episode
- all 163 were confirmed as iOS by `lamp.analytics`
- among 65 episodes without SensorKit evidence:
  - 41 were Android
  - 21 were iOS
  - 3 were unknown

Therefore:

> Absence of SensorKit data must not be used as an Android label.

## OS-stratified Wi-Fi association

### GPS-specific onset

Adjusted previous-hour Wi-Fi effects:

| OS | OR | 95% CI |
|---|---:|---:|
| iOS | 0.160 | 0.065–0.396 |
| Android | 0.275 | 0.036–2.084 |

Android vs iOS interaction:

- interaction OR ratio = 1.72
- p = 0.634

There is therefore no evidence that the GPS-specific Wi-Fi association differs by OS. The Android estimate is substantially less precise because fewer informative Android episodes were available.

### Shared onset

| OS | OR | 95% CI |
|---|---:|---:|
| iOS | 3.44 | 2.57–4.62 |
| Android | 1.25 | 0.51–3.07 |

Android vs iOS interaction:

- interaction OR ratio = 0.364
- p = 0.032

The positive Wi-Fi association with shared interruption was therefore stronger on iOS.

This should not yet be interpreted mechanistically because Wi-Fi sensing behavior differs substantially across operating systems.

## Platform differences in Wi-Fi sensing

Episode-level mean Wi-Fi evidence:

- iOS: 80.7% of hours
- Android: 42.8% of hours

Median episode-level Wi-Fi evidence:

- iOS: 86.1%
- Android: 37.0%

Raw Wi-Fi identifier counts also differed markedly across platforms. For example, during pooled no-onset hours:

- Android mean unique Wi-Fi IDs ≈ 51.5
- iOS mean unique Wi-Fi IDs ≈ 1.1

Therefore raw Wi-Fi counts are not directly comparable across iOS and Android. Binary within-platform evidence is the more defensible representation.

## Episode-balanced within-episode contrasts

For each episode, Wi-Fi evidence during onset hours was compared with that same episode's no-onset hours.

### GPS-specific onset

iOS:

- mean difference = -27.49 percentage points
- median = -23.95 pp
- 95% bootstrap CI = -37.79 to -17.33 pp
- 59.0% of informative episodes had lower Wi-Fi evidence at onset

Android:

- mean difference = -5.76 pp
- median = 0
- 95% CI = -16.18 to 2.95 pp

The iOS GPS-specific signal therefore remains strong after episode balancing.

### Shared onset

iOS:

- mean difference = -0.45 pp
- 95% CI = -3.01 to 2.05 pp

Android:

- mean difference = -3.82 pp
- 95% CI = -9.45 to 0.16 pp

The strong positive adjusted shared-onset association therefore should not be interpreted from pooled prevalence alone; it depends on the multivariable within-episode model structure.

## Event-centered iOS temporal profile

For iOS GPS-specific onsets, episode-balanced Wi-Fi evidence declined in the hours approaching onset:

| Relative hour | Wi-Fi evidence |
|---:|---:|
| -6 | 55.0% |
| -5 | 55.2% |
| -4 | 51.9% |
| -3 | 45.1% |
| -2 | 43.0% |
| -1 | 44.3% |
| 0 | 32.9% |
| +1 | 42.4% |
| +6 | 47.9% |

This supports a temporal contextual signature preceding GPS-specific missingness.

## GPS recovery analysis

Among 564 iOS GPS-specific onset events:

- 560/564 (99.3%) recovered within 12 hours
- median time to GPS recovery = 1 hour
- event-weighted Wi-Fi evidence at onset = 9.2%
- Wi-Fi evidence at GPS recovery = 13.4%

Among events with Wi-Fi absent at onset, only 8.9% had Wi-Fi evidence present at the hour when GPS recovered.

Therefore the data do **not** support the simple mechanism:

> Wi-Fi disappears → GPS cannot function → GPS returns only when Wi-Fi returns.

GPS generally recovered while Wi-Fi evidence remained absent.

The difference between the 32.9% event-centered onset estimate and the 9.2% recovery-analysis estimate is expected:

- 32.9% is episode-balanced
- 9.2% is event-weighted across all 564 onsets

Episodes with many recurrent onsets therefore contribute more heavily to the latter.

## Final interpretation

The completed analysis supports the following conclusions.

1. **Passive-sensing missingness is heterogeneous.** GPS-specific dropout and shared GPS+ACC interruption have different multimodal signatures.

2. **GPS-specific missingness has a strong preceding Wi-Fi-detection signature.** The association persists within episodes, after adjustment, after removing onset-heavy episodes, across lag analyses, and within confirmed iOS episodes.

3. **The signal is Wi-Fi-specific rather than generic nearby-device sensing.** Bluetooth evidence does not show the same association.

4. **The GPS-specific association is not demonstrably iOS-exclusive.** Android estimates point in the same direction, although they are less precise.

5. **Wi-Fi sensing itself is platform dependent.** Raw Wi-Fi counts and prevalence should therefore not be assumed comparable between iOS and Android.

6. **Wi-Fi absence is a contextual marker, not an established cause of GPS failure.** GPS usually recovers without Wi-Fi evidence returning.

7. The observed multimodal patterns are inconsistent with a simple unconditional MCAR interpretation.

8. These results do **not** by themselves establish MAR or MNAR, which require additional assumptions and/or sensitivity analyses.

## Language to use

Preferred:

> GPS-specific missingness exhibited a reproducible contextual signature characterized by reduced preceding Wi-Fi detection despite preserved accelerometer observability.

Also acceptable:

> Passive-sensing missingness was systematically patterned across modalities and differed between GPS-specific and broader sensor interruptions.

Avoid:

- "Wi-Fi causes GPS dropout."
- "GPS requires Wi-Fi."
- "The phone disconnected from Wi-Fi."
- "Wi-Fi was turned off."
- "GPS recovered when Wi-Fi reconnected."
- "The missingness is MAR."
- "The missingness is MNAR."

## Reproducible workflows

Primary workflows:

- `workflows/build_cohort_hourly_context.py`
- `workflows/model_gps_missingness_onsets.py`
- `workflows/model_gps_nearby_lag_profile.py`
- `workflows/model_gps_nearby_state.py`
- `workflows/build_lamp_analytics_episode_os.py`
- `workflows/model_gps_wifi_true_os_interaction.py`
- `workflows/audit_wifi_evidence_by_true_os.py`
- `workflows/audit_wifi_within_episode_by_true_os.py`
- `workflows/audit_ios_gps_wifi_event_profile.py`

Supporting LAMP analytics inspection workflows:

- `workflows/probe_lamp_analytics_os.py`
- `workflows/find_local_lamp_analytics.py`
- `workflows/inspect_lamp_analytics_structure.py`
- `workflows/run_probe_lamp_analytics_with_proxy.py`

## Key outputs

- `quality/hourly_missingness/gps_onset_contextual_risk_set.parquet`
- `quality/hourly_missingness/onset_models/`
- `quality/hourly_missingness/onset_models/os_stratified/`
- `quality/hourly_missingness/onset_models/os_interaction/`
- `quality/hourly_missingness/onset_models/true_os_interaction/`
- `inventory/risk_episode_lamp_analytics_os.csv`
- `inventory/lamp_analytics_phone_os_events.parquet`

## Decision

This missingness-mechanism branch is considered complete for the current RAPHE stage.

Further analyses should only be added if required for a specific manuscript question or external validation. The next RAPHE work should proceed to the next framework component rather than further elaborating the Wi-Fi mechanism.
