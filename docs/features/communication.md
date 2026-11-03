# Communication, anchors and social rhythm (Phase 6)

## Communication — `features/communication.py`

`compute_communication_features(telephony, *, window_start_ms, window_end_ms, device_os=None,
source_observed_ms=None, stream_active_in_episode=False, min_observed_ms=43_200_000, suffix="")`,
with wrappers `compute_daily_communication(…, date, tz, day_start="00:00")` and
`compute_weekly_communication(…, date, tz, window_days=7, min_observed_ms=4 × 12 h)`. Weekly
keys carry the suffix `_7d`.

| Feature | Unit | Definition |
|---|---|---|
| `call_count` (`_incoming`, `_outgoing`, `_missed`) | count | calls by direction |
| `call_duration_ms` (`_incoming`, `_outgoing`) | ms | summed duration of connected calls |
| `sms_count` (`_incoming`, `_outgoing`) | count | messages by direction |
| `n_unique_contacts` | count | distinct `contact_hash` values |
| `communication_reciprocity` | fraction | contacts with outgoing **and** incoming/missed events ÷ contacts |
| `communication_entropy_nats` | nats | Shannon entropy of events per contact |
| `outgoing_fraction` | fraction | outgoing ÷ (incoming + outgoing); missed and `unknown` excluded |

Rows with `unknown` channel or direction are counted in `telephony_n_unknown` and excluded.
Rows without a hash count towards totals but not contact measures.

**Statuses.**
- `device_os == "ios"` → `source_absent` (iOS exposes no call or SMS log).
- A window with no usable rows → `observed_zero` counts **only** when
  `stream_active_in_episode` and `source_observed_ms ≥ min_observed_ms`. Otherwise
  `no_raw_data`, or `insufficient_observability` if only `unknown` rows exist.
- Ratios with an empty denominator → `not_applicable`.

## Anchors — `features/anchors.py`

`build_daily_anchors(*, date, tz, sleep=None, screen_events=None, gps_edges=None, home=None,
telephony=None, earliest_local="04:00")` returns long rows (`date_local`, `anchor`,
`timestamp_ms`, `clock_hours`):

| Anchor | Source |
|---|---|
| `sleep_offset` | `detect_sleep_window` for the night ending on `date` (`observed` or `censored`) |
| `first_unlock` | first `unlock`, else first `screen_on` |
| `first_home_departure` | first supported GPS edge from inside to outside the home radius; needs an `observed` home |
| `first_communication`, `last_communication` | first and last call/SMS |

All except `sleep_offset` are searched in the anchor day, from 04:00 to 04:00 the next day, so
late-night events belong to the evening before. Anchors that did not happen are absent rather
than imputed.

## Social rhythm — `features/social_rhythm.py`

`social_rhythm_regularity(anchors_long, *, end_date=None, window_days=7,
min_days_per_anchor=4, regular_threshold_min=45.0, min_anchors=2)`. This is a passive
analogue of the Social Rhythm Metric (Monk et al. 1990).

- For each anchor with ≥ 4 occurrences in the 7-day window, the habitual time is the circular
  mean. A hit is a day within ±45 min of it on the 24 h circle.
- `<anchor>_hit_fraction` and `<anchor>_circular_sd_h` (`circular_sd_hours`).
- `social_rhythm_regularity` is the mean hit fraction over eligible anchors, in [0, 1].
  `social_rhythm_mean_circular_sd_h` is the mean circular SD. Both need ≥ 2 eligible anchors.
- Times spread evenly round the clock (zero resultant length) have no habitual time; that
  anchor is not eligible.

## References

- Monk TH, Flaherty JF, Frank E, Hoskinson K, Kupfer DJ (1990). The Social Rhythm Metric: an
  instrument to quantify the daily rhythms of life. *J Nerv Ment Dis* 178:120–126.
- Shannon CE (1948). A mathematical theory of communication. *Bell Syst Tech J* 27:379–423.
