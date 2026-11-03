# Screen features

Module: `features/screen_bouts.py`.

## Bouts — `build_screen_bouts(screen, *, window_start_ms, window_end_ms, max_bout_ms=14_400_000, debounce_ms=1000)`

- On events: `screen_on`, `unlock`; off events: `screen_off`, `lock`; `unknown` is ignored.
  The encoding comes from the canonical `event` column and is not re-inferred.
- A bout runs from the first on event while off to the next off event. Repeated on events
  (`screen_on` → `unlock`) do not restart it.
- Off-to-on gaps shorter than `debounce_ms` merge two bouts.
- **Censoring, never dropping.** `censor_reason` ∈ {`left_window` (on at the window start,
  from earlier events or a leading off), `right_window` (still on at the end),
  `exceeds_max_bout` (longer than 4 h, which usually means a missed off event)}. Censored
  bouts have `bout_status = censored`. `counted_duration_ms = min(duration_ms, max_bout_ms)`.
  Legacy Cortex dropped bouts > 2 h; that is not reproduced (brief rule 5).

## Supported time — `screen_supported_duration_ms(screen, *, max_silence_ms=43_200_000)`

Screen data are event-driven, so silence is ambiguous. Time between consecutive screen or
battery events (or an event and the window edge) counts as observed when the gap is
≤ 12 h. A night without events is supported; a day without any event is not.

## Daily features — `compute_daily_screen_features(bouts, *, supported_duration_ms, …)`

| Feature | Unit | Definition |
|---|---|---|
| `screen_on_ms` | ms | Σ `counted_duration_ms` |
| `n_bouts` | count | number of bouts |
| `median_bout_ms` | ms | median length of uncensored bouts |
| `micro_bout_rate_per_hour` | 1/h | bouts < `micro_bout_ms` (30 s) per **supported** hour |
| `screen_fragmentation_index` | 1/h | bouts per screen-on hour |
| `nocturnal_screen_on_ms` | ms | screen-on time within local 22:00–06:00 |
| `nocturnal_n_bouts` | count | bouts overlapping 22:00–06:00 |

The minimum supported time is `min_supported_ms` (12 h). Below it, every value is NaN /
`insufficient_observability`. A supported day with no bouts gives `observed_zero` for the
totals and `not_applicable` for the median and fragmentation index. A total that includes a
censored bout is `censored`. The nocturnal window is DST-aware: on a spring-forward day,
00:00–06:00 is 5 elapsed hours.

## Limitations

The screen being on is not the same as attention. Supported time is a heuristic. Use
`device_usage` (iOS SensorKit) where the screen stream is absent.
