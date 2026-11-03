# Daily feature driver and CLI (Phase 9)

## CLI

```bash
cortex2 features --config study.yaml --start 2026-05-04 --end 2026-05-31
```

`--start` / `--end` are inclusive local dates and both are optional. The study config needs
an `adapter` section in addition to the fields `cortex2 inventory` uses:

```yaml
name: my_study
timezone: America/New_York
day_start: "00:00"
output_dir: out
windows:
  path: windows.csv            # participant_id, mindlamp_uid, clinical_start, clinical_end
adapter:
  kind: mindlamp               # or generic_csv (then: config_path: generic.yaml)
  passive_root: raw            # raw/<prefix>/<prefix>_<uid>_*.csv
  participant_map: map.csv     # optional
  source_manifest_dir: null    # optional
```

Relative paths resolve against the config file. Episodes are built from the clinical windows
with `build_episodes_from_windows` (inclusive end dates). Output is
`<output_dir>/features/daily_features_long.parquet`.

## Library

`features/daily_driver.py`:

- `iter_daily_features(adapter, episodes, config, *, start_date=None, end_date=None,
  params=None)` yields one long table per episode, reading streams through
  `canonical.reader.EpisodeReader`.
- `write_daily_features(...)` concatenates the tables and writes the parquet.
- `compute_episode_daily_features(reader, episode, *, study, tz, day_start, …)` handles a
  single episode.
- `DriverParameters` holds the thresholds passed to the feature functions; its `hash()` is
  `parameter_set_hash`.

## Output (long)

| Column | Meaning |
|---|---|
| `study`, `participant_id`, `episode_id`, `date_local` | identity |
| `feature`, `value`, `unit` | value is NaN unless the status is value-bearing |
| `feature_status` | `FeatureStatus` value |
| `observability` | `observed_duration_ms` ÷ the family's window (local day; night for sleep) |
| `observed_duration_ms` | evidence time for the family (worn minutes, screen-supported time, supported GPS edges, night coverage) |
| `n_observations` | raw rows for the family that day |
| `completeness_weight` | `docs/features/missingness.md` |
| `algorithm`, `algorithm_version`, `parameter_set_hash` | provenance |

## Families per local day

| Family | Inputs | Features |
|---|---|---|
| accelerometer | `accelerometer` | ENMO, non-wear, fragmentation |
| screen | `screen` (events before the day carry state in) | bouts and rates |
| sleep | `screen` + `accelerometer` (night 18:00–12:00 ending on the date) | phone sleep window |
| wearable_sleep | `sleep` | HealthKit totals |
| gps | `gps` → jitter filter → intervals (60 s max gap) → semantics → plausibility; episode-wide place candidates | mobility, activity space, location, home time, novelty, routine |
| communication | `telephony`; logger evidence = screen-supported time | counts, contacts, ratios |

Multi-day constructs (IS/IV/RA, circadian movement, social rhythm, baselines, clinical
alignment, surveys, cognition) are not emitted per day. Compute them from this table or from
their own functions.

## Known limitations

- Episodes from clinical windows are UTC-midnight bounded (`build_episodes_from_windows`), so
  in a non-UTC study the first and last local days are partial (`open_questions.md` #23).
  Restrict with `--start`/`--end` to full days.
- `device_os` is not populated by the CLI, so telephony never reports `source_absent` from the
  CLI path (#5). Add a `device_os` column to the episodes to enable it.
- Each stream is read whole for an episode. Run very long accelerometer episodes one at a time.
