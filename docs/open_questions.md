# Open questions and brief-vs-repo conflicts

The feature-expansion brief says that where it conflicts with the repository, the
repository's existing design documents win; each conflict is written down here with the
resolution that was implemented. Items are flagged in the PR description.

| # | Topic | Conflict | Implemented resolution | Status |
|---|---|---|---|---|
| 1 | Protected branch | Brief says "do not push to `main`"; the repository's default branch is `master` (there is no `main`). | `master` treated as protected; work on `feature/cortex2-feature-expansion`, draft PR into `master`. | resolved |
| 2 | Baseline not green | Brief: "`pytest -q` is green at HEAD (161 test functions)". 161 is correct, but under pandas 3.1 10 of 165 collected tests failed, and ruff 0.17 reported 68 findings. | Phase 0 fixes the datetime-resolution bug and the lint findings (see CHANGELOG, Phase 0). | resolved |
| 3 | Same bug in a workflow | `workflows/build_episode_ios_sensorkit_evidence.py` also converts with `dt.astype("int64") / 1e6`. | Not changed: library code must not depend on `workflows/`, and the study scripts are out of scope. The owner should switch it to `datetime_to_epoch_ms`. | open |
| 4 | Contact hashing | Brief: "salted-sha256 approach used in `_nearby_identifier_hash`". That function uses **blake2b (16-byte digest)** over `study_id\|source_platform\|type\|address`, unsalted apart from the study id. | Repo wins: one shared `_pseudonymous_hash(kind, value)` (the existing blake2b scheme). Nearby-device tokens are unchanged. If a secret salt is wanted, add it once to `_pseudonymous_hash`; that changes every token, so it needs a re-run. | resolved, decision for owner |
| 5 | Device OS | Brief: telephony `source_absent` "when the episode platform is iOS". Episodes carry `source_platform` (= "mindlamp"), not the OS. | Optional `device_os` on the episodes schema and a `device_os=` keyword on features. `source_absent` only when `device_os == "ios"`. An unknown OS with no records gives `no_raw_data`. Populating `device_os` (e.g. from `workflows/build_lamp_analytics_episode_os.py`) is a study step. | resolved |
| 6 | Battery | Brief: add `battery.yaml` or ride on screen. | Ride on `screen.battery_level_fraction`; no battery schema. | resolved |
| 7 | New-stream export columns | No real mindLAMP exports for telephony/survey/cognitive/steps/sleep were available. | Accepted column names (documented per adapter method) follow mindLAMP field names: `call_type`/`sms_type`, `call_duration` in seconds, `call_trace`/`sms_trace`, activity `temporal_slices` as `question`/`value`/`duration`, HealthKit sleep codes. **Confirm against a real export.** | open |
| 8 | `unknown` category | Brief lists `direction` ∈ {incoming, outgoing, missed} and `representation` ∈ {in_bed, in_sleep, in_awake}. | `unknown` is also allowed, matching the existing screen schema (`allowed_events` includes `unknown`), so one unmapped code does not abort a whole file. Features ignore `unknown` rows and count them. | resolved |
