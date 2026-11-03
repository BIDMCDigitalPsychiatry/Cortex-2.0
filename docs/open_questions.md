# Open questions and brief-vs-repo conflicts

The feature-expansion brief says that where it conflicts with the repository, the
repository's existing design documents win; each conflict is written down here with the
resolution that was implemented. Items are flagged in the PR description.

| # | Topic | Conflict | Implemented resolution | Status |
|---|---|---|---|---|
| 1 | Protected branch | Brief says "do not push to `main`"; the repository's default branch is `master` (there is no `main`). | `master` treated as protected; work on `feature/cortex2-feature-expansion`, draft PR into `master`. | resolved |
| 2 | Baseline not green | Brief: "`pytest -q` is green at HEAD (161 test functions)". 161 is correct, but under pandas 3.1 10 of 165 collected tests failed, and ruff 0.17 reported 68 findings. | Phase 0 fixes the datetime-resolution bug and the lint findings (see CHANGELOG, Phase 0). | resolved |
| 3 | Same bug in a workflow | `workflows/build_episode_ios_sensorkit_evidence.py` also converts with `dt.astype("int64") / 1e6`. | Not changed: library code must not depend on `workflows/`, and the study scripts are out of scope. The owner should switch it to `datetime_to_epoch_ms`. | open |
