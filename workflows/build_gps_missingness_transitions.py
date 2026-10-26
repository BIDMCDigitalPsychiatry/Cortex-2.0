from pathlib import Path

import pandas as pd

from cortex2_engine.missingness.transitions import (
    add_hourly_missingness_transitions,
)


ROOT = Path(
    "../cortex2_outputs/digital_clinic/quality/"
    "hourly_missingness"
)

EPISODE_DIR = ROOT / "episodes"

RUNS_PATH = ROOT / "gap_runs.parquet"

OUTPUT = (
    ROOT
    / "gps_missingness_transitions.parquet"
)


# ============================================================
# Boundary audit of existing run table
# ============================================================

runs = pd.read_parquet(
    RUNS_PATH
)

gps_runs = runs[
    runs["gap_pattern"].eq(
        "gps_specific_gap"
    )
].copy()


print("=" * 70)
print("GPS-SPECIFIC RUNS BY SOURCE-BOUNDARY CERTAINTY")
print("=" * 70)


boundary_summary = (
    gps_runs.groupby(
        "any_boundary_unknown",
        dropna=False,
    )
    .agg(
        runs=(
            "run_id",
            "size",
        ),

        participants=(
            "participant_id",
            "nunique",
        ),

        total_hours=(
            "duration_hours",
            "sum",
        ),

        median_run_hours=(
            "duration_hours",
            "median",
        ),

        median_supporting_acc_dq=(
            "mean_supporting_dq",
            "median",
        ),

        runs_with_screen_activity=(
            "screen_event_hours",
            lambda s: int(
                s.gt(0).sum()
            ),
        ),
    )
)

print(
    boundary_summary.to_string()
)


# ============================================================
# Build transition-level table
# ============================================================

files = sorted(
    EPISODE_DIR.glob("*.parquet")
)

frames = []


for path in files:

    x = pd.read_parquet(
        path
    )

    x = add_hourly_missingness_transitions(
        x
    )

    frames.append(x)


out = pd.concat(
    frames,
    ignore_index=True,
    sort=False,
)

out.to_parquet(
    OUTPUT,
    index=False,
)


print(
    "\n" + "=" * 70
)
print("GPS GAP TRANSITIONS")
print("=" * 70)


eligible = out[
    out["transition_eligible"]
].copy()

print(
    "Transition-eligible hours:",
    f"{len(eligible):,}"
)

print(
    "GPS gap onsets:",
    int(
        eligible[
            "gps_gap_onset"
        ].sum()
    )
)

print(
    "GPS-specific gap onsets:",
    int(
        eligible[
            "gps_specific_gap_onset"
        ].sum()
    )
)

print(
    "Shared GPS+ACC gap onsets:",
    int(
        eligible[
            "shared_gap_onset"
        ].sum()
    )
)

print(
    "GPS recoveries:",
    int(
        eligible[
            "gps_recovery"
        ].sum()
    )
)


# ============================================================
# Boundary-known primary subset
# ============================================================

known = eligible[
    eligible["boundary_known"]
].copy()

onsets = known[
    known["gps_gap_onset"]
].copy()


print(
    "\n" + "=" * 70
)
print("BOUNDARY-KNOWN GPS GAP ONSETS")
print("=" * 70)

print(
    "Onsets:",
    len(onsets)
)

print(
    "GPS-specific:",
    int(
        onsets[
            "gps_specific_gap_onset"
        ].sum()
    )
)

print(
    "Shared:",
    int(
        onsets[
            "shared_gap_onset"
        ].sum()
    )
)

print(
    "Onsets with screen event in current hour:",
    int(
        onsets[
            "screen_event_observed"
        ].sum()
    )
)

print(
    "Onsets preceded by screen event:",
    int(
        onsets[
            "prev_screen_event_observed"
        ]
        .fillna(False)
        .sum()
    )
)


gps_specific = onsets[
    onsets[
        "gps_specific_gap_onset"
    ]
].copy()


print(
    "\nGPS-specific onset supporting ACC DQ:"
)

print(
    gps_specific[
        "acc_dq"
    ]
    .describe(
        percentiles=[
            .25,
            .50,
            .75,
            .90,
            .95,
        ]
    )
    .to_string()
)


print(
    "\nACC DQ in hour BEFORE GPS-specific onset:"
)

print(
    gps_specific[
        "prev_acc_dq"
    ]
    .describe(
        percentiles=[
            .25,
            .50,
            .75,
            .90,
            .95,
        ]
    )
    .to_string()
)


print(
    "\nParticipants with boundary-known GPS-specific onsets:",
    gps_specific[
        "participant_id"
    ].nunique()
)


print("\nSaved:")
print(OUTPUT)
