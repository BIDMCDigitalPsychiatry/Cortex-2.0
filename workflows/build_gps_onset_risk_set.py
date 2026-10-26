from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "../cortex2_outputs/digital_clinic/quality/"
    "hourly_missingness/gps_missingness_transitions.parquet"
)

OUTPUT = Path(
    "../cortex2_outputs/digital_clinic/quality/"
    "hourly_missingness/gps_onset_risk_set.parquet"
)


x = pd.read_parquet(INPUT)

x = x.sort_values(
    [
        "episode_id",
        "hour_index",
    ]
).copy()


# ------------------------------------------------------------
# Primary analysis:
# known source boundary + valid consecutive transition
# ------------------------------------------------------------

risk = x[
    x["transition_eligible"]
    &
    x["boundary_known"]
    &
    x["prev_gps_observed"]
    .fillna(False)
].copy()


# ------------------------------------------------------------
# Competing transition outcome
# ------------------------------------------------------------

risk["onset_type"] = "no_onset"

risk.loc[
    risk["gps_specific_gap_onset"],
    "onset_type",
] = "gps_specific_onset"

risk.loc[
    risk["shared_gap_onset"],
    "onset_type",
] = "shared_gap_onset"


# Sanity check
if (
    risk["gps_specific_gap_onset"]
    &
    risk["shared_gap_onset"]
).any():
    raise RuntimeError(
        "GPS-specific and shared onset overlap."
    )


# ------------------------------------------------------------
# Lagged context only
# ------------------------------------------------------------

risk["prev_screen_active"] = (
    risk[
        "prev_screen_event_observed"
    ]
    .fillna(False)
    .astype(bool)
)


# Previous-hour quality
risk["prev_gps_dq"] = pd.to_numeric(
    risk["prev_gps_dq"],
    errors="coerce",
)

risk["prev_acc_dq"] = pd.to_numeric(
    risk["prev_acc_dq"],
    errors="coerce",
)


# ------------------------------------------------------------
# Participant-specific history up to, but not including,
# the current hour
# ------------------------------------------------------------

risk["prior_gps_specific_onsets"] = (
    risk.groupby(
        "episode_id",
        sort=False,
    )["gps_specific_gap_onset"]
    .transform(
        lambda s: (
            s.astype(int)
            .cumsum()
            .shift(
                fill_value=0
            )
        )
    )
)

risk["prior_shared_onsets"] = (
    risk.groupby(
        "episode_id",
        sort=False,
    )["shared_gap_onset"]
    .transform(
        lambda s: (
            s.astype(int)
            .cumsum()
            .shift(
                fill_value=0
            )
        )
    )
)


# ------------------------------------------------------------
# Calendar context
#
# These are UTC descriptors only.
# Do NOT interpret them as participant-local behavioral time.
# ------------------------------------------------------------

t = pd.to_datetime(
    risk["hour_start_utc"],
    utc=True,
)

risk["utc_hour"] = (
    t.dt.hour
)

risk["utc_day_of_week"] = (
    t.dt.dayofweek
)

risk["utc_weekend"] = (
    risk["utc_day_of_week"]
    .ge(5)
)


# ------------------------------------------------------------
# Episode-relative time
# ------------------------------------------------------------

risk["episode_hour"] = (
    risk["hour_index"]
)

risk["episode_day"] = (
    risk["hour_index"]
    // 24
)


# ------------------------------------------------------------
# Binary outcomes, useful for later models
# ------------------------------------------------------------

risk["any_gps_onset"] = (
    risk["onset_type"]
    .ne("no_onset")
)

risk["gps_specific_onset"] = (
    risk["onset_type"]
    .eq("gps_specific_onset")
)

risk["shared_onset"] = (
    risk["onset_type"]
    .eq("shared_gap_onset")
)


risk.to_parquet(
    OUTPUT,
    index=False,
)


print("=" * 70)
print("GPS ONSET RISK SET")
print("=" * 70)

print(
    "Risk hours:",
    f"{len(risk):,}"
)

print(
    "\nOutcome:"
)

print(
    risk[
        "onset_type"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nOutcome percentages:"
)

print(
    (
        100
        * risk[
            "onset_type"
        ].value_counts(
            normalize=True
        )
    )
    .round(3)
    .to_string()
)


print(
    "\nParticipants:"
)

print(
    risk[
        "participant_id"
    ].nunique()
)


print(
    "\nParticipants with GPS-specific onset:"
)

print(
    risk.loc[
        risk[
            "gps_specific_onset"
        ],
        "participant_id",
    ].nunique()
)


print(
    "\nPrevious-hour ACC DQ by outcome:"
)

print(
    risk.groupby(
        "onset_type"
    )[
        "prev_acc_dq"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
        ]
    )
    .to_string()
)


print(
    "\nPrevious-hour screen activity by outcome:"
)

print(
    pd.crosstab(
        risk[
            "onset_type"
        ],
        risk[
            "prev_screen_active"
        ],
        normalize="index",
    )
    .round(3)
    .to_string()
)


print(
    "\nPrior GPS-specific onset history:"
)

print(
    risk.groupby(
        "onset_type"
    )[
        "prior_gps_specific_onsets"
    ]
    .agg(
        [
            "mean",
            "median",
            "max",
        ]
    )
    .round(3)
    .to_string()
)


print("\nSaved:")
print(OUTPUT)
