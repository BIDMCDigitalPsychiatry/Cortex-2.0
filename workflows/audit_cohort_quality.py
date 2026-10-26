from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "../cortex2_outputs/digital_clinic/quality"
)

gps = pd.read_parquet(
    ROOT / "gps_daily.parquet"
)

acc = pd.read_parquet(
    ROOT / "accelerometer_daily.parquet"
)

screen = pd.read_parquet(
    ROOT / "screen_daily.parquet"
)


KEY = [
    "participant_id",
    "episode_id",
    "source_participant_id",
    "date",
    "study_day",
]


# ------------------------------------------------------------
# Basic integrity
# ------------------------------------------------------------

for name, df in [
    ("gps", gps),
    ("accelerometer", acc),
    ("screen", screen),
]:
    if df.duplicated(KEY).any():
        raise RuntimeError(
            f"Duplicate daily keys in {name}"
        )

    print(
        f"{name}: "
        f"{len(df):,} rows | "
        f"{df['episode_id'].nunique()} episodes"
    )


# ------------------------------------------------------------
# Stream-level audit
# ------------------------------------------------------------

def audit_scalar_dq(
    df,
    name,
):
    print(
        "\n" + "=" * 70
    )
    print(name.upper())
    print("=" * 70)

    print("\nSource status:")
    print(
        df["source_status"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nDQ availability by source status:")
    x = (
        df.assign(
            dq_available=(
                df["data_quality"]
                .notna()
            ),
            dq_zero=(
                df["data_quality"]
                .eq(0)
            ),
            has_observations=(
                df["n_observations"]
                .gt(0)
            ),
        )
        .groupby(
            "source_status",
            dropna=False,
        )
        .agg(
            days=(
                "study_day",
                "size",
            ),
            dq_available=(
                "dq_available",
                "sum",
            ),
            dq_zero=(
                "dq_zero",
                "sum",
            ),
            days_with_observations=(
                "has_observations",
                "sum",
            ),
            mean_dq=(
                "data_quality",
                "mean",
            ),
            median_dq=(
                "data_quality",
                "median",
            ),
        )
    )

    print(
        x.to_string()
    )

    unavailable = (
        df["source_unavailable"]
        | df["partial_source"]
    )

    if (
        df.loc[
            unavailable,
            "data_quality",
        ]
        .notna()
        .any()
    ):
        raise RuntimeError(
            f"{name}: unavailable source "
            "days contain final DQ."
        )

    observable_empty = (
        df["source_status"]
        .eq(
            "within_declared_boundary"
        )
        &
        df["n_observations"]
        .eq(0)
    )

    if not (
        df.loc[
            observable_empty,
            "data_quality",
        ]
        .fillna(-1)
        .eq(0)
        .all()
    ):
        raise RuntimeError(
            f"{name}: available empty days "
            "are not consistently DQ=0."
        )


audit_scalar_dq(
    gps,
    "gps",
)

audit_scalar_dq(
    acc,
    "accelerometer",
)


print(
    "\n" + "=" * 70
)
print("SCREEN")
print("=" * 70)

screen_audit = (
    screen.assign(
        has_events=(
            screen["n_events"]
            .gt(0)
        )
    )
    .groupby(
        "source_status",
        dropna=False,
    )
    .agg(
        days=(
            "study_day",
            "size",
        ),
        days_with_events=(
            "has_events",
            "sum",
        ),
        total_events=(
            "n_events",
            "sum",
        ),
    )
)

print(
    screen_audit.to_string()
)


# ------------------------------------------------------------
# Joint daily table
# ------------------------------------------------------------

gps_joint = gps[
    KEY
    + [
        "source_status",
        "boundary_unknown",
        "source_unavailable",
        "n_observations",
        "data_quality",
    ]
].rename(
    columns={
        "source_status":
            "gps_source_status",
        "boundary_unknown":
            "gps_boundary_unknown",
        "source_unavailable":
            "gps_source_unavailable",
        "n_observations":
            "gps_n_observations",
        "data_quality":
            "gps_data_quality",
    }
)

acc_joint = acc[
    KEY
    + [
        "source_status",
        "boundary_unknown",
        "source_unavailable",
        "n_observations",
        "data_quality",
    ]
].rename(
    columns={
        "source_status":
            "acc_source_status",
        "boundary_unknown":
            "acc_boundary_unknown",
        "source_unavailable":
            "acc_source_unavailable",
        "n_observations":
            "acc_n_observations",
        "data_quality":
            "acc_data_quality",
    }
)

screen_joint = screen[
    KEY
    + [
        "source_status",
        "boundary_unknown",
        "source_unavailable",
        "n_events",
        "observed",
    ]
].rename(
    columns={
        "source_status":
            "screen_source_status",
        "boundary_unknown":
            "screen_boundary_unknown",
        "source_unavailable":
            "screen_source_unavailable",
        "n_events":
            "screen_n_events",
        "observed":
            "screen_observed",
    }
)


joint = gps_joint.merge(
    acc_joint,
    on=KEY,
    how="outer",
    validate="one_to_one",
)

joint = joint.merge(
    screen_joint,
    on=KEY,
    how="outer",
    validate="one_to_one",
)


joint["gps_observed"] = (
    joint["gps_n_observations"]
    .gt(0)
)

joint["acc_observed"] = (
    joint["acc_n_observations"]
    .gt(0)
)

joint["screen_event_observed"] = (
    joint["screen_n_events"]
    .gt(0)
)


joint["all_three_observed"] = (
    joint["gps_observed"]
    &
    joint["acc_observed"]
    &
    joint["screen_event_observed"]
)


joint["any_boundary_unknown"] = (
    joint["gps_boundary_unknown"]
    |
    joint["acc_boundary_unknown"]
    |
    joint["screen_boundary_unknown"]
)


joint["any_source_unavailable"] = (
    joint["gps_source_unavailable"]
    |
    joint["acc_source_unavailable"]
    |
    joint["screen_source_unavailable"]
)


joint["all_three_within_declared_boundary"] = (
    joint["gps_source_status"]
    .eq("within_declared_boundary")
    &
    joint["acc_source_status"]
    .eq("within_declared_boundary")
    &
    joint["screen_source_status"]
    .eq("within_declared_boundary")
)


print(
    "\n" + "=" * 70
)
print("JOINT DAILY OBSERVABILITY")
print("=" * 70)

print(
    "Daily rows:",
    f"{len(joint):,}",
)

print(
    "GPS observed days:",
    f"{joint['gps_observed'].sum():,}",
)

print(
    "ACC observed days:",
    f"{joint['acc_observed'].sum():,}",
)

print(
    "Screen-event observed days:",
    f"{joint['screen_event_observed'].sum():,}",
)

print(
    "All three observed days:",
    f"{joint['all_three_observed'].sum():,}",
)

print(
    "All-three within declared boundary:",
    f"{joint['all_three_within_declared_boundary'].sum():,}",
)

print(
    "Any boundary unknown:",
    f"{joint['any_boundary_unknown'].sum():,}",
)

print(
    "Any source unavailable:",
    f"{joint['any_source_unavailable'].sum():,}",
)


# ------------------------------------------------------------
# Participant / episode summary
# ------------------------------------------------------------

records = []

for episode_id, z in joint.groupby(
    "episode_id",
    sort=False,
):
    z = z.sort_values(
        "study_day"
    )

    gps_known = z[
        z["gps_source_status"]
        .eq(
            "within_declared_boundary"
        )
    ]

    acc_known = z[
        z["acc_source_status"]
        .eq(
            "within_declared_boundary"
        )
    ]

    screen_known = z[
        z["screen_source_status"]
        .eq(
            "within_declared_boundary"
        )
    ]

    def fraction(
        numerator,
        denominator,
    ):
        if denominator == 0:
            return np.nan

        return (
            numerator
            / denominator
        )

    records.append({
        "participant_id":
            z.iloc[0][
                "participant_id"
            ],

        "episode_id":
            episode_id,

        "source_participant_id":
            z.iloc[0][
                "source_participant_id"
            ],

        "episode_days":
            len(z),

        "gps_known_available_days":
            len(gps_known),

        "gps_observed_known_days":
            int(
                gps_known[
                    "gps_observed"
                ].sum()
            ),

        "gps_observed_fraction_known":
            fraction(
                int(
                    gps_known[
                        "gps_observed"
                    ].sum()
                ),
                len(gps_known),
            ),

        "gps_mean_dq_known":
            gps_known[
                "gps_data_quality"
            ].mean(),

        "acc_known_available_days":
            len(acc_known),

        "acc_observed_known_days":
            int(
                acc_known[
                    "acc_observed"
                ].sum()
            ),

        "acc_observed_fraction_known":
            fraction(
                int(
                    acc_known[
                        "acc_observed"
                    ].sum()
                ),
                len(acc_known),
            ),

        "acc_mean_dq_known":
            acc_known[
                "acc_data_quality"
            ].mean(),

        "screen_known_available_days":
            len(screen_known),

        "screen_observed_known_days":
            int(
                screen_known[
                    "screen_event_observed"
                ].sum()
            ),

        "screen_observed_fraction_known":
            fraction(
                int(
                    screen_known[
                        "screen_event_observed"
                    ].sum()
                ),
                len(screen_known),
            ),

        "all_three_observed_days":
            int(
                z[
                    "all_three_observed"
                ].sum()
            ),

        "boundary_unknown_days":
            int(
                z[
                    "any_boundary_unknown"
                ].sum()
            ),

        "source_unavailable_days":
            int(
                z[
                    "any_source_unavailable"
                ].sum()
            ),
    })


participant = pd.DataFrame(
    records
)


ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

joint_path = (
    ROOT
    / "joint_daily_quality.parquet"
)

participant_path = (
    ROOT
    / "participant_quality_summary.csv"
)

joint.to_parquet(
    joint_path,
    index=False,
)

participant.to_csv(
    participant_path,
    index=False,
)


print(
    "\nParticipant summary:"
)

print(
    participant[
        [
            "episode_days",
            "gps_observed_fraction_known",
            "acc_observed_fraction_known",
            "screen_observed_fraction_known",
            "all_three_observed_days",
            "boundary_unknown_days",
        ]
    ]
    .describe()
    .to_string()
)


print("\nSaved:")
print(joint_path)
print(participant_path)
