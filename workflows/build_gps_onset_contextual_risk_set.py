from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "../raphe_outputs/digital_clinic"
)

QUALITY = ROOT / "quality"

RISK_PATH = (
    QUALITY
    / "hourly_missingness"
    / "gps_onset_risk_set.parquet"
)

CONTEXT_DIR = (
    QUALITY
    / "hourly_context"
    / "episodes"
)

OUTPUT_PATH = (
    QUALITY
    / "hourly_missingness"
    / "gps_onset_contextual_risk_set.parquet"
)


# ============================================================
# LOAD RISK SET
# ============================================================

risk = pd.read_parquet(
    RISK_PATH
)

print(
    "Risk rows:",
    f"{len(risk):,}",
)

print(
    "Episodes:",
    risk["episode_id"].nunique(),
)

print(
    "GPS-specific onsets:",
    int(
        risk[
            "gps_specific_onset"
        ].sum()
    ),
)

print(
    "Shared onsets:",
    int(
        risk[
            "shared_onset"
        ].sum()
    ),
)


# ============================================================
# LOAD COMPLETE HOURLY CONTEXT
#
# IMPORTANT:
# Lagging happens here, BEFORE restricting to risk rows.
# ============================================================

files = sorted(
    CONTEXT_DIR.glob(
        "*.parquet"
    )
)

print(
    "Context episode files:",
    len(files),
)


columns = [
    "episode_id",
    "hour_index",
    "start_ms",
    "end_ms",

    "screen_episode_evidence",
    "screen_event_count",
    "screen_event_observed",

    "battery_observation_count",
    "battery_mean",
    "battery_min",
    "battery_max",
    "battery_first",
    "battery_last",
    "battery_within_hour_delta",

    "device_usage_episode_evidence",
    "device_usage_raw_record_count",
    "device_usage_record_count",
    "device_usage_duplicate_record_count",
    "device_usage_max_duplicate_multiplicity",
    "device_usage_conflict_timestamp_count",
    "device_usage_observed",
    "device_reported_duration_ms",
    "unlock_duration_ms",
    "unlock_count",
    "screen_wake_count",

    "nearby_device_episode_evidence",
    "nearby_record_count",
    "nearby_observation_present",
    "wifi_record_count",
    "bluetooth_record_count",
    "unique_nearby_id_count",
    "unique_wifi_id_count",
    "unique_bluetooth_id_count",
]


frames = []

for i, path in enumerate(
    files,
    start=1,
):

    x = pd.read_parquet(
        path,
        columns=columns,
    )

    frames.append(
        x
    )

    if (
        i % 50 == 0
        or i == len(files)
    ):
        print(
            f"Loaded context: "
            f"{i}/{len(files)}",
            flush=True,
        )


context = pd.concat(
    frames,
    ignore_index=True,
)


context["episode_id"] = (
    context["episode_id"]
    .astype(str)
)


context = context.sort_values(
    [
        "episode_id",
        "hour_index",
    ]
).reset_index(
    drop=True
)


if context[
    [
        "episode_id",
        "hour_index",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate episode/hour rows "
        "in hourly context."
    )


# Hour length is useful because final episode hours
# can be partial.
context[
    "hour_duration_ms"
] = (
    context["end_ms"]
    - context["start_ms"]
)


# ============================================================
# CURRENT-HOUR CONTEXT
# ============================================================

episode_columns = [
    "screen_episode_evidence",
    "device_usage_episode_evidence",
    "nearby_device_episode_evidence",
]


measurement_columns = [
    "screen_event_count",
    "screen_event_observed",

    "battery_observation_count",
    "battery_mean",
    "battery_min",
    "battery_max",
    "battery_first",
    "battery_last",
    "battery_within_hour_delta",

    "device_usage_raw_record_count",
    "device_usage_record_count",
    "device_usage_duplicate_record_count",
    "device_usage_max_duplicate_multiplicity",
    "device_usage_conflict_timestamp_count",
    "device_usage_observed",
    "device_reported_duration_ms",
    "unlock_duration_ms",
    "unlock_count",
    "screen_wake_count",

    "nearby_record_count",
    "nearby_observation_present",
    "wifi_record_count",
    "bluetooth_record_count",
    "unique_nearby_id_count",
    "unique_wifi_id_count",
    "unique_bluetooth_id_count",
]


join = context[
    [
        "episode_id",
        "hour_index",
        "hour_duration_ms",
    ]
    + episode_columns
    + measurement_columns
].copy()


join = join.rename(
    columns={
        c: f"ctx_{c}"
        for c in episode_columns
    }
)


join = join.rename(
    columns={
        c: f"ctx_current_{c}"
        for c in (
            [
                "hour_duration_ms",
            ]
            + measurement_columns
        )
    }
)


# ============================================================
# PREVIOUS-HOUR CONTEXT
# ============================================================

lag_source = context[
    [
        "episode_id",
        "hour_index",
        "hour_duration_ms",
    ]
    + measurement_columns
].copy()


lag_columns = (
    [
        "hour_duration_ms",
    ]
    + measurement_columns
)


lag_source[
    lag_columns
] = (
    lag_source
    .groupby(
        "episode_id",
        sort=False,
    )[
        lag_columns
    ]
    .shift(1)
)


lag_source = lag_source.rename(
    columns={
        c: f"ctx_prev_{c}"
        for c in lag_columns
    }
)


join = join.merge(
    lag_source,
    on=[
        "episode_id",
        "hour_index",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# DERIVED PREVIOUS-HOUR CONTEXT
# ============================================================

join[
    "ctx_prev_battery_observed"
] = (
    join[
        "ctx_prev_battery_observation_count"
    ]
    .fillna(0)
    .gt(0)
)


join[
    "ctx_prev_device_usage_observed"
] = (
    join[
        "ctx_prev_device_usage_record_count"
    ]
    .fillna(0)
    .gt(0)
)


join[
    "ctx_prev_nearby_observed"
] = (
    join[
        "ctx_prev_nearby_record_count"
    ]
    .fillna(0)
    .gt(0)
)


join[
    "ctx_prev_all_three_observed"
] = (
    join[
        "ctx_prev_battery_observed"
    ]
    &
    join[
        "ctx_prev_device_usage_observed"
    ]
    &
    join[
        "ctx_prev_nearby_observed"
    ]
)


# Device usage is reported over time intervals.
# Preserve the fraction of the previous hour represented
# by those records rather than assuming exactly four records.

denominator = (
    join[
        "ctx_prev_hour_duration_ms"
    ]
)

numerator = (
    join[
        "ctx_prev_device_reported_duration_ms"
    ]
)

join[
    "ctx_prev_device_reported_fraction"
] = np.where(
    denominator.gt(0)
    & numerator.notna(),
    numerator
    / denominator,
    np.nan,
)


# ============================================================
# JOIN TO GPS ONSET RISK SET
# ============================================================

merged = risk.merge(
    join,
    on=[
        "episode_id",
        "hour_index",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
)


if len(merged) != len(risk):
    raise RuntimeError(
        "Risk-set row count changed "
        "during context merge."
    )


missing_context = int(
    merged["_merge"]
    .ne("both")
    .sum()
)


print(
    "\nRisk rows without matching "
    "hourly context:",
    missing_context,
)


merged = merged.drop(
    columns="_merge"
)


# ============================================================
# CURRENT-HOUR POSITIVE EVIDENCE
# ============================================================

merged[
    "ctx_current_battery_observed"
] = (
    merged[
        "ctx_current_battery_observation_count"
    ]
    .fillna(0)
    .gt(0)
)


merged[
    "ctx_current_device_usage_observed"
] = (
    merged[
        "ctx_current_device_usage_record_count"
    ]
    .fillna(0)
    .gt(0)
)


merged[
    "ctx_current_nearby_observed"
] = (
    merged[
        "ctx_current_nearby_record_count"
    ]
    .fillna(0)
    .gt(0)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

merged.to_parquet(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "PRE-ONSET CONTEXT COVERAGE"
)

print(
    "=" * 72
)


def summarize_subset(
    name,
    df,
):

    n = len(df)

    print(
        f"\n{name}"
    )

    print(
        "Hours:",
        f"{n:,}",
    )

    if n == 0:
        return

    measures = [
        (
            "Prior battery",
            "ctx_prev_battery_observed",
        ),
        (
            "Prior device usage",
            "ctx_prev_device_usage_observed",
        ),
        (
            "Prior nearby device",
            "ctx_prev_nearby_observed",
        ),
        (
            "All three",
            "ctx_prev_all_three_observed",
        ),
    ]

    for label, col in measures:

        count = int(
            df[col]
            .fillna(False)
            .sum()
        )

        pct = (
            100
            * count
            / n
        )

        print(
            f"{label:24s} "
            f"{count:8,} "
            f"({pct:6.2f}%)"
        )


summarize_subset(
    "ALL RISK HOURS",
    merged,
)


summarize_subset(
    "GPS-SPECIFIC ONSETS",
    merged[
        merged[
            "gps_specific_onset"
        ]
    ],
)


summarize_subset(
    "SHARED ONSETS",
    merged[
        merged[
            "shared_onset"
        ]
    ],
)


summarize_subset(
    "NO ONSET",
    merged[
        ~merged[
            "any_gps_onset"
        ]
    ],
)


# ============================================================
# CONCURRENT EVIDENCE DURING ONSETS
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "CONCURRENT POSITIVE EVIDENCE DURING GPS ONSET"
)

print(
    "=" * 72
)


for outcome in [
    "gps_specific_onset",
    "shared_onset",
]:

    z = merged[
        merged[outcome]
    ]

    print(
        f"\n{outcome}"
    )

    print(
        "Onsets:",
        f"{len(z):,}",
    )

    for label, col in [
        (
            "Battery observed",
            "ctx_current_battery_observed",
        ),
        (
            "Device usage observed",
            "ctx_current_device_usage_observed",
        ),
        (
            "Nearby device observed",
            "ctx_current_nearby_observed",
        ),
        (
            "Screen event observed",
            "ctx_current_screen_event_observed",
        ),
    ]:

        count = int(
            z[col]
            .fillna(False)
            .sum()
        )

        pct = (
            100
            * count
            / len(z)
            if len(z)
            else np.nan
        )

        print(
            f"{label:24s} "
            f"{count:8,} "
            f"({pct:6.2f}%)"
        )


# ============================================================
# SIMPLE PRE-ONSET DESCRIPTIVES
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "PREVIOUS-HOUR DESCRIPTIVES BY OUTCOME"
)

print(
    "=" * 72
)


variables = [
    "prev_acc_dq",

    "ctx_prev_battery_last",
    "ctx_prev_battery_within_hour_delta",

    "ctx_prev_device_reported_fraction",
    "ctx_prev_unlock_duration_ms",
    "ctx_prev_unlock_count",
    "ctx_prev_screen_wake_count",

    "ctx_prev_nearby_record_count",
    "ctx_prev_unique_wifi_id_count",
    "ctx_prev_unique_bluetooth_id_count",
]


for outcome in [
    "no_onset",
    "gps_specific_onset",
    "shared_gap_onset",
]:

    if outcome == "no_onset":
        z = merged[
            ~merged[
                "any_gps_onset"
            ]
        ]

    else:
        z = merged[
            merged[outcome]
        ]

    print(
        f"\n--- {outcome} ---"
    )

    for col in variables:

        values = pd.to_numeric(
            z[col],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            print(
                f"{col}: no observations"
            )
            continue

        print(
            f"{col}: "
            f"n={len(values):,}, "
            f"mean={values.mean():.4f}, "
            f"median={values.median():.4f}"
        )


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    "Shape:",
    merged.shape,
)
