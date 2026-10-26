from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

x = pd.read_parquet(PATH)


def outcome_frame(name):

    if name == "no_onset":
        return x[
            ~x["any_gps_onset"]
        ].copy()

    if name == "gps_specific":
        return x[
            x["gps_specific_onset"]
        ].copy()

    if name == "shared":
        return x[
            x["shared_onset"]
        ].copy()

    raise ValueError(name)


print("=" * 72)
print("DEVICE-USAGE INTERVAL AUDIT")
print("=" * 72)

for name in [
    "no_onset",
    "gps_specific",
    "shared",
]:

    z = outcome_frame(name)

    v = pd.to_numeric(
        z[
            "ctx_prev_device_reported_fraction"
        ],
        errors="coerce",
    ).dropna()

    print(f"\n{name}")

    print(
        "n:",
        f"{len(v):,}",
    )

    if not len(v):
        continue

    print(
        "mean:",
        round(v.mean(), 4),
    )

    print(
        "median:",
        round(v.median(), 4),
    )

    print(
        "p95:",
        round(
            v.quantile(.95),
            4,
        ),
    )

    print(
        "max:",
        round(v.max(), 4),
    )

    for threshold in [
        1.01,
        1.25,
        1.50,
        2.00,
    ]:

        n = int(
            (v > threshold).sum()
        )

        print(
            f"> {threshold:.2f}: "
            f"{n:,} "
            f"({100*n/len(v):.2f}%)"
        )


print("\n" + "=" * 72)
print("PREVIOUS-HOUR PRESENCE BY OUTCOME")
print("=" * 72)

presence = [
    "ctx_prev_battery_observed",
    "ctx_prev_device_usage_observed",
    "ctx_prev_nearby_observed",
    "prev_screen_active",
]

for name in [
    "no_onset",
    "gps_specific",
    "shared",
]:

    z = outcome_frame(name)

    print(f"\n{name}")

    for col in presence:

        count = int(
            z[col]
            .fillna(False)
            .sum()
        )

        print(
            f"{col:38s} "
            f"{count:8,} "
            f"({100*count/len(z):6.2f}%)"
        )


print("\n" + "=" * 72)
print("WITHIN-EPISODE CENTERED CONTEXT AT ONSET")
print("=" * 72)

variables = [
    "prev_acc_dq",

    "ctx_prev_battery_last",
    "ctx_prev_battery_within_hour_delta",

    "ctx_prev_unlock_duration_ms",
    "ctx_prev_unlock_count",
    "ctx_prev_screen_wake_count",

    "ctx_prev_device_reported_fraction",
]


work = x.copy()


# Presence variables are especially important because
# a zero nearby count is not interpreted as "nothing nearby".
work[
    "ctx_prev_nearby_observed_num"
] = (
    work[
        "ctx_prev_nearby_observed"
    ]
    .astype(float)
)

work[
    "ctx_prev_device_usage_observed_num"
] = (
    work[
        "ctx_prev_device_usage_observed"
    ]
    .astype(float)
)

work[
    "ctx_prev_battery_observed_num"
] = (
    work[
        "ctx_prev_battery_observed"
    ]
    .astype(float)
)


variables += [
    "ctx_prev_nearby_observed_num",
    "ctx_prev_device_usage_observed_num",
    "ctx_prev_battery_observed_num",
]


for col in variables:

    values = pd.to_numeric(
        work[col],
        errors="coerce",
    )

    episode_mean = (
        values
        .groupby(
            work["episode_id"]
        )
        .transform("mean")
    )

    work[
        f"{col}__within"
    ] = (
        values
        - episode_mean
    )


for outcome_col in [
    "gps_specific_onset",
    "shared_onset",
]:

    z = work[
        work[outcome_col]
    ]

    print(
        f"\n--- {outcome_col} ---"
    )

    for col in variables:

        v = (
            z[
                f"{col}__within"
            ]
            .dropna()
        )

        if not len(v):
            continue

        print(
            f"{col}: "
            f"n={len(v):,}, "
            f"mean_within_delta="
            f"{v.mean():.4f}, "
            f"median="
            f"{v.median():.4f}"
        )


print("\n" + "=" * 72)
print("GPS-SPECIFIC ONSET PARTICIPANT CONCENTRATION")
print("=" * 72)

gps = x[
    x["gps_specific_onset"]
]

counts = (
    gps.groupby(
        "episode_id"
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    "Episodes with >=1 onset:",
    len(counts),
)

print(
    "Median onsets / episode:",
    counts.median(),
)

print(
    "90th percentile:",
    counts.quantile(.90),
)

print(
    "Maximum:",
    counts.max(),
)

print(
    "\nTop 15 episodes:"
)

print(
    counts.head(15).to_string()
)
