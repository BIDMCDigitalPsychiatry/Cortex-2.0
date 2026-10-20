from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex_longitudinal/data/"
    "benchmark_5_full/"
    "participant_daily"
)

FEATURES = [
    "hometime",
    "entropy",
    "trip_distance",
    "trip_duration",
]

THRESHOLDS = [
    1.00,
    0.99,
    0.95,
    0.90,
    0.75,
    0.50,
]


# ============================================================
# LOAD
# ============================================================

frames = []

for path in sorted(
    ROOT.glob("*.csv")
):

    df = pd.read_csv(
        path
    )

    df["source_file"] = (
        path.name
    )

    frames.append(
        df
    )


if not frames:
    raise RuntimeError(
        f"No participant daily files found in {ROOT}"
    )


x = pd.concat(
    frames,
    ignore_index=True,
)


x["gps_data_quality"] = pd.to_numeric(
    x["gps_data_quality"],
    errors="coerce",
)


for feature in FEATURES:

    x[feature] = pd.to_numeric(
        x[feature],
        errors="coerce",
    )


# ============================================================
# BASIC INVENTORY
# ============================================================

print("=" * 80)
print("GPS FEATURE REFERENCE-DAY AUDIT")
print("=" * 80)

print(
    "Participants:",
    x["participant_id"].nunique(),
)

print(
    "Participant-days:",
    len(x),
)

print(
    "Days with GPS DQ:",
    int(
        x[
            "gps_data_quality"
        ].notna().sum()
    ),
)


# ============================================================
# DQ DISTRIBUTION
# ============================================================

q = (
    x[
        "gps_data_quality"
    ]
    .dropna()
)

print()
print("=" * 80)
print("GPS DATA-QUALITY DISTRIBUTION")
print("=" * 80)

for p in [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]:

    print(
        f"q{int(p * 100):02d}: "
        f"{q.quantile(p):.4f}"
    )


# ============================================================
# THRESHOLD COUNTS
# ============================================================

print()
print("=" * 80)
print("POTENTIAL REFERENCE DAYS")
print("=" * 80)

rows = []

for threshold in THRESHOLDS:

    d = x[
        x["gps_data_quality"]
        >= threshold
    ].copy()

    row = {
        "gps_dq_threshold":
            threshold,

        "days":
            len(d),

        "participants":
            d[
                "participant_id"
            ].nunique(),
    }

    for feature in FEATURES:

        row[
            f"{feature}_available"
        ] = int(
            d[
                feature
            ]
            .notna()
            .sum()
        )

    rows.append(
        row
    )


threshold_table = pd.DataFrame(
    rows
)

print(
    threshold_table.to_string(
        index=False
    )
)


# ============================================================
# PARTICIPANT-SPECIFIC COUNTS
# ============================================================

print()
print("=" * 80)
print("NEAR-COMPLETE DAYS BY PARTICIPANT")
print("=" * 80)

participant_rows = []

for participant_id, g in x.groupby(
    "participant_id"
):

    row = {
        "participant_id":
            participant_id,

        "days_total":
            len(g),

        "median_gps_dq":
            g[
                "gps_data_quality"
            ].median(),
    }

    for threshold in [
        0.99,
        0.95,
        0.90,
    ]:

        row[
            f"days_ge_{int(threshold*100)}"
        ] = int(
            (
                g[
                    "gps_data_quality"
                ]
                >= threshold
            ).sum()
        )

    participant_rows.append(
        row
    )


participant_table = pd.DataFrame(
    participant_rows
)

print(
    participant_table.to_string(
        index=False,
        float_format=lambda v:
            f"{v:.4f}",
    )
)


# ============================================================
# FEATURE AVAILABILITY / RANGE
# ============================================================

print()
print("=" * 80)
print("FEATURE AVAILABILITY AND RANGE")
print("=" * 80)

feature_rows = []

for feature in FEATURES:

    s = (
        x[
            feature
        ]
        .dropna()
    )

    feature_rows.append({
        "feature":
            feature,

        "available_days":
            len(s),

        "missing_days":
            int(
                x[
                    feature
                ]
                .isna()
                .sum()
            ),

        "median":
            s.median()
            if len(s)
            else np.nan,

        "p05":
            s.quantile(0.05)
            if len(s)
            else np.nan,

        "p95":
            s.quantile(0.95)
            if len(s)
            else np.nan,

        "min":
            s.min()
            if len(s)
            else np.nan,

        "max":
            s.max()
            if len(s)
            else np.nan,
    })


feature_table = pd.DataFrame(
    feature_rows
)

print(
    feature_table.to_string(
        index=False,
        float_format=lambda v:
            f"{v:.4f}",
    )
)


# ============================================================
# DAYS WITH ALL FOUR GPS FEATURES
# ============================================================

x[
    "all_gps_features_available"
] = (
    x[
        FEATURES
    ]
    .notna()
    .all(
        axis=1
    )
)


print()
print("=" * 80)
print("ALL-FOUR-FEATURE REFERENCE CANDIDATES")
print("=" * 80)

for threshold in [
    1.00,
    0.99,
    0.95,
    0.90,
]:

    d = x[
        (
            x[
                "gps_data_quality"
            ]
            >= threshold
        )
        & x[
            "all_gps_features_available"
        ]
    ]

    print(
        f"GPS DQ >= {threshold:.2f}: "
        f"{len(d)} days across "
        f"{d['participant_id'].nunique()} participants"
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
