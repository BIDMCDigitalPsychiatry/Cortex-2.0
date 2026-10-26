from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

RISK_PATH = (
    ROOT
    / "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

OS_PATH = (
    ROOT
    / "inventory/"
    "risk_episode_lamp_analytics_os.csv"
)

OUT = (
    ROOT
    / "quality/hourly_missingness/"
    "onset_models/true_os_interaction/"
    "wifi_evidence_by_os.csv"
)


# ============================================================
# LOAD
# ============================================================

x = pd.read_parquet(
    RISK_PATH
)

os_map = pd.read_csv(
    OS_PATH
)

x = x.merge(
    os_map[
        [
            "episode_id",
            "phone_os",
        ]
    ],
    on="episode_id",
    how="left",
    validate="many_to_one",
)

x = x[
    x["phone_os"].isin(
        [
            "iOS",
            "Android",
        ]
    )
].copy()


# ============================================================
# WIFI / BLUETOOTH EVIDENCE
# ============================================================

x["prev_wifi"] = (
    pd.to_numeric(
        x[
            "ctx_prev_unique_wifi_id_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
)

x["prev_bluetooth"] = (
    pd.to_numeric(
        x[
            "ctx_prev_unique_bluetooth_id_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
)


# ============================================================
# OUTCOME STATE
# ============================================================

x["state"] = "no_onset"

x.loc[
    x["gps_specific_onset"],
    "state",
] = "gps_specific"

x.loc[
    x["shared_onset"],
    "state",
] = "shared"


# ============================================================
# POOLED HOURLY DESCRIPTION
# ============================================================

hourly = (
    x.groupby(
        [
            "phone_os",
            "state",
        ]
    )
    .agg(
        hours=(
            "episode_id",
            "size",
        ),

        episodes=(
            "episode_id",
            "nunique",
        ),

        wifi_hours=(
            "prev_wifi",
            "sum",
        ),

        bluetooth_hours=(
            "prev_bluetooth",
            "sum",
        ),

        mean_wifi_ids=(
            "ctx_prev_unique_wifi_id_count",
            "mean",
        ),

        median_wifi_ids=(
            "ctx_prev_unique_wifi_id_count",
            "median",
        ),

        mean_bluetooth_ids=(
            "ctx_prev_unique_bluetooth_id_count",
            "mean",
        ),

        median_bluetooth_ids=(
            "ctx_prev_unique_bluetooth_id_count",
            "median",
        ),
    )
    .reset_index()
)

hourly[
    "wifi_evidence_pct"
] = (
    100
    * hourly["wifi_hours"]
    / hourly["hours"]
)

hourly[
    "bluetooth_evidence_pct"
] = (
    100
    * hourly["bluetooth_hours"]
    / hourly["hours"]
)


# ============================================================
# EPISODE-BALANCED WIFI PREVALENCE
# ============================================================

episode_state = (
    x.groupby(
        [
            "phone_os",
            "episode_id",
            "state",
        ]
    )
    .agg(
        hours=(
            "episode_id",
            "size",
        ),

        wifi_fraction=(
            "prev_wifi",
            "mean",
        ),

        bluetooth_fraction=(
            "prev_bluetooth",
            "mean",
        ),
    )
    .reset_index()
)


balanced = (
    episode_state.groupby(
        [
            "phone_os",
            "state",
        ]
    )
    .agg(
        episodes=(
            "episode_id",
            "nunique",
        ),

        mean_episode_wifi_fraction=(
            "wifi_fraction",
            "mean",
        ),

        median_episode_wifi_fraction=(
            "wifi_fraction",
            "median",
        ),

        mean_episode_bluetooth_fraction=(
            "bluetooth_fraction",
            "mean",
        ),

        median_episode_bluetooth_fraction=(
            "bluetooth_fraction",
            "median",
        ),
    )
    .reset_index()
)


balanced[
    "mean_episode_wifi_pct"
] = (
    100
    * balanced[
        "mean_episode_wifi_fraction"
    ]
)

balanced[
    "median_episode_wifi_pct"
] = (
    100
    * balanced[
        "median_episode_wifi_fraction"
    ]
)


# ============================================================
# OVERALL EPISODE DISTRIBUTION
# ============================================================

episode_overall = (
    x.groupby(
        [
            "phone_os",
            "episode_id",
        ]
    )
    .agg(
        wifi_fraction=(
            "prev_wifi",
            "mean",
        ),

        bluetooth_fraction=(
            "prev_bluetooth",
            "mean",
        ),
    )
    .reset_index()
)


overall = (
    episode_overall.groupby(
        "phone_os"
    )
    .agg(
        episodes=(
            "episode_id",
            "nunique",
        ),

        mean_wifi_fraction=(
            "wifi_fraction",
            "mean",
        ),

        median_wifi_fraction=(
            "wifi_fraction",
            "median",
        ),

        p25_wifi_fraction=(
            "wifi_fraction",
            lambda s:
                s.quantile(0.25),
        ),

        p75_wifi_fraction=(
            "wifi_fraction",
            lambda s:
                s.quantile(0.75),
        ),
    )
    .reset_index()
)


for c in [
    "mean_wifi_fraction",
    "median_wifi_fraction",
    "p25_wifi_fraction",
    "p75_wifi_fraction",
]:
    overall[c] *= 100


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("POOLED PREVIOUS-HOUR WIFI EVIDENCE")
print("=" * 80)

print(
    hourly[
        [
            "phone_os",
            "state",
            "hours",
            "episodes",
            "wifi_evidence_pct",
            "bluetooth_evidence_pct",
            "mean_wifi_ids",
            "median_wifi_ids",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda v:
            f"{v:.2f}",
    )
)


print()
print("=" * 80)
print("EPISODE-BALANCED WIFI EVIDENCE")
print("=" * 80)

print(
    balanced[
        [
            "phone_os",
            "state",
            "episodes",
            "mean_episode_wifi_pct",
            "median_episode_wifi_pct",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda v:
            f"{v:.2f}",
    )
)


print()
print("=" * 80)
print("OVERALL WIFI EVIDENCE BY OS")
print("=" * 80)

print(
    overall.to_string(
        index=False,
        float_format=lambda v:
            f"{v:.2f}",
    )
)


hourly.to_csv(
    OUT,
    index=False,
)

print()
print("Saved:")
print(OUT)
