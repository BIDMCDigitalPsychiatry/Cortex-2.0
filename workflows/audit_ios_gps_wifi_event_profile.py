from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "../raphe_outputs/digital_clinic"
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

OUTDIR = (
    ROOT
    / "quality/hourly_missingness/"
    "onset_models/true_os_interaction"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
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
    x["phone_os"] == "iOS"
].copy()


# ============================================================
# IDENTIFY HOUR COLUMN
# ============================================================

time_candidates = [
    "hour_start_utc",
    "hour_start_ms",
    "hour_ms",
    "timestamp",
    "hour_start",
    "hour_utc",
    "datetime_utc",
]

time_col = next(
    (
        c
        for c in time_candidates
        if c in x.columns
    ),
    None,
)

if time_col is None:

    possible = [
        c
        for c in x.columns
        if (
            "hour" in c.lower()
            or "time" in c.lower()
        )
    ]

    raise RuntimeError(
        "Could not identify hourly time column. "
        f"Possible columns: {possible}"
    )


# ============================================================
# IDENTIFY CURRENT GPS OBSERVABILITY
# ============================================================

gps_observed_candidates = [
    "gps_observed",
    "gps_hour_observed",
    "gps_evidence",
    "gps_present",
    "gps_available",
]

gps_missing_candidates = [
    "gps_missing",
    "gps_hour_missing",
    "gps_absent",
]


gps_observed_col = next(
    (
        c
        for c in gps_observed_candidates
        if c in x.columns
    ),
    None,
)

gps_missing_col = next(
    (
        c
        for c in gps_missing_candidates
        if c in x.columns
    ),
    None,
)


def as_bool(s):

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    numeric = pd.to_numeric(
        s,
        errors="coerce",
    )

    if numeric.notna().any():
        return numeric.fillna(0).gt(0)

    return (
        s.astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "yes",
                "observed",
                "present",
                "available",
                "1",
            ]
        )
    )


if gps_observed_col is not None:

    x["gps_observed_now"] = as_bool(
        x[gps_observed_col]
    )

elif gps_missing_col is not None:

    x["gps_observed_now"] = ~as_bool(
        x[gps_missing_col]
    )

else:

    possible = [
        c
        for c in x.columns
        if "gps" in c.lower()
    ]

    raise RuntimeError(
        "Could not identify current GPS observability. "
        f"GPS-related columns: {possible}"
    )


# ============================================================
# CURRENT WIFI EVIDENCE
# ============================================================

if (
    "ctx_current_unique_wifi_id_count"
    not in x.columns
):

    possible = [
        c
        for c in x.columns
        if (
            "wifi" in c.lower()
            and "prev" not in c.lower()
        )
    ]

    raise RuntimeError(
        "ctx_unique_wifi_id_count not found. "
        f"Possible current Wi-Fi columns: {possible}"
    )


x["wifi_now"] = (
    pd.to_numeric(
        x[
            "ctx_current_unique_wifi_id_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
)


# ============================================================
# SORT HOURLY DATA
# ============================================================

x = x.sort_values(
    [
        "episode_id",
        time_col,
    ]
).reset_index(
    drop=True
)


# ============================================================
# EVENT-CENTERED WINDOWS: -6 TO +6 HOURS
# ============================================================

WINDOW = 6

profile_rows = []
recovery_rows = []


for episode_id, g in x.groupby(
    "episode_id",
    sort=False,
):

    g = g.reset_index(
        drop=True
    )

    onset_positions = np.flatnonzero(
        g[
            "gps_specific_onset"
        ].astype(bool)
        .to_numpy()
    )


    for event_number, pos in enumerate(
        onset_positions,
        start=1,
    ):

        event_id = (
            f"{episode_id}__gpsspec_{event_number}"
        )


        # --------------------------------------------
        # Event-centered profile
        # --------------------------------------------

        for offset in range(
            -WINDOW,
            WINDOW + 1,
        ):

            j = pos + offset

            if (
                j < 0
                or j >= len(g)
            ):
                continue

            profile_rows.append({
                "episode_id":
                    episode_id,

                "event_id":
                    event_id,

                "offset_hour":
                    offset,

                "wifi_evidence":
                    float(
                        g.loc[
                            j,
                            "wifi_now",
                        ]
                    ),

                "gps_observed":
                    float(
                        g.loc[
                            j,
                            "gps_observed_now",
                        ]
                    ),
            })


        # --------------------------------------------
        # First GPS recovery within 12 hours
        # --------------------------------------------

        recovery_offset = np.nan
        recovery_wifi = np.nan

        for offset in range(
            1,
            13,
        ):

            j = pos + offset

            if j >= len(g):
                break

            if bool(
                g.loc[
                    j,
                    "gps_observed_now",
                ]
            ):

                recovery_offset = offset

                recovery_wifi = float(
                    g.loc[
                        j,
                        "wifi_now",
                    ]
                )

                break


        recovery_rows.append({
            "episode_id":
                episode_id,

            "event_id":
                event_id,

            "wifi_at_onset":
                float(
                    g.loc[
                        pos,
                        "wifi_now",
                    ]
                ),

            "recovery_offset_hours":
                recovery_offset,

            "wifi_at_recovery":
                recovery_wifi,
        })


profile = pd.DataFrame(
    profile_rows
)

recovery = pd.DataFrame(
    recovery_rows
)


# ============================================================
# EPISODE-BALANCED PROFILE
# ============================================================

episode_profile = (
    profile.groupby(
        [
            "episode_id",
            "offset_hour",
        ]
    )
    .agg(
        wifi_fraction=(
            "wifi_evidence",
            "mean",
        ),

        gps_fraction=(
            "gps_observed",
            "mean",
        ),
    )
    .reset_index()
)


summary_rows = []

rng = np.random.default_rng(
    42
)


for offset, g in episode_profile.groupby(
    "offset_hour"
):

    vals = (
        g["wifi_fraction"]
        .dropna()
        .to_numpy()
    )

    boots = []

    for _ in range(
        5000
    ):

        boots.append(
            rng.choice(
                vals,
                size=len(vals),
                replace=True,
            ).mean()
        )


    summary_rows.append({
        "offset_hour":
            offset,

        "episodes":
            len(vals),

        "wifi_pct":
            100
            * vals.mean(),

        "wifi_ci_low":
            100
            * np.quantile(
                boots,
                0.025,
            ),

        "wifi_ci_high":
            100
            * np.quantile(
                boots,
                0.975,
            ),

        "gps_observed_pct":
            100
            * g[
                "gps_fraction"
            ].mean(),
    })


summary = pd.DataFrame(
    summary_rows
).sort_values(
    "offset_hour"
)


# ============================================================
# RECOVERY SUMMARY
# ============================================================

recovered = recovery.dropna(
    subset=[
        "recovery_offset_hours"
    ]
).copy()


print("=" * 80)
print("IOS GPS-SPECIFIC EVENT-CENTERED PROFILE")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda v:
            f"{v:.2f}",
    )
)


print()
print("=" * 80)
print("GPS RECOVERY WITHIN 12 HOURS")
print("=" * 80)

print(
    "GPS-specific onset events:",
    len(recovery),
)

print(
    "Recovered within 12h:",
    len(recovered),
    f"({100 * len(recovered) / len(recovery):.1f}%)",
)

if len(recovered):

    print(
        "Median time to recovery:",
        f"{recovered['recovery_offset_hours'].median():.1f} h",
    )

    print(
        "Wi-Fi present at onset:",
        f"{100 * recovery['wifi_at_onset'].mean():.1f}%",
    )

    print(
        "Wi-Fi present at recovery:",
        f"{100 * recovered['wifi_at_recovery'].mean():.1f}%",
    )

    absent_onset = recovered[
        recovered[
            "wifi_at_onset"
        ] == 0
    ]

    if len(absent_onset):

        print(
            "Among events with Wi-Fi absent at onset, "
            "Wi-Fi present when GPS recovered:",
            f"{100 * absent_onset['wifi_at_recovery'].mean():.1f}%",
            f"(n={len(absent_onset)})",
        )


# ============================================================
# SAVE
# ============================================================

profile.to_csv(
    OUTDIR
    / "ios_gps_specific_event_profile_events.csv",
    index=False,
)

summary.to_csv(
    OUTDIR
    / "ios_gps_specific_event_profile_summary.csv",
    index=False,
)

recovery.to_csv(
    OUTDIR
    / "ios_gps_specific_recovery.csv",
    index=False,
)


print()
print("Saved:")
print(OUTDIR)
