from pathlib import Path
import numpy as np
import pandas as pd


DC = Path("/shared/Digital Clinic")

HOME_DIR = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time"
)

SEG_FILE = (
    HOME_DIR
    / "phq_temporal_windows/"
    "phq8_phq9_home_time_strict_14d_cohort.csv"
)

ANCHOR_FILE = (
    HOME_DIR
    / "participant_home_anchors.csv"
)

DAILY_FILE = (
    HOME_DIR
    / "daily_home_time.csv"
)


# ============================================================
# SEGMENTS
# ============================================================

seg = pd.read_csv(
    SEG_FILE,
    low_memory=False,
)

seg["assessment_date"] = pd.to_datetime(
    seg["assessment_date"],
    errors="coerce",
).dt.floor("D")

seg["home_14d_all_pct"] = pd.to_numeric(
    seg["home_14d_all_pct"],
    errors="coerce",
)

seg["is_zero_segment"] = np.isclose(
    seg["home_14d_all_pct"],
    0,
    atol=1e-9,
)

zero_ids = set(
    seg.loc[
        seg["is_zero_segment"],
        "participant_id",
    ]
)

all_ids = set(
    seg["participant_id"]
)


# ============================================================
# HOME ANCHORS
# ============================================================

anchors = pd.read_csv(
    ANCHOR_FILE,
    low_memory=False,
)

anchor_cols = [
    "participant_id",
    "home_status",
    "home_reason",
    "home_confidence",
    "home_n_valid_nights",
    "home_n_nights_at_home",
    "nighttime_observed_duration_ms",
    "home_nighttime_duration_ms",
    "home_nighttime_fraction",
]

missing = [
    c for c in anchor_cols
    if c not in anchors.columns
]

if missing:
    raise RuntimeError(
        f"Missing anchor columns: {missing}"
    )

anchors = anchors[
    anchor_cols
].copy()

for c in [
    "home_confidence",
    "home_n_valid_nights",
    "home_n_nights_at_home",
    "nighttime_observed_duration_ms",
    "home_nighttime_duration_ms",
    "home_nighttime_fraction",
]:
    anchors[c] = pd.to_numeric(
        anchors[c],
        errors="coerce",
    )

anchors = anchors[
    anchors["participant_id"].isin(
        all_ids
    )
].copy()

anchors["group"] = np.where(
    anchors["participant_id"].isin(
        zero_ids
    ),
    "ever 0% segment",
    "never 0% segment",
)


print("=" * 110)
print("HOME-ANCHOR QUALITY: ZERO-SEGMENT VS OTHER PARTICIPANTS")
print("=" * 110)

summary = (
    anchors.groupby("group")
    .agg(
        n_participants=(
            "participant_id",
            "nunique",
        ),
        mean_home_confidence=(
            "home_confidence",
            "mean",
        ),
        median_home_confidence=(
            "home_confidence",
            "median",
        ),
        min_home_confidence=(
            "home_confidence",
            "min",
        ),
        median_valid_nights=(
            "home_n_valid_nights",
            "median",
        ),
        median_nights_at_home=(
            "home_n_nights_at_home",
            "median",
        ),
        median_home_night_fraction=(
            "home_nighttime_fraction",
            "median",
        ),
    )
)

print(
    summary.round(3).to_string()
)


# ============================================================
# ANCHOR STATUS / REASON
# ============================================================

print()
print("=" * 110)
print("HOME-ANCHOR STATUS")
print("=" * 110)

status = (
    anchors.groupby(
        [
            "group",
            "home_status",
        ],
        dropna=False,
    )
    .size()
    .rename("n")
    .reset_index()
)

print(
    status.to_string(
        index=False
    )
)

print()
print("=" * 110)
print("HOME-ANCHOR REASON")
print("=" * 110)

reason = (
    anchors.groupby(
        [
            "group",
            "home_reason",
        ],
        dropna=False,
    )
    .size()
    .rename("n")
    .reset_index()
)

print(
    reason.to_string(
        index=False
    )
)


# ============================================================
# DAILY DATA
# ============================================================

daily = pd.read_csv(
    DAILY_FILE,
    low_memory=False,
)

daily["date"] = pd.to_datetime(
    daily["date"],
    errors="coerce",
).dt.floor("D")

daily["time_at_home_pct"] = pd.to_numeric(
    daily["time_at_home_pct"],
    errors="coerce",
)

daily["home_duration_ms"] = pd.to_numeric(
    daily["home_duration_ms"],
    errors="coerce",
)

daily["raw_home_duration_ms"] = pd.to_numeric(
    daily["raw_home_duration_ms"],
    errors="coerce",
)

daily = daily[
    daily["participant_id"].isin(
        all_ids
    )
].copy()

daily["participant_group"] = np.where(
    daily["participant_id"].isin(
        zero_ids
    ),
    "ever 0% segment",
    "never 0% segment",
)

daily["is_zero_day"] = (
    daily["time_at_home_pct"].notna()
    & np.isclose(
        daily["time_at_home_pct"],
        0,
        atol=1e-9,
    )
)

daily["is_hundred_day"] = (
    daily["time_at_home_pct"].notna()
    & np.isclose(
        daily["time_at_home_pct"],
        100,
        atol=1e-9,
    )
)


# ============================================================
# DAILY STATUS DISTRIBUTION
# ============================================================

print()
print("=" * 110)
print("DAILY HOME-TIME STATUS BY PARTICIPANT GROUP")
print("=" * 110)

status_daily = (
    daily.groupby(
        [
            "participant_group",
            "home_time_status",
        ],
        dropna=False,
    )
    .size()
    .rename("n_days")
    .reset_index()
)

print(
    status_daily.to_string(
        index=False
    )
)


# ============================================================
# ZERO DAYS ONLY
# ============================================================

zero_days = daily[
    daily["is_zero_day"]
].copy()

print()
print("=" * 110)
print("EXACT-0% DAILY VALUES")
print("=" * 110)

print(
    "Total exact-0% days:",
    len(zero_days),
)

print(
    "Participants contributing exact-0% days:",
    zero_days[
        "participant_id"
    ].nunique(),
)

print()
print("Status distribution among exact-0% days:")

print(
    zero_days[
        "home_time_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# RAW VS FINAL HOME DURATION
# ============================================================

print()
print("=" * 110)
print("RAW/FILTERED HOME DURATION ON EXACT-0% DAYS")
print("=" * 110)

print(
    "Exact-0% days with raw_home_duration_ms > 0:",
    int(
        (
            zero_days[
                "raw_home_duration_ms"
            ] > 0
        ).sum()
    ),
)

print(
    "Exact-0% days with final home_duration_ms > 0:",
    int(
        (
            zero_days[
                "home_duration_ms"
            ] > 0
        ).sum()
    ),
)

print(
    "Median raw home duration (hours):",
    round(
        zero_days[
            "raw_home_duration_ms"
        ].median()
        / 3_600_000,
        3,
    ),
)

print(
    "Median final home duration (hours):",
    round(
        zero_days[
            "home_duration_ms"
        ].median()
        / 3_600_000,
        3,
    ),
)


# ============================================================
# ARE ZERO DAYS LONG RUNS?
# ============================================================

print()
print("=" * 110)
print("LONGEST CONSECUTIVE RUNS OF 0% DAYS")
print("=" * 110)

runs = []

for pid, g in daily.groupby(
    "participant_id"
):

    g = g.sort_values(
        "date"
    ).copy()

    if g.empty:
        continue

    max_run = 0
    run = 0
    prev_date = None

    for _, r in g.iterrows():

        d = r["date"]
        z = bool(
            r["is_zero_day"]
        )

        if (
            z
            and (
                prev_date is None
                or d - prev_date
                == pd.Timedelta(days=1)
            )
        ):
            run += 1

        elif z:
            run = 1

        else:
            run = 0

        max_run = max(
            max_run,
            run,
        )

        prev_date = d

    runs.append(
        {
            "participant_id":
                pid,
            "group":
                (
                    "ever 0% segment"
                    if pid in zero_ids
                    else "never 0% segment"
                ),
            "max_zero_day_run":
                max_run,
        }
    )

runs = pd.DataFrame(
    runs
)

run_summary = (
    runs.groupby("group")
    .agg(
        n_participants=(
            "participant_id",
            "nunique",
        ),
        median_max_zero_run=(
            "max_zero_day_run",
            "median",
        ),
        mean_max_zero_run=(
            "max_zero_day_run",
            "mean",
        ),
        max_zero_run=(
            "max_zero_day_run",
            "max",
        ),
    )
)

print(
    run_summary.round(2).to_string()
)


# ============================================================
# SAVE AGGREGATES
# ============================================================

summary.reset_index().to_csv(
    HOME_DIR
    / "phq_temporal_windows/"
    "home_anchor_quality_extreme_groups.csv",
    index=False,
)

run_summary.reset_index().to_csv(
    HOME_DIR
    / "phq_temporal_windows/"
    "home_zero_day_run_summary.csv",
    index=False,
)

print()
print(
    "No participant IDs or row-level patient data were printed."
)
