from pathlib import Path
import numpy as np
import pandas as pd


DC = Path("/shared/Digital Clinic")

SEGMENT_FILE = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/phq_temporal_windows/"
    "phq8_phq9_home_time_strict_14d_cohort.csv"
)

DAILY_FILE = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/daily_home_time.csv"
)

OUT = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/phq_temporal_windows"
)


# ============================================================
# LOAD SEGMENTS
# ============================================================

seg = pd.read_csv(
    SEGMENT_FILE,
    low_memory=False,
)

seg["assessment_date"] = pd.to_datetime(
    seg["assessment_date"],
    errors="coerce",
).dt.floor("D")

HOME_COLS = {
    "All 14 days":
        "home_14d_all_pct",

    "Weekdays":
        "home_14d_weekday_matched_pct",

    "Weekends":
        "home_14d_weekend_pct",
}


# ============================================================
# SEGMENT-LEVEL EXTREMES
# ============================================================

print("=" * 110)
print("14-DAY HOME-TIME EXTREME-VALUE AUDIT")
print("=" * 110)

print("\nTotal eligible segments:", len(seg))
print(
    "Participants:",
    seg["participant_id"].nunique(),
)

summary_rows = []

for label, col in HOME_COLS.items():

    x = pd.to_numeric(
        seg[col],
        errors="coerce",
    )

    valid = x.notna()

    zero = valid & np.isclose(
        x,
        0.0,
        atol=1e-9,
    )

    hundred = valid & np.isclose(
        x,
        100.0,
        atol=1e-9,
    )

    near_zero = (
        valid
        & (x > 0)
        & (x <= 5)
    )

    near_hundred = (
        valid
        & (x < 100)
        & (x >= 95)
    )

    print()
    print(label)
    print("-" * 70)

    print(
        "Valid segments:",
        int(valid.sum()),
    )

    print(
        "Exactly 0%:",
        int(zero.sum()),
        f"({100 * zero.sum() / valid.sum():.1f}%)"
        if valid.sum() else "",
    )

    print(
        "Exactly 100%:",
        int(hundred.sum()),
        f"({100 * hundred.sum() / valid.sum():.1f}%)"
        if valid.sum() else "",
    )

    print(
        ">0% to <=5%:",
        int(near_zero.sum()),
    )

    print(
        ">=95% to <100%:",
        int(near_hundred.sum()),
    )

    print(
        "Participants contributing exact 0%:",
        seg.loc[
            zero,
            "participant_id",
        ].nunique(),
    )

    print(
        "Participants contributing exact 100%:",
        seg.loc[
            hundred,
            "participant_id",
        ].nunique(),
    )

    summary_rows.append(
        {
            "window": label,
            "n_segments": int(valid.sum()),
            "n_zero": int(zero.sum()),
            "pct_zero": (
                100 * zero.sum() / valid.sum()
                if valid.sum()
                else np.nan
            ),
            "n_hundred": int(hundred.sum()),
            "pct_hundred": (
                100 * hundred.sum() / valid.sum()
                if valid.sum()
                else np.nan
            ),
            "participants_zero":
                seg.loc[
                    zero,
                    "participant_id",
                ].nunique(),

            "participants_hundred":
                seg.loc[
                    hundred,
                    "participant_id",
                ].nunique(),
        }
    )


# ============================================================
# LOAD DAILY HOME TIME
# ============================================================

daily = pd.read_csv(
    DAILY_FILE,
    low_memory=False,
)

daily["date"] = pd.to_datetime(
    daily["date"],
    errors="coerce",
).dt.floor("D")


def first_existing(options):

    for c in options:
        if c in daily.columns:
            return c

    return None


HOME_DAILY_COL = first_existing(
    [
        "time_at_home_fraction",
        "daily_home_fraction",
        "home_fraction",
        "home_time_fraction",
        "time_at_home_pct",
        "home_time_pct",
    ]
)

if HOME_DAILY_COL is None:
    raise RuntimeError(
        "Could not identify daily home-time column."
    )


daily[HOME_DAILY_COL] = pd.to_numeric(
    daily[HOME_DAILY_COL],
    errors="coerce",
)

finite = (
    daily[HOME_DAILY_COL]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .dropna()
)

if finite.max() <= 1.05:
    daily["daily_home_pct"] = (
        daily[HOME_DAILY_COL] * 100
    )
else:
    daily["daily_home_pct"] = (
        daily[HOME_DAILY_COL]
    )


# Look for underlying support / duration variables.
SUPPORT_COL = first_existing(
    [
        "supported_duration_s",
        "daily_supported_duration_s",
        "total_supported_duration_s",
        "supported_seconds",
        "observed_duration_s",
    ]
)

HOME_DURATION_COL = first_existing(
    [
        "home_duration_s",
        "daily_home_duration_s",
        "time_at_home_s",
        "home_supported_duration_s",
    ]
)

STATUS_COL = first_existing(
    [
        "home_status",
        "status",
    ]
)

REASON_COL = first_existing(
    [
        "home_reason",
        "reason",
    ]
)


print()
print("=" * 110)
print("DAILY HOME-TIME TABLE")
print("=" * 110)

print(
    "Daily home column:",
    HOME_DAILY_COL,
)

print(
    "Support-duration column:",
    SUPPORT_COL
    if SUPPORT_COL
    else "not found",
)

print(
    "Home-duration column:",
    HOME_DURATION_COL
    if HOME_DURATION_COL
    else "not found",
)

print(
    "Status column:",
    STATUS_COL
    if STATUS_COL
    else "not found",
)

print(
    "Reason column:",
    REASON_COL
    if REASON_COL
    else "not found",
)


if SUPPORT_COL:
    daily[SUPPORT_COL] = pd.to_numeric(
        daily[SUPPORT_COL],
        errors="coerce",
    )

if HOME_DURATION_COL:
    daily[HOME_DURATION_COL] = pd.to_numeric(
        daily[HOME_DURATION_COL],
        errors="coerce",
    )


# ============================================================
# RECONSTRUCT THE 14 DAILY RECORDS FOR EACH SEGMENT
# ============================================================

records = []

daily_lookup = {
    pid: g.set_index("date")
    for pid, g in daily.groupby(
        "participant_id"
    )
}

for _, r in seg.iterrows():

    pid = r["participant_id"]

    if pid not in daily_lookup:
        continue

    assessment_date = r[
        "assessment_date"
    ]

    dates = pd.date_range(
        end=(
            assessment_date
            - pd.Timedelta(days=1)
        ),
        periods=14,
        freq="D",
    )

    g = daily_lookup[
        pid
    ].reindex(
        dates
    )

    hpct = pd.to_numeric(
        g["daily_home_pct"],
        errors="coerce",
    )

    rec = {
        "participant_id": pid,
        "assessment_date":
            assessment_date,

        "segment_home_pct":
            r["home_14d_all_pct"],

        "valid_days":
            int(hpct.notna().sum()),

        "days_exact_zero":
            int(
                np.isclose(
                    hpct.dropna(),
                    0,
                    atol=1e-9,
                ).sum()
            ),

        "days_exact_hundred":
            int(
                np.isclose(
                    hpct.dropna(),
                    100,
                    atol=1e-9,
                ).sum()
            ),

        "daily_min_pct":
            hpct.min(),

        "daily_median_pct":
            hpct.median(),

        "daily_max_pct":
            hpct.max(),
    }

    if SUPPORT_COL:

        support = pd.to_numeric(
            g[SUPPORT_COL],
            errors="coerce",
        )

        rec[
            "mean_supported_hours"
        ] = (
            support.mean()
            / 3600
        )

        rec[
            "min_supported_hours"
        ] = (
            support.min()
            / 3600
        )

        rec[
            "median_supported_hours"
        ] = (
            support.median()
            / 3600
        )

    if HOME_DURATION_COL:

        hd = pd.to_numeric(
            g[
                HOME_DURATION_COL
            ],
            errors="coerce",
        )

        rec[
            "mean_home_hours"
        ] = (
            hd.mean()
            / 3600
        )

    records.append(rec)


audit = pd.DataFrame(
    records
)

audit[
    "segment_class"
] = "non-extreme"

audit.loc[
    np.isclose(
        audit[
            "segment_home_pct"
        ],
        0,
        atol=1e-9,
    ),
    "segment_class",
] = "exact 0%"

audit.loc[
    np.isclose(
        audit[
            "segment_home_pct"
        ],
        100,
        atol=1e-9,
    ),
    "segment_class",
] = "exact 100%"


# ============================================================
# WHAT DO EXTREME SEGMENTS LOOK LIKE?
# ============================================================

print()
print("=" * 110)
print("UNDERLYING DAILY PATTERN FOR ENTIRE-14-DAY SEGMENTS")
print("=" * 110)

cols = [
    "segment_class",
    "segment_home_pct",
    "valid_days",
    "days_exact_zero",
    "days_exact_hundred",
    "daily_min_pct",
    "daily_median_pct",
    "daily_max_pct",
]

if SUPPORT_COL:
    cols += [
        "mean_supported_hours",
        "median_supported_hours",
        "min_supported_hours",
    ]

group_summary = (
    audit[
        cols
    ]
    .groupby(
        "segment_class"
    )
    .agg(
        n_segments=(
            "segment_home_pct",
            "size",
        ),
        mean_segment_home_pct=(
            "segment_home_pct",
            "mean",
        ),
        median_valid_days=(
            "valid_days",
            "median",
        ),
        median_zero_days=(
            "days_exact_zero",
            "median",
        ),
        median_hundred_days=(
            "days_exact_hundred",
            "median",
        ),
        median_daily_min_pct=(
            "daily_min_pct",
            "median",
        ),
        median_daily_home_pct=(
            "daily_median_pct",
            "median",
        ),
        median_daily_max_pct=(
            "daily_max_pct",
            "median",
        ),
    )
)

if SUPPORT_COL:

    support_summary = (
        audit.groupby(
            "segment_class"
        )
        .agg(
            mean_daily_supported_hours=(
                "mean_supported_hours",
                "mean",
            ),
            median_daily_supported_hours=(
                "median_supported_hours",
                "median",
            ),
            median_min_daily_supported_hours=(
                "min_supported_hours",
                "median",
            ),
        )
    )

    group_summary = (
        group_summary.join(
            support_summary
        )
    )

print(
    group_summary
    .round(2)
    .to_string()
)


# ============================================================
# EXTREMES BY PARTICIPANT
# ============================================================

print()
print("=" * 110)
print("ARE EXTREMES CONCENTRATED IN A FEW PARTICIPANTS?")
print("=" * 110)

for cls in [
    "exact 0%",
    "exact 100%",
]:

    x = audit[
        audit[
            "segment_class"
        ].eq(cls)
    ]

    print()
    print(cls)

    print(
        "Segments:",
        len(x),
    )

    print(
        "Participants:",
        x[
            "participant_id"
        ].nunique(),
    )

    if len(x):

        counts = (
            x.groupby(
                "participant_id"
            )
            .size()
        )

        print(
            "Maximum extreme segments from one participant:",
            int(
                counts.max()
            ),
        )

        print(
            "Participants with >1 extreme segment:",
            int(
                (counts > 1).sum()
            ),
        )


# ============================================================
# VERIFY SEGMENT MEDIAN
# ============================================================

diff = (
    pd.to_numeric(
        audit[
            "segment_home_pct"
        ],
        errors="coerce",
    )
    - pd.to_numeric(
        audit[
            "daily_median_pct"
        ],
        errors="coerce",
    )
).abs()

print()
print("=" * 110)
print("SEGMENT RECONSTRUCTION CHECK")
print("=" * 110)

print(
    "Segments reconstructed:",
    len(audit),
)

print(
    "Max absolute difference between stored 14-day value "
    "and reconstructed daily median:",
    round(
        diff.max(),
        10,
    ),
)

print(
    "Segments with difference > 1e-6:",
    int(
        (diff > 1e-6).sum()
    ),
)


# ============================================================
# SAVE AGGREGATE OUTPUTS
# ============================================================

pd.DataFrame(
    summary_rows
).to_csv(
    OUT
    / "home_time_extreme_segment_counts.csv",
    index=False,
)

group_summary.reset_index().to_csv(
    OUT
    / "home_time_extreme_daily_support_summary.csv",
    index=False,
)

print()
print("Saved aggregate summaries to:")
print(
    OUT
    / "home_time_extreme_segment_counts.csv"
)
print(
    OUT
    / "home_time_extreme_daily_support_summary.csv"
)

print()
print(
    "No participant IDs or row-level patient records were printed."
)
