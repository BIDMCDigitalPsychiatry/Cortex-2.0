from pathlib import Path
import numpy as np
import pandas as pd


DC = Path("/shared/Digital Clinic")

SEG_FILE = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time/phq_temporal_windows/"
    "phq8_phq9_home_time_strict_14d_cohort.csv"
)

HOME_DIR = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "home_time"
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

seg["is_hundred_segment"] = np.isclose(
    seg["home_14d_all_pct"],
    100,
    atol=1e-9,
)


# ============================================================
# PARTICIPANT LONGITUDINAL POSITION
# ============================================================

seg = seg.sort_values(
    [
        "participant_id",
        "assessment_date",
    ]
).copy()

seg["segment_number"] = (
    seg.groupby("participant_id")
    .cumcount()
    + 1
)

seg["n_segments_participant"] = (
    seg.groupby("participant_id")[
        "participant_id"
    ]
    .transform("size")
)

first_assessment = (
    seg.groupby("participant_id")[
        "assessment_date"
    ]
    .transform("min")
)

seg["days_since_first_assessment"] = (
    seg["assessment_date"]
    - first_assessment
).dt.days


# ============================================================
# COMPARE EXTREME VS NON-EXTREME
# ============================================================

seg["group"] = np.where(
    seg["is_zero_segment"],
    "exact 0%",
    np.where(
        seg["is_hundred_segment"],
        "exact 100%",
        "non-extreme",
    ),
)

print("=" * 110)
print("LONGITUDINAL POSITION OF HOME-TIME EXTREMES")
print("=" * 110)

summary = (
    seg.groupby("group")
    .agg(
        n_segments=(
            "participant_id",
            "size",
        ),
        n_participants=(
            "participant_id",
            "nunique",
        ),
        median_segment_number=(
            "segment_number",
            "median",
        ),
        mean_segment_number=(
            "segment_number",
            "mean",
        ),
        median_days_since_first_assessment=(
            "days_since_first_assessment",
            "median",
        ),
        mean_days_since_first_assessment=(
            "days_since_first_assessment",
            "mean",
        ),
        median_total_segments_per_participant=(
            "n_segments_participant",
            "median",
        ),
    )
)

print(
    summary.round(2).to_string()
)


# ============================================================
# DO ZERO-SEGMENT PARTICIPANTS ALSO HAVE NORMAL SEGMENTS?
# ============================================================

zero_participants = set(
    seg.loc[
        seg["is_zero_segment"],
        "participant_id",
    ]
)

zp = seg[
    seg["participant_id"].isin(
        zero_participants
    )
].copy()

participant_patterns = (
    zp.groupby("participant_id")
    .agg(
        n_segments=(
            "participant_id",
            "size",
        ),
        n_zero=(
            "is_zero_segment",
            "sum",
        ),
        min_home_pct=(
            "home_14d_all_pct",
            "min",
        ),
        max_home_pct=(
            "home_14d_all_pct",
            "max",
        ),
        median_home_pct=(
            "home_14d_all_pct",
            "median",
        ),
    )
)

participant_patterns[
    "has_nonzero_segment"
] = (
    participant_patterns[
        "max_home_pct"
    ] > 0
)

participant_patterns[
    "has_high_home_segment"
] = (
    participant_patterns[
        "max_home_pct"
    ] >= 50
)


print()
print("=" * 110)
print("ZERO-SEGMENT PARTICIPANT PATTERNS")
print("=" * 110)

print(
    "Participants with at least one 0% segment:",
    len(participant_patterns),
)

print(
    "Also had at least one non-zero segment:",
    int(
        participant_patterns[
            "has_nonzero_segment"
        ].sum()
    ),
)

print(
    "Also had at least one segment >=50% home:",
    int(
        participant_patterns[
            "has_high_home_segment"
        ].sum()
    ),
)

print(
    "Participants whose every segment was 0%:",
    int(
        (
            participant_patterns[
                "max_home_pct"
            ] == 0
        ).sum()
    ),
)


# ============================================================
# ORDER OF ZERO SEGMENTS
# ============================================================

print()
print("=" * 110)
print("WHERE DO 0% SEGMENTS OCCUR WITHIN PARTICIPANT FOLLOW-UP?")
print("=" * 110)

zero = seg[
    seg["is_zero_segment"]
].copy()

zero["relative_segment_position"] = (
    zero["segment_number"]
    / zero["n_segments_participant"]
)

print(
    "Median segment number:",
    round(
        zero[
            "segment_number"
        ].median(),
        2,
    )
)

print(
    "Median relative position (0=early, 1=late):",
    round(
        zero[
            "relative_segment_position"
        ].median(),
        3,
    )
)

print(
    "0% segments occurring in first participant segment:",
    int(
        (
            zero["segment_number"] == 1
        ).sum()
    ),
)

print(
    "0% segments occurring in final participant segment:",
    int(
        (
            zero["segment_number"]
            == zero[
                "n_segments_participant"
            ]
        ).sum()
    ),
)


# ============================================================
# FIND HOME-ANCHOR FILES + COLUMNS
# ============================================================

print()
print("=" * 110)
print("HOME-INFERENCE METADATA SOURCES")
print("=" * 110)

keywords = [
    "confidence",
    "night",
    "home",
    "reason",
    "status",
]

candidate_files = []

for path in HOME_DIR.glob("*"):

    if path.suffix.lower() not in {
        ".csv",
        ".parquet",
    }:
        continue

    try:

        if path.suffix.lower() == ".csv":
            cols = pd.read_csv(
                path,
                nrows=0,
            ).columns.tolist()

        else:
            import pyarrow.parquet as pq
            cols = pq.read_schema(
                path
            ).names

        matches = [
            c
            for c in cols
            if any(
                k in c.lower()
                for k in keywords
            )
        ]

        if matches:
            candidate_files.append(
                (
                    path,
                    matches,
                )
            )

    except Exception:
        pass


for path, cols in candidate_files:

    print("\nFILE:")
    print(path)

    print("Relevant columns:")

    for c in cols:
        print(" ", repr(c))


print()
print(
    "No participant IDs or patient-level rows were printed."
)
