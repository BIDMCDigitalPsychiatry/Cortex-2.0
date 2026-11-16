from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DC = Path("/shared/Digital Clinic")
TZ = "America/New_York"
WINDOW_DAYS = 7

MAP_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/inventory/"
    "raphe_participant_map.csv"
)

if not MAP_PATH.exists():
    MAP_PATH = (
        DC
        / "raphe_outputs/digital_clinic/inventory/"
        "raphe_participant_map.csv"
    )

SURVEY_PATH = (
    DC / "active/survey_tabular_long.csv"
)

HOME_ANCHOR_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "participant_home_anchors.csv"
)

# This must be the FULL-COHORT assignment-level file.
PLACE_ASSIGNMENTS_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "place_assignments.parquet"
)

OUT_DIR = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "phq_diurnal_profiles"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ASSESSMENT_OUT = (
    OUT_DIR
    / f"phq9_low_high_assessments_{WINDOW_DAYS}d.csv"
)

HOURLY_OUT = (
    OUT_DIR
    / f"hourly_home_checkpoints_{WINDOW_DAYS}d.parquet"
)

PROFILE_OUT = (
    OUT_DIR
    / f"phq9_diurnal_home_profile_summary_{WINDOW_DAYS}d.csv"
)

DIFF_OUT = (
    OUT_DIR
    / f"phq9_diurnal_home_profile_difference_{WINDOW_DAYS}d.csv"
)

FIG_OUT = (
    OUT_DIR
    / f"phq9_diurnal_home_profile_low_vs_high_{WINDOW_DAYS}d.png"
)

PDF_OUT = (
    OUT_DIR
    / f"phq9_diurnal_home_profile_low_vs_high_{WINDOW_DAYS}d.pdf"
)

LOW_LABEL = "Low PHQ-9 (0–4)"
HIGH_LABEL = "High PHQ-9 (≥10)"

# ============================================================
# HELPERS
# ============================================================

def identify_phq_item(text):

    t = str(text).strip().lower()

    if "little interest" in t:
        return "phq1"

    if "feeling down" in t or "depressed" in t or "hopeless" in t:
        return "phq2"

    if (
        "trouble falling" in t
        or "staying asleep" in t
        or "sleeping too much" in t
    ):
        return "phq3"

    if "feeling tired" in t or "little energy" in t:
        return "phq4"

    if "poor appetite" in t or "overeating" in t:
        return "phq5"

    if "feeling bad about yourself" in t or "failure" in t:
        return "phq6"

    if "trouble concentrating" in t:
        return "phq7"

    if (
        "moving or speaking so slowly" in t
        or "fidgety" in t
        or "restless" in t
    ):
        return "phq8"

    if "better off dead" in t or "hurting yourself" in t:
        return "phq9"

    return None


def first_existing(columns, candidates):

    for c in candidates:
        if c in columns:
            return c

    return None


def parse_local_time(series):

    x = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return x.dt.tz_convert(TZ)


# ============================================================
# BASIC FILE CHECKS
# ============================================================

for p in [
    MAP_PATH,
    SURVEY_PATH,
    HOME_ANCHOR_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{p}"
        )

if not PLACE_ASSIGNMENTS_PATH.exists():

    print()
    print("=" * 110)
    print("FULL-COHORT PLACE ASSIGNMENTS FILE NOT FOUND")
    print("=" * 110)
    print()
    print(
        "Expected:"
    )
    print(
        PLACE_ASSIGNMENTS_PATH
    )
    print()
    print(
        "Do NOT rerun the full home-time workflow yet."
    )
    print(
        "We first need to check whether the assignment table "
        "already exists elsewhere or export it from the existing "
        "home-time workflow."
    )

    raise SystemExit(1)

# ============================================================
# PARTICIPANT MAP
# ============================================================

mapping = pd.read_csv(
    MAP_PATH,
    low_memory=False,
)

mapping[
    "source_participant_id"
] = (
    mapping[
        "source_participant_id"
    ]
    .astype(str)
    .str.strip()
)

mapping[
    "participant_id"
] = (
    mapping[
        "participant_id"
    ]
    .astype(str)
    .str.strip()
)

duplicate_sources = (
    mapping
    .groupby(
        "source_participant_id"
    )[
        "participant_id"
    ]
    .nunique()
)

duplicate_sources = (
    duplicate_sources[
        duplicate_sources > 1
    ]
)

if len(
    duplicate_sources
):

    print(
        "Duplicate source mappings detected:"
    )

    for source_id in (
        duplicate_sources.index
    ):

        aliases = sorted(
            mapping.loc[
                mapping[
                    "source_participant_id"
                ].eq(
                    source_id
                ),
                "participant_id",
            ]
            .astype(str)
            .unique()
        )

        print(
            f"  one source maps to multiple IDs "
            f"-> retaining {aliases[0]}"
        )

mapping = (
    mapping
    .sort_values(
        [
            "source_participant_id",
            "participant_id",
        ]
    )
    .drop_duplicates(
        subset=[
            "source_participant_id"
        ],
        keep="first",
    )
    .copy()
)

# ============================================================
# BUILD PHQ-9 ASSESSMENTS
# ============================================================

survey = pd.read_csv(
    SURVEY_PATH,
    low_memory=False,
)

dep = survey[
    survey[
        "activity_name"
    ].eq(
        "Weekly Depression Survey"
    )
].copy()

dep[
    "source_participant_id"
] = (
    dep[
        "participant_id"
    ]
    .astype(str)
    .str.strip()
)

dep[
    "numeric_value"
] = pd.to_numeric(
    dep[
        "value"
    ],
    errors="coerce",
)

dep[
    "phq_item"
] = (
    dep[
        "item"
    ]
    .apply(
        identify_phq_item
    )
)

symptom_items = [
    f"phq{i}"
    for i in range(
        1,
        10,
    )
]

dep = dep[
    dep[
        "phq_item"
    ].isin(
        symptom_items
    )
    &
    dep[
        "numeric_value"
    ].between(
        0,
        3,
    )
].copy()

dep = dep.merge(
    mapping[
        [
            "source_participant_id",
            "participant_id",
        ]
    ],
    on="source_participant_id",
    how="inner",
)

dep[
    "assessment_datetime"
] = (
    pd.to_datetime(
        dep[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )
    .dt.tz_convert(
        TZ
    )
)

dep = dep.dropna(
    subset=[
        "assessment_datetime"
    ]
).copy()

# Use exact assessment timestamp as assessment identity.
pivot = (
    dep
    .pivot_table(
        index=[
            "participant_id",
            "assessment_datetime",
        ],
        columns="phq_item",
        values="numeric_value",
        aggfunc="first",
    )
    .reset_index()
)

for c in symptom_items:
    if c not in pivot.columns:
        pivot[c] = np.nan

pivot[
    "phq9_total"
] = (
    pivot[
        symptom_items
    ]
    .sum(
        axis=1,
        min_count=9,
    )
)

pivot = pivot[
    pivot[
        "phq9_total"
    ].notna()
].copy()

pivot[
    "assessment_date"
] = (
    pivot[
        "assessment_datetime"
    ]
    .dt.floor(
        "D"
    )
)

pivot[
    "severity_group"
] = np.select(
    [
        pivot[
            "phq9_total"
        ].between(
            0,
            4,
        ),

        pivot[
            "phq9_total"
        ].ge(
            10
        ),
    ],
    [
        LOW_LABEL,
        HIGH_LABEL,
    ],
    default=np.nan,
)

assessments = pivot[
    pivot[
        "severity_group"
    ].notna()
].copy()

assessments = (
    assessments
    .sort_values(
        [
            "participant_id",
            "assessment_datetime",
        ]
    )
    .reset_index(
        drop=True
    )
)

assessments[
    "assessment_id"
] = np.arange(
    len(
        assessments
    )
)

print()
print("=" * 110)
print("PHQ-9 ASSESSMENTS")
print("=" * 110)

print(
    "Total low/high assessments:",
    len(
        assessments
    ),
)

print(
    "Participants:",
    assessments[
        "participant_id"
    ].nunique(),
)

print()
print(
    assessments[
        "severity_group"
    ]
    .value_counts()
    .to_string()
)

assessments.to_csv(
    ASSESSMENT_OUT,
    index=False,
)

# ============================================================
# HOME ANCHORS
# ============================================================

anchors = pd.read_csv(
    HOME_ANCHOR_PATH,
    low_memory=False,
)

anchors[
    "participant_id"
] = (
    anchors[
        "participant_id"
    ]
    .astype(str)
    .str.strip()
)

if (
    "home_place_candidate_id"
    not in anchors.columns
):

    raise RuntimeError(
        "home_place_candidate_id missing from anchor file."
    )

anchors[
    "home_place_candidate_id"
] = (
    anchors[
        "home_place_candidate_id"
    ]
    .astype(str)
)

# ============================================================
# PLACE ASSIGNMENTS
# ============================================================

place = pd.read_parquet(
    PLACE_ASSIGNMENTS_PATH
)

place[
    "participant_id"
] = (
    place[
        "participant_id"
    ]
    .astype(str)
    .str.strip()
)

print()
print(
    "Place assignment columns:"
)

print(
    list(
        place.columns
    )
)

start_col = first_existing(
    place.columns,
    [
        "start_local",
        "interval_start_local",
        "start_datetime_local",
        "local_start",
        "start_dt",
        "start",
        "segment_start",
        "start_time",
    ],
)

end_col = first_existing(
    place.columns,
    [
        "end_local",
        "interval_end_local",
        "end_datetime_local",
        "local_end",
        "end_dt",
        "end",
        "segment_end",
        "end_time",
    ],
)

candidate_col = first_existing(
    place.columns,
    [
        "place_candidate_id",
        "assigned_candidate_id",
        "candidate_id",
        "cluster_id",
    ],
)

if start_col is None:
    raise RuntimeError(
        "Could not identify assignment start column."
    )

if end_col is None:
    raise RuntimeError(
        "Could not identify assignment end column."
    )

if candidate_col is None:
    raise RuntimeError(
        "Could not identify place candidate column."
    )

print()
print(
    "Using assignment start:",
    start_col,
)

print(
    "Using assignment end:",
    end_col,
)

print(
    "Using place ID:",
    candidate_col,
)

place[
    "start_dt"
] = parse_local_time(
    place[
        start_col
    ]
)

place[
    "end_dt"
] = parse_local_time(
    place[
        end_col
    ]
)

place[
    candidate_col
] = (
    place[
        candidate_col
    ]
    .astype(str)
)

place = place.merge(
    anchors[
        [
            "participant_id",
            "home_place_candidate_id",
        ]
    ],
    on="participant_id",
    how="inner",
)

place[
    "at_home"
] = (
    place[
        candidate_col
    ].eq(
        place[
            "home_place_candidate_id"
        ]
    )
    .astype(int)
)

keep_ids = set(
    assessments[
        "participant_id"
    ].unique()
)

place = place[
    place[
        "participant_id"
    ].isin(
        keep_ids
    )
].copy()

place = place.dropna(
    subset=[
        "start_dt",
        "end_dt",
    ]
).copy()

place = place[
    place[
        "end_dt"
    ]
    >
    place[
        "start_dt"
    ]
].copy()

print()
print("=" * 110)
print("PLACE ASSIGNMENTS")
print("=" * 110)

print(
    "Rows:",
    len(
        place
    ),
)

print(
    "Participants:",
    place[
        "participant_id"
    ].nunique(),
)

# ============================================================
# BUILD HOURLY HOME-PRESENCE CHECKPOINTS
# ============================================================

hourly_parts = []

for pid, assess_sub in (
    assessments.groupby(
        "participant_id"
    )
):

    p = place[
        place[
            "participant_id"
        ].eq(
            pid
        )
    ].copy()

    if p.empty:
        continue

    p = p[
        [
            "start_dt",
            "end_dt",
            "at_home",
        ]
    ].sort_values(
        "start_dt"
    )

    first_day = (
        assess_sub[
            "assessment_date"
        ].min()
        -
        pd.Timedelta(
            days=WINDOW_DAYS
        )
    )

    last_day = (
        assess_sub[
            "assessment_date"
        ].max()
        +
        pd.Timedelta(
            days=1
        )
    )

    checkpoints = pd.DataFrame(
        {
            "timestamp":
                pd.date_range(
                    start=first_day,
                    end=last_day,
                    freq="1h",
                    inclusive="left",
                    tz=TZ,
                )
        }
    )

    m = pd.merge_asof(
        checkpoints,
        p,
        left_on="timestamp",
        right_on="start_dt",
        direction="backward",
    )

    # Checkpoint must actually fall inside assignment interval.
    m = m[
        m[
            "end_dt"
        ].notna()
        &
        (
            m[
                "timestamp"
            ]
            <
            m[
                "end_dt"
            ]
        )
    ].copy()

    if m.empty:
        continue

    m[
        "participant_id"
    ] = pid

    m[
        "date"
    ] = (
        m[
            "timestamp"
        ]
        .dt.floor(
            "D"
        )
    )

    m[
        "hour"
    ] = (
        m[
            "timestamp"
        ]
        .dt.hour
    )

    m[
        "is_weekend"
    ] = (
        m[
            "timestamp"
        ]
        .dt.dayofweek
        .ge(
            5
        )
    )

    hourly_parts.append(
        m[
            [
                "participant_id",
                "timestamp",
                "date",
                "hour",
                "is_weekend",
                "at_home",
            ]
        ]
    )

if not hourly_parts:
    raise RuntimeError(
        "No hourly checkpoints could be constructed."
    )

hourly = pd.concat(
    hourly_parts,
    ignore_index=True,
)

hourly.to_parquet(
    HOURLY_OUT,
    index=False,
)

print()
print("=" * 110)
print("HOURLY HOME-PRESENCE DATA")
print("=" * 110)

print(
    "Rows:",
    len(
        hourly
    ),
)

print(
    "Participants:",
    hourly[
        "participant_id"
    ].nunique(),
)

# ============================================================
# ALIGN PREVIOUS 7 DAYS TO EACH PHQ ASSESSMENT
#
# IMPORTANT:
# Assessment day is EXCLUDED.
#
# For a PHQ on date D:
# window = D-7 through D-1
# ============================================================

profile_rows = []

for a in assessments.itertuples(
    index=False
):

    h = hourly[
        hourly[
            "participant_id"
        ].eq(
            a.participant_id
        )
    ].copy()

    if h.empty:
        continue

    window_start = (
        a.assessment_date
        -
        pd.Timedelta(
            days=WINDOW_DAYS
        )
    )

    window_end = (
        a.assessment_date
        -
        pd.Timedelta(
            days=1
        )
    )

    h = h[
        (
            h[
                "date"
            ]
            >=
            window_start
        )
        &
        (
            h[
                "date"
            ]
            <=
            window_end
        )
    ].copy()

    if h.empty:
        continue

    for is_weekend, context in [
        (
            False,
            "Weekdays",
        ),
        (
            True,
            "Weekends",
        ),
    ]:

        hh = h[
            h[
                "is_weekend"
            ].eq(
                is_weekend
            )
        ].copy()

        if hh.empty:
            continue

        tmp = (
            hh
            .groupby(
                "hour",
                as_index=False,
            )[
                "at_home"
            ]
            .mean()
            .rename(
                columns={
                    "at_home":
                        "home_probability"
                }
            )
        )

        tmp[
            "assessment_id"
        ] = a.assessment_id

        tmp[
            "participant_id"
        ] = a.participant_id

        tmp[
            "assessment_date"
        ] = a.assessment_date

        tmp[
            "phq9_total"
        ] = a.phq9_total

        tmp[
            "severity_group"
        ] = a.severity_group

        tmp[
            "context"
        ] = context

        tmp[
            "n_days_available"
        ] = (
            hh[
                "date"
            ].nunique()
        )

        profile_rows.append(
            tmp
        )

profile = pd.concat(
    profile_rows,
    ignore_index=True,
)

# ============================================================
# SUMMARY CURVES
# ============================================================

summary = (
    profile
    .groupby(
        [
            "context",
            "severity_group",
            "hour",
        ],
        as_index=False,
    )
    .agg(
        mean_home_probability=(
            "home_probability",
            "mean",
        ),
        sd_home_probability=(
            "home_probability",
            "std",
        ),
        n_assessments=(
            "assessment_id",
            "nunique",
        ),
        n_participants=(
            "participant_id",
            "nunique",
        ),
    )
)

summary[
    "se"
] = (
    summary[
        "sd_home_probability"
    ]
    /
    np.sqrt(
        summary[
            "n_assessments"
        ]
    )
)

summary[
    "mean_home_pct"
] = (
    100
    *
    summary[
        "mean_home_probability"
    ]
)

summary[
    "ci_low_pct"
] = (
    100
    *
    (
        summary[
            "mean_home_probability"
        ]
        -
        1.96
        *
        summary[
            "se"
        ]
    )
)

summary[
    "ci_high_pct"
] = (
    100
    *
    (
        summary[
            "mean_home_probability"
        ]
        +
        1.96
        *
        summary[
            "se"
        ]
    )
)

summary.to_csv(
    PROFILE_OUT,
    index=False,
)

# ============================================================
# HIGH - LOW DIFFERENCE CURVES
# ============================================================

low = summary[
    summary[
        "severity_group"
    ].eq(
        LOW_LABEL
    )
].copy()

high = summary[
    summary[
        "severity_group"
    ].eq(
        HIGH_LABEL
    )
].copy()

diff = low.merge(
    high,
    on=[
        "context",
        "hour",
    ],
    suffixes=(
        "_low",
        "_high",
    ),
)

diff[
    "difference_pct"
] = (
    diff[
        "mean_home_pct_high"
    ]
    -
    diff[
        "mean_home_pct_low"
    ]
)

diff.to_csv(
    DIFF_OUT,
    index=False,
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 11

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        17,
        5.5,
    ),
)

for ax, context in zip(
    axes[:2],
    [
        "Weekdays",
        "Weekends",
    ],
):

    sub = summary[
        summary[
            "context"
        ].eq(
            context
        )
    ]

    for group in [
        LOW_LABEL,
        HIGH_LABEL,
    ]:

        g = sub[
            sub[
                "severity_group"
            ].eq(
                group
            )
        ].sort_values(
            "hour"
        )

        ax.plot(
            g[
                "hour"
            ],
            g[
                "mean_home_pct"
            ],
            linewidth=2,
            marker="o",
            markersize=3,
            label=group,
        )

        ax.fill_between(
            g[
                "hour"
            ],
            g[
                "ci_low_pct"
            ],
            g[
                "ci_high_pct"
            ],
            alpha=0.15,
        )

    ax.set_title(
        context
    )

    ax.set_xlabel(
        "Hour of day"
    )

    ax.set_ylabel(
        "Probability of being at home (%)"
    )

    ax.set_xticks(
        [
            0,
            4,
            8,
            12,
            16,
            20,
            23,
        ]
    )

    ax.set_xlim(
        0,
        23,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

axes[
    0
].legend(
    frameon=False
)

ax = axes[
    2
]

for context, style in [
    (
        "Weekdays",
        "-",
    ),
    (
        "Weekends",
        "--",
    ),
]:

    g = diff[
        diff[
            "context"
        ].eq(
            context
        )
    ].sort_values(
        "hour"
    )

    ax.plot(
        g[
            "hour"
        ],
        g[
            "difference_pct"
        ],
        linewidth=2,
        linestyle=style,
        label=context,
    )

ax.axhline(
    0,
    linestyle=":",
    linewidth=1,
)

ax.set_title(
    "High PHQ-9 − low PHQ-9"
)

ax.set_xlabel(
    "Hour of day"
)

ax.set_ylabel(
    "Difference in home probability (%)"
)

ax.set_xticks(
    [
        0,
        4,
        8,
        12,
        16,
        20,
        23,
    ]
)

ax.set_xlim(
    0,
    23,
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

ax.legend(
    frameon=False
)

fig.suptitle(
    "Diurnal home-presence profile by depression severity\n"
    "Previous 7 days before weekly PHQ-9 assessment",
    fontsize=17,
)

fig.tight_layout()

fig.savefig(
    FIG_OUT,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

plt.close(
    fig
)

print()
print("=" * 110)
print("SAVED")
print("=" * 110)

print(
    ASSESSMENT_OUT
)

print(
    HOURLY_OUT
)

print(
    PROFILE_OUT
)

print(
    DIFF_OUT
)

print(
    FIG_OUT
)

print(
    PDF_OUT
)
