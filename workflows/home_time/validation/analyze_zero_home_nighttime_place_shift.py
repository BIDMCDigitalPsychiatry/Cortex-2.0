from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time"
)

SEG_FILE = (
    ROOT
    / "phq_temporal_windows"
    / "phq8_phq9_home_time_strict_14d_cohort.csv"
)

DAILY_FILE = ROOT / "daily_home_time.csv"
ANCHOR_FILE = ROOT / "participant_home_anchors.csv"

ASSIGN_FILE = (
    ROOT
    / "zero_home_diagnostic"
    / "place_assignments.parquet"
)

OUT_DIR = (
    ROOT
    / "zero_home_diagnostic"
)

LOCAL_TZ = "America/New_York"

# Same nighttime observability rule used by home inference
MIN_NIGHT_OBSERVED_MS = 2 * 60 * 60 * 1000


# ============================================================
# LOAD DATA
# ============================================================

seg = pd.read_csv(
    SEG_FILE,
    low_memory=False,
)

seg["home_14d_all_pct"] = pd.to_numeric(
    seg["home_14d_all_pct"],
    errors="coerce",
)

zero_ids = set(
    seg.loc[
        np.isclose(
            seg["home_14d_all_pct"],
            0.0,
            atol=1e-9,
        ),
        "participant_id",
    ]
    .astype(str)
    .unique()
)


daily = pd.read_csv(
    DAILY_FILE,
    low_memory=False,
)

daily["participant_id"] = (
    daily["participant_id"].astype(str)
)

daily["date"] = pd.to_datetime(
    daily["date"],
    errors="coerce",
).dt.floor("D")

daily["time_at_home_pct"] = pd.to_numeric(
    daily["time_at_home_pct"],
    errors="coerce",
)

daily = daily[
    daily["participant_id"].isin(
        zero_ids
    )
].copy()


anchors = pd.read_csv(
    ANCHOR_FILE,
    low_memory=False,
)

anchors["participant_id"] = (
    anchors["participant_id"].astype(str)
)

anchors["home_place_candidate_id"] = pd.to_numeric(
    anchors["home_place_candidate_id"],
    errors="coerce",
)

anchors = anchors[
    anchors["participant_id"].isin(
        zero_ids
    )
].copy()

home_map = dict(
    zip(
        anchors["participant_id"],
        anchors["home_place_candidate_id"],
    )
)


assign = pd.read_parquet(
    ASSIGN_FILE
)

assign["participant_id"] = (
    assign["participant_id"].astype(str)
)

for c in [
    "episode_start_ms",
    "episode_end_ms",
    "place_candidate_id",
]:
    assign[c] = pd.to_numeric(
        assign[c],
        errors="coerce",
    )

assign = assign[
    assign["participant_id"].isin(
        zero_ids
    )
].copy()

assign = assign[
    assign[
        "episode_start_ms"
    ].notna()
    & assign[
        "episode_end_ms"
    ].notna()
    & assign[
        "place_candidate_id"
    ].notna()
].copy()


# ============================================================
# LONGEST CONSECUTIVE ZERO RUN
# ============================================================

def longest_zero_run(g):

    g = (
        g.sort_values("date")
        .drop_duplicates("date")
        .copy()
    )

    g["is_zero"] = (
        g["time_at_home_pct"].notna()
        & np.isclose(
            g["time_at_home_pct"],
            0.0,
            atol=1e-9,
        )
    )

    best = []
    current = []

    previous_date = None

    for _, r in g.iterrows():

        d = r["date"]

        if not r["is_zero"]:
            current = []
            previous_date = d
            continue

        if (
            current
            and previous_date is not None
            and d - previous_date
            != pd.Timedelta(days=1)
        ):
            current = []

        current.append(d)

        if len(current) > len(best):
            best = current.copy()

        previous_date = d

    if not best:
        return None

    return {
        "run_start": best[0],
        "run_end": best[-1],
        "run_days": len(best),
    }


runs = {}

for pid, g in daily.groupby(
    "participant_id"
):

    x = longest_zero_run(g)

    if x is not None:
        runs[pid] = x


# ============================================================
# NIGHTTIME PLACE OCCUPANCY
# ============================================================

def nighttime_summary(
    participant_assignments,
    dates,
    home_place_id,
):

    winner_rows = []
    all_place_duration = []

    starts = (
        participant_assignments[
            "episode_start_ms"
        ]
        .to_numpy(dtype=float)
    )

    ends = (
        participant_assignments[
            "episode_end_ms"
        ]
        .to_numpy(dtype=float)
    )

    places = (
        participant_assignments[
            "place_candidate_id"
        ]
        .to_numpy(dtype=float)
    )

    for date in dates:

        local_date = pd.Timestamp(
            date
        ).date()

        night_start = pd.Timestamp(
            f"{local_date} 00:00:00"
        ).tz_localize(
            LOCAL_TZ
        )

        night_end = pd.Timestamp(
            f"{local_date} 06:00:00"
        ).tz_localize(
            LOCAL_TZ
        )

        ns = (
            night_start
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )

        ne = (
            night_end
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )

        overlap = np.maximum(
            0,
            np.minimum(
                ends,
                ne,
            )
            - np.maximum(
                starts,
                ns,
            ),
        )

        observed_ms = float(
            overlap.sum()
        )

        if (
            observed_ms
            < MIN_NIGHT_OBSERVED_MS
        ):
            continue

        night = pd.DataFrame(
            {
                "place_candidate_id":
                    places,
                "duration_ms":
                    overlap,
            }
        )

        night = (
            night[
                night["duration_ms"] > 0
            ]
            .groupby(
                "place_candidate_id",
                as_index=False,
            )[
                "duration_ms"
            ]
            .sum()
            .sort_values(
                [
                    "duration_ms",
                    "place_candidate_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        if night.empty:
            continue

        winner = int(
            night.iloc[0][
                "place_candidate_id"
            ]
        )

        winner_rows.append(
            {
                "date": date,
                "winner": winner,
                "observed_ms":
                    observed_ms,
            }
        )

        night = night.copy()
        night["date"] = date

        all_place_duration.append(
            night
        )

    if not winner_rows:

        return {
            "n_valid_nights": 0,
            "home_winner_fraction":
                np.nan,
            "top_alt_winner_fraction":
                np.nan,
            "n_distinct_winners":
                np.nan,
            "home_duration_fraction":
                np.nan,
            "top_alt_duration_fraction":
                np.nan,
        }

    winners = pd.DataFrame(
        winner_rows
    )

    n_valid = len(winners)

    home_winner_fraction = (
        winners["winner"]
        .eq(home_place_id)
        .mean()
    )

    winner_counts = (
        winners[
            "winner"
        ]
        .value_counts()
    )

    alt_counts = (
        winner_counts[
            winner_counts.index
            != home_place_id
        ]
    )

    top_alt_winner_fraction = (
        alt_counts.iloc[0]
        / n_valid
        if len(alt_counts)
        else 0.0
    )

    all_duration = pd.concat(
        all_place_duration,
        ignore_index=True,
    )

    by_place = (
        all_duration.groupby(
            "place_candidate_id"
        )[
            "duration_ms"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    total_duration = (
        by_place.sum()
    )

    home_duration = (
        by_place.get(
            home_place_id,
            0.0,
        )
    )

    alt_duration = (
        by_place[
            by_place.index
            != home_place_id
        ]
    )

    top_alt_duration = (
        alt_duration.iloc[0]
        if len(alt_duration)
        else 0.0
    )

    return {
        "n_valid_nights":
            n_valid,

        "home_winner_fraction":
            home_winner_fraction,

        "top_alt_winner_fraction":
            top_alt_winner_fraction,

        "n_distinct_winners":
            winners[
                "winner"
            ].nunique(),

        "home_duration_fraction":
            (
                home_duration
                / total_duration
                if total_duration > 0
                else np.nan
            ),

        "top_alt_duration_fraction":
            (
                top_alt_duration
                / total_duration
                if total_duration > 0
                else np.nan
            ),
    }


# ============================================================
# PARTICIPANT-LEVEL DIAGNOSTIC
# ============================================================

rows = []

for pid in sorted(
    zero_ids
):

    if pid not in runs:
        continue

    home_id = home_map.get(
        pid,
        np.nan,
    )

    if not np.isfinite(
        home_id
    ):
        continue

    home_id = int(
        home_id
    )

    g = assign[
        assign[
            "participant_id"
        ].eq(pid)
    ].copy()

    if g.empty:
        continue

    run = runs[pid]

    run_dates = pd.date_range(
        run["run_start"],
        run["run_end"],
        freq="D",
    )

    prior_dates = pd.date_range(
        end=(
            run["run_start"]
            - pd.Timedelta(days=1)
        ),
        periods=14,
        freq="D",
    )

    prior = nighttime_summary(
        g,
        prior_dates,
        home_id,
    )

    during = nighttime_summary(
        g,
        run_dates,
        home_id,
    )

    rows.append(
        {
            "run_days":
                run["run_days"],

            "prior_valid_nights":
                prior[
                    "n_valid_nights"
                ],

            "prior_home_winner_fraction":
                prior[
                    "home_winner_fraction"
                ],

            "prior_top_alt_winner_fraction":
                prior[
                    "top_alt_winner_fraction"
                ],

            "prior_n_distinct_winners":
                prior[
                    "n_distinct_winners"
                ],

            "prior_home_duration_fraction":
                prior[
                    "home_duration_fraction"
                ],

            "prior_top_alt_duration_fraction":
                prior[
                    "top_alt_duration_fraction"
                ],

            "run_valid_nights":
                during[
                    "n_valid_nights"
                ],

            "run_home_winner_fraction":
                during[
                    "home_winner_fraction"
                ],

            "run_top_alt_winner_fraction":
                during[
                    "top_alt_winner_fraction"
                ],

            "run_n_distinct_winners":
                during[
                    "n_distinct_winners"
                ],

            "run_home_duration_fraction":
                during[
                    "home_duration_fraction"
                ],

            "run_top_alt_duration_fraction":
                during[
                    "top_alt_duration_fraction"
                ],
        }
    )


result = pd.DataFrame(
    rows
)


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

print("=" * 110)
print(
    "NIGHTTIME PLACE SHIFT DURING LONGEST ZERO-HOME RUN"
)
print("=" * 110)

print(
    "Participants analyzed:",
    len(result),
)

print()

summary_vars = [
    "run_days",

    "prior_valid_nights",
    "run_valid_nights",

    "prior_home_winner_fraction",
    "run_home_winner_fraction",

    "prior_top_alt_winner_fraction",
    "run_top_alt_winner_fraction",

    "prior_n_distinct_winners",
    "run_n_distinct_winners",

    "prior_home_duration_fraction",
    "run_home_duration_fraction",

    "prior_top_alt_duration_fraction",
    "run_top_alt_duration_fraction",
]

summary = (
    result[
        summary_vars
    ]
    .describe(
        percentiles=[
            .25,
            .50,
            .75,
        ]
    )
    .T
)

print(
    summary[
        [
            "count",
            "mean",
            "25%",
            "50%",
            "75%",
            "min",
            "max",
        ]
    ]
    .round(3)
    .to_string()
)


# ============================================================
# PAIRED CHANGE
# ============================================================

print()
print("=" * 110)
print(
    "WITHIN-PARTICIPANT CHANGE: BEFORE → ZERO-HOME RUN"
)
print("=" * 110)

comparisons = {
    "Home wins nights":
        (
            "prior_home_winner_fraction",
            "run_home_winner_fraction",
        ),

    "Top alternative wins nights":
        (
            "prior_top_alt_winner_fraction",
            "run_top_alt_winner_fraction",
        ),

    "Home share of nighttime duration":
        (
            "prior_home_duration_fraction",
            "run_home_duration_fraction",
        ),

    "Top alternative share of nighttime duration":
        (
            "prior_top_alt_duration_fraction",
            "run_top_alt_duration_fraction",
        ),

    "Distinct nightly winners":
        (
            "prior_n_distinct_winners",
            "run_n_distinct_winners",
        ),
}

comparison_rows = []

for label, (
    before_col,
    during_col,
) in comparisons.items():

    x = result[
        [
            before_col,
            during_col,
        ]
    ].dropna()

    delta = (
        x[during_col]
        - x[before_col]
    )

    print()
    print(label)

    print(
        "  N:",
        len(x),
    )

    print(
        "  Median before:",
        round(
            x[
                before_col
            ].median(),
            3,
        ),
    )

    print(
        "  Median during:",
        round(
            x[
                during_col
            ].median(),
            3,
        ),
    )

    print(
        "  Median change:",
        round(
            delta.median(),
            3,
        ),
    )

    comparison_rows.append(
        {
            "metric":
                label,

            "n":
                len(x),

            "median_before":
                x[
                    before_col
                ].median(),

            "median_during":
                x[
                    during_col
                ].median(),

            "median_change":
                delta.median(),
        }
    )


# ============================================================
# DIRECTIONAL PATTERNS
# No clinical threshold imposed.
# ============================================================

print()
print("=" * 110)
print(
    "DIRECTION OF NIGHTTIME PLACE CHANGE"
)
print("=" * 110)

valid = result[
    result[
        "run_valid_nights"
    ] > 0
].copy()

print(
    "Participants with valid nighttime evidence during run:",
    len(valid),
)

print(
    "Participants where home won fewer nights during run "
    "than before:",
    int(
        (
            valid[
                "run_home_winner_fraction"
            ]
            <
            valid[
                "prior_home_winner_fraction"
            ]
        ).sum()
    ),
)

print(
    "Participants where top alternative won more nights "
    "during run than before:",
    int(
        (
            valid[
                "run_top_alt_winner_fraction"
            ]
            >
            valid[
                "prior_top_alt_winner_fraction"
            ]
        ).sum()
    ),
)

print(
    "Participants where top alternative won more nights "
    "than original home during zero run:",
    int(
        (
            valid[
                "run_top_alt_winner_fraction"
            ]
            >
            valid[
                "run_home_winner_fraction"
            ]
        ).sum()
    ),
)


# ============================================================
# SAVE AGGREGATE ONLY
# ============================================================

pd.DataFrame(
    comparison_rows
).to_csv(
    OUT_DIR
    / "zero_home_nighttime_shift_summary.csv",
    index=False,
)

summary.reset_index(
    names="metric"
).to_csv(
    OUT_DIR
    / "zero_home_nighttime_shift_descriptives.csv",
    index=False,
)

print()
print(
    "Saved aggregate summaries:"
)

print(
    OUT_DIR
    / "zero_home_nighttime_shift_summary.csv"
)

print(
    OUT_DIR
    / "zero_home_nighttime_shift_descriptives.csv"
)

print()
print(
    "No participant IDs, coordinates, or patient-level "
    "location records were printed or saved."
)
