from pathlib import Path
import pandas as pd
import numpy as np

SURVEY_PATH = Path(
    "/shared/Digital Clinic/active/survey_tabular_long.csv"
)

MAP_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "inventory/raphe_participant_map.csv"
)

if not MAP_PATH.exists():
    MAP_PATH = Path(
        "/shared/Digital Clinic/raphe_outputs/digital_clinic/"
        "inventory/raphe_participant_map.csv"
    )

OUT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "daily_functioning/"
    "daily_functioning_duplicate_audit.csv"
)

# ============================================================
# LOAD PARTICIPANT MAP
# ============================================================

mapping = pd.read_csv(
    MAP_PATH,
    low_memory=False,
)

mapping["source_participant_id"] = (
    mapping["source_participant_id"]
    .astype(str)
    .str.strip()
)

mapping["participant_id"] = (
    mapping["participant_id"]
    .astype(str)
    .str.strip()
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
        subset=["source_participant_id"],
        keep="first",
    )
)

# ============================================================
# LOAD FUNCTIONING RESPONSES
# ============================================================

survey = pd.read_csv(
    SURVEY_PATH,
    low_memory=False,
)

func = survey[
    survey["activity_name"]
    .astype(str)
    .str.strip()
    .eq("Daily Survey")
    &
    survey["item"]
    .astype(str)
    .str.contains(
        "manage my day-to-day life",
        case=False,
        na=False,
    )
].copy()

func["source_participant_id"] = (
    func["participant_id"]
    .astype(str)
    .str.strip()
)

func = func.merge(
    mapping[
        [
            "source_participant_id",
            "participant_id",
        ]
    ],
    on="source_participant_id",
    how="inner",
    suffixes=(
        "_mindlamp",
        "_clinical",
    ),
)

func = func.rename(
    columns={
        "participant_id_clinical":
            "participant_id"
    }
)

func["functioning_score"] = pd.to_numeric(
    func["value"],
    errors="coerce",
)

func["assessment_datetime"] = pd.to_datetime(
    func["datetime"],
    utc=True,
    errors="coerce",
)

func["assessment_local"] = (
    func["assessment_datetime"]
    .dt.tz_convert("America/New_York")
)

func["assessment_date"] = (
    func["assessment_local"]
    .dt.date
)

func = func.dropna(
    subset=[
        "participant_id",
        "functioning_score",
        "assessment_local",
    ]
).copy()

func = func.sort_values(
    [
        "participant_id",
        "assessment_local",
    ]
)

# ============================================================
# DAILY DUPLICATE AUDIT
# ============================================================

rows = []

for (pid, date), g in func.groupby(
    [
        "participant_id",
        "assessment_date",
    ]
):
    if len(g) <= 1:
        continue

    scores = g["functioning_score"].to_numpy()

    times = (
        g["assessment_local"]
        .sort_values()
    )

    span_minutes = (
        times.iloc[-1]
        - times.iloc[0]
    ).total_seconds() / 60

    rows.append(
        {
            "participant_id": pid,
            "date": date,
            "n_responses": len(g),
            "n_unique_scores": g[
                "functioning_score"
            ].nunique(),
            "all_scores_same": (
                g["functioning_score"].nunique()
                == 1
            ),
            "first_score": scores[0],
            "last_score": scores[-1],
            "score_range": (
                np.max(scores)
                - np.min(scores)
            ),
            "time_span_minutes": span_minutes,
        }
    )

dup = pd.DataFrame(rows)

print("=" * 100)
print("DAILY FUNCTIONING DUPLICATE-RESPONSE AUDIT")
print("=" * 100)

print("\nDuplicate participant-days:", len(dup))

if len(dup):

    print(
        "Participants with >=1 duplicate day:",
        dup["participant_id"].nunique(),
    )

    print(
        "\nDays where all duplicate scores are identical:",
        int(dup["all_scores_same"].sum()),
        f"({100*dup['all_scores_same'].mean():.1f}%)",
    )

    print(
        "Days where scores differ:",
        int((~dup["all_scores_same"]).sum()),
        f"({100*(~dup['all_scores_same']).mean():.1f}%)",
    )

    print("\nNumber of responses per duplicate day:")
    print(
        dup["n_responses"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nScore range on duplicate days:")
    print(
        dup["score_range"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTime span between first and last response (minutes):")
    print(
        dup["time_span_minutes"]
        .describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
            ]
        )
        .to_string()
    )

    print("\nDiffering-score duplicate days only:")
    different = dup[
        ~dup["all_scores_same"]
    ]

    if len(different):
        print(
            different[
                [
                    "n_responses",
                    "score_range",
                    "time_span_minutes",
                ]
            ]
            .describe()
            .to_string()
        )

dup.to_csv(
    OUT,
    index=False,
)

print("\nSaved:")
print(" ", OUT)
