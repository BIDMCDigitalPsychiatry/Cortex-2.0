"""Build 7-day GPS windows preceding weekly clinical assessments.

Each clinical observation is one participant + assessment date.
GPS window = assessment date - 6 days through assessment date, inclusive.

Depression:
- 9 numeric symptom items, score 0-27
- functional difficulty item excluded

Anxiety:
- 7 numeric symptom items, score 0-21
- functional difficulty item excluded
"""

from pathlib import Path

import numpy as np
import pandas as pd


DC = Path("/shared/Digital Clinic")

MAP_PATH = (
    DC
    / "cortex2_outputs/digital_clinic/inventory/"
    "raphe_participant_map.csv"
)

# Backward-compatible fallback.
if not MAP_PATH.exists():
    MAP_PATH = (
        DC
        / "raphe_outputs/digital_clinic/inventory/"
        "raphe_participant_map.csv"
    )

SURVEY_PATH = (
    DC
    / "active/survey_tabular_long.csv"
)

OUT = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ASSESSMENT_OUT = (
    OUT
    / "gps_clinical_weekly_assessments.csv"
)

DAY_OUT = (
    OUT
    / "gps_clinical_weekly_requested_days.parquet"
)


CALIBRATION = {
    "LZH628","YLA357","YHU414","MSY516",
    "AWU341","LPE511","IRA765","BCU709",
    "SSC418","ESE381","CMA782","CCH811",
    "CAL651","HPO706","NKA705","JHE615",
}

HOLDOUT = {
    "JSC434","AKL446","BPH745","LKU748",
    "TTO350","VTE817","MFR349","SKL382",
    "ODI718","JDE488","SXU679","PDO805",
    "EAB653","MSI522","AVO733","PPH714",
}

COHORT = CALIBRATION | HOLDOUT


mapping = pd.read_csv(
    MAP_PATH
)

mapping = mapping[
    mapping["participant_id"].isin(
        COHORT
    )
].copy()

if len(mapping) != 32:
    raise RuntimeError(
        f"Expected mappings for 32 participants, "
        f"found {len(mapping)}."
    )

mapping["cohort"] = (
    mapping["participant_id"]
    .apply(
        lambda x:
        "calibration"
        if x in CALIBRATION
        else "holdout"
    )
)


survey = pd.read_csv(
    SURVEY_PATH,
    low_memory=False,
)

survey = survey[
    survey["activity_name"].isin(
        [
            "Weekly Depression Survey",
            "Weekly Anxiety Survey",
        ]
    )
].copy()

survey["datetime"] = pd.to_datetime(
    survey["datetime"],
    utc=True,
    errors="coerce",
)

survey["numeric_value"] = pd.to_numeric(
    survey["value"],
    errors="coerce",
)

survey = survey.merge(
    mapping[
        [
            "participant_id",
            "source_participant_id",
            "cohort",
        ]
    ],
    left_on="participant_id",
    right_on="source_participant_id",
    how="inner",
    suffixes=("_mindlamp", "_clinical"),
)

survey["assessment_date"] = (
    survey["datetime"]
    .dt.floor("D")
    .dt.tz_localize(None)
)


# ============================================================
# DEPRESSION SCORE
# ============================================================

dep = survey[
    survey["activity_name"].eq(
        "Weekly Depression Survey"
    )
    & survey["numeric_value"].between(
        0,
        3,
    )
].copy()

dep = (
    dep.groupby(
        [
            "participant_id_clinical",
            "source_participant_id",
            "cohort",
            "timestamp",
            "assessment_date",
        ],
        as_index=False,
    )
    .agg(
        depression_score=(
            "numeric_value",
            "sum",
        ),
        depression_items=(
            "numeric_value",
            "count",
        ),
    )
)

dep.loc[
    dep["depression_items"] != 9,
    "depression_score",
] = np.nan


# ============================================================
# ANXIETY SCORE
# ============================================================

anx = survey[
    survey["activity_name"].eq(
        "Weekly Anxiety Survey"
    )
    & survey["numeric_value"].between(
        0,
        3,
    )
].copy()

anx = (
    anx.groupby(
        [
            "participant_id_clinical",
            "source_participant_id",
            "cohort",
            "timestamp",
            "assessment_date",
        ],
        as_index=False,
    )
    .agg(
        anxiety_score=(
            "numeric_value",
            "sum",
        ),
        anxiety_items=(
            "numeric_value",
            "count",
        ),
    )
)

anx.loc[
    anx["anxiety_items"] != 7,
    "anxiety_score",
] = np.nan


# ============================================================
# ONE ROW PER PARTICIPANT + ASSESSMENT DATE
# ============================================================

dep_date = (
    dep.groupby(
        [
            "participant_id_clinical",
            "source_participant_id",
            "cohort",
            "assessment_date",
        ],
        as_index=False,
    )
    .agg(
        depression_score=(
            "depression_score",
            "first",
        ),
    )
)

anx_date = (
    anx.groupby(
        [
            "participant_id_clinical",
            "source_participant_id",
            "cohort",
            "assessment_date",
        ],
        as_index=False,
    )
    .agg(
        anxiety_score=(
            "anxiety_score",
            "first",
        ),
    )
)

assessments = dep_date.merge(
    anx_date,
    on=[
        "participant_id_clinical",
        "source_participant_id",
        "cohort",
        "assessment_date",
    ],
    how="outer",
)

assessments = assessments.rename(
    columns={
        "participant_id_clinical":
            "participant_id",
        "source_participant_id":
            "mindlamp_uid",
    }
)

assessments = assessments.sort_values(
    [
        "participant_id",
        "assessment_date",
    ]
).reset_index(drop=True)

assessments["clinical_window_id"] = (
    assessments["participant_id"]
    + "__"
    + assessments[
        "assessment_date"
    ].dt.strftime("%Y%m%d")
)

if assessments[
    [
        "participant_id",
        "assessment_date",
    ]
].duplicated().any():
    raise RuntimeError(
        "Duplicate participant-assessment dates."
    )


# ============================================================
# CREATE PRECEDING 7 DAYS
# ============================================================

rows = []

for _, a in assessments.iterrows():

    for offset in range(-6, 1):

        date = (
            a["assessment_date"]
            + pd.Timedelta(
                days=offset
            )
        )

        start = pd.Timestamp(
            date,
            tz="UTC",
        )

        end = (
            start
            + pd.Timedelta(days=1)
        )

        rows.append(
            {
                "clinical_window_id":
                    a["clinical_window_id"],

                "participant_id":
                    a["participant_id"],

                "mindlamp_uid":
                    a["mindlamp_uid"],

                "cohort":
                    a["cohort"],

                "assessment_date":
                    a["assessment_date"],

                "date":
                    date,

                "window_day":
                    offset,

                "start_ms":
                    int(
                        start.timestamp()
                        * 1000
                    ),

                "end_ms":
                    int(
                        end.timestamp()
                        * 1000
                    ),
            }
        )

days = pd.DataFrame(rows)

assessments.to_csv(
    ASSESSMENT_OUT,
    index=False,
)

days.to_parquet(
    DAY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("GPS CLINICAL WEEKLY WINDOW MANIFEST")
print("=" * 80)

print(
    "Participants:",
    assessments[
        "participant_id"
    ].nunique(),
)

print(
    "Unique clinical dates:",
    len(assessments),
)

print(
    "Depression assessments:",
    assessments[
        "depression_score"
    ].notna().sum(),
)

print(
    "Anxiety assessments:",
    assessments[
        "anxiety_score"
    ].notna().sum(),
)

print(
    "Requested participant-days:",
    len(days),
)

print(
    "Unique participant-days:",
    days[
        [
            "participant_id",
            "date",
        ]
    ]
    .drop_duplicates()
    .shape[0],
)

print("\nBy cohort:")

print(
    assessments.groupby(
        "cohort"
    )
    .agg(
        participants=(
            "participant_id",
            "nunique",
        ),
        clinical_dates=(
            "clinical_window_id",
            "size",
        ),
        depression=(
            "depression_score",
            lambda x:
                x.notna().sum(),
        ),
        anxiety=(
            "anxiety_score",
            lambda x:
                x.notna().sum(),
        ),
    )
    .to_string()
)

print(
    "\nAssessment date range:",
    assessments[
        "assessment_date"
    ].min(),
    "→",
    assessments[
        "assessment_date"
    ].max(),
)

print(
    "\nSaved assessments:",
    ASSESSMENT_OUT,
)

print(
    "Saved requested days:",
    DAY_OUT,
)
