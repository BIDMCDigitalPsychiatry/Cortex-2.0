from pathlib import Path
import numpy as np
import pandas as pd

DC = Path("/shared/Digital Clinic")

SURVEY = DC / "active/survey_tabular_long.csv"

MAP = (
    DC
    / "cortex2_outputs/digital_clinic/inventory/"
    "raphe_participant_map.csv"
)
if not MAP.exists():
    MAP = (
        DC
        / "raphe_outputs/digital_clinic/inventory/"
        "raphe_participant_map.csv"
    )

DAILY_HOME = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "daily_home_time.csv"
)

OUT_DIR = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "phq_diurnal_profiles"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "diurnal_export_participants.txt"


def identify_phq_item(text):
    t = str(text).strip().lower()

    if "little interest" in t:
        return "phq1"
    if "feeling down" in t or "hopeless" in t:
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


mapping = pd.read_csv(MAP, low_memory=False)

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
        "source_participant_id",
        keep="first",
    )
)

survey = pd.read_csv(
    SURVEY,
    low_memory=False,
)

dep = survey[
    survey["activity_name"].eq(
        "Weekly Depression Survey"
    )
].copy()

dep["source_participant_id"] = (
    dep["participant_id"]
    .astype(str)
    .str.strip()
)

dep["score"] = pd.to_numeric(
    dep["value"],
    errors="coerce",
)

dep["phq_item"] = dep["item"].apply(
    identify_phq_item
)

items = [
    f"phq{i}"
    for i in range(1, 10)
]

dep = dep[
    dep["phq_item"].isin(items)
    & dep["score"].between(0, 3)
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
    suffixes=(
        "_mindlamp",
        "_clinical",
    ),
)

dep = dep.rename(
    columns={
        "participant_id_clinical":
            "participant_id"
    }
)

dep["assessment_datetime"] = pd.to_datetime(
    dep["datetime"],
    errors="coerce",
    utc=True,
)

wide = (
    dep.pivot_table(
        index=[
            "participant_id",
            "assessment_datetime",
        ],
        columns="phq_item",
        values="score",
        aggfunc="first",
    )
    .reset_index()
)

for c in items:
    if c not in wide.columns:
        wide[c] = np.nan

wide["phq9"] = wide[items].sum(
    axis=1,
    min_count=9,
)

wide = wide[
    wide["phq9"].notna()
].copy()

wide = wide[
    wide["phq9"].le(4)
    | wide["phq9"].ge(10)
].copy()

# Only participants already represented in the valid
# home-time cohort are relevant.
home = pd.read_csv(
    DAILY_HOME,
    usecols=["participant_id"],
    low_memory=False,
)

home_ids = set(
    home["participant_id"]
    .astype(str)
    .str.strip()
    .unique()
)

target_ids = sorted(
    set(
        wide["participant_id"]
        .astype(str)
        .unique()
    )
    & home_ids
)

OUT.write_text(
    "\n".join(target_ids) + "\n"
)

print("=" * 100)
print("PHQ DIURNAL TARGET PARTICIPANTS")
print("=" * 100)

print(
    "\nLow/high PHQ assessments:",
    len(wide),
)

print(
    "Low/high PHQ participants:",
    wide["participant_id"].nunique(),
)

print(
    "Participants also in home-time cohort:",
    len(target_ids),
)

print("\nSaved:")
print(OUT)
