from pathlib import Path
import inspect
import importlib.util
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/shared/Digital Clinic")

OUT_DIR = (
    ROOT
    / "cortex2_outputs"
    / "digital_clinic"
    / "multires_clinical_benchmark"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEGRADATION_PY = (
    ROOT
    / "Cortex-2.0"
    / "src"
    / "cortex2_engine"
    / "features"
    / "gps_degradation.py"
)

SURVEY_LONG = ROOT / "active" / "survey_tabular_long.csv"

DAILY_EMA = (
    ROOT
    / "cortex2_outputs"
    / "digital_clinic"
    / "clinical"
    / "daily_ema"
    / "daily_ema_gps_features.parquet"
)

WEEKLY_GPS = (
    ROOT
    / "cortex2_outputs"
    / "digital_clinic"
    / "feature_reliability"
    / "gps_clinical"
    / "gps_clinical_weekly_features_with_home.parquet"
)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 1. GPS DEGRADATION API
# ============================================================

section("GPS DEGRADATION API")

spec = importlib.util.spec_from_file_location(
    "gps_degradation",
    DEGRADATION_PY,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

api_rows = []

for name, obj in inspect.getmembers(module):
    if (
        inspect.isfunction(obj)
        and not name.startswith("_")
        and obj.__module__ == module.__name__
    ):
        signature = str(inspect.signature(obj))
        doc = inspect.getdoc(obj) or ""

        api_rows.append({
            "function": name,
            "signature": signature,
            "docstring": doc,
        })

        print(f"{name}{signature}")

pd.DataFrame(api_rows).to_csv(
    OUT_DIR / "gps_degradation_api.csv",
    index=False,
)


# ============================================================
# 2. WEEKLY CLINICAL TARGETS
# ============================================================

section("WEEKLY CLINICAL TARGETS")

weekly = pd.read_parquet(
    WEEKLY_GPS,
    columns=[
        "participant_id",
        "depression_score",
        "anxiety_score",
    ],
)

weekly_summary = []

for col in [
    "depression_score",
    "anxiety_score",
]:
    values = pd.to_numeric(
        weekly[col],
        errors="coerce",
    )

    row = {
        "target": col,
        "n_observations": int(values.notna().sum()),
        "n_participants": int(
            weekly.loc[
                values.notna(),
                "participant_id",
            ].nunique()
        ),
        "min": values.min(),
        "median": values.median(),
        "max": values.max(),
    }

    weekly_summary.append(row)

    print(
        f"{col}: "
        f"{row['n_observations']:,} observations | "
        f"{row['n_participants']:,} participants | "
        f"range {row['min']}–{row['max']}"
    )

pd.DataFrame(weekly_summary).to_csv(
    OUT_DIR / "weekly_clinical_target_summary.csv",
    index=False,
)


# ============================================================
# 3. DAILY EMA TARGETS
# ============================================================

section("DAILY EMA TARGETS")

ema = pd.read_parquet(
    DAILY_EMA,
    columns=[
        "participant_id",
        "date",
        "anxiety",
        "mood",
        "daily_function",
    ],
)

ema_summary = []

for col in [
    "anxiety",
    "mood",
    "daily_function",
]:
    values = pd.to_numeric(
        ema[col],
        errors="coerce",
    )

    row = {
        "target": col,
        "n_observations": int(values.notna().sum()),
        "n_participants": int(
            ema.loc[
                values.notna(),
                "participant_id",
            ].nunique()
        ),
        "min": values.min(),
        "median": values.median(),
        "max": values.max(),
    }

    ema_summary.append(row)

    print(
        f"{col}: "
        f"{row['n_observations']:,} observations | "
        f"{row['n_participants']:,} participants | "
        f"range {row['min']}–{row['max']}"
    )

pd.DataFrame(ema_summary).to_csv(
    OUT_DIR / "daily_ema_target_summary.csv",
    index=False,
)


# ============================================================
# 4. SURVEY STRUCTURE
# ============================================================

section("SURVEY STRUCTURE")

survey = pd.read_csv(
    SURVEY_LONG,
    low_memory=False,
)

print(f"Rows: {len(survey):,}")
print(
    f"Participants: "
    f"{survey['participant_id'].nunique():,}"
)
print(f"Columns: {len(survey.columns):,}")

print("\nColumns:")
print(", ".join(survey.columns))


# ============================================================
# 5. SURVEY ITEM INVENTORY
# ============================================================

if "item" in survey.columns:

    items = (
        survey["item"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
        .rename_axis("item")
        .reset_index(name="n_rows")
    )

    items.to_csv(
        OUT_DIR / "survey_item_inventory.csv",
        index=False,
    )

    section("LIKELY CLINICAL SURVEY ITEMS")

    pattern = (
        r"phq|gad|depress|anx|mood|sleep|stress|"
        r"function|suic|self.?harm|lonely|well.?being|"
        r"interest|pleasure|energy|appetite|concentr|"
        r"restless|worry|relax|irritable"
    )

    clinical_items = items[
        items["item"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    ].copy()

    clinical_items.to_csv(
        OUT_DIR / "survey_likely_clinical_items.csv",
        index=False,
    )

    print(
        f"Unique survey item labels: {len(items):,}"
    )
    print(
        f"Likely clinical items: {len(clinical_items):,}"
    )

    if not clinical_items.empty:
        print()
        print(
            clinical_items
            .head(40)
            .to_string(index=False)
        )


# ============================================================
# 6. CLINICAL TARGET REGISTRY
# ============================================================

section("CLINICAL TARGET REGISTRY")

registry = pd.DataFrame([
    {
        "target": "PHQ-9 total",
        "column": "depression_score",
        "timescale": "questionnaire",
        "level": "total",
    },
    {
        "target": "GAD-7 total",
        "column": "anxiety_score",
        "timescale": "questionnaire",
        "level": "total",
    },
    {
        "target": "Daily anxiety EMA",
        "column": "anxiety",
        "timescale": "daily",
        "level": "EMA",
    },
    {
        "target": "Daily mood EMA",
        "column": "mood",
        "timescale": "daily",
        "level": "EMA",
    },
    {
        "target": "Daily functioning EMA",
        "column": "daily_function",
        "timescale": "daily",
        "level": "EMA",
    },
])

registry.to_csv(
    OUT_DIR / "clinical_target_registry.csv",
    index=False,
)

print(registry.to_string(index=False))


# ============================================================
# DONE
# ============================================================

section("DONE")

print("Saved detailed outputs to:")
print(OUT_DIR)

print("\nFor the next step, paste only:")
print("  1. GPS DEGRADATION API")
print("  2. LIKELY CLINICAL SURVEY ITEMS")
