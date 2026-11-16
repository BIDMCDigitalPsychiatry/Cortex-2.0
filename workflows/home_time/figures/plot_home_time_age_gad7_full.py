from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_CANDIDATES = [
    Path("/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/home_time/paper_replication/paper_replication_analysis_dataset.csv"),
    Path("/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/home_time/phq_temporal_windows/phq8_phq9_home_time_all_assessments.csv"),
    Path("/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/home_time/phq_temporal_windows/phq8_phq9_home_time_strict_14d_cohort.csv"),
]

WEEKLY_FEATURES = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/gps_clinical_weekly_features_with_home.parquet"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/feature_reliability/gps_clinical/home_time/paper_replication"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "home_time_age_gad7_scatter_full_cohort.png"
OUT_PDF = OUT_DIR / "home_time_age_gad7_scatter_full_cohort.pdf"
OUT_CSV = OUT_DIR / "home_time_age_gad7_analysis_dataset.csv"
OUT_SUMMARY = OUT_DIR / "home_time_age_gad7_panel_counts.csv"

# ============================================================
# HELPERS
# ============================================================

def pick_base_file():
    for path in BASE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any base analysis dataset. Checked:\n"
        + "\n".join(str(p) for p in BASE_CANDIDATES)
    )

def normalize_date(series):
    s = pd.to_datetime(series, errors="coerce")
    try:
        # works for tz-aware datetimes
        s = s.dt.tz_convert(None)
    except Exception:
        pass
    try:
        # harmless for tz-naive; converts if still tz-aware in some cases
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s.dt.normalize()

def first_existing_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label}. Tried: {candidates}")

def add_scatter_with_fit(ax, df, xcol, ycol, title, xlabel, panel_label):
    sub = df[[xcol, ycol]].dropna().copy()
    n = len(sub)

    ax.scatter(
        sub[xcol],
        sub[ycol],
        s=10,
        alpha=0.45,
        linewidths=0
    )

    if n >= 2 and sub[xcol].nunique() >= 2:
        x = sub[xcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ys = intercept + slope * xs
        ax.plot(xs, ys, linewidth=1.5)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Home time (%)")
    ax.set_ylim(0, 100)
    ax.text(
        0.01, 1.04, panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom"
    )
    ax.text(
        0.02, 0.96, f"N = {n}",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top"
    )

    return n

# ============================================================
# LOAD BASE DATASET
# ============================================================

base_path = pick_base_file()
df = pd.read_csv(base_path)

required_home_cols = [
    "home_14d_all_pct",
    "home_14d_weekday_matched_pct",
    "home_14d_weekend_pct",
]
for c in required_home_cols:
    if c not in df.columns:
        raise KeyError(f"Missing required home-time column: {c}")

participant_col = first_existing_column(
    df,
    ["participant_id", "clinic_id"],
    "participant identifier"
)

assessment_col = first_existing_column(
    df,
    ["assessment_date", "timestamp", "date"],
    "assessment date"
)

if "age" not in df.columns:
    raise KeyError("Base dataset does not contain 'age'.")

df = df.copy()
df["_assessment_date_norm"] = normalize_date(df[assessment_col])

# ============================================================
# LOAD WEEKLY FEATURES AND EXTRACT GAD-7
# ============================================================

weekly = pd.read_parquet(WEEKLY_FEATURES).copy()

weekly_participant_col = first_existing_column(
    weekly,
    ["participant_id", "clinic_id"],
    "weekly participant identifier"
)

weekly_date_col = first_existing_column(
    weekly,
    ["assessment_date", "timestamp", "date"],
    "weekly assessment date"
)

weekly_gad_col = first_existing_column(
    weekly,
    ["anxiety_score", "gad7_total", "gad_score"],
    "GAD-7 column"
)

weekly["_assessment_date_norm"] = normalize_date(weekly[weekly_date_col])

gad = weekly[
    [weekly_participant_col, "_assessment_date_norm", weekly_gad_col]
].copy()

gad = gad.rename(
    columns={
        weekly_participant_col: participant_col,
        weekly_gad_col: "gad7_total"
    }
)

# In case the same participant/date appears more than once, keep one value.
gad = (
    gad.dropna(subset=[participant_col, "_assessment_date_norm"])
       .groupby([participant_col, "_assessment_date_norm"], as_index=False)["gad7_total"]
       .mean()
)

# ============================================================
# MERGE
# ============================================================

merged = df.merge(
    gad,
    on=[participant_col, "_assessment_date_norm"],
    how="left",
    validate="m:1"
)

# Save merged analysis table
merged.to_csv(OUT_CSV, index=False)

# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("HOME TIME + AGE + GAD-7")
print("=" * 100)
print("Base dataset:", base_path)
print("Rows in base dataset:", len(df))
print("Participants in base dataset:", df[participant_col].nunique())
print()
print("Weekly features:", WEEKLY_FEATURES)
print("Rows in weekly features:", len(weekly))
print("Participants in weekly features:", weekly[weekly_participant_col].nunique())
print()
print("Merged rows:", len(merged))
print("Merged participants:", merged[participant_col].nunique())
print("Rows with age:", merged["age"].notna().sum())
print("Rows with GAD-7:", merged["gad7_total"].notna().sum())
print("Participants with any GAD-7:", merged.loc[merged["gad7_total"].notna(), participant_col].nunique())

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

panel_counts = []

# Top row: age
panel_counts.append({
    "panel": "A",
    "predictor": "age",
    "outcome": "home_14d_all_pct",
    "n": add_scatter_with_fit(
        axes[0, 0], merged, "age", "home_14d_all_pct",
        "Entire 14-day period", "Age at enrollment", "(A)"
    )
})
panel_counts.append({
    "panel": "B",
    "predictor": "age",
    "outcome": "home_14d_weekday_matched_pct",
    "n": add_scatter_with_fit(
        axes[0, 1], merged, "age", "home_14d_weekday_matched_pct",
        "Weekdays only", "Age at enrollment", "(B)"
    )
})
panel_counts.append({
    "panel": "C",
    "predictor": "age",
    "outcome": "home_14d_weekend_pct",
    "n": add_scatter_with_fit(
        axes[0, 2], merged, "age", "home_14d_weekend_pct",
        "Weekends only", "Age at enrollment", "(C)"
    )
})

# Bottom row: GAD-7
panel_counts.append({
    "panel": "D",
    "predictor": "gad7_total",
    "outcome": "home_14d_all_pct",
    "n": add_scatter_with_fit(
        axes[1, 0], merged, "gad7_total", "home_14d_all_pct",
        "Entire 14-day period", "GAD-7 total score", "(D)"
    )
})
panel_counts.append({
    "panel": "E",
    "predictor": "gad7_total",
    "outcome": "home_14d_weekday_matched_pct",
    "n": add_scatter_with_fit(
        axes[1, 1], merged, "gad7_total", "home_14d_weekday_matched_pct",
        "Weekdays only", "GAD-7 total score", "(E)"
    )
})
panel_counts.append({
    "panel": "F",
    "predictor": "gad7_total",
    "outcome": "home_14d_weekend_pct",
    "n": add_scatter_with_fit(
        axes[1, 2], merged, "gad7_total", "home_14d_weekend_pct",
        "Weekends only", "GAD-7 total score", "(F)"
    )
})

fig.suptitle(
    "Home time in relation to age and anxiety symptom severity",
    fontsize=20,
    y=1.02
)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

pd.DataFrame(panel_counts).to_csv(OUT_SUMMARY, index=False)

print()
print("Saved:")
print(" ", OUT_PNG)
print(" ", OUT_PDF)
print(" ", OUT_CSV)
print(" ", OUT_SUMMARY)
