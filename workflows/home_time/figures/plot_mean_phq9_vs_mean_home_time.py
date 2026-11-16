from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

# =============================================================================
# PATHS
# =============================================================================

DATA_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_clinical_weekly_features_with_home.parquet"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "paper_replication"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_OUT = (
    OUT_DIR
    / "mean_phq9_vs_mean_home_time.png"
)

PDF_OUT = (
    OUT_DIR
    / "mean_phq9_vs_mean_home_time.pdf"
)

DATA_OUT = (
    OUT_DIR
    / "mean_phq9_vs_mean_home_time_participant_data.csv"
)

STATS_OUT = (
    OUT_DIR
    / "mean_phq9_vs_mean_home_time_stats.csv"
)

# =============================================================================
# LOAD WEEKLY CLINICAL + HOME-TIME DATA
# =============================================================================

df = pd.read_parquet(
    DATA_PATH
)

print("=" * 100)
print("MEAN PHQ-9 vs MEAN HOME TIME")
print("=" * 100)

print("\nInput rows:", len(df))
print(
    "Input participants:",
    df["participant_id"].nunique(),
)

required = {
    "participant_id",
    "depression_score",
    "time_at_home_pct",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )

# Existing clinical workflow:
# depression_score = complete PHQ-9 symptom total, 0–27.
df["phq9"] = pd.to_numeric(
    df["depression_score"],
    errors="coerce",
)

df["home_time_pct"] = pd.to_numeric(
    df["time_at_home_pct"],
    errors="coerce",
)

# =============================================================================
# IMPORTANT:
#
# Restrict FIRST to weekly observations where both PHQ-9 and home time exist.
#
# This ensures the participant's mean PHQ-9 and mean home time are calculated
# from the same set of clinical weeks.
# =============================================================================

paired = df.dropna(
    subset=[
        "participant_id",
        "phq9",
        "home_time_pct",
    ]
).copy()

# PHQ-9 validity check
paired = paired[
    paired["phq9"].between(
        0,
        27,
    )
].copy()

# Home-time validity check
paired = paired[
    paired["home_time_pct"].between(
        0,
        100,
    )
].copy()

print("\nWeekly paired observations:", len(paired))
print(
    "Participants with >=1 paired week:",
    paired["participant_id"].nunique(),
)

# =============================================================================
# PARTICIPANT-LEVEL AVERAGES
# =============================================================================

participant = (
    paired.groupby(
        "participant_id",
        as_index=False,
    )
    .agg(
        mean_phq9=(
            "phq9",
            "mean",
        ),
        mean_home_time_pct=(
            "home_time_pct",
            "mean",
        ),
        median_phq9=(
            "phq9",
            "median",
        ),
        median_home_time_pct=(
            "home_time_pct",
            "median",
        ),
        n_paired_weeks=(
            "phq9",
            "size",
        ),
    )
)

participant.to_csv(
    DATA_OUT,
    index=False,
)

print("\nParticipant-level analysis:")
print("N participants:", len(participant))

print("\nPaired weeks per participant:")
print(
    participant["n_paired_weeks"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
    .to_string()
)

print("\nMean PHQ-9 across participants:")
print(
    participant["mean_phq9"]
    .describe()
    .to_string()
)

print("\nMean home time across participants:")
print(
    participant["mean_home_time_pct"]
    .describe()
    .to_string()
)

# =============================================================================
# CORRELATIONS
# =============================================================================

x = participant[
    "mean_phq9"
].to_numpy()

y = participant[
    "mean_home_time_pct"
].to_numpy()

pearson_r, pearson_p = pearsonr(
    x,
    y,
)

spearman_rho, spearman_p = spearmanr(
    x,
    y,
)

reg = linregress(
    x,
    y,
)

print()
print("=" * 100)
print("CORRELATION RESULTS")
print("=" * 100)

print(
    f"\nPearson r = {pearson_r:.4f}"
)
print(
    f"Pearson p = {pearson_p:.6g}"
)

print(
    f"\nSpearman rho = {spearman_rho:.4f}"
)
print(
    f"Spearman p = {spearman_p:.6g}"
)

print(
    f"\nLinear slope = {reg.slope:.4f} "
    "percentage points home time per +1 PHQ-9 point"
)

print(
    f"95% visual interpretation should remain descriptive; "
    f"this is a participant-level between-person analysis."
)

# =============================================================================
# SAVE STATS
# =============================================================================

stats = pd.DataFrame(
    [
        {
            "n_participants":
                len(participant),

            "n_weekly_pairs":
                len(paired),

            "median_paired_weeks_per_participant":
                participant[
                    "n_paired_weeks"
                ].median(),

            "mean_phq9":
                participant[
                    "mean_phq9"
                ].mean(),

            "mean_home_time_pct":
                participant[
                    "mean_home_time_pct"
                ].mean(),

            "pearson_r":
                pearson_r,

            "pearson_p":
                pearson_p,

            "spearman_rho":
                spearman_rho,

            "spearman_p":
                spearman_p,

            "linear_slope_home_pct_per_phq9":
                reg.slope,

            "linear_intercept":
                reg.intercept,
        }
    ]
)

stats.to_csv(
    STATS_OUT,
    index=False,
)

# =============================================================================
# FIGURE
# =============================================================================

plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 11

fig, ax = plt.subplots(
    figsize=(7.3, 6.0)
)

ax.scatter(
    participant["mean_phq9"],
    participant["mean_home_time_pct"],
    s=38,
    alpha=0.65,
)

# Regression line
x_line = np.linspace(
    participant[
        "mean_phq9"
    ].min(),
    participant[
        "mean_phq9"
    ].max(),
    200,
)

y_line = (
    reg.intercept
    + reg.slope * x_line
)

ax.plot(
    x_line,
    y_line,
    linewidth=1.6,
)

ax.set_xlabel(
    "Mean PHQ-9 score across all paired weeks"
)

ax.set_ylabel(
    "Mean home time across all paired weeks (%)"
)

ax.set_title(
    "Average depression severity and home time across participants"
)

ax.set_ylim(
    0,
    100,
)

# Statistics annotation
text = (
    f"N = {len(participant)} participants\n"
    f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}\n"
    f"Spearman ρ = {spearman_rho:.3f}, p = {spearman_p:.3g}"
)

ax.text(
    0.03,
    0.97,
    text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

fig.tight_layout()

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(PNG_OUT)
print(PDF_OUT)
print(DATA_OUT)
print(STATS_OUT)
