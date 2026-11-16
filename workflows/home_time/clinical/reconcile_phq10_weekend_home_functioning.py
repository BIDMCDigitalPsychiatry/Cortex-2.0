from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

warnings.filterwarnings("ignore")

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)

WEEKEND_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "temporal_functioning"
    / "weekend_home_following_week_functioning.csv"
)

CLINICAL_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "paper_replication"
    / "home_time_age_gad7_analysis_dataset.csv"
)

OUT = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "temporal_functioning"
)

FIG_OUT = OUT / "phq10_weekend_home_functioning_reconciliation.png"
DATA_OUT = OUT / "phq10_weekend_home_functioning_reconciliation_data.csv"
STATS_OUT = OUT / "phq10_weekend_home_functioning_reconciliation_stats.csv"

# ============================================================
# LOAD + PHQ
# ============================================================

df = pd.read_csv(WEEKEND_FILE, low_memory=False)

clin = pd.read_csv(
    CLINICAL_FILE,
    low_memory=False
)

clin["phq9_total"] = pd.to_numeric(
    clin["phq9_total"],
    errors="coerce"
)

phq = (
    clin.groupby("participant_id")
    .agg(
        mean_phq9=("phq9_total", "mean"),
        n_phq=("phq9_total", "count"),
    )
    .reset_index()
)

df = df.merge(
    phq,
    on="participant_id",
    how="left"
)

# ============================================================
# EXACT PHQ >= 10 SAMPLE
# ============================================================

x = df[
    df["mean_phq9"] >= 10
].copy()

x = x.dropna(
    subset=[
        "participant_id",
        "weekend_home_pct",
        "following_week_functioning",
        "study_time_weeks",
    ]
)

x["weekend_home_10"] = (
    x["weekend_home_pct"] / 10
)

# IMPORTANT:
# Every analysis below uses this exact x dataframe.
x.to_csv(DATA_OUT, index=False)

print()
print("=" * 100)
print("EXACT PHQ-9 >= 10 ANALYSIS SAMPLE")
print("=" * 100)

print("Weekend windows:", len(x))
print("Participants:", x["participant_id"].nunique())
print(
    "Median windows per participant:",
    x.groupby("participant_id").size().median()
)

print(
    "Weekend home-time range:",
    f"{x['weekend_home_pct'].min():.1f}%",
    "to",
    f"{x['weekend_home_pct'].max():.1f}%"
)

# ============================================================
# 1. CONTINUOUS LINEAR GEE
# ============================================================

linear = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "weekend_home_10 + study_time_weeks"
    ),
    groups="participant_id",
    data=x,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

lin_beta = float(
    linear.params["weekend_home_10"]
)

lin_se = float(
    linear.bse["weekend_home_10"]
)

lin_p = float(
    linear.pvalues["weekend_home_10"]
)

# ============================================================
# 2. QUADRATIC / NONLINEAR GEE
# ============================================================

home_mean = x["weekend_home_pct"].mean()

x["home_c_10"] = (
    x["weekend_home_pct"] - home_mean
) / 10

x["home_c_10_sq"] = (
    x["home_c_10"] ** 2
)

quad = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "home_c_10 + home_c_10_sq + "
        "study_time_weeks"
    ),
    groups="participant_id",
    data=x,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

quad_linear_p = float(
    quad.pvalues["home_c_10"]
)

quad_curve_p = float(
    quad.pvalues["home_c_10_sq"]
)

try:
    quad_joint = quad.wald_test(
        "home_c_10 = 0, home_c_10_sq = 0",
        scalar=True
    )
    quad_joint_p = float(quad_joint.pvalue)
except Exception:
    quad_joint_p = np.nan

# ============================================================
# 3. EXACT MEDIAN SPLIT
# ============================================================

median_threshold = float(
    x["weekend_home_pct"].median()
)

x["high_home"] = (
    x["weekend_home_pct"]
    > median_threshold
).astype(int)

median_model = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "high_home + study_time_weeks"
    ),
    groups="participant_id",
    data=x,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

median_beta = float(
    median_model.params["high_home"]
)

median_se = float(
    median_model.bse["high_home"]
)

median_p = float(
    median_model.pvalues["high_home"]
)

group_summary = (
    x.groupby("high_home")
    .agg(
        mean_functioning=(
            "following_week_functioning",
            "mean"
        ),
        sd_functioning=(
            "following_week_functioning",
            "std"
        ),
        n_windows=(
            "following_week_functioning",
            "size"
        ),
        n_participants=(
            "participant_id",
            "nunique"
        ),
        mean_home_pct=(
            "weekend_home_pct",
            "mean"
        ),
        min_home_pct=(
            "weekend_home_pct",
            "min"
        ),
        max_home_pct=(
            "weekend_home_pct",
            "max"
        ),
    )
    .reset_index()
)

group_summary["se"] = (
    group_summary["sd_functioning"]
    / np.sqrt(group_summary["n_windows"])
)

group_summary["ci95"] = (
    1.96 * group_summary["se"]
)

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 100)
print("RECONCILIATION")
print("=" * 100)

print()
print("1. CONTINUOUS GEE")
print(
    "beta per +10pp weekend home stay:",
    f"{lin_beta:.4f}"
)
print(
    "95% CI:",
    f"{lin_beta - 1.96*lin_se:.4f}",
    "to",
    f"{lin_beta + 1.96*lin_se:.4f}"
)
print("p =", f"{lin_p:.4f}")

print()
print("2. NONLINEAR GEE")
print(
    "linear component p =",
    f"{quad_linear_p:.4f}"
)
print(
    "quadratic component p =",
    f"{quad_curve_p:.4f}"
)
print(
    "joint nonlinear model p =",
    f"{quad_joint_p:.4f}"
)

print()
print("3. MEDIAN SPLIT")
print(
    "Actual threshold:",
    f"{median_threshold:.2f}% weekend home stay"
)
print(
    "High-vs-low beta:",
    f"{median_beta:.4f}"
)
print(
    "95% CI:",
    f"{median_beta - 1.96*median_se:.4f}",
    "to",
    f"{median_beta + 1.96*median_se:.4f}"
)
print("p =", f"{median_p:.4f}")

print()
print(group_summary.to_string(index=False))

# ============================================================
# PREDICTION CURVES
# ============================================================

grid = np.linspace(
    np.percentile(
        x["weekend_home_pct"], 2.5
    ),
    np.percentile(
        x["weekend_home_pct"], 97.5
    ),
    200
)

pred_linear = pd.DataFrame(
    {
        "weekend_home_10": grid / 10,
        "study_time_weeks":
            x["study_time_weeks"].median(),
    }
)

linear_pred = linear.predict(
    pred_linear
)

pred_quad = pd.DataFrame(
    {
        "home_c_10":
            (grid - home_mean) / 10,
        "study_time_weeks":
            x["study_time_weeks"].median(),
    }
)

pred_quad["home_c_10_sq"] = (
    pred_quad["home_c_10"] ** 2
)

quad_pred = quad.predict(
    pred_quad
)

# ============================================================
# PARTICIPANT-LEVEL MEANS FOR VISUAL OVERLAY
#
# This prevents participants with many weekends from visually
# dominating the scatter.
# ============================================================

person = (
    x.groupby("participant_id")
    .agg(
        mean_weekend_home_pct=(
            "weekend_home_pct",
            "mean"
        ),
        mean_following_functioning=(
            "following_week_functioning",
            "mean"
        ),
        n_windows=(
            "following_week_functioning",
            "size"
        ),
    )
    .reset_index()
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.8)
)

# ------------------------------------------------------------
# A. RAW CONTINUOUS RELATIONSHIP
# ------------------------------------------------------------

ax = axes[0]

ax.scatter(
    person["mean_weekend_home_pct"],
    person["mean_following_functioning"],
    s=38,
    alpha=0.65,
)

ax.plot(
    grid,
    linear_pred,
    linewidth=2.2,
)

ax.set_xlabel(
    "Mean weekend home stay (%)"
)

ax.set_ylabel(
    "Mean following-week functioning"
)

ax.set_title(
    "A. Continuous relationship\n"
    f"GEE p = {lin_p:.3f}"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# B. LINEAR VS NONLINEAR
# ------------------------------------------------------------

ax = axes[1]

ax.scatter(
    person["mean_weekend_home_pct"],
    person["mean_following_functioning"],
    s=30,
    alpha=0.40,
)

ax.plot(
    grid,
    linear_pred,
    linewidth=2,
    label="Linear"
)

ax.plot(
    grid,
    quad_pred,
    linewidth=2,
    linestyle="--",
    label="Quadratic"
)

ax.set_xlabel(
    "Mean weekend home stay (%)"
)

ax.set_ylabel(
    "Mean following-week functioning"
)

ax.set_title(
    "B. Shape of relationship\n"
    f"quadratic p = {quad_curve_p:.3f}"
)

ax.legend(
    frameon=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# C. EXACT MEDIAN SPLIT
# ------------------------------------------------------------

ax = axes[2]

pos = np.arange(2)

ax.bar(
    pos,
    group_summary["mean_functioning"],
    yerr=group_summary["ci95"],
    capsize=4,
    edgecolor="black",
    linewidth=0.7,
)

labels = [
    (
        f"≤ {median_threshold:.1f}%\n"
        f"n={int(group_summary.iloc[0]['n_participants'])}"
    ),
    (
        f"> {median_threshold:.1f}%\n"
        f"n={int(group_summary.iloc[1]['n_participants'])}"
    ),
]

ax.set_xticks(pos)
ax.set_xticklabels(labels)

for i, r in group_summary.iterrows():

    ax.text(
        i,
        r["mean_functioning"]
        + r["ci95"] + 0.04,
        f'{r["mean_functioning"]:.2f}',
        ha="center",
    )

ax.set_ylabel(
    "Following-week functioning"
)

ax.set_xlabel(
    "Weekend home stay"
)

ax.set_title(
    "C. Exact median comparison\n"
    f"threshold = {median_threshold:.1f}%, p = {median_p:.3f}"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Reconciling weekend home stay and subsequent functioning\n"
    "within the identical PHQ-9 ≥10 sample",
    fontsize=14
)

fig.tight_layout(
    rect=[0, 0, 1, 0.90]
)

fig.savefig(
    FIG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

# ============================================================
# SAVE STATS
# ============================================================

stats = pd.DataFrame(
    [
        {
            "analysis": "continuous_GEE",
            "threshold": np.nan,
            "beta": lin_beta,
            "ci_low": lin_beta - 1.96*lin_se,
            "ci_high": lin_beta + 1.96*lin_se,
            "p_value": lin_p,
        },
        {
            "analysis": "quadratic_component",
            "threshold": np.nan,
            "beta": quad.params["home_c_10_sq"],
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": quad_curve_p,
        },
        {
            "analysis": "median_split",
            "threshold": median_threshold,
            "beta": median_beta,
            "ci_low": median_beta - 1.96*median_se,
            "ci_high": median_beta + 1.96*median_se,
            "p_value": median_p,
        },
    ]
)

stats.to_csv(
    STATS_OUT,
    index=False
)

print()
print("Saved:")
print(FIG_OUT)
print(DATA_OUT)
print(STATS_OUT)
