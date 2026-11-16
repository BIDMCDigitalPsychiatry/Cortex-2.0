from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
import matplotlib.pyplot as plt

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

OUT.mkdir(parents=True, exist_ok=True)

STATS_OUT = (
    OUT
    / "weekend_home_functioning_elevated_depression_deep_stats.csv"
)

DATA_OUT = (
    OUT
    / "weekend_home_functioning_elevated_depression_deep_data.csv"
)

FIG_OUT = (
    OUT
    / "weekend_home_functioning_elevated_depression_deep.png"
)

# ============================================================
# LOAD WEEKEND DATA
# ============================================================

df = pd.read_csv(
    WEEKEND_FILE,
    low_memory=False
)

# ============================================================
# PHQ STATUS
# ============================================================

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

# Elevated depression subgroup
x = df[
    df["mean_phq9"] >= 10
].copy()

x = x.dropna(
    subset=[
        "weekend_home_pct",
        "following_week_functioning",
    ]
)

print()
print("=" * 100)
print("ELEVATED DEPRESSION SUBGROUP")
print("=" * 100)
print("Windows:", len(x))
print(
    "Participants:",
    x["participant_id"].nunique()
)
print(
    "Median windows/participant:",
    x.groupby("participant_id").size().median()
)

# ============================================================
# PARTICIPANT-SPECIFIC DECOMPOSITION
# ============================================================

x["person_mean_weekend_home"] = (
    x.groupby("participant_id")["weekend_home_pct"]
    .transform("mean")
)

x["weekend_home_within"] = (
    x["weekend_home_pct"]
    - x["person_mean_weekend_home"]
)

x["weekend_home_within_10"] = (
    x["weekend_home_within"] / 10
)

x["weekend_home_between_10"] = (
    x["person_mean_weekend_home"] / 10
)

# Overall continuous predictor
x["weekend_home_10"] = (
    x["weekend_home_pct"] / 10
)

# Centered predictor for nonlinear model
center = x["weekend_home_pct"].mean()

x["home_centered_10"] = (
    (x["weekend_home_pct"] - center) / 10
)

x["home_centered_10_sq"] = (
    x["home_centered_10"] ** 2
)

x.to_csv(
    DATA_OUT,
    index=False
)

rows = []

# ============================================================
# 1. CONTINUOUS GEE
# ============================================================

m1 = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "weekend_home_10 + study_time_weeks"
    ),
    groups="participant_id",
    data=x,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

beta = float(
    m1.params["weekend_home_10"]
)

se = float(
    m1.bse["weekend_home_10"]
)

p = float(
    m1.pvalues["weekend_home_10"]
)

rows.append(
    {
        "analysis": "continuous",
        "term": "weekend_home_10",
        "beta": beta,
        "ci_low": beta - 1.96*se,
        "ci_high": beta + 1.96*se,
        "p_value": p,
        "n_windows": len(x),
        "n_participants": x["participant_id"].nunique(),
    }
)

# ============================================================
# 2. WITHIN-PERSON + BETWEEN-PERSON
# ============================================================

m2 = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "weekend_home_within_10 + "
        "weekend_home_between_10 + "
        "study_time_weeks"
    ),
    groups="participant_id",
    data=x,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

for term in [
    "weekend_home_within_10",
    "weekend_home_between_10",
]:

    beta = float(
        m2.params[term]
    )

    se = float(
        m2.bse[term]
    )

    p = float(
        m2.pvalues[term]
    )

    rows.append(
        {
            "analysis": "within_between",
            "term": term,
            "beta": beta,
            "ci_low": beta - 1.96*se,
            "ci_high": beta + 1.96*se,
            "p_value": p,
            "n_windows": len(x),
            "n_participants": x["participant_id"].nunique(),
        }
    )

# ============================================================
# 3. NONLINEAR / QUADRATIC GEE
#
# Tests whether the relationship is curved rather than forcing
# a straight line.
# ============================================================

m3 = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "home_centered_10 + "
        "home_centered_10_sq + "
        "study_time_weeks"
    ),
    groups="participant_id",
    data=x,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

for term in [
    "home_centered_10",
    "home_centered_10_sq",
]:

    beta = float(
        m3.params[term]
    )

    se = float(
        m3.bse[term]
    )

    p = float(
        m3.pvalues[term]
    )

    rows.append(
        {
            "analysis": "quadratic",
            "term": term,
            "beta": beta,
            "ci_low": beta - 1.96*se,
            "ci_high": beta + 1.96*se,
            "p_value": p,
            "n_windows": len(x),
            "n_participants": x["participant_id"].nunique(),
        }
    )

# Joint test: linear + quadratic terms
try:
    joint = m3.wald_test(
        "home_centered_10 = 0, home_centered_10_sq = 0",
        scalar=True
    )

    joint_p = float(
        joint.pvalue
    )

except Exception:
    joint_p = np.nan

rows.append(
    {
        "analysis": "quadratic_joint",
        "term": "linear_plus_quadratic",
        "beta": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": joint_p,
        "n_windows": len(x),
        "n_participants": x["participant_id"].nunique(),
    }
)

results = pd.DataFrame(
    rows
)

results.to_csv(
    STATS_OUT,
    index=False
)

# ============================================================
# PRINT
# ============================================================

print()
print("=" * 100)
print("RESULTS")
print("=" * 100)

print(
    results.to_string(
        index=False,
        float_format=lambda z: f"{z:.4f}"
    )
)

print()
print("Interpretation:")
print(
    "continuous beta = change in following-week functioning "
    "per +10 percentage points weekend home time"
)

print(
    "within-person beta = effect of spending +10pp more time "
    "at home than that person's own usual weekend"
)

print(
    "between-person beta = difference associated with having "
    "a 10pp higher typical weekend home-time level"
)

# ============================================================
# FIGURE
#
# Panel A: continuous relationship using grouped deciles for
# visualization only.
#
# Panel B: within-person deviation bins.
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.6)
)

# ------------------------------------------------------------
# A. Overall relationship
# ------------------------------------------------------------

plot_a = x.copy()

try:
    plot_a["home_bin"] = pd.qcut(
        plot_a["weekend_home_pct"],
        q=5,
        duplicates="drop"
    )

    agg_a = (
        plot_a.groupby(
            "home_bin",
            observed=True
        )
        .agg(
            home_pct=("weekend_home_pct", "mean"),
            functioning=("following_week_functioning", "mean"),
            sd=("following_week_functioning", "std"),
            n=("following_week_functioning", "size"),
        )
        .reset_index()
    )

    agg_a["se"] = (
        agg_a["sd"] / np.sqrt(agg_a["n"])
    )

    axes[0].errorbar(
        agg_a["home_pct"],
        agg_a["functioning"],
        yerr=1.96 * agg_a["se"],
        marker="o",
        linewidth=2,
        capsize=4,
    )

except Exception:
    pass

axes[0].set_xlabel(
    "Weekend home time (%)"
)

axes[0].set_ylabel(
    "Following-week functioning"
)

axes[0].set_title(
    "A. Overall relationship\n"
    f"continuous GEE p={m1.pvalues['weekend_home_10']:.3f}"
)

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# ------------------------------------------------------------
# B. Within-person deviations
# ------------------------------------------------------------

plot_b = x.copy()

try:
    plot_b["within_bin"] = pd.qcut(
        plot_b["weekend_home_within"],
        q=5,
        duplicates="drop"
    )

    agg_b = (
        plot_b.groupby(
            "within_bin",
            observed=True
        )
        .agg(
            home_deviation=("weekend_home_within", "mean"),
            functioning=("following_week_functioning", "mean"),
            sd=("following_week_functioning", "std"),
            n=("following_week_functioning", "size"),
        )
        .reset_index()
    )

    agg_b["se"] = (
        agg_b["sd"] / np.sqrt(agg_b["n"])
    )

    axes[1].errorbar(
        agg_b["home_deviation"],
        agg_b["functioning"],
        yerr=1.96 * agg_b["se"],
        marker="o",
        linewidth=2,
        capsize=4,
    )

except Exception:
    pass

axes[1].axvline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.5"
)

axes[1].set_xlabel(
    "Weekend home time relative to personal usual level (pp)"
)

axes[1].set_ylabel(
    "Following-week functioning"
)

axes[1].set_title(
    "B. Within-person relationship\n"
    f"GEE p={m2.pvalues['weekend_home_within_10']:.3f}"
)

axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.suptitle(
    "Weekend home stay and following-week functioning\n"
    "among participants with elevated depression",
    fontsize=13
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

print()
print("Saved:")
print(STATS_OUT)
print(DATA_OUT)
print(FIG_OUT)
