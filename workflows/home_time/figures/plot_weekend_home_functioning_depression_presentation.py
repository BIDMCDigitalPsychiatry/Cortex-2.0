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

PNG = OUT / "weekend_home_functioning_phq10_presentation.png"
CSV = OUT / "weekend_home_functioning_phq10_presentation_stats.csv"

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    WEEKEND_FILE,
    low_memory=False
)

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
# ELEVATED DEPRESSION SUBGROUP
# ============================================================

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
print("=" * 90)
print("PHQ-9 >= 10 SUBGROUP")
print("=" * 90)
print("Participants:", x["participant_id"].nunique())
print("Weekend windows:", len(x))

# ============================================================
# HELPER
# ============================================================

def summarize_groups(data, q):

    z = data.copy()

    z["group"] = pd.qcut(
        z["weekend_home_pct"],
        q=q,
        labels=False,
        duplicates="drop"
    )

    if z["group"].nunique() != q:
        raise RuntimeError(
            f"Could not form {q} unique groups."
        )

    summary = (
        z.groupby("group")
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
        )
        .reset_index()
    )

    summary["se"] = (
        summary["sd_functioning"]
        / np.sqrt(summary["n_windows"])
    )

    summary["ci95"] = (
        1.96 * summary["se"]
    )

    z["group_numeric"] = (
        z["group"].astype(float)
    )

    gee = smf.gee(
        formula=(
            "following_week_functioning ~ "
            "group_numeric + study_time_weeks"
        ),
        groups="participant_id",
        data=z,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()

    trend_p = float(
        gee.pvalues["group_numeric"]
    )

    trend_beta = float(
        gee.params["group_numeric"]
    )

    return z, summary, trend_p, trend_beta


# ============================================================
# MEDIAN + TERTILES
# ============================================================

median_data, median_summary, median_p, median_beta = (
    summarize_groups(x, 2)
)

tertile_data, tertile_summary, tertile_p, tertile_beta = (
    summarize_groups(x, 3)
)

# ============================================================
# SAVE STATS
# ============================================================

stats = []

for label, summary, p, beta in [
    (
        "Median split",
        median_summary,
        median_p,
        median_beta,
    ),
    (
        "Tertiles",
        tertile_summary,
        tertile_p,
        tertile_beta,
    ),
]:

    for _, r in summary.iterrows():

        stats.append(
            {
                "analysis": label,
                "group": int(r["group"]) + 1,
                "mean_functioning": r["mean_functioning"],
                "ci95": r["ci95"],
                "n_windows": int(r["n_windows"]),
                "n_participants": int(r["n_participants"]),
                "mean_weekend_home_pct": r["mean_home_pct"],
                "trend_beta": beta,
                "trend_p": p,
            }
        )

pd.DataFrame(
    stats
).to_csv(
    CSV,
    index=False
)

# ============================================================
# PRESENTATION FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.8)
)

# ------------------------------------------------------------
# A. MEDIAN SPLIT
# ------------------------------------------------------------

ax = axes[0]

pos = np.arange(2)

ax.bar(
    pos,
    median_summary["mean_functioning"],
    yerr=median_summary["ci95"],
    capsize=4,
    edgecolor="black",
    linewidth=0.7,
)

ax.set_xticks(pos)

ax.set_xticklabels([
    "Lower\nweekend home stay",
    "Higher\nweekend home stay",
])

for i, r in median_summary.iterrows():

    ax.text(
        i,
        r["mean_functioning"] + r["ci95"] + 0.05,
        f'{r["mean_functioning"]:.2f}',
        ha="center",
        fontsize=11,
    )

ax.set_ylabel(
    "Following-week functioning"
)

ax.set_title(
    "A. Median split\n"
    f"p = {median_p:.3f}"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# B. TERTILES
# ------------------------------------------------------------

ax = axes[1]

pos = np.arange(3)

ax.bar(
    pos,
    tertile_summary["mean_functioning"],
    yerr=tertile_summary["ci95"],
    capsize=4,
    edgecolor="black",
    linewidth=0.7,
)

ax.set_xticks(pos)

ax.set_xticklabels([
    "Low",
    "Medium",
    "High",
])

for i, r in tertile_summary.iterrows():

    ax.text(
        i,
        r["mean_functioning"] + r["ci95"] + 0.05,
        f'{r["mean_functioning"]:.2f}',
        ha="center",
        fontsize=11,
    )

ax.set_title(
    "B. Tertiles\n"
    f"trend p = {tertile_p:.3f}"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# COMMON TITLE
# ------------------------------------------------------------

fig.suptitle(
    "Weekend home stay and following-week functioning\n"
    "among participants with elevated depression (PHQ-9 ≥10)",
    fontsize=15
)

fig.text(
    0.5,
    0.01,
    (
        "Higher functioning scores indicate better functioning. "
        "Bars show group means with 95% confidence intervals."
    ),
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.05, 1, 0.88]
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("Median split p:", f"{median_p:.4f}")
print("Tertile trend p:", f"{tertile_p:.4f}")

print()
print("Saved:")
print(PNG)
print(CSV)
