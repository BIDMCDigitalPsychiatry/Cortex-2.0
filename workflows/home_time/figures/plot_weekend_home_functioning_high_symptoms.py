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

PNG = OUT / "weekend_home_functioning_high_symptoms.png"
CSV = OUT / "weekend_home_functioning_high_symptoms_stats.csv"

# ============================================================
# LOAD WEEKEND -> FOLLOWING WEEK DATA
# ============================================================

df = pd.read_csv(
    WEEKEND_FILE,
    low_memory=False
)

# ============================================================
# PARTICIPANT-LEVEL PHQ/GAD STATUS
# ============================================================

clin = pd.read_csv(
    CLINICAL_FILE,
    low_memory=False
)

clin["phq9_total"] = pd.to_numeric(
    clin["phq9_total"],
    errors="coerce"
)

clin["gad7_total"] = pd.to_numeric(
    clin["gad7_total"],
    errors="coerce"
)

symptoms = (
    clin.groupby("participant_id")
    .agg(
        mean_phq9=("phq9_total", "mean"),
        mean_gad7=("gad7_total", "mean"),
        n_phq=("phq9_total", "count"),
        n_gad=("gad7_total", "count"),
    )
    .reset_index()
)

symptoms["high_depression"] = (
    symptoms["mean_phq9"] >= 10
)

symptoms["high_anxiety"] = (
    symptoms["mean_gad7"] >= 10
)

symptoms["high_either"] = (
    symptoms["high_depression"]
    | symptoms["high_anxiety"]
)

df = df.merge(
    symptoms,
    on="participant_id",
    how="left"
)

print()
print("=" * 100)
print("HIGH-SYMPTOM SUBGROUPS")
print("=" * 100)

for col in [
    "high_depression",
    "high_anxiety",
    "high_either",
]:
    sub = df[df[col] == True]

    print(
        f"{col:20s}",
        "participants =",
        sub["participant_id"].nunique(),
        "| weekend windows =",
        len(sub)
    )


# ============================================================
# FIGURE
# ============================================================

SPLITS = [
    (2, "Median split"),
    (3, "Tertiles"),
    (4, "Quartiles"),
    (5, "Quintiles"),
]

SUBGROUPS = [
    (
        "high_depression",
        "Elevated depression (mean PHQ-9 ≥10)"
    ),
    (
        "high_anxiety",
        "Elevated anxiety (mean GAD-7 ≥10)"
    ),
]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8.5

fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 7.5)
)

stats_rows = []

for row_i, (group_col, row_title) in enumerate(
    SUBGROUPS
):

    base = df[
        df[group_col] == True
    ].copy()

    for col_i, (q, split_name) in enumerate(
        SPLITS
    ):

        ax = axes[row_i, col_i]

        x = base[
            [
                "participant_id",
                "weekend_home_pct",
                "following_week_functioning",
                "study_time_weeks",
            ]
        ].dropna().copy()

        if len(x) < q * 4:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        try:
            x["home_group"] = pd.qcut(
                x["weekend_home_pct"],
                q=q,
                labels=False,
                duplicates="drop"
            )
        except Exception:
            ax.set_axis_off()
            continue

        if x["home_group"].nunique() != q:
            ax.set_axis_off()
            continue

        summary = (
            x.groupby("home_group")
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

        # Clustered GEE trend test
        x["group_numeric"] = (
            x["home_group"].astype(float)
        )

        trend_p = np.nan
        beta = np.nan

        try:
            gee = smf.gee(
                formula=(
                    "following_week_functioning ~ "
                    "group_numeric + study_time_weeks"
                ),
                groups="participant_id",
                data=x,
                family=sm.families.Gaussian(),
                cov_struct=Exchangeable(),
            ).fit()

            beta = float(
                gee.params["group_numeric"]
            )

            trend_p = float(
                gee.pvalues["group_numeric"]
            )

        except Exception:
            pass

        stats_rows.append(
            {
                "subgroup": group_col,
                "split": split_name,
                "n_windows": len(x),
                "n_participants": x[
                    "participant_id"
                ].nunique(),
                "trend_beta": beta,
                "trend_p": trend_p,
            }
        )

        pos = np.arange(q)

        ax.bar(
            pos,
            summary["mean_functioning"],
            yerr=summary["ci95"],
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

        for i, r in summary.iterrows():

            ax.text(
                i,
                r["mean_functioning"]
                + r["ci95"],
                f'{r["mean_functioning"]:.2f}',
                ha="center",
                va="bottom",
                fontsize=7,
            )

        if q == 2:
            labels = ["Low", "High"]
        else:
            labels = [
                f"Q{i+1}"
                for i in range(q)
            ]

        ax.set_xticks(pos)
        ax.set_xticklabels(labels)

        ax.set_title(
            f"{split_name}\ntrend p = {trend_p:.4f}",
            fontsize=8.5
        )

        if col_i == 0:
            ax.set_ylabel(
                f"{row_title}\n\n"
                "Following-week functioning"
            )

        if row_i == 1:
            ax.set_xlabel(
                "Weekend home time"
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


fig.suptitle(
    "Weekend home stay and following-week functioning "
    "in participants with elevated symptoms",
    fontsize=13
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

pd.DataFrame(
    stats_rows
).to_csv(
    CSV,
    index=False
)

print()
print("=" * 100)
print("TREND RESULTS")
print("=" * 100)

print(
    pd.DataFrame(
        stats_rows
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Saved:")
print(PNG)
print(CSV)
