from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "clinical_comparisons/home_time_phq9_analysis.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG_OUT = OUT_DIR / "home_time_phq9_low_vs_high.png"
CSV_OUT = OUT_DIR / "home_time_phq9_low_vs_high_stats.csv"


# ============================================================
# CHANGE THESE ONLY IF YOUR COLUMN NAMES DIFFER
# ============================================================

PHQ_COL = "phq9_total"

HOME_COLUMNS = {
    "Entire 14-day period": "home_time_pct",
    "Weekdays only": "weekday_home_time_pct",
    "Weekends only": "weekend_home_time_pct",
}


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_CSV,
    low_memory=False,
)

df[PHQ_COL] = pd.to_numeric(
    df[PHQ_COL],
    errors="coerce",
)


# ============================================================
# LOW / HIGH PHQ-9 FILTER
#
# Low:  0–4
# Mild: 5–9  --> EXCLUDED
# High: >=10
# ============================================================

df["phq_group"] = np.select(
    [
        df[PHQ_COL].between(0, 4, inclusive="both"),
        df[PHQ_COL] >= 10,
    ],
    [
        "Low (0–4)",
        "High (≥10)",
    ],
    default=np.nan,
)

analysis = df[
    df["phq_group"].notna()
].copy()

print("=" * 80)
print("PHQ-9 LOW VS HIGH SENSITIVITY ANALYSIS")
print("=" * 80)
print()
print(
    analysis["phq_group"]
    .value_counts()
    .to_string()
)
print()
print(
    "Excluded mild PHQ-9 (5–9):",
    int(
        df[PHQ_COL]
        .between(5, 9, inclusive="both")
        .sum()
    ),
)


# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12.8, 4.3),
    sharey=True,
)

stats = []


for ax, (title, home_col) in zip(
    axes,
    HOME_COLUMNS.items(),
):

    d = analysis[
        [
            PHQ_COL,
            home_col,
            "phq_group",
        ]
    ].copy()

    d[home_col] = pd.to_numeric(
        d[home_col],
        errors="coerce",
    )

    d = d.dropna(
        subset=[
            PHQ_COL,
            home_col,
        ]
    )

    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    ax.scatter(
        d[PHQ_COL],
        d[home_col],
        s=16,
        alpha=0.50,
        edgecolors="none",
    )

    # --------------------------------------------------------
    # Regression using ONLY low + high observations
    # --------------------------------------------------------

    if (
        len(d) >= 3
        and d[PHQ_COL].nunique() >= 2
    ):

        fit = linregress(
            d[PHQ_COL],
            d[home_col],
        )

        xx = np.linspace(
            d[PHQ_COL].min(),
            d[PHQ_COL].max(),
            200,
        )

        yy = (
            fit.intercept
            + fit.slope * xx
        )

        ax.plot(
            xx,
            yy,
            linewidth=1.8,
        )

        stats.append(
            {
                "period": title,
                "n": len(d),
                "n_low": int(
                    (
                        d["phq_group"]
                        == "Low (0–4)"
                    ).sum()
                ),
                "n_high": int(
                    (
                        d["phq_group"]
                        == "High (≥10)"
                    ).sum()
                ),
                "slope_home_pct_per_phq_point":
                    fit.slope,
                "intercept": fit.intercept,
                "r": fit.rvalue,
                "r_squared":
                    fit.rvalue ** 2,
                "p_value": fit.pvalue,
                "slope_se": fit.stderr,
            }
        )

        ax.text(
            0.04,
            0.96,
            (
                f"β = {fit.slope:.2f}%/point\n"
                f"r = {fit.rvalue:.2f}\n"
                f"p = {fit.pvalue:.3g}\n"
                f"n = {len(d)}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
        )

    # --------------------------------------------------------
    # Show excluded mild region
    # --------------------------------------------------------

    ax.axvspan(
        5,
        9,
        alpha=0.08,
    )

    ax.text(
        7,
        3,
        "Mild excluded",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90,
    )

    ax.set_title(
        title,
        fontsize=11,
    )

    ax.set_xlabel(
        "PHQ-9 total score"
    )

    ax.set_xlim(
        -0.5,
        max(
            27,
            analysis[PHQ_COL].max() + 1,
        ),
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


axes[0].set_ylabel(
    "Home time (%)"
)

fig.suptitle(
    "Home time vs PHQ-9: low and high symptom groups only",
    fontsize=13,
)

fig.text(
    0.5,
    0.01,
    "Low = PHQ-9 0–4; mild scores 5–9 excluded; high = PHQ-9 ≥10",
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[
        0,
        0.06,
        1,
        0.92,
    ]
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# SAVE STATISTICS
# ============================================================

stats_df = pd.DataFrame(
    stats
)

stats_df.to_csv(
    CSV_OUT,
    index=False,
)

print()
print(stats_df.to_string(index=False))
print()
print("Saved:")
print(PNG_OUT)
print(CSV_OUT)
