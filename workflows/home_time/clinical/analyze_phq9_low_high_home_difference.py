from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/paper_replication"
)

INPUT = ROOT / "paper_replication_analysis_dataset.csv"

OUT_CSV = ROOT / "phq9_low_high_home_time_difference_gee.csv"
OUT_PNG = ROOT / "phq9_low_high_home_time_difference_gee.png"
OUT_PDF = ROOT / "phq9_low_high_home_time_difference_gee.pdf"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT,
    low_memory=False,
)

df["phq9_total"] = pd.to_numeric(
    df["phq9_total"],
    errors="coerce",
)


# ============================================================
# DEFINE LOW / HIGH
#
# Low  = PHQ-9 0–4
# Mild = PHQ-9 5–9 excluded
# High = PHQ-9 >=10
# ============================================================

analysis = df[
    (
        df["phq9_total"].between(
            0,
            4,
            inclusive="both",
        )
    )
    |
    (
        df["phq9_total"] >= 10
    )
].copy()

analysis["phq_high"] = (
    analysis["phq9_total"] >= 10
).astype(int)


# ============================================================
# OUTCOMES
# ============================================================

outcomes = [
    (
        "home_14d_all_pct",
        "Entire 14-day period",
    ),
    (
        "home_14d_weekday_matched_pct",
        "Weekdays",
    ),
    (
        "home_14d_weekend_pct",
        "Weekends",
    ),
]


# ============================================================
# PARTICIPANT-CLUSTERED GEE
#
# beta for phq_high =
# High PHQ-9 minus Low PHQ-9 home-time percentage points
# ============================================================

rows = []

for outcome, label in outcomes:

    d = analysis[
        [
            "participant_id",
            "phq_high",
            outcome,
        ]
    ].copy()

    d[outcome] = pd.to_numeric(
        d[outcome],
        errors="coerce",
    )

    d = d.dropna()

    X = sm.add_constant(
        d[["phq_high"]]
    )

    model = sm.GEE(
        endog=d[outcome],
        exog=X,
        groups=d["participant_id"],
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )

    result = model.fit()

    beta = float(
        result.params["phq_high"]
    )

    se = float(
        result.bse["phq_high"]
    )

    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    p_value = float(
        result.pvalues["phq_high"]
    )

    low = d[
        d["phq_high"].eq(0)
    ][outcome]

    high = d[
        d["phq_high"].eq(1)
    ][outcome]

    rows.append(
        {
            "time_frame": label,

            "n_segments":
                len(d),

            "n_participants":
                d[
                    "participant_id"
                ].nunique(),

            "n_low_segments":
                len(low),

            "n_high_segments":
                len(high),

            "mean_low_home_pct":
                low.mean(),

            "mean_high_home_pct":
                high.mean(),

            "unadjusted_mean_difference":
                high.mean()
                - low.mean(),

            "gee_high_minus_low_pp":
                beta,

            "ci_low":
                ci_low,

            "ci_high":
                ci_high,

            "p_value":
                p_value,
        }
    )


results = pd.DataFrame(
    rows
)

results.to_csv(
    OUT_CSV,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 95)
print("PHQ-9 LOW VS HIGH — PARTICIPANT-CLUSTERED GEE")
print("High = PHQ-9 >=10; Low = PHQ-9 0–4; scores 5–9 excluded")
print("=" * 95)

print(
    results[
        [
            "time_frame",
            "n_segments",
            "n_participants",
            "mean_low_home_pct",
            "mean_high_home_pct",
            "gee_high_minus_low_pp",
            "ci_low",
            "ci_high",
            "p_value",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# DIFFERENCE-ONLY FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10.5

fig, ax = plt.subplots(
    figsize=(7.2, 4.5)
)

y = np.arange(
    len(results)
)

beta = results[
    "gee_high_minus_low_pp"
].to_numpy()

low = results[
    "ci_low"
].to_numpy()

high = results[
    "ci_high"
].to_numpy()

xerr = np.vstack(
    [
        beta - low,
        high - beta,
    ]
)

ax.errorbar(
    beta,
    y,
    xerr=xerr,
    fmt="o",
    markersize=8,
    capsize=5,
    linewidth=1.6,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.4",
)

ax.set_yticks(
    y
)

ax.set_yticklabels(
    results[
        "time_frame"
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Difference in home time: high − low PHQ-9 (percentage points)"
)

ax.set_title(
    "Home-time difference between high and low PHQ-9 groups"
)

# Add estimates and p-values
for i, row in results.iterrows():

    ax.text(
        row["ci_high"] + 0.5,
        i,
        (
            f"{row['gee_high_minus_low_pp']:+.1f} pp"
            f"   p={row['p_value']:.3f}"
        ),
        va="center",
        fontsize=9,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.grid(
    axis="x",
    alpha=0.15,
)

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print()
print("Saved:")
print(OUT_CSV)
print(OUT_PNG)
print(OUT_PDF)
