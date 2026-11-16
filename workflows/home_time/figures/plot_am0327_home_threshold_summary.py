from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "home_reference_stability/AMO327"
)

INPUT = ROOT / "AMO327_home_threshold_sensitivity.csv"

OUT_PNG = ROOT / "AMO327_home_threshold_summary.png"
OUT_PDF = ROOT / "AMO327_home_threshold_summary.pdf"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

calibration_order = [
    7,
    10,
    14,
    21,
    30,
    48,
]


# ============================================================
# SUMMARIZE ACROSS ALL THRESHOLD COMBINATIONS
# ============================================================

rows = []

for days in calibration_order:

    d = df[
        df["calibration_days"].eq(days)
    ].copy()

    inferred = d[
        d["home_status"].eq("observed")
    ]

    distances = (
        inferred[
            "distance_to_reference_home_m"
        ]
        .dropna()
        .astype(float)
    )

    rows.append(
        {
            "calibration_days": days,

            "n_threshold_combinations":
                len(d),

            "n_inferred":
                len(inferred),

            "inference_success_pct":
                100 * len(inferred) / len(d),

            "median_distance_m":
                distances.median()
                if len(distances)
                else np.nan,

            "min_distance_m":
                distances.min()
                if len(distances)
                else np.nan,

            "max_distance_m":
                distances.max()
                if len(distances)
                else np.nan,
        }
    )

summary = pd.DataFrame(rows)

print()
print("=" * 90)
print("AMO327 THRESHOLD-ROBUSTNESS SUMMARY")
print("=" * 90)
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# PLOT
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10.5

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.4),
)


# ------------------------------------------------------------
# A. Spatial convergence
# ------------------------------------------------------------

ax = axes[0]

x = summary[
    "calibration_days"
].to_numpy()

median = summary[
    "median_distance_m"
].to_numpy()

low = summary[
    "min_distance_m"
].to_numpy()

high = summary[
    "max_distance_m"
].to_numpy()

ax.fill_between(
    x,
    low,
    high,
    alpha=0.18,
)

ax.plot(
    x,
    median,
    marker="o",
    linewidth=2,
)

for xx, yy in zip(x, median):

    ax.text(
        xx,
        yy + 0.035,
        f"{yy:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

ax.set_xlabel(
    "Calibration period (days)"
)

ax.set_ylabel(
    "Distance to 48-day reference home (m)"
)

ax.set_title(
    "A. Home-location convergence"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(
    0.03,
    0.96,
    (
        "Line = median across all threshold combinations\n"
        "Shaded range = min–max across combinations"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.5,
)


# ------------------------------------------------------------
# B. Ability to infer home
# ------------------------------------------------------------

ax = axes[1]

success = summary[
    "inference_success_pct"
].to_numpy()

ax.plot(
    x,
    success,
    marker="o",
    linewidth=2,
)

for xx, yy in zip(x, success):

    ax.text(
        xx,
        yy + 2.5,
        f"{yy:.0f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

ax.set_ylim(
    0,
    108,
)

ax.set_xlabel(
    "Calibration period (days)"
)

ax.set_ylabel(
    "Threshold combinations yielding a home (%)"
)

ax.set_title(
    "B. Robustness of successful inference"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    (
        "AMO327: robustness of home inference across nighttime evidence thresholds"
    ),
    fontsize=14,
)

fig.text(
    0.5,
    0.01,
    (
        "Threshold grid: 0.5–4 h minimum nighttime observation "
        "and 2–7 minimum valid nights."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        0.92,
    ]
)

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
print(OUT_PNG)
print(OUT_PDF)
