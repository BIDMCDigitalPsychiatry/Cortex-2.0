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

OUT = ROOT / "AMO327_home_threshold_factors.png"

df = pd.read_csv(INPUT)

df = df[
    df["home_status"].eq("observed")
].copy()

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10.5


def summarize(data, xcol):
    return (
        data
        .groupby(xcol)["distance_to_reference_home_m"]
        .agg(
            median="median",
            minimum="min",
            maximum="max",
            n="count",
        )
        .reset_index()
        .sort_values(xcol)
    )


cal = summarize(
    df,
    "calibration_days",
)

hours = summarize(
    df,
    "min_night_observation_hours",
)

nights = summarize(
    df,
    "min_valid_nights",
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.4),
)


# ============================================================
# PANEL A — CALIBRATION DURATION
# ============================================================

ax = axes[0]

ax.fill_between(
    cal["calibration_days"],
    cal["minimum"],
    cal["maximum"],
    alpha=0.18,
)

ax.plot(
    cal["calibration_days"],
    cal["median"],
    marker="o",
    linewidth=2,
)

for _, r in cal.iterrows():
    ax.text(
        r["calibration_days"],
        r["median"] + 0.035,
        f'{r["median"]:.2f}',
        ha="center",
        fontsize=8.5,
    )

ax.set_title(
    "A. Calibration duration"
)

ax.set_xlabel(
    "Calibration period (days)"
)

ax.set_ylabel(
    "Distance to 48-day reference home (m)"
)


# ============================================================
# PANEL B — NIGHTTIME OBSERVATION
# ============================================================

ax = axes[1]

ax.plot(
    hours["min_night_observation_hours"],
    hours["median"],
    marker="o",
    linewidth=2,
)

for _, r in hours.iterrows():
    ax.text(
        r["min_night_observation_hours"],
        r["median"] + 0.035,
        f'{r["median"]:.2f}',
        ha="center",
        fontsize=8.5,
    )

ax.set_title(
    "B. Nighttime GPS requirement"
)

ax.set_xlabel(
    "Minimum observed nighttime GPS (hours/night)"
)

ax.set_ylabel(
    "Distance to 48-day reference home (m)"
)


# ============================================================
# PANEL C — NUMBER OF VALID NIGHTS
# ============================================================

ax = axes[2]

ax.plot(
    nights["min_valid_nights"],
    nights["median"],
    marker="o",
    linewidth=2,
)

for _, r in nights.iterrows():
    ax.text(
        r["min_valid_nights"],
        r["median"] + 0.035,
        f'{r["median"]:.2f}',
        ha="center",
        fontsize=8.5,
    )

ax.set_title(
    "C. Valid-night requirement"
)

ax.set_xlabel(
    "Minimum number of valid nights"
)

ax.set_ylabel(
    "Distance to 48-day reference home (m)"
)


# ============================================================
# STYLE
# ============================================================

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(
        -0.03,
        max(
            0.8,
            df["distance_to_reference_home_m"].max() + 0.1
        ),
    )

fig.suptitle(
    "AMO327: factors affecting stability of inferred home location",
    fontsize=14,
)

fig.text(
    0.5,
    0.01,
    (
        "Points show median distance to the 48-day reference home. "
        "Panel A also shows the min–max range across threshold settings."
    ),
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
    OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("Saved:")
print(OUT)
