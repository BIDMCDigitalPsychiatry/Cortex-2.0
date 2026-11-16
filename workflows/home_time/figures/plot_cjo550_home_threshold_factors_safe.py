from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PID = "CJO550"

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "home_reference_stability"
)

PDIR = ROOT / PID
OUT = PDIR / f"{PID}_home_threshold_factors.png"

# ------------------------------------------------------------
# Find the needed CSVs by their columns rather than filename
# ------------------------------------------------------------

conv = None
sens = None

for f in PDIR.rglob("*.csv"):
    try:
        x = pd.read_csv(f, low_memory=False)
    except Exception:
        continue

    cols = set(x.columns)

    if {
        "calibration_label",
        "distance_to_full_home_m"
    }.issubset(cols):
        conv = x.copy()
        print("Convergence CSV:", f)

    if {
        "min_night_observation_hours",
        "min_valid_nights"
    }.issubset(cols):
        # Find a distance column
        dist_candidates = [
            c for c in x.columns
            if "distance" in c.lower()
            and c.lower().endswith("_m")
        ]

        if dist_candidates:
            sens = x.copy()
            sens_dist_col = dist_candidates[0]
            print("Sensitivity CSV:", f)
            print("Distance column:", sens_dist_col)

if conv is None:
    raise RuntimeError(
        "Could not find convergence CSV for CJO550."
    )

if sens is None:
    raise RuntimeError(
        "Could not find threshold-sensitivity CSV for CJO550."
    )

# ------------------------------------------------------------
# PANEL A — calibration duration
# ------------------------------------------------------------

conv["distance_to_full_home_m"] = pd.to_numeric(
    conv["distance_to_full_home_m"],
    errors="coerce"
)

# Exclude Full study because its distance is necessarily zero
a = conv[
    conv["calibration_label"].astype(str) != "Full study"
].copy()

a["calibration_days"] = pd.to_numeric(
    a["calibration_label"]
    .astype(str)
    .str.extract(r"(\d+)")[0],
    errors="coerce"
)

a = a[
    np.isfinite(a["calibration_days"])
    &
    np.isfinite(a["distance_to_full_home_m"])
].sort_values("calibration_days")


# ------------------------------------------------------------
# Sensitivity data
# ------------------------------------------------------------

sens[sens_dist_col] = pd.to_numeric(
    sens[sens_dist_col],
    errors="coerce"
)

sens["min_night_observation_hours"] = pd.to_numeric(
    sens["min_night_observation_hours"],
    errors="coerce"
)

sens["min_valid_nights"] = pd.to_numeric(
    sens["min_valid_nights"],
    errors="coerce"
)

sens = sens[
    np.isfinite(sens[sens_dist_col])
].copy()


# ------------------------------------------------------------
# PANEL B — nighttime observation requirement
# ------------------------------------------------------------

b = (
    sens.groupby(
        "min_night_observation_hours",
        as_index=False
    )[sens_dist_col]
    .median()
    .rename(
        columns={
            sens_dist_col: "median_distance_m"
        }
    )
)

b = b[
    np.isfinite(b["min_night_observation_hours"])
    &
    np.isfinite(b["median_distance_m"])
].sort_values("min_night_observation_hours")


# ------------------------------------------------------------
# PANEL C — valid-night requirement
# ------------------------------------------------------------

c = (
    sens.groupby(
        "min_valid_nights",
        as_index=False
    )[sens_dist_col]
    .median()
    .rename(
        columns={
            sens_dist_col: "median_distance_m"
        }
    )
)

c = c[
    np.isfinite(c["min_valid_nights"])
    &
    np.isfinite(c["median_distance_m"])
].sort_values("min_valid_nights")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(
    1, 3,
    figsize=(12.8, 4.1)
)

# A
ax = axes[0]

ax.plot(
    a["calibration_days"],
    a["distance_to_full_home_m"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Calibration duration (days)")
ax.set_ylabel("Distance to full-study home (m)")
ax.set_title("A. Calibration duration")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# B
ax = axes[1]

ax.plot(
    b["min_night_observation_hours"],
    b["median_distance_m"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Minimum nighttime GPS (hours/night)")
ax.set_ylabel("Median distance to reference home (m)")
ax.set_title("B. Nighttime GPS requirement")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# C
ax = axes[2]

ax.plot(
    c["min_valid_nights"],
    c["median_distance_m"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Minimum number of valid nights")
ax.set_ylabel("Median distance to reference home (m)")
ax.set_title("C. Valid-night requirement")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


fig.suptitle(
    f"{PID}: factors affecting stability of inferred home location",
    fontsize=13
)

fig.text(
    0.5,
    0.01,
    "Distance is measured relative to the participant's full-study inferred home.",
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.05, 1, 0.92]
)

fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("Saved:", OUT)
