from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from cortex2_engine.features.gps_pipeline_qc import (
    GPSQCConfig, clean_gps_fixes
)
from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig, _bridge
)

sys.path.insert(0, str(Path.cwd() / "workflows"))
from run_gps_imputed_clinical_features import (
    build_participant_donors
)

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)
RESULTS = Path("outputs/am0327_final_gps_comparison")
OUT = Path.home() / "cortex2_visualizations/outputs"

window = "AMO327__20231212"
target = pd.Timestamp("2023-12-12T15:41:48Z").value // 10**6

status = pd.read_csv(RESULTS / "AMO327_window_status.csv")
diag = pd.read_csv(RESULTS / "AMO327_gap_diagnostics.csv")

s = status.loc[status.clinical_window_id.eq(window)]
if len(s) != 1:
    raise RuntimeError("Clinical window not uniquely identified.")
s = s.iloc[0]

# Recover exactly the saved imputation-0 donor selection.
g = diag.loc[
    diag.clinical_window_id.eq(window)
    & diag.imputation.eq(0)
    & diag.start_ms.eq(target)
    & diag.gap_type.eq("trip")
    & diag.imputed.eq(True)
]

if len(g) != 1:
    raise RuntimeError("Saved December 12 imputation not found.")
g = g.iloc[0]

if int(g.n_candidate_donors) != 6:
    raise RuntimeError("Candidate count differs from Panel C.")

if pd.isna(g.donor_index):
    raise RuntimeError("Selected donor index missing.")

start_ms = int(s.window_start_ms)
end_ms = int(s.window_end_ms)

raw = pq.read_table(
    ROOT / "gps_clinical_weekly_raw_points.parquet",
    filters=[("participant_id", "=", "AMO327")]
).to_pandas()

raw = raw.loc[
    raw.timestamp.ge(start_ms)
    & raw.timestamp.lt(end_ms),
    ["timestamp", "latitude", "longitude", "accuracy"]
].copy()

cfg = ImputationConfig(
    m=20, stay_compatibility_multiplier=1.0
)
clean = clean_gps_fixes(
    raw, GPSQCConfig(max_gap_ms=60_000)
).sort_values("timestamp")

trip_donors, stay_donors, stats = build_participant_donors(
    clean.loc[
        clean.timestamp.ge(start_ms)
        & clean.timestamp.lt(end_ms)
    ]
)

donor_idx = int(g.donor_index)
if not 0 <= donor_idx < len(trip_donors):
    raise RuntimeError("Saved donor index does not match donor list.")

donor = trip_donors[donor_idx]

# Recover the two actual GPS observations surrounding the gap.
t0 = int(g.start_ms)
t1 = int(g.end_ms)

left_rows = clean.loc[clean.timestamp.eq(t0)]
right_rows = clean.loc[clean.timestamp.eq(t1)]

if len(left_rows) != 1 or len(right_rows) != 1:
    raise RuntimeError("Observed gap endpoints do not match saved data.")

left = left_rows.iloc[0]
right = right_rows.iloc[0]

# This is the EXACT reconstruction function used by the pipeline.
imputed = _bridge(left, right, donor, cfg)

if imputed is None or imputed.empty:
    raise RuntimeError("Saved donor could not reproduce the bridge.")

if not (
    imputed.timestamp.gt(t0).all()
    and imputed.timestamp.lt(t1).all()
    and imputed.imputed.all()
):
    raise RuntimeError("Reconstructed gap failed provenance checks.")

R = 6371008.8

def local_xy(frame, lat0, lon0):
    lat = frame.latitude.to_numpy(float)
    lon = frame.longitude.to_numpy(float)
    x = R * np.cos(np.radians(lat0)) * np.radians(lon-lon0)
    y = R * np.radians(lat-lat0)
    return x, y

# Donor displayed relative to its own starting position.
dx, dy = local_xy(
    donor,
    float(donor.latitude.iloc[0]),
    float(donor.longitude.iloc[0])
)

# Gap reconstruction displayed relative to the true gap start.
gap = pd.concat([
    pd.DataFrame([left]),
    imputed,
    pd.DataFrame([right])
], ignore_index=True)

gx, gy = local_xy(
    gap, float(left.latitude), float(left.longitude)
)

# Nearby real GPS data, displayed without crossing the gap.
before = clean.loc[
    clean.timestamp.between(t0-180_000, t0)
]
after = clean.loc[
    clean.timestamp.between(t1, t1+180_000)
]


# Align the ORIGINAL donor for display using the SAME
# rigid rotation and uniform scale as _bridge().
# This makes geometric similarity directly visible.
lat0 = float(left.latitude)

donor_x = (
    R * np.cos(np.deg2rad(lat0))
    * np.deg2rad(
        donor.longitude.to_numpy(float)
        - float(donor.longitude.iloc[0])
    )
)
donor_y = (
    R * np.deg2rad(
        donor.latitude.to_numpy(float)
        - float(donor.latitude.iloc[0])
    )
)

source_end = np.array(
    [donor_x[-1], donor_y[-1]], dtype=float
)
target_end = np.array(
    [gx[-1], gy[-1]], dtype=float
)

source_norm = np.linalg.norm(source_end)
target_norm = np.linalg.norm(target_end)

if source_norm <= 1e-6 or target_norm <= 1e-6:
    raise RuntimeError(
        "Cannot show a comparable aligned donor for this gap."
    )

angle = (
    np.arctan2(target_end[1], target_end[0])
    - np.arctan2(source_end[1], source_end[0])
)
scale = target_norm / source_norm

if not np.isfinite(scale) or scale > 4:
    raise RuntimeError("Donor transformation is out of range.")

cc, ss = np.cos(angle), np.sin(angle)

aligned_x = scale * (cc * donor_x - ss * donor_y)
aligned_y = scale * (ss * donor_x + cc * donor_y)

# Verify that the displayed donor connects the same endpoints.
if np.linalg.norm(
    np.array([aligned_x[-1], aligned_y[-1]]) - target_end
) > 1.0:
    raise RuntimeError("Donor endpoint alignment failed.")


# ------------------------------------------------------------
# Diagnostic: original donor / transformed donor / resampled
# ------------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)

def length_m(x, y):
    return float(np.hypot(np.diff(x), np.diff(y)).sum())

raw_length = length_m(dx, dy)
aligned_length = length_m(aligned_x, aligned_y)
imputed_length = length_m(gx, gy)

retained_pct = (
    100 * imputed_length / aligned_length
    if aligned_length > 0 else np.nan
)

lost_pct = 100 - retained_pct

# Check whether the generated positions are consistent with
# temporal interpolation of the full transformed donor.
donor_t = donor.timestamp.to_numpy(dtype=float)
donor_tau = (donor_t - donor_t[0]) / (donor_t[-1] - donor_t[0])

gap_t = gap.timestamp.to_numpy(dtype=float)
gap_tau = (gap_t - gap_t[0]) / (gap_t[-1] - gap_t[0])

expected_x = np.interp(gap_tau, donor_tau, aligned_x)
expected_y = np.interp(gap_tau, donor_tau, aligned_y)

max_position_difference = float(np.max(np.hypot(
    expected_x - gx, expected_y - gy
)))

print("\nAMO327 DONOR-RESAMPLING DIAGNOSTICS")
print("-" * 48)
print("Gap start:", pd.to_datetime(t0, unit="ms", utc=True))
print("Gap end:  ", pd.to_datetime(t1, unit="ms", utc=True))
print("Gap duration (minutes):", round((t1-t0)/60000, 3))
print("Saved donor index:", donor_idx)
print("Original donor observations:", len(donor))
print("Final total trajectory positions:", len(gap))
print("Synthetic interior positions:", len(imputed))
print()
print(f"Original donor path:       {raw_length:.2f} m")
print(f"Full transformed path:     {aligned_length:.2f} m")
print(f"Final resampled path:      {imputed_length:.2f} m")
print(f"Path length retained:      {retained_pct:.1f}%")
print(f"Path length lost:          {lost_pct:.1f}%")
print(
    "Max difference from temporal interpolation:",
    f"{max_position_difference:.3f} m"
)

if max_position_difference > 2.0:
    print(
        "WARNING: reconstructed points do not match the "
        "expected resampling closely. Investigate before "
        "attributing the difference to sampling."
    )

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.subplots_adjust(
    left=.065, right=.985,
    top=.81, bottom=.17, wspace=.30
)

GREEN = "#238A78"
ORANGE = "#D58B32"
BLUE = "#2679B2"

# 1: original observed donor, native geographic orientation.
a = axes[0]
a.plot(dx, dy, color=GREEN, lw=2.4)
a.scatter(dx, dy, s=7, color=GREEN, alpha=.65)
a.scatter(
    [dx[0], dx[-1]], [dy[0], dy[-1]],
    s=55, color=BLUE, zorder=5
)
a.set_title(
    "1  Original observed donor",
    loc="left", fontsize=12, fontweight="bold"
)

# 2: all donor coordinates after rotation and uniform scaling.
b = axes[1]
b.plot(aligned_x, aligned_y, color=GREEN, lw=2.5)
b.scatter(
    [aligned_x[0], aligned_x[-1]],
    [aligned_y[0], aligned_y[-1]],
    s=55, color=BLUE, zorder=5
)
b.set_title(
    "2  Full transformed donor",
    loc="left", fontsize=12, fontweight="bold"
)

# 3: full transformed geometry in faint green,
# with actual generated GPS positions connected in orange.
c = axes[2]
c.plot(
    aligned_x, aligned_y,
    color=GREEN, lw=3.2, alpha=.27,
    label="Full transformed donor"
)
c.plot(
    gx, gy, "--", color=ORANGE,
    lw=2.5, marker="o", markersize=4,
    label="Resampled reconstruction"
)
c.scatter(
    [gx[0], gx[-1]],
    [gy[0], gy[-1]],
    s=55, color=BLUE, zorder=5
)
c.set_title(
    "3  Final resampled reconstruction",
    loc="left", fontsize=12, fontweight="bold"
)
c.legend(loc="lower center", fontsize=8, frameon=True)

# Same bounds for panels 2 and 3 so they can be compared directly.
all_x = np.r_[aligned_x, gx]
all_y = np.r_[aligned_y, gy]
pad = max(30.0, .08 * max(np.ptp(all_x), np.ptp(all_y)))

for ax in (b, c):
    ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
    ax.set_ylim(all_y.min()-pad, all_y.max()+pad)

for ax in axes:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E6EDF2", lw=.7)
    ax.set_xlabel("Relative east–west (m)")
    ax.set_ylabel("Relative north–south (m)")
    ax.tick_params(labelsize=8)

fig.suptitle(
    "AMO327 | Donor transformation and resampling",
    fontsize=18, fontweight="bold", y=.965
)

fig.text(
    .5, .065,
    f"Full transformed: {aligned_length:.1f} m"
    f"    |    Resampled: {imputed_length:.1f} m"
    f"    |    Length lost: {lost_pct:.1f}%",
    ha="center", fontsize=11, color="#354F63"
)

path = OUT / "AMO327_donor_resampling_diagnostic.png"
fig.savefig(path, dpi=210, facecolor="white")
plt.close(fig)

print("\nSaved:", path)
