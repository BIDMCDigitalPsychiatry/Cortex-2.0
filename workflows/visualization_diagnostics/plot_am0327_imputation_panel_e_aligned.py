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

fig, (a, b) = plt.subplots(
    1, 2, figsize=(11.8, 4.3),
    gridspec_kw={"width_ratios": [1, 1]}
)
fig.subplots_adjust(
    left=.065, right=.985,
    top=.83, bottom=.17, wspace=.15
)

GREEN = "#258A78"
ORANGE = "#D58B32"
BLUE = "#2478AD"

# E1: original observed donor geometry, transformed for
# visual comparison. No synthetic observations added.
a.plot(
    aligned_x, aligned_y,
    color=GREEN, lw=2.7, zorder=3
)
a.scatter(
    [aligned_x[0], aligned_x[-1]],
    [aligned_y[0], aligned_y[-1]],
    color=BLUE, s=42, zorder=5
)

# E2: faint reference donor and the actual _bridge() output.
b.plot(
    aligned_x, aligned_y,
    color=GREEN, lw=4, alpha=.22, zorder=2
)

b.plot(
    gx, gy,
    color=ORANGE, lw=2.6,
    linestyle=(0, (4, 3)),
    zorder=3
)

b.scatter(
    gx[1:-1], gy[1:-1],
    color=ORANGE, s=12, zorder=4
)
b.scatter(
    [gx[0], gx[-1]],
    [gy[0], gy[-1]],
    color=BLUE, s=46, zorder=5
)

a.set_title(
    "E1  Donor shape (aligned)",
    loc="left", fontsize=12, weight="bold"
)
b.set_title(
    "E2  Reconstructed gap",
    loc="left", fontsize=12, weight="bold"
)

# Identical geographic scale and bounds in both panels.
all_x = np.r_[aligned_x, gx]
all_y = np.r_[aligned_y, gy]

pad = max(
    25.0,
    .10 * max(np.ptp(all_x), np.ptp(all_y))
)

bounds = (
    all_x.min()-pad, all_x.max()+pad,
    all_y.min()-pad, all_y.max()+pad
)

for ax in (a, b):
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E8EEF1", lw=.7)
    ax.set_xlabel("East–west (m)", fontsize=9)
    ax.set_ylabel("North–south (m)", fontsize=9)
    ax.tick_params(labelsize=8)

fig.suptitle(
    "E  Donor-based GPS reconstruction",
    fontsize=16, weight="bold", y=.97
)

fig.text(
    .5, .055,
    "Green: aligned donor    "
    "Orange: imputed GPS    "
    "Blue: endpoints",
    ha="center", fontsize=10, color="#526C7C"
)

OUT.mkdir(parents=True, exist_ok=True)

panel_path = OUT / "AMO327_20231212_imputation_panel_E_aligned.png"
fig.savefig(panel_path, dpi=210, facecolor="white")
plt.close(fig)

# Append E to a NEW copy of the existing A-D figure.
# The original scientific figure remains unchanged.
original_path = OUT / "AMO327_20231212_01_observed_gps_to_trips.png"
combined_path = OUT / "AMO327_20231212_01_with_aligned_imputation_E.png"

if not original_path.exists():
    raise RuntimeError("Existing A-D figure not found.")

with Image.open(original_path) as src, Image.open(panel_path) as extra:
    upper = src.convert("RGB")
    lower = extra.convert("RGB")

    new_height = round(lower.height * upper.width / lower.width)
    lower = lower.resize(
        (upper.width, new_height),
        Image.Resampling.LANCZOS
    )

    combined = Image.new(
        "RGB",
        (upper.width, upper.height + lower.height),
        "white"
    )
    combined.paste(upper, (0, 0))
    combined.paste(lower, (0, upper.height))
    combined.save(combined_path)

print("Selected donor index:", donor_idx)
print("Original donor fixes:", len(donor))
print("Reconstructed GPS positions:", len(imputed))
print("Gap duration (min):", round(float(g.gap_ms)/60000, 2))
print("Saved Panel E:", panel_path)
print("Saved expanded figure:", combined_path)
print("Original A-D figure unchanged.")
