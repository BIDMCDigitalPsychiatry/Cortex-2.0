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

fig, (a, b) = plt.subplots(
    1, 2, figsize=(15.8, 5.0)
)
fig.subplots_adjust(
    left=.08, right=.96, top=.74,
    bottom=.17, wspace=.25
)

# Panel E1: genuine observed donor geometry.
a.plot(dx, dy, color="#238A79", lw=2.3, zorder=2)
a.scatter(dx, dy, s=9, color="#238A79", alpha=.75, zorder=3)
a.scatter(
    [dx[0], dx[-1]], [dy[0], dy[-1]],
    s=65, color=["#236FA4", "#C88730"],
    edgecolors="white", zorder=5
)
a.set_title(
    "E1  Original observed donor trip",
    loc="left", fontsize=13, weight="bold"
)

# Panel E2: actual _bridge() result, not invented geometry.
for frame in (before, after):
    if len(frame) > 1:
        x, y = local_xy(
            frame, float(left.latitude), float(left.longitude)
        )
        a_color = "#2F7DAE"
        b.plot(x, y, color=a_color, lw=1.6, alpha=.8)
        b.scatter(x, y, s=8, color=a_color, alpha=.6)

b.plot(
    gx, gy, linestyle=(0, (3, 3)),
    color="#D58A32", lw=2.2, zorder=4
)
b.scatter(
    gx[1:-1], gy[1:-1],
    s=17, color="#D58A32", zorder=5
)
b.scatter(
    [gx[0], gx[-1]], [gy[0], gy[-1]],
    s=80, color="#236FA4",
    edgecolors="white", zorder=7
)

for x, y, label in [
    (gx[0], gy[0], "Last observed GPS"),
    (gx[-1], gy[-1], "Next observed GPS")
]:
    b.annotate(
        label, (x, y), xytext=(7, 9),
        textcoords="offset points", fontsize=9.5,
        color="#234F6F",
        bbox=dict(facecolor="white", edgecolor="none",
                  alpha=.9, pad=1)
    )

b.set_title(
    "E2  Donor adapted to the missing interval",
    loc="left", fontsize=13, weight="bold"
)

for ax in (a, b):
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color="#E8EDF1", lw=.8)
    ax.set_xlabel("Relative east–west position (m)")
    ax.set_ylabel("Relative north–south position (m)")
    ax.set_axisbelow(True)

fig.suptitle(
    "E  How the missing GPS trajectory was estimated",
    fontsize=19, weight="bold", y=.965
)

fig.text(
    .5, .855,
    f"AMO327  |  Dec 12, 2023, 15:41 UTC"
    f"  |  {float(g.gap_ms)/60000:.1f}-minute gap"
    f"  |  6 eligible donors"
    f"  |  saved donor #{donor_idx}",
    ha="center", fontsize=11, color="#536D7D"
)

fig.text(
    .5, .05,
    "Green: real recorded donor trip"
    "   |   Blue: observed GPS"
    "   |   Orange: reconstructed positions (not measured GPS)",
    ha="center", fontsize=10, color="#536D7D"
)

OUT.mkdir(parents=True, exist_ok=True)

panel_path = OUT / "AMO327_20231212_imputation_panel_E.png"
fig.savefig(panel_path, dpi=210, facecolor="white")
plt.close(fig)

# Append E to a NEW copy of the existing A-D figure.
# The original scientific figure remains unchanged.
original_path = OUT / "AMO327_20231212_01_observed_gps_to_trips.png"
combined_path = OUT / "AMO327_20231212_01_with_imputation_panel_E.png"

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
