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


from dataclasses import replace
from matplotlib.patches import Rectangle

from cortex2_engine.features.gps_intervals import build_gps_intervals
from cortex2_engine.features.gps_spatial_semantics import (
    add_gps_spatial_semantics
)
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes
)
from cortex2_engine.features.gps_place_candidates import (
    build_place_candidates
)
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility
)
from cortex2_engine.features.gps_trip_features import (
    build_place_visits, build_observed_trips
)

# ------------------------------------------------------------
# 1. Recover the full observed trip containing Panel C's gap.
#    Use the same place/trip detection as the clinical workflow.
# ------------------------------------------------------------

edges = build_gps_intervals(
    clean,
    window_start_ms=int(clean.timestamp.min()),
    window_end_ms=int(clean.timestamp.max()) + 1,
    max_gap_ms=60_000,
)

spatial = add_gps_spatial_semantics(edges.copy())

supported = spatial.loc[
    spatial.supported.fillna(False)
].copy()

episodes = build_anchor_compatible_location_episodes(supported)

assignments, places = build_place_candidates(
    episodes, compatibility_multiplier=1.0
)

visits = build_place_visits(
    assignments,
    min_endpoint_dwell_ms=600_000
)

trajectory = add_gps_trajectory_plausibility(
    spatial,
    max_plausible_speed_kmh=1000.0
)

trips = build_observed_trips(
    trajectory, visits, places
)

# The actual Panel C gap must fall inside the selected trip.
matching = trips.loc[
    (trips.trip_start_ms <= t0)
    & (trips.trip_end_ms >= t1)
].copy()

if matching.empty:
    raise RuntimeError(
        "Cannot identify the original trip containing the "
        "December 12 Panel C gap. No substitute trip was plotted."
    )

matching["duration_ms"] = (
    matching.trip_end_ms - matching.trip_start_ms
)

trip = matching.sort_values("duration_ms").iloc[0]

trip_start = int(trip.trip_start_ms)
trip_end = int(trip.trip_end_ms)

observed_trip = clean.loc[
    clean.timestamp.between(trip_start, trip_end)
].sort_values("timestamp").copy()

if len(observed_trip) < 3:
    raise RuntimeError("Insufficient observations for full trip.")

# ------------------------------------------------------------
# 2. Reconstruct the SAME gap using the SAME selected donor.
# ------------------------------------------------------------

resolutions = [60, 15, 5]
paths = {}

for seconds in resolutions:
    local_cfg = replace(
        cfg,
        max_supported_gap_ms=seconds * 1000
    )

    part = _bridge(left, right, donor, local_cfg)

    if part is None or part.empty:
        raise RuntimeError(
            f"Reconstruction failed at {seconds} seconds."
        )

    paths[seconds] = pd.concat(
        [
            pd.DataFrame([left]),
            part,
            pd.DataFrame([right])
        ],
        ignore_index=True
    )

# Verify the 60-second path reproduces the saved reconstruction.
expected = imputed.sort_values("timestamp").reset_index(drop=True)
actual = paths[60].iloc[1:-1].reset_index(drop=True)

if len(expected) != len(actual):
    raise RuntimeError("60s reconstruction differs from saved Panel E.")

for col in ("timestamp", "latitude", "longitude"):
    if not np.allclose(
        expected[col].to_numpy(float),
        actual[col].to_numpy(float),
        rtol=0,
        atol=1e-7 if col != "timestamp" else 0
    ):
        raise RuntimeError(
            f"60s reconstruction mismatch in {col}."
        )

# ------------------------------------------------------------
# 3. Project everything into the same local metric coordinates.
# ------------------------------------------------------------

lat0 = float(left.latitude)
lon0 = float(left.longitude)

def to_xy(frame):
    return local_xy(frame, lat0, lon0)

# Original trip, excluding unsupported temporal connections.
observed_trip = observed_trip.reset_index(drop=True)

# NaNs prevent Matplotlib from drawing over any missing interval.
ox, oy = to_xy(observed_trip)
times = observed_trip.timestamp.to_numpy(np.int64)

large_gaps = np.flatnonzero(
    np.diff(times) > cfg.max_supported_gap_ms
)

plot_x = []
plot_y = []

for i in range(len(observed_trip)):
    if i > 0 and (i - 1) in set(large_gaps):
        plot_x.append(np.nan)
        plot_y.append(np.nan)

    plot_x.append(ox[i])
    plot_y.append(oy[i])

plot_x = np.asarray(plot_x, dtype=float)
plot_y = np.asarray(plot_y, dtype=float)

reconstruction_xy = {
    seconds: to_xy(frame)
    for seconds, frame in paths.items()
}

# Endpoints: last observed before gap, first observed after gap.
ex, ey = to_xy(
    pd.DataFrame([left, right])
)

styles = {
    60: ("#C94F55", "60 s"),
    15: ("#329D79", "15 s"),
    5:  ("#7863BB", "5 s"),
}

# ------------------------------------------------------------
# 4. Plot the full trip and a magnified view of the exact gap.
# ------------------------------------------------------------


# ------------------------------------------------------------
# Compare December 7's complete observed route with the
# actual December 12 gap endpoints. No imputation.
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path.home() / "cortex2_visualizations/outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Panel B's approximately 20:32–21:13 UTC observed trip,
# with two minutes of context on either side.
dec7_start = pd.Timestamp("2023-12-07T20:30:00Z").value // 10**6
dec7_end = pd.Timestamp("2023-12-07T21:15:00Z").value // 10**6

dec7 = clean.loc[
    clean.timestamp.between(dec7_start, dec7_end)
].sort_values("timestamp").reset_index(drop=True)

if len(dec7) < 3:
    raise RuntimeError("Insufficient Dec 7 GPS observations")

lat0 = float(left.latitude)
lon0 = float(left.longitude)

def project(frame):
    return local_xy(frame, lat0, lon0)

def observed_line(frame):
    x, y = project(frame)
    t = frame.timestamp.to_numpy(np.int64)
    for k in np.flatnonzero(np.diff(t) > 60000)[::-1]:
        x = np.insert(x, k + 1, np.nan)
        y = np.insert(y, k + 1, np.nan)
    return x, y

R = 6371008.8

def distances_to(lat, lon, frame):
    lat1 = np.radians(frame.latitude.to_numpy(float))
    lon1 = np.radians(frame.longitude.to_numpy(float))
    lat2 = np.radians(float(lat))
    lon2 = np.radians(float(lon))
    h = (
        np.sin((lat1-lat2)/2)**2
        + np.cos(lat1)*np.cos(lat2)
        * np.sin((lon1-lon2)/2)**2
    )
    return 2*R*np.arcsin(np.sqrt(np.clip(h, 0, 1)))

# Distances from every Dec 7 observation to each real gap endpoint.
d_start = distances_to(left.latitude, left.longitude, dec7)
d_end = distances_to(right.latitude, right.longitude, dec7)

# Find the best ordered pair i < j.
# This is a diagnostic nearest-coordinate match,
# not a new donor eligibility threshold.
n = len(dec7)
suffix_best = np.empty(n, dtype=int)
best = n-1

for k in range(n-1, -1, -1):
    if d_end[k] < d_end[best]:
        best = k
    suffix_best[k] = best

indices = np.arange(n-1)
ends = suffix_best[indices+1]
scores = d_start[:-1] + d_end[ends]

i = int(np.argmin(scores))
j = int(ends[i])

matched = dec7.iloc[i:j+1].copy()

if len(matched) < 2:
    raise RuntimeError("Ordered matched segment too short")

t = matched.timestamp.to_numpy(np.int64)
dt = np.diff(t)

# Show exact geographic proximity and temporal support.
print()
print("DEC 7 OBSERVED ROUTE VS DEC 12 MISSING GAP")
print("=" * 56)
print("Dec 7 observations:", len(dec7))
print("Dec 12 gap duration:",
      round((t1-t0)/60000, 2), "min")
print()
print("Nearest Dec 7 fix to gap start:",
      round(float(d_start.min()), 1), "m")
print("Nearest Dec 7 fix to gap end:",
      round(float(d_end.min()), 1), "m")
print()
print("BEST CHRONOLOGICALLY ORDERED PAIR")
print("Start match distance:", round(float(d_start[i]), 1), "m")
print("End match distance:", round(float(d_end[j]), 1), "m")
print("Matched start:",
      pd.to_datetime(int(t[0]), unit="ms", utc=True))
print("Matched end:  ",
      pd.to_datetime(int(t[-1]), unit="ms", utc=True))
print("Matched duration:", round((t[-1]-t[0])/60000, 2), "min")
print("Matched observations:", len(matched))
print("Largest observed sampling gap:",
      round(float(dt.max())/1000, 2), "s")
print("Intervals over 60s:", int((dt > 60000).sum()))

mx, my = project(matched)
matched_distance = float(
    np.hypot(np.diff(mx), np.diff(my))[dt <= 60000].sum()
)
print("Supported polyline length:",
      round(matched_distance, 1), "m")
print()

# Compare actual geographic paths on exactly the same axes.
fig, (a, b) = plt.subplots(
    1, 2, figsize=(14.5, 6),
    gridspec_kw={"width_ratios": [1.05, 1]}
)
fig.subplots_adjust(
    left=.075, right=.97,
    top=.79, bottom=.15, wspace=.23
)

target = observed_trip.sort_values("timestamp")
tx, ty = observed_line(target)
dx, dy = observed_line(dec7)
sx, sy = observed_line(matched)

gap_x, gap_y = project(pd.DataFrame([left, right]))

for ax in (a, b):
    ax.plot(
        tx/1000, ty/1000,
        color="#9BAAB4", lw=2.2,
        label="Dec 12 observed trip"
    )
    ax.plot(
        dx/1000, dy/1000,
        color="#2878AE", lw=1.6, alpha=.65,
        label="Dec 7 observed route"
    )
    ax.plot(
        sx/1000, sy/1000,
        color="#238B78", lw=3,
        label="Dec 7 candidate section"
    )
    ax.scatter(
        gap_x/1000, gap_y/1000,
        color="#D48A2E", s=70,
        edgecolors="white", linewidths=.8,
        zorder=8, label="Dec 12 gap endpoints"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E9EEF1")
    ax.set_xlabel("East–west (km)")
    ax.set_ylabel("North–south (km)")

a.set_title(
    "A  Original geographic trajectories",
    loc="left", fontsize=13, weight="bold"
)
b.set_title(
    "B  Candidate donor section",
    loc="left", fontsize=13, weight="bold"
)

# Panel B zooms to the matched section and target gap endpoints.
xx = np.r_[sx[np.isfinite(sx)], gap_x]
yy = np.r_[sy[np.isfinite(sy)], gap_y]

pad = max(.08, .12*max(np.ptp(xx), np.ptp(yy))/1000)

b.set_xlim(xx.min()/1000-pad, xx.max()/1000+pad)
b.set_ylim(yy.min()/1000-pad, yy.max()/1000+pad)

handles, labels = a.get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center",
    bbox_to_anchor=(.5, .865),
    ncol=4, frameon=False, fontsize=9.5
)

fig.suptitle(
    "AMO327 | Does the Dec 7 trip contain the Dec 12 missing route?",
    fontsize=17, fontweight="bold", y=.97
)

out = OUT / "AMO327_Dec7_route_vs_Dec12_gap.png"
fig.savefig(out, dpi=210, facecolor="white")
plt.close(fig)

print("Saved:", out)
