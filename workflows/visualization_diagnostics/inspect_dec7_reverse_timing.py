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


# Reverse-direction matching: start point occurs LATER
# than end point in the original Dec 7 recording.
prefix_best = np.empty(n, dtype=int)
best = 0

for k in range(n):
    if d_end[k] < d_end[best]:
        best = k
    prefix_best[k] = best

start_indices = np.arange(1, n)
end_indices = prefix_best[start_indices - 1]

scores_reverse = (
    d_start[start_indices] + d_end[end_indices]
)
best_pos = int(np.argmin(scores_reverse))
i_rev = int(start_indices[best_pos])
j_rev = int(end_indices[best_pos])

reverse_section = dec7.iloc[j_rev:i_rev+1].iloc[::-1].copy()
reverse_times = reverse_section.timestamp.to_numpy(np.int64)
support_gaps = np.abs(np.diff(reverse_times))

print("\\nBEST REVERSE-DIRECTION MATCH")
print("=" * 45)
print("Start endpoint mismatch:", round(float(d_start[i_rev]), 1), "m")
print("End endpoint mismatch:", round(float(d_end[j_rev]), 1), "m")
print("Original Dec 7 timestamps:")
print("  Earlier:", pd.to_datetime(int(dec7.timestamp.iloc[j_rev]), unit="ms", utc=True))
print("  Later:  ", pd.to_datetime(int(dec7.timestamp.iloc[i_rev]), unit="ms", utc=True))
print("Reverse section duration:", round(
    (reverse_times[0] - reverse_times[-1]) / 60000, 2
), "min")
print("Number of recorded fixes:", len(reverse_section))
print("Largest sampling interval:", round(
    float(support_gaps.max()) / 1000, 2
), "s")
print("Sampling intervals over 60s:", int((support_gaps > 60000).sum()))

# ------------------------------------------------------------
# Movement timing and cumulative distance of the actual
# December 7 reverse-direction candidate section.
# ------------------------------------------------------------

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Reverse matching was computed above. Restore the actual
# December 7 chronological recording order for timing analysis.
section = (
    reverse_section
    .sort_values("timestamp")
    .reset_index(drop=True)
)

ts = section.timestamp.to_numpy(dtype=np.int64)
elapsed_min = (ts - ts[0]) / 60000.0

x, y = project(section)

interval_s = np.diff(ts) / 1000.0
step_m = np.hypot(np.diff(x), np.diff(y))

if np.any(interval_s <= 0):
    raise RuntimeError("Non-increasing GPS timestamps")

if np.any(interval_s > 60):
    raise RuntimeError("Unexpected unsupported interval")

cumulative_m = np.r_[0.0, np.cumsum(step_m)]

duration_min = float(elapsed_min[-1])
total_m = float(cumulative_m[-1])
endpoint_m = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))
mean_speed_kmh = total_m / (duration_min * 60) * 3.6

# Each two-minute interval's traveled distance, interpolating
# cumulative distance at the time boundaries.
bin_edges = np.r_[
    np.arange(0, duration_min, 2.0),
    duration_min
]
bin_edges = np.unique(bin_edges)

bin_distances = np.diff(
    np.interp(bin_edges, elapsed_min, cumulative_m)
)
bin_durations = np.diff(bin_edges)
bin_speeds = bin_distances / (bin_durations * 60) * 3.6

print("\nDECEMBER 7 MATCHED SECTION — MOVEMENT TIMING")
print("=" * 58)
print("Start:", pd.to_datetime(ts[0], unit="ms", utc=True))
print("End:  ", pd.to_datetime(ts[-1], unit="ms", utc=True))
print(f"Duration: {duration_min:.2f} min")
print(f"GPS observations: {len(section)}")
print(f"Largest sampling gap: {interval_s.max():.2f} s")
print(f"Raw observed polyline: {total_m:.1f} m")
print(f"Endpoint separation: {endpoint_m:.1f} m")
print(f"Mean polyline speed: {mean_speed_kmh:.1f} km/h")
print(f"Route directness: {endpoint_m/total_m:.3f}")
print()

print("TWO-MINUTE MOVEMENT SUMMARY")
print("Elapsed (min)     Path distance (m)    Mean speed (km/h)")
for start, end, dist, speed in zip(
    bin_edges[:-1], bin_edges[1:], bin_distances, bin_speeds
):
    print(
        f"{start:5.1f}–{end:5.1f}"
        f"           {dist:8.1f}"
        f"              {speed:6.1f}"
    )

fig, axes = plt.subplots(
    1, 3, figsize=(16.5, 5.3),
    gridspec_kw={"width_ratios": [1.15, 1, 1]}
)
fig.subplots_adjust(
    left=.065, right=.975,
    top=.79, bottom=.19, wspace=.30
)

a, b, c = axes

# A: real GPS path colored by elapsed time.
points = np.column_stack([x/1000, y/1000])
segments = np.stack([points[:-1], points[1:]], axis=1)

norm = Normalize(vmin=0, vmax=duration_min)
lines = LineCollection(
    segments,
    cmap="viridis",
    norm=norm,
    linewidths=2.7
)
lines.set_array((elapsed_min[:-1]+elapsed_min[1:])/2)
a.add_collection(lines)

a.scatter(
    [x[0]/1000], [y[0]/1000],
    s=65, marker="o", color="#247CAB",
    edgecolors="white", zorder=5
)
a.scatter(
    [x[-1]/1000], [y[-1]/1000],
    s=65, marker="s", color="#B7792B",
    edgecolors="white", zorder=5
)

# Plot the two actual December 12 gap endpoints for context.
gx, gy = project(pd.DataFrame([left, right]))
a.scatter(
    gx/1000, gy/1000, marker="x",
    s=65, color="#CF573A",
    linewidths=2, zorder=6
)

pad = max(0.08, 0.1 * max(np.ptp(x), np.ptp(y)) / 1000)
a.set_xlim(x.min()/1000-pad, x.max()/1000+pad)
a.set_ylim(y.min()/1000-pad, y.max()/1000+pad)
a.set_aspect("equal", adjustable="box")
a.set_xlabel("East–west (km)")
a.set_ylabel("North–south (km)")
a.set_title("A  Recorded route over time",
            loc="left", fontsize=12, weight="bold")

colorbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap="viridis"),
    ax=a, fraction=.045, pad=.025
)
colorbar.set_label("Elapsed time (min)")

# B: cumulative GPS-derived path length.
b.plot(
    elapsed_min, cumulative_m/1000,
    color="#267EA9", linewidth=2.4
)
b.set_xlim(0, duration_min)
b.set_ylim(bottom=0)
b.set_xlabel("Elapsed time (min)")
b.set_ylabel("Cumulative path length (km)")
b.set_title("B  Cumulative traveled distance",
            loc="left", fontsize=12, weight="bold")

# C: distance covered in fixed two-minute intervals.
centers = (bin_edges[:-1]+bin_edges[1:])/2

c.bar(
    centers, bin_distances,
    width=bin_durations*.82,
    color="#328A7A", alpha=.85
)
c.set_xlim(0, duration_min)
c.set_ylim(bottom=0)
c.set_xlabel("Elapsed time (min)")
c.set_ylabel("Distance per interval (m)")
c.set_title("C  Movement by two-minute interval",
            loc="left", fontsize=12, weight="bold")

for ax in (a, b, c):
    ax.grid(color="#E7EDF0", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

fig.suptitle(
    "AMO327 | December 7 matched-route movement timing",
    fontsize=18, weight="bold", y=.965
)

fig.text(
    .5, .055,
    "Original GPS observations only. Path distance includes "
    "GPS measurement variation; no stop classification applied.",
    ha="center", fontsize=9.5, color="#647B89"
)

output = (
    Path.home()
    / "cortex2_visualizations/outputs"
    / "AMO327_Dec7_reverse_route_timing.png"
)

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=210, facecolor="white")
plt.close(fig)

print("\nSaved:", output)
