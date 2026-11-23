"""Paired temporal-resampling benchmark for saved AMO327 trip donors.

Same gap, same donor, same endpoints; only the synthetic time grid changes.
Full transformed donor geometry is the reference, NOT true missing travel.
"""

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig, _bridge, EARTH_RADIUS_M,
)
from cortex2_engine.features.gps_pipeline_qc import (
    GPSQCConfig, clean_gps_fixes,
)

sys.path.insert(0, str(Path.cwd() / "workflows"))
from run_gps_imputed_clinical_features import (
    build_participant_donors, RAW, REQUESTED,
)

ROOT = (
    Path.home() / "cortex2_visualizations/outputs"
    / "AMO327_all_gap_resampling"
)
OUT = ROOT / "trip_resampling_comparison"
OUT.mkdir(parents=True, exist_ok=True)

INTERVALS_S = [60, 30, 15, 10, 5]
KEYS = ["clinical_window_id", "gap_start_ms", "gap_end_ms"]

saved = pd.read_csv(ROOT / "all_imputation_draws.csv")
saved = saved.loc[saved.donor_type.eq("trip")].copy()

if saved.empty:
    raise RuntimeError("No saved trip-donor reconstructions.")

cfg = ImputationConfig(
    m=20, stay_compatibility_multiplier=1.0
)


def path_length(x, y):
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def metric_coordinates(frame, lat0, lon0):
    r = EARTH_RADIUS_M
    coslat = np.cos(np.deg2rad(lat0))
    x = r * np.deg2rad(
        frame.longitude.to_numpy(float) - lon0
    ) * coslat
    y = r * np.deg2rad(
        frame.latitude.to_numpy(float) - lat0
    )
    return x, y


def full_transformed_donor(donor, left, right):
    """Duplicate the geometry transformation in _bridge, without resampling."""
    d = donor.reset_index(drop=True)

    lat0, lon0 = float(left.latitude), float(left.longitude)
    x, y = metric_coordinates(d, lat0, lon0)
    x -= x[0]
    y -= y[0]

    t = d.timestamp.to_numpy(dtype=float)
    tau = (t - t[0]) / (t[-1] - t[0])

    xe, ye = metric_coordinates(
        pd.DataFrame([right]), lat0, lon0
    )
    target = np.array([xe[0], ye[0]], dtype=float)
    source = np.array([x[-1], y[-1]], dtype=float)

    tn = float(np.linalg.norm(target))
    dn = float(np.linalg.norm(source))
    threshold = max(
        1.0, float(left.accuracy + right.accuracy)
    )

    if tn <= threshold:
        if dn > threshold:
            raise ValueError("Incompatible near-zero donor")
        x = x + tau * (target[0] - x[-1])
        y = y + tau * (target[1] - y[-1])
    else:
        if dn <= 1e-6:
            raise ValueError("Zero donor endpoint displacement")

        angle = (
            np.arctan2(target[1], target[0])
            - np.arctan2(source[1], source[0])
        )
        scale = tn / dn

        if not np.isfinite(scale) or scale > 4.0:
            raise ValueError("Invalid donor scaling")

        cc, ss = np.cos(angle), np.sin(angle)
        xx = scale * (cc*x - ss*y)
        yy = scale * (ss*x + cc*y)
        x, y = xx, yy

    return path_length(x, y)


def reconstructed_length(donor, left, right, seconds):
    # Use the unmodified production bridge function.
    local_cfg = replace(
        cfg, max_supported_gap_ms=int(seconds * 1000)
    )
    piece = _bridge(left, right, donor, local_cfg)

    if piece is None or piece.empty:
        raise RuntimeError(
            f"Bridge failed with interval {seconds}s"
        )

    completed = pd.concat(
        [pd.DataFrame([left]), piece, pd.DataFrame([right])],
        ignore_index=True
    )

    x, y = metric_coordinates(
        completed,
        float(left.latitude),
        float(left.longitude)
    )
    return path_length(x, y), len(piece)


requested = pq.read_table(
    REQUESTED,
    filters=[("participant_id", "=", "AMO327")]
).to_pandas()

raw = pq.read_table(
    RAW,
    filters=[("participant_id", "=", "AMO327")]
).to_pandas()

records = []
n_verified = 0

for window_id, days in requested.groupby(
    "clinical_window_id", sort=False
):
    subset = saved.loc[
        saved.clinical_window_id.astype(str).eq(str(window_id))
    ]

    if subset.empty:
        continue

    days = days.drop_duplicates("date").sort_values("start_ms")
    start = int(days.start_ms.min())
    end = int(days.end_ms.max())

    inside = raw.loc[
        raw.timestamp.ge(start) & raw.timestamp.lt(end),
        ["timestamp", "latitude", "longitude", "accuracy"]
    ].copy()

    clean = clean_gps_fixes(
        inside, GPSQCConfig(max_gap_ms=60_000)
    )
    donors, stays, _ = build_participant_donors(clean)

    indexed = clean.set_index("timestamp", drop=False)

    for row in subset.itertuples(index=False):
        left = indexed.loc[int(row.gap_start_ms)]
        right = indexed.loc[int(row.gap_end_ms)]
        donor = donors[int(row.donor_index)]

        full_m = full_transformed_donor(donor, left, right)

        # Verify we recovered the same donor and geometry
        # as the previous saved experiment.
        if not np.isclose(
            full_m, row.full_transformed_m, atol=1.0, rtol=0
        ):
            raise RuntimeError(
                f"Transformed donor mismatch: {row.gap_start_ms}"
            )

        for seconds in INTERVALS_S:
            distance_m, n_synthetic = reconstructed_length(
                donor, left, right, seconds
            )

            if seconds == 60:
                if not np.isclose(
                    distance_m, row.resampled_m,
                    atol=1.0, rtol=0
                ):
                    raise RuntimeError(
                        f"60s baseline mismatch: {row.gap_start_ms}"
                    )
                n_verified += 1

            records.append({
                "clinical_window_id": str(window_id),
                "gap_start_ms": int(row.gap_start_ms),
                "gap_end_ms": int(row.gap_end_ms),
                "imputation": int(row.imputation),
                "donor_index": int(row.donor_index),
                "duration_min": float(row.duration_min),
                "endpoint_distance_m": float(
                    row.endpoint_distance_m
                ),
                "interval_s": seconds,
                "full_transformed_m": full_m,
                "reconstructed_m": distance_m,
                "n_synthetic": n_synthetic,
                "loss_pct": (
                    100 * (full_m-distance_m) / full_m
                    if full_m > 0 else np.nan
                ),
                "route_directness": (
                    float(row.endpoint_distance_m) / full_m
                    if full_m > 0 else np.nan
                ),
            })

results = pd.DataFrame(records)

# Each draw is the same saved trip-donor selection
# evaluated at each of the five resolutions.
results.to_csv(
    OUT / "trip_resampling_all_draws.csv", index=False
)

# Gap-level medians prevent repeated draws being treated
# as independent missing intervals.
gap_resolution = results.groupby(
    KEYS + ["interval_s"], as_index=False
).agg(
    median_loss_pct=("loss_pct", "median"),
    median_distance_m=("reconstructed_m", "median"),
    median_full_m=("full_transformed_m", "median"),
    median_synthetic_points=("n_synthetic", "median"),
    median_directness=("route_directness", "median"),
    n_draws=("imputation", "size"),
)

gap_resolution.to_csv(
    OUT / "trip_resampling_gap_level.csv", index=False
)

summary = gap_resolution.groupby("interval_s").agg(
    n_gaps=("median_loss_pct", "size"),
    median_loss_pct=("median_loss_pct", "median"),
    mean_loss_pct=("median_loss_pct", "mean"),
    p90_loss_pct=(
        "median_loss_pct", lambda s: s.quantile(.90)
    ),
    gaps_loss_gt20_pct=(
        "median_loss_pct", lambda s: 100*(s > 20).mean()
    ),
    median_synthetic_points=("median_synthetic_points", "median"),
).reset_index()

summary.to_csv(
    OUT / "trip_resampling_summary.csv", index=False
)

# All gaps are paired: compute within-gap change vs 60s.
wide = gap_resolution.pivot(
    index=KEYS, columns="interval_s",
    values="median_loss_pct"
)

if wide[INTERVALS_S].isna().any().any():
    raise RuntimeError("Some paired resolutions are missing.")

paired = wide.reset_index().copy()
paired["improvement_60_to_5_pp"] = paired[60] - paired[5]
paired.to_csv(
    OUT / "trip_resampling_paired_gaps.csv", index=False
)

high_loss = paired.loc[paired[60] > 20].copy()
high_loss.to_csv(
    OUT / "high_loss_trip_gaps.csv", index=False
)

print("\nAMO327 — TRIP DONOR RESAMPLING")
print("=" * 55)
print("Verified original 60s reconstruction draws:", n_verified)
print("Unique trip-donor gaps:", len(paired))

print("\nPER-GAP RESULTS BY RESAMPLING INTERVAL")
print(summary.round(2).to_string(index=False))

print("\nPAIRED IMPROVEMENT: 60s TO 5s")
print(
    "Median reduction in loss:",
    f"{paired.improvement_60_to_5_pp.median():.2f} percentage points"
)
print(
    "Gaps with reduced loss:",
    f"{100*(paired.improvement_60_to_5_pp > 0).mean():.1f}%"
)
print(
    "High-loss gaps (>20% at 60s):", len(high_loss)
)
if len(high_loss):
    print(
        "Their median loss at 5s:",
        f"{high_loss[5].median():.2f}%"
    )

# Figure: paired, gap-weighted comparison.
fig, (a, b) = plt.subplots(
    1, 2, figsize=(12, 4.7)
)
fig.subplots_adjust(
    left=.08, right=.97, top=.82,
    bottom=.16, wspace=.30
)

plot_summary = summary.sort_values("interval_s", ascending=False)

a.plot(
    plot_summary.interval_s,
    plot_summary.median_loss_pct,
    "-o", color="#267FAD", linewidth=2.4,
    markersize=7
)
a.set_xticks(INTERVALS_S)
a.invert_xaxis()
a.set_xlabel("Resampling interval (seconds)")
a.set_ylabel("Median path-length loss (%)")
a.set_title(
    "A  Geometry retention across resolutions",
    loc="left", fontweight="bold", fontsize=12
)

b.scatter(
    paired[60], paired[5], s=38,
    color="#238B7A", alpha=.8,
    edgecolors="white", linewidths=.5
)

max_val = max(
    float(paired[60].max()),
    float(paired[5].max()),
    1.0
)
b.plot(
    [0, max_val], [0, max_val],
    "--", color="#8797A3", linewidth=1
)
b.set_xlim(0, max_val * 1.04)
b.set_ylim(0, max_val * 1.04)
b.set_aspect("equal", adjustable="box")
b.set_xlabel("Path loss at 60s (%)")
b.set_ylabel("Path loss at 5s (%)")
b.set_title(
    "B  Paired comparison across gaps",
    loc="left", fontweight="bold", fontsize=12
)

for ax in (a, b):
    ax.grid(color="#E7ECEF", linewidth=.7)
    ax.set_axisbelow(True)

fig.suptitle(
    "AMO327 | Does denser resampling preserve donor-trip geometry?",
    fontsize=15.5, fontweight="bold", y=.965
)

fig.text(
    .5, .05,
    "Same observed gap and donor at every resolution; "
    "loss is relative to transformed donor geometry, not true travel.",
    ha="center", fontsize=9, color="#617786"
)

figure_path = OUT / "trip_resampling_resolution_comparison.png"
fig.savefig(figure_path, dpi=210, facecolor="white")
plt.close(fig)

print("\nSaved:", OUT)
print("Figure:", figure_path)
