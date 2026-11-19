from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cortex2_engine.features.gps_pipeline_qc import (
    GPSQCConfig, clean_gps_fixes
)
from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig, impute_timelines, _bridge, _haversine,
    find_long_gaps, EARTH_RADIUS_M
)

sys.path.insert(0, str(Path.cwd() / "workflows"))
from run_gps_imputed_clinical_features import (
    build_participant_donors, RAW, REQUESTED
)

PARTICIPANT = "AMO327"
OUT = (
    Path.home()
    / "cortex2_visualizations/outputs"
    / "AMO327_all_gap_resampling"
)
OUT.mkdir(parents=True, exist_ok=True)

cfg = ImputationConfig(
    m=20,
    stay_compatibility_multiplier=1.0
)
qc = GPSQCConfig(max_gap_ms=cfg.max_supported_gap_ms)

def distance(x, y):
    return float(np.hypot(np.diff(x), np.diff(y)).sum())

def transform_full(donor, left, right):
    """Reproduce _bridge's full-resolution donor transformation."""
    d = donor.sort_values("timestamp").reset_index(drop=True)
    lat0 = float(left.latitude)
    lon0 = float(left.longitude)
    coslat = np.cos(np.deg2rad(lat0))
    R = EARTH_RADIUS_M

    dt = float(right.timestamp - left.timestamp)
    donor_dt = float(d.timestamp.iloc[-1] - d.timestamp.iloc[0])
    tau = (
        (d.timestamp.to_numpy(float) - float(d.timestamp.iloc[0]))
        / donor_dt
    )

    x = R * np.deg2rad(d.longitude.to_numpy(float) - lon0) * coslat
    y = R * np.deg2rad(d.latitude.to_numpy(float) - lat0)
    x -= x[0]
    y -= y[0]

    xe = R * np.deg2rad(float(right.longitude) - lon0) * coslat
    ye = R * np.deg2rad(float(right.latitude) - lat0)

    donor_end = np.array([x[-1], y[-1]])
    target_end = np.array([xe, ye])
    dn = float(np.linalg.norm(donor_end))
    tn = float(np.linalg.norm(target_end))

    threshold = max(
        1.0, float(left.accuracy + right.accuracy)
    )

    if tn <= threshold:
        if dn > threshold:
            raise ValueError("Incompatible near-zero donor")
        x = x + tau * (xe - x[-1])
        y = y + tau * (ye - y[-1])
    else:
        if dn <= 1e-6:
            raise ValueError("Zero donor displacement")
        angle = (
            np.arctan2(ye, xe)
            - np.arctan2(donor_end[1], donor_end[0])
        )
        scale = tn / dn
        if not np.isfinite(scale) or scale > 4:
            raise ValueError("Invalid donor scaling")
        c, s = np.cos(angle), np.sin(angle)
        x, y = (
            scale * (c*x - s*y),
            scale * (s*x + c*y)
        )

    return tau, x, y

def evaluate(donor, left, right):
    reconstructed = _bridge(left, right, donor, cfg)
    if reconstructed is None:
        raise RuntimeError("Saved donor could not be reconstructed")

    tau, x, y = transform_full(donor, left, right)
    full_m = distance(x, y)

    # Measure the final generated trajectory including observed endpoints.
    R = EARTH_RADIUS_M
    lat0 = float(left.latitude)
    lon0 = float(left.longitude)
    coslat = np.cos(np.deg2rad(lat0))

    f = pd.concat([
        pd.DataFrame([left]),
        reconstructed,
        pd.DataFrame([right])
    ], ignore_index=True)

    fx = R * np.deg2rad(f.longitude.to_numpy(float) - lon0) * coslat
    fy = R * np.deg2rad(f.latitude.to_numpy(float) - lat0)
    final_m = distance(fx, fy)

    # Verify that actual reconstructed positions match temporal
    # interpolation of the full transformed donor.
    t = f.timestamp.to_numpy(float)
    grid = (t - t[0]) / (t[-1] - t[0])

    px = np.interp(grid, tau, x)
    py = np.interp(grid, tau, y)

    mismatch = float(np.max(np.hypot(px-fx, py-fy)))

    duration_min = (
        float(right.timestamp - left.timestamp) / 60000
    )

    return dict(
        full_transformed_m=full_m,
        resampled_m=final_m,
        lost_m=full_m-final_m,
        loss_pct=100*(full_m-final_m)/full_m if full_m > 0 else np.nan,
        reconstructed_speed_kmh=(
            final_m / (duration_min*60) * 3.6
        ),
        n_original_donor_fixes=len(donor),
        n_imputed_fixes=len(reconstructed),
        max_interpolation_mismatch_m=mismatch
    )

requested = pq.read_table(
    REQUESTED,
    filters=[("participant_id", "=", PARTICIPANT)]
).to_pandas()

raw = pq.read_table(
    RAW,
    filters=[("participant_id", "=", PARTICIPANT)]
).to_pandas()

if requested.empty or raw.empty:
    raise RuntimeError("Missing AMO327 input data")

rows = []
window_summaries = []

for window_id, days in requested.groupby(
    "clinical_window_id", sort=False
):
    days = days.drop_duplicates("date").sort_values("start_ms")
    start_ms = int(days.start_ms.min())
    end_ms = int(days.end_ms.max())

    inside = raw.loc[
        (raw.timestamp >= start_ms)
        & (raw.timestamp < end_ms),
        ["timestamp", "latitude", "longitude", "accuracy"]
    ].copy()

    clean = clean_gps_fixes(inside, qc)

    if len(clean) < 2:
        continue

    donors, stays, _ = build_participant_donors(clean)

    # Exact multiple-imputation algorithm and configuration.
    timelines, diagnostics = impute_timelines(
        clean, donors,
        config=cfg,
        stay_donors=stays,
        window_start_ms=start_ms,
        window_end_ms=end_ms
    )

    # One lookup for the actual observed gap boundary points.
    indexed = clean.set_index("timestamp", drop=False)

    observed_gaps = find_long_gaps(
        clean, cfg.max_supported_gap_ms
    )

    success = 0
    failures = 0

    for rec in diagnostics.itertuples(index=False):
        if not bool(rec.imputed):
            failures += 1
            continue

        if rec.gap_type not in ("trip", "stay"):
            continue

        donor_list = donors if rec.gap_type == "trip" else stays
        idx = int(rec.donor_index)

        left = indexed.loc[int(rec.start_ms)]
        right = indexed.loc[int(rec.end_ms)]
        donor = donor_list[idx]

        result = evaluate(donor, left, right)

        if result["max_interpolation_mismatch_m"] > 2.0:
            raise RuntimeError(
                f"Interpolation mismatch in window {window_id}, "
                f"gap {rec.start_ms}"
            )

        rows.append(dict(
            clinical_window_id=str(window_id),
            gap_start_ms=int(rec.start_ms),
            gap_end_ms=int(rec.end_ms),
            gap_start_utc=str(pd.to_datetime(
                rec.start_ms, unit="ms", utc=True
            )),
            imputation=int(rec.imputation),
            donor_type=rec.gap_type,
            donor_index=idx,
            candidates=int(rec.n_candidate_donors),
            duration_min=float(rec.gap_ms)/60000,
            endpoint_distance_m=float(_haversine(
                left.latitude, left.longitude,
                right.latitude, right.longitude
            )),
            **result
        ))
        success += 1

    window_summaries.append(dict(
        window=str(window_id),
        unique_observed_gaps=len(observed_gaps),
        trip_donors=len(donors),
        stay_donors=len(stays),
        successful_draws=success,
        unsuccessful_diagnostics=failures
    ))

if not rows:
    raise RuntimeError("No successful donor reconstructions found")

draws = pd.DataFrame(rows)

# Summarize by UNIQUE gap, not just individual imputation draws.
gap_keys = [
    "clinical_window_id", "gap_start_ms", "gap_end_ms"
]

gaps = draws.groupby(gap_keys, as_index=False).agg(
    start_utc=("gap_start_utc", "first"),
    duration_min=("duration_min", "first"),
    n_successful_draws=("imputation", "count"),
    n_distinct_donors=("donor_index", "nunique"),
    median_loss_pct=("loss_pct", "median"),
    mean_loss_pct=("loss_pct", "mean"),
    min_loss_pct=("loss_pct", "min"),
    max_loss_pct=("loss_pct", "max"),
    median_full_m=("full_transformed_m", "median"),
    median_resampled_m=("resampled_m", "median"),
    median_speed_kmh=("reconstructed_speed_kmh", "median")
)

draws.to_csv(OUT / "all_imputation_draws.csv", index=False)
gaps.to_csv(OUT / "unique_gap_summary.csv", index=False)
pd.DataFrame(window_summaries).to_csv(
    OUT / "window_counts.csv", index=False
)

def fmt(x):
    return f"{x:.1f}%"

print("\nAMO327 — DONOR RESAMPLING VALIDATION")
print("=" * 52)
print("Clinical windows:", len(window_summaries))
print("Unique gaps with successful reconstruction:", len(gaps))
print("Successful reconstruction draws:", len(draws))
print()
print("PATH LENGTH LOST — PER UNIQUE GAP")
print("Median:", fmt(gaps.median_loss_pct.median()))
print("Mean:", fmt(gaps.median_loss_pct.mean()))
print("90th percentile:", fmt(gaps.median_loss_pct.quantile(.90)))
print("Maximum:", fmt(gaps.median_loss_pct.max()))
print()
print("PERCENT OF GAPS WITH MEDIAN LOSS")
for cutoff in (1, 5, 10, 20):
    value = 100 * (gaps.median_loss_pct > cutoff).mean()
    print(f"  >{cutoff:2d}% loss: {value:.1f}%")
print()
print("MEDIAN RECONSTRUCTED SPEED")
print(f"{gaps.median_speed_kmh.median():.1f} km/h")
print()
print("FIVE GAPS WITH LARGEST MEDIAN LOSS")
print(gaps.nlargest(5, "median_loss_pct")[
    ["start_utc", "duration_min", "n_successful_draws",
     "median_loss_pct", "median_speed_kmh"]
].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# Scientific summary figure.
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.subplots_adjust(
    left=.08, right=.97, top=.81,
    bottom=.17, wspace=.29
)

a, b = axes

# Each faint point is an imputation; solid points summarize gaps.
a.scatter(
    draws.duration_min, draws.loss_pct,
    s=11, color="#A2B7C6", alpha=.24,
    label="Individual imputations"
)
a.scatter(
    gaps.duration_min, gaps.median_loss_pct,
    s=38, color="#176C95", alpha=.9,
    edgecolors="white", linewidths=.5,
    label="Median per gap", zorder=5
)
a.axhline(0, color="#758995", lw=.9, ls="--")
a.set_xlabel("GPS gap duration (minutes)")
a.set_ylabel("Path length lost during resampling (%)")
a.set_title(
    "A  Loss by gap duration", loc="left",
    fontsize=12, weight="bold"
)
a.legend(fontsize=8, frameon=False)

# Show distribution of median loss across unique gaps.
b.hist(
    gaps.median_loss_pct,
    bins=min(16, max(5, int(np.sqrt(len(gaps))))),
    color="#278779",
    edgecolor="white",
    linewidth=1
)
b.axvline(
    gaps.median_loss_pct.median(),
    color="#D18A34", lw=2, ls="--",
    label=f"Median: {gaps.median_loss_pct.median():.1f}%"
)
b.set_xlabel("Median path length lost per gap (%)")
b.set_ylabel("Number of unique gaps")
b.set_title(
    "B  Across reconstructed gaps", loc="left",
    fontsize=12, weight="bold"
)
b.legend(fontsize=9, frameon=False)

for ax in axes:
    ax.grid(color="#E9EEF1", lw=.6)
    ax.set_axisbelow(True)

fig.suptitle(
    "AMO327 | Donor reconstruction: effect of temporal resampling",
    fontsize=16, weight="bold", y=.96
)

fig.text(
    .5, .045,
    "Loss compares full transformed donor geometry with the "
    "resampled candidate trajectory; it is not reconstruction error.",
    ha="center", fontsize=9.5, color="#587080"
)

fig.savefig(
    OUT / "resampling_across_gaps.png",
    dpi=200, facecolor="white"
)
plt.close(fig)

print("\nResults saved to:", OUT)
