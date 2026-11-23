"""Small AMO327 masked-trip validation of GPS donor imputation.

Observed withheld GPS is an imperfect reference, not surveyed ground truth.
No production code or original data are modified.
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
    ImputationConfig, _bridge, _haversine, EARTH_RADIUS_M,
)
from cortex2_engine.features.gps_pipeline_qc import (
    GPSQCConfig, clean_gps_fixes,
)

sys.path.insert(0, str(Path.cwd() / "workflows"))
from run_gps_imputed_clinical_features import (
    RAW, REQUESTED, build_participant_donors,
)

PARTICIPANT = "AMO327"
DURATIONS_MIN = (2, 5, 10)
MAX_TESTS_PER_DURATION = 3
RESOLUTIONS_S = (60, 15, 5)

OUT = (
    Path.home() / "cortex2_visualizations/outputs"
    / "AMO327_masked_imputation_pilot"
)
OUT.mkdir(parents=True, exist_ok=True)

cfg = ImputationConfig(m=20, stay_compatibility_multiplier=1.0)
qc = GPSQCConfig(max_gap_ms=60_000)
rng = np.random.default_rng(20261117)


def xy(df, lat0, lon0):
    r = EARTH_RADIUS_M
    x = r * np.cos(np.deg2rad(lat0)) * np.deg2rad(
        df.longitude.to_numpy(float) - lon0
    )
    y = r * np.deg2rad(df.latitude.to_numpy(float) - lat0)
    return x, y


def path_length(df):
    x, y = xy(
        df, float(df.latitude.iloc[0]), float(df.longitude.iloc[0])
    )
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def eligible_donors(donors, left, right):
    """Apply the trip-donor eligibility checks from impute_timelines."""
    dt = float(right.timestamp - left.timestamp)
    endpoint = float(_haversine(
        left.latitude, left.longitude,
        right.latitude, right.longitude
    ))

    eligible = []
    for j, d in enumerate(donors):
        if len(d) < cfg.min_donor_fixes:
            continue

        times = d.timestamp.to_numpy(np.int64)
        if np.diff(times).max(initial=0) > cfg.max_supported_gap_ms:
            continue

        duration = float(times[-1] - times[0])
        if duration <= 0:
            continue

        ratio = dt / duration
        if not (
            1 / cfg.maximum_time_stretch
            <= ratio <= cfg.maximum_time_stretch
        ):
            continue

        donor_endpoint = float(_haversine(
            d.latitude.iloc[0], d.longitude.iloc[0],
            d.latitude.iloc[-1], d.longitude.iloc[-1]
        ))

        if abs(donor_endpoint - endpoint) > cfg.endpoint_distance_tolerance_m:
            continue

        if _bridge(left, right, d, cfg) is not None:
            eligible.append((j, d))

    return eligible


def make_candidates(trips, duration_min):
    """Choose internal gaps from genuinely continuous observed trips."""
    candidates = []
    seen = set()
    target_ms = int(duration_min * 60_000)

    for trip in trips:
        d = trip.sort_values("timestamp").reset_index(drop=True)
        ts = d.timestamp.to_numpy(np.int64)

        if len(ts) < 12:
            continue
        if np.diff(ts).max(initial=0) > 60_000:
            continue
        if ts[-1] - ts[0] < target_ms + 120_000:
            continue

        # Center the artificial gap within the observed trip.
        midpoint = int((ts[0] + ts[-1]) // 2)
        intended_left = midpoint - target_ms // 2
        intended_right = intended_left + target_ms

        li = int(np.argmin(abs(ts - intended_left)))
        ri = int(np.argmin(abs(ts - intended_right)))

        if ri <= li + 3:
            continue
        if li < 2 or ri >= len(ts) - 2:
            continue
        if abs((ts[ri] - ts[li]) - target_ms) > 30_000:
            continue

        key = (int(ts[li]), int(ts[ri]))
        if key in seen:
            continue
        seen.add(key)

        candidates.append((
            d, li, ri,
            float(ts[-1] - ts[0])
        ))

    # Prefer longer continuous, well-observed source trips.
    candidates.sort(key=lambda z: z[3], reverse=True)
    return candidates


requested = pq.read_table(
    REQUESTED, filters=[("participant_id", "=", PARTICIPANT)]
).to_pandas()

raw = pq.read_table(
    RAW, filters=[("participant_id", "=", PARTICIPANT)]
).to_pandas()

if requested.empty or raw.empty:
    raise RuntimeError("AMO327 raw or requested-day data missing.")

# Select one clinical window for a low-cost pilot.
windows = []

for window_id, days in requested.groupby(
    "clinical_window_id", sort=False
):
    days = days.drop_duplicates("date").sort_values("start_ms")
    start, end = int(days.start_ms.min()), int(days.end_ms.max())

    inside = raw.loc[
        raw.timestamp.ge(start) & raw.timestamp.lt(end),
        ["timestamp", "latitude", "longitude", "accuracy"]
    ].copy()

    cleaned = clean_gps_fixes(inside, qc)
    if len(cleaned) < 2:
        continue

    trip_donors, _, _ = build_participant_donors(cleaned)

    potential = sum(
        len(make_candidates(trip_donors, duration))
        for duration in DURATIONS_MIN
    )

    windows.append(
        (potential, str(window_id), cleaned, trip_donors)
    )

if not windows:
    raise RuntimeError("No usable clinical windows.")

windows.sort(key=lambda z: z[0], reverse=True)
_, window_id, observed, full_trips = windows[0]

print("Selected clinical window:", window_id)
print("Observed GPS fixes:", len(observed))
print("Original trip donors:", len(full_trips))

records = []
test_count = 0
comparison_files = []

for duration_min in DURATIONS_MIN:
    candidates = make_candidates(full_trips, duration_min)
    successes = 0

    for original_trip, li, ri, _ in candidates:
        if successes >= MAX_TESTS_PER_DURATION:
            break

        left = original_trip.iloc[li]
        right = original_trip.iloc[ri]
        t0, t1 = int(left.timestamp), int(right.timestamp)

        # Entire observed original trip is excluded as a donor source.
        original_start = int(original_trip.timestamp.iloc[0])
        original_end = int(original_trip.timestamp.iloc[-1])

        # Temporarily remove only the internal GPS observations.
        masked = observed.loc[
            ~observed.timestamp.between(
                t0 + 1, t1 - 1
            )
        ].copy()

        # Ensure endpoint observations remain present.
        if (
            int(masked.timestamp.eq(t0).sum()) != 1
            or int(masked.timestamp.eq(t1).sum()) != 1
        ):
            continue

        truth = original_trip.iloc[li:ri+1].copy()

        if len(truth) < 5:
            continue

        # Rebuild donors AFTER masking to avoid hidden-data leakage.
        masked_trip_donors, _, _ = build_participant_donors(masked)

        # Strict safeguard: no donor can overlap the source trip.
        donor_pool = [
            d for d in masked_trip_donors
            if (
                int(d.timestamp.iloc[-1]) < original_start
                or int(d.timestamp.iloc[0]) > original_end
            )
        ]

        eligible = eligible_donors(donor_pool, left, right)

        if not eligible:
            continue

        # Select once; use the SAME donor at all resolutions.
        chosen_j, donor = eligible[
            int(rng.integers(len(eligible)))
        ]

        reference_m = path_length(truth)
        endpoint_m = float(_haversine(
            left.latitude, left.longitude,
            right.latitude, right.longitude
        ))

        reference_x, reference_y = xy(
            truth, float(left.latitude), float(left.longitude)
        )

        paths = {}
        for resolution in RESOLUTIONS_S:
            local_cfg = replace(
                cfg,
                max_supported_gap_ms=resolution * 1000
            )

            piece = _bridge(
                left, right, donor, local_cfg
            )

            if piece is None:
                raise RuntimeError(
                    "Same donor failed at finer resolution."
                )

            reconstructed = pd.concat([
                pd.DataFrame([left]),
                piece,
                pd.DataFrame([right])
            ], ignore_index=True)

            paths[resolution] = reconstructed.copy()
            predicted_m = path_length(reconstructed)
            rx, ry = xy(
                reconstructed,
                float(left.latitude),
                float(left.longitude)
            )

            # Compare positions at the held-out observation times.
            ref_t = truth.timestamp.to_numpy(float)
            pred_t = reconstructed.timestamp.to_numpy(float)

            est_x = np.interp(ref_t, pred_t, rx)
            est_y = np.interp(ref_t, pred_t, ry)
            point_errors = np.hypot(
                reference_x-est_x, reference_y-est_y
            )

            records.append(dict(
                clinical_window_id=window_id,
                test_id=test_count,
                duration_requested_min=duration_min,
                actual_duration_min=(t1-t0)/60000,
                gap_start_ms=t0,
                gap_end_ms=t1,
                n_withheld_fixes=len(truth)-2,
                n_eligible_donors=len(eligible),
                selected_donor_index=chosen_j,
                resolution_s=resolution,
                reference_distance_m=reference_m,
                endpoint_distance_m=endpoint_m,
                reconstructed_distance_m=predicted_m,
                distance_abs_error_m=abs(
                    predicted_m-reference_m
                ),
                distance_bias_m=predicted_m-reference_m,
                median_position_error_m=float(
                    np.median(point_errors)
                ),
                n_synthetic_fixes=len(piece),
            ))


        # Save ONE panel-C-style comparison for this masked gap:
        # truth + straight-line + 60s + 15s + 5s.
        compare_dir = OUT / "shape_comparison_single_gap"
        compare_dir.mkdir(parents=True, exist_ok=True)

        lat0, lon0 = float(left.latitude), float(left.longitude)
        tx, ty = xy(truth, lat0, lon0)

        coords = {r: xy(df, lat0, lon0) for r, df in paths.items()}

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 6.2))

        # True withheld path
        ax.plot(tx, ty, color="#7F8C8D", lw=3.0, label="Original withheld path", zorder=2)
        ax.scatter(tx, ty, color="#7F8C8D", s=10, alpha=0.5, zorder=2)

        # Straight-line baseline
        ax.plot(
            [tx[0], tx[-1]], [ty[0], ty[-1]],
            "--", color="#D58B32", lw=2.4, label="Straight-line baseline", zorder=3
        )

        # Reconstructions
        style_map = {
            60: dict(color="#C44E52", lw=2.2, ls='-', label="60 s reconstruction"),
            15: dict(color="#55A868", lw=2.2, ls='-', label="15 s reconstruction"),
            5:  dict(color="#4C72B0", lw=2.6, ls='-', label="5 s reconstruction"),
        }

        for r in [60, 15, 5]:
            if r in coords:
                px, py = coords[r]
                ax.plot(px, py, **style_map[r], zorder=4)
                ax.scatter(px[1:-1], py[1:-1], s=12, color=style_map[r]["color"], alpha=.75, zorder=4)

        # Endpoints
        ax.scatter(
            [tx[0], tx[-1]], [ty[0], ty[-1]],
            s=62, color="#236A9A", edgecolors="white", linewidth=1.0,
            label="Observed gap endpoints", zorder=5
        )

        all_x = [tx]
        all_y = [ty]
        for r in coords:
            all_x.append(coords[r][0])
            all_y.append(coords[r][1])
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        pad = max(20.0, 0.08 * max(np.ptp(all_x), np.ptp(all_y), 1.0))
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)

        ax.set_xlabel("East–west (m)")
        ax.set_ylabel("North–south (m)")

        gap_min = (t1 - t0) / 60000
        ax.set_title(
            f"Masked gap {test_count+1}: original path vs reconstructions\n"
            f"{gap_min:.1f} min hidden interval",
            fontsize=13, weight="bold"
        )

        ax.text(
            0.02, 0.02,
            f"Window: {window_id} | eligible donors: {len(eligible)}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#D0D7DE", pad=3)
        )

        ax.legend(loc="best", fontsize=9, frameon=True)
        fig.tight_layout()

        out_path = compare_dir / f"gap_{test_count+1:02d}_comparison.png"
        fig.savefig(out_path, dpi=180, facecolor="white")
        plt.close(fig)
        comparison_files.append(out_path)

        successes += 1
        test_count += 1

    print(
        f"{duration_min}-minute gaps: "
        f"{successes} successful tests "
        f"({len(candidates)} source candidates)"
    )

if not records:
    raise RuntimeError(
        "No eligible masked reconstructions. "
        "Inputs were not modified; inspect donor availability."
    )

result = pd.DataFrame(records)
result.to_csv(
    OUT / "masked_gap_results.csv", index=False
)

summary = result.groupby("resolution_s").agg(
    n_tests=("test_id", "nunique"),
    median_distance_abs_error_m=(
        "distance_abs_error_m", "median"
    ),
    mean_distance_abs_error_m=(
        "distance_abs_error_m", "mean"
    ),
    median_position_error_m=(
        "median_position_error_m", "median"
    ),
    median_synthetic_fixes=(
        "n_synthetic_fixes", "median"
    ),
).reset_index()

summary.to_csv(
    OUT / "masked_gap_summary.csv", index=False
)

print("\nAMO327 MASKED-GAP VALIDATION")
print("=" * 52)
print("Completed masked tests:", result.test_id.nunique())
print("\nRESULTS BY RESOLUTION")
print(summary.round(2).to_string(index=False))

baseline = (
    result.drop_duplicates("test_id")
    .assign(
        endpoint_abs_error_m=lambda d: abs(
            d.endpoint_distance_m-d.reference_distance_m
        )
    )
)

print(
    "\nStraight-line baseline median absolute error:",
    round(baseline.endpoint_abs_error_m.median(), 2),
    "m"
)

# Paired figure: each resolution uses exactly the same tests.
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
fig.subplots_adjust(
    left=.085, right=.97, top=.82,
    bottom=.18, wspace=.29
)

s = summary.sort_values("resolution_s", ascending=False)

axes[0].plot(
    s.resolution_s,
    s.median_distance_abs_error_m,
    "-o", linewidth=2.2,
)
axes[0].axhline(
    baseline.endpoint_abs_error_m.median(),
    linestyle="--", color="gray",
    label="Endpoint-distance baseline"
)
axes[0].set_xticks(RESOLUTIONS_S)
axes[0].invert_xaxis()
axes[0].set_xlabel("Resampling interval (seconds)")
axes[0].set_ylabel("Median distance absolute error (m)")
axes[0].set_title(
    "A  Held-out distance error",
    loc="left", weight="bold"
)
axes[0].legend(frameon=False, fontsize=9)

axes[1].plot(
    s.resolution_s,
    s.median_position_error_m,
    "-o", linewidth=2.2,
)
axes[1].set_xticks(RESOLUTIONS_S)
axes[1].invert_xaxis()
axes[1].set_xlabel("Resampling interval (seconds)")
axes[1].set_ylabel("Median positional deviation (m)")
axes[1].set_title(
    "B  Agreement with withheld GPS",
    loc="left", weight="bold"
)

for ax in axes:
    ax.grid(alpha=.20)
    ax.set_axisbelow(True)

fig.suptitle(
    "AMO327 | Masked observed-trip reconstruction",
    fontsize=15, weight="bold", y=.97
)

fig.text(
    .5, .055,
    "Reference: withheld observed GPS trajectory; "
    "same donor selected across resolutions.",
    ha="center", fontsize=9, color="#607586"
)

fig.savefig(
    OUT / "masked_gap_validation.png",
    dpi=210, facecolor="white"
)
plt.close(fig)

print("\nResults saved:", OUT)


print("Saved single-gap comparison figures:", len(comparison_files))
for p in comparison_files:
    print("FIGURE:", p)
