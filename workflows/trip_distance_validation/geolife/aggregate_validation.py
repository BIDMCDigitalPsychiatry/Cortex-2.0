from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("outputs/trip_distance_validation/geolife/full_benchmark")
SHARDS = BASE / "shards"
OUT = BASE / "aggregated"
OUT.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {
    "clean_downsample": "clean_downsample_distance_m",
    "raw_all": "raw_all_distance_m",
    "raw_supported": "raw_supported_distance_m",
    "legacy_supported": "legacy_supported_displacement_m",
    "smooth_0.5": "smooth_0.5_m",
    "smooth_1": "smooth_1_m",
    "smooth_2": "smooth_2_m",
}

CONFIG = [
    "sampling_interval_s",
    "accuracy68_added_m",
]

KEY = [
    "segment_id",
    "sampling_interval_s",
    "accuracy68_added_m",
    "replicate",
]

def metrics(frame):
    e = frame["error_m"].to_numpy(dtype=float)
    ref = frame["reference_distance_m"].to_numpy(dtype=float)

    return pd.Series({
        "n": len(frame),
        "n_segments": frame["segment_id"].nunique(),
        "mae_m": np.mean(np.abs(e)),
        "rmse_m": np.sqrt(np.mean(e**2)),
        "bias_m": np.mean(e),
        "mape_pct": np.mean(
            np.abs(e / ref) * 100
        ),
        "median_ape_pct": np.median(
            np.abs(e / ref) * 100
        ),
    })

def summarize(frame, groups):
    return (
        frame.groupby(
            groups,
            dropna=False,
            observed=True,
        )
        .apply(metrics)
        .reset_index()
    )

files = sorted(SHARDS.glob("*.parquet"))
if not files:
    raise RuntimeError(f"No Parquet shards found in {SHARDS}")

print(f"Loading {len(files)} shards...")
df = pd.concat(
    (pd.read_parquet(p) for p in files),
    ignore_index=True,
)

print(f"Total records: {len(df):,}")
print(f"Unique segments: {df.segment_id.nunique():,}")
print(f"Unique participants: {df.user_id.nunique():,}")

duplicates = df.duplicated(KEY).sum()
if duplicates:
    raise RuntimeError(f"Found {duplicates} duplicate experiment keys")

expected = df.segment_id.nunique() * 4 * 3 * 20
print(f"Expected records under full design: {expected:,}")
if len(df) != expected:
    print("WARNING: Record count differs from full design.")

# Check experimental coverage.
design = (
    df.groupby(
        ["segment_id"] + CONFIG,
        observed=True,
    )
    .agg(n_replicates=("replicate", "nunique"))
    .reset_index()
)
design.to_csv(OUT / "design_coverage.csv", index=False)

print("Replicate counts:")
print(design.n_replicates.value_counts().sort_index().to_string())

# Coverage of each estimator.
coverage_cols = [
    "supported_fraction",
    "smoothing_coverage_0.5",
    "smoothing_coverage_1",
    "smoothing_coverage_2",
]

coverage = (
    df.groupby(CONFIG, observed=True)[coverage_cols]
    .agg(["mean", "median", "min"])
)
coverage.to_csv(OUT / "coverage_summary.csv")

# Convert to long format, preserving paired experiment keys.
id_cols = KEY + [
    "user_id",
    "transportation_mode",
    "reference_distance_m",
    "supported_fraction",
    "smoothing_coverage_0.5",
    "smoothing_coverage_1",
    "smoothing_coverage_2",
]

long = df[id_cols + list(ESTIMATORS.values())].melt(
    id_vars=id_cols,
    value_vars=list(ESTIMATORS.values()),
    var_name="estimator_column",
    value_name="estimated_distance_m",
)

reverse = {v: k for k, v in ESTIMATORS.items()}
long["estimator"] = long["estimator_column"].map(reverse)

# Supported-region estimators cannot automatically be interpreted
# as full-trip distances when coverage is incomplete.
long["evaluation_scope"] = "full_trip"

supported = long.estimator.isin([
    "raw_supported",
    "legacy_supported",
])

long.loc[
    supported &
    (long.supported_fraction < 0.999999),
    "evaluation_scope"
] = "incomplete_coverage"

for name, column in [
    ("smooth_0.5", "smoothing_coverage_0.5"),
    ("smooth_1", "smoothing_coverage_1"),
    ("smooth_2", "smoothing_coverage_2"),
]:
    long.loc[
        (long.estimator == name) &
        (long[column] < 0.999999),
        "evaluation_scope"
    ] = "incomplete_coverage"

# The reference is a whole-trip distance. Only compare estimates
# with this target when full-trip coverage is established.
long = long[
    np.isfinite(long.estimated_distance_m) &
    np.isfinite(long.reference_distance_m) &
    (long.reference_distance_m > 0)
].copy()

long["error_m"] = (
    long.estimated_distance_m -
    long.reference_distance_m
)
long["abs_error_m"] = long.error_m.abs()
long["ape_pct"] = (
    100 * long.abs_error_m /
    long.reference_distance_m
)

long.to_parquet(
    OUT / "validation_records.parquet",
    index=False,
)

valid = long[long.evaluation_scope == "full_trip"].copy()

# Overall metrics, separately for each estimator.
overall = summarize(valid, ["estimator"])
overall.sort_values("mae_m").to_csv(
    OUT / "overall_metrics.csv",
    index=False,
)

# Sampling interval x positional error.
by_configuration = summarize(
    valid,
    CONFIG + ["estimator"],
)
by_configuration.to_csv(
    OUT / "metrics_by_configuration.csv",
    index=False,
)

# Transportation modes.
by_mode = summarize(
    valid,
    ["transportation_mode", "estimator"],
)
by_mode.to_csv(
    OUT / "metrics_by_transportation_mode.csv",
    index=False,
)

# Combined breakdown.
detailed = summarize(
    valid,
    ["transportation_mode"] + CONFIG + ["estimator"],
)
detailed.to_csv(
    OUT / "metrics_detailed.csv",
    index=False,
)

# Paired comparisons: use only experiments for which BOTH
# estimators have a valid full-trip estimate.
wide = valid.pivot(
    index=KEY,
    columns="estimator",
    values="abs_error_m",
)

pairs = []
for baseline in ["raw_all", "legacy_supported"]:
    for candidate in ["smooth_0.5", "smooth_1", "smooth_2"]:
        paired = wide[[baseline, candidate]].dropna().reset_index()

        paired["improvement_m"] = (
            paired[baseline] - paired[candidate]
        )

        pairs.append(
            paired[KEY + ["improvement_m"]].assign(
                baseline=baseline,
                candidate=candidate,
            )
        )

paired = pd.concat(pairs, ignore_index=True)
paired.to_csv(OUT / "paired_improvements.csv", index=False)

# Average replicates within each segment/configuration first,
# avoiding treating repeated noise realizations as independent
# trajectories.
segment_pairs = (
    paired.groupby(
        ["segment_id"] + CONFIG + ["baseline", "candidate"],
        observed=True,
    )
    .agg(mean_improvement_m=("improvement_m", "mean"))
    .reset_index()
)

segment_pairs.to_csv(
    OUT / "segment_level_improvements.csv",
    index=False,
)

paired_summary = (
    segment_pairs.groupby(
        ["baseline", "candidate"] + CONFIG,
        observed=True,
    )
    .agg(
        n_segments=("segment_id", "nunique"),
        mean_improvement_m=("mean_improvement_m", "mean"),
        median_improvement_m=("mean_improvement_m", "median"),
        fraction_segments_improved=(
            "mean_improvement_m",
            lambda x: (x > 0).mean(),
        ),
    )
    .reset_index()
)

paired_summary.to_csv(
    OUT / "paired_improvement_summary.csv",
    index=False,
)

# Track excluded estimates so incomplete coverage is transparent.
scope_counts = (
    long.groupby(
        ["estimator", "evaluation_scope"],
        observed=True,
    )
    .size()
    .rename("n")
    .reset_index()
)
scope_counts.to_csv(OUT / "evaluation_scope_counts.csv", index=False)

print("\n" + "=" * 80)
print("OVERALL VALIDATION METRICS")
print("=" * 80)
print(
    overall.sort_values("mae_m").round(3).to_string(index=False)
)

print("\n" + "=" * 80)
print("EVALUATION SCOPE COUNTS")
print("=" * 80)
print(scope_counts.to_string(index=False))

print("\n" + "=" * 80)
print("AGGREGATION COMPLETED")
print("=" * 80)
print(f"Output directory: {OUT}")
