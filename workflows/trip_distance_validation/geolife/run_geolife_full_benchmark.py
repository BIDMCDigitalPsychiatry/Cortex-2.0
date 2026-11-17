from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cortex2_engine.features.gps_trip_features import (
    compute_smoothed_trip_path_distance,
)


SAMPLE_FILE = Path(
    "outputs/trip_distance_validation/geolife/"
    "validation_sample/validation_sample.csv"
)

OUT_DIR = Path(
    "outputs/trip_distance_validation/geolife/"
    "full_benchmark"
)

SHARD_DIR = OUT_DIR / "shards"

SMOKE_SCRIPT = Path(
    "workflows/trip_distance_validation/geolife/"
    "run_sampling_degradation_benchmark.py"
)

SAMPLING_INTERVALS_S = [
    5,
    10,
    30,
    60,
]

# Controlled ADDITIONAL positional-error conditions.
# These are not claims about native GeoLife GPS accuracy.
ACCURACY68_LEVELS_M = [
    5.0,
    15.0,
    30.0,
]

ACCELERATION_NOISE_DENSITY_GRID = [
    0.5,
    1.0,
    2.0,
]

N_REPLICATES = 20

BASE_SEED = 20261116


def load_smoke_helpers():
    spec = importlib.util.spec_from_file_location(
        "geolife_smoke_helpers",
        SMOKE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load helper script: {SMOKE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_reference(
    path: Path,
) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(
            path
        )
    else:
        df = pd.read_csv(
            path
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    return (
        df.dropna(
            subset=[
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def deterministic_seed(
    *parts,
) -> int:
    text = "|".join(
        str(x)
        for x in (
            BASE_SEED,
            *parts,
        )
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def run_segment(
    item,
    helpers,
) -> pd.DataFrame:
    reference = load_reference(
        Path(item.output_file)
    )

    if len(reference) < 2:
        raise ValueError(
            f"{item.segment_id}: fewer than 2 reference fixes."
        )

    reference_distance_m = (
        helpers.raw_path_distance_m(
            reference
        )
    )

    duration_s = (
        reference["timestamp"].iloc[-1]
        - reference["timestamp"].iloc[0]
    ).total_seconds()

    rows = []

    for interval_s in (
        SAMPLING_INTERVALS_S
    ):
        sampled_clean = (
            helpers.deterministic_downsample(
                reference,
                interval_s,
            )
        )

        clean_downsample_distance_m = (
            helpers.raw_path_distance_m(
                sampled_clean
            )
        )

        clean_sampling_bias_pct = (
            (
                clean_downsample_distance_m
                - reference_distance_m
            )
            / reference_distance_m
            * 100.0
            if reference_distance_m > 0
            else np.nan
        )

        for accuracy68_m in (
            ACCURACY68_LEVELS_M
        ):
            for replicate in range(
                N_REPLICATES
            ):
                seed = deterministic_seed(
                    item.segment_id,
                    interval_s,
                    accuracy68_m,
                    replicate,
                )

                rng = (
                    np.random.default_rng(
                        seed
                    )
                )

                degraded = (
                    helpers.add_controlled_noise(
                        sampled_clean,
                        accuracy68_m,
                        rng,
                    )
                )

                raw_all_m = (
                    helpers.raw_path_distance_m(
                        degraded
                    )
                )

                edges = (
                    helpers.build_cortex_edges(
                        degraded
                    )
                )

                raw_supported_m = (
                    helpers.supported_raw_distance_m(
                        edges
                    )
                )

                support = (
                    helpers.supported_fraction(
                        edges
                    )
                )

                legacy_m = (
                    helpers.legacy_displacement_m(
                        edges
                    )
                )

                start_ms = int(
                    degraded["timestamp"]
                    .astype("int64")
                    .iloc[0]
                    // 1_000_000
                )

                end_ms = (
                    int(
                        degraded["timestamp"]
                        .astype("int64")
                        .iloc[-1]
                        // 1_000_000
                    )
                    + 1
                )

                row = {
                    "segment_id":
                        item.segment_id,
                    "user_id":
                        str(item.user_id),
                    "transportation_mode":
                        item.transportation_mode,

                    "reference_n_points":
                        len(reference),
                    "reference_duration_s":
                        duration_s,
                    "reference_distance_m":
                        reference_distance_m,

                    "sampling_interval_s":
                        interval_s,

                    "clean_sample_n_points":
                        len(sampled_clean),
                    "clean_downsample_distance_m":
                        clean_downsample_distance_m,
                    "clean_sampling_bias_pct":
                        clean_sampling_bias_pct,

                    "accuracy68_added_m":
                        accuracy68_m,
                    "replicate":
                        replicate,

                    "supported_fraction":
                        support,

                    "raw_all_distance_m":
                        raw_all_m,

                    "raw_supported_distance_m":
                        raw_supported_m,

                    "legacy_supported_displacement_m":
                        legacy_m,
                }

                for noise_density in (
                    ACCELERATION_NOISE_DENSITY_GRID
                ):
                    smooth = (
                        compute_smoothed_trip_path_distance(
                            edges,
                            trip_start_ms=start_ms,
                            trip_end_ms=end_ms,
                            acceleration_noise_density_m_s1p5=(
                                noise_density
                            ),
                        )
                    )

                    key = (
                        f"{noise_density:g}"
                    )

                    row[
                        f"smooth_{key}_m"
                    ] = smooth[
                        "path_distance_smooth_obs_m"
                    ]

                    row[
                        f"smoothing_coverage_{key}"
                    ] = smooth[
                        "smoothing_coverage_fraction"
                    ]

                rows.append(
                    row
                )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SHARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    helpers = (
        load_smoke_helpers()
    )

    sample = pd.read_csv(
        SAMPLE_FILE,
        dtype={
            "user_id": str,
        },
    )

    print("=" * 100)
    print("GEOLIFE FULL EXTERNAL VALIDATION BENCHMARK")
    print("=" * 100)

    print(
        f"Reference blocks: {len(sample):,}"
    )

    print(
        f"Sampling intervals: "
        f"{SAMPLING_INTERVALS_S}"
    )

    print(
        f"Added accuracy conditions: "
        f"{ACCURACY68_LEVELS_M}"
    )

    print(
        f"Noise replicates per condition: "
        f"{N_REPLICATES}"
    )

    expected_rows = (
        len(sample)
        * len(SAMPLING_INTERVALS_S)
        * len(ACCURACY68_LEVELS_M)
        * N_REPLICATES
    )

    print(
        f"Expected benchmark rows: "
        f"{expected_rows:,}"
    )

    completed = 0
    failed = []

    for item in tqdm(
        sample.itertuples(
            index=False
        ),
        total=len(sample),
        desc="Reference blocks",
    ):
        shard = (
            SHARD_DIR
            / f"{item.segment_id}.parquet"
        )

        # Makes the benchmark restartable.
        if shard.exists():
            completed += 1
            continue

        try:
            result = run_segment(
                item,
                helpers,
            )

            result.to_parquet(
                shard,
                index=False,
            )

            completed += 1

        except Exception as exc:
            failed.append(
                {
                    "segment_id":
                        item.segment_id,
                    "error":
                        repr(exc),
                }
            )

    print()
    print("=" * 100)
    print("RUN SUMMARY")
    print("=" * 100)

    print(
        f"Completed blocks: "
        f"{completed:,}/{len(sample):,}"
    )

    print(
        f"Failed blocks: "
        f"{len(failed):,}"
    )

    if failed:
        failed_df = pd.DataFrame(
            failed
        )

        failed_df.to_csv(
            OUT_DIR
            / "failed_blocks.csv",
            index=False,
        )

        print()
        print(
            failed_df.head(
                20
            ).to_string(
                index=False
            )
        )

    shard_files = sorted(
        SHARD_DIR.glob(
            "*.parquet"
        )
    )

    if not shard_files:
        raise RuntimeError(
            "No benchmark shards were produced."
        )

    print()
    print(
        f"Saved shards: "
        f"{len(shard_files):,}"
    )

    print(
        f"Output directory: "
        f"{OUT_DIR}"
    )


if __name__ == "__main__":
    main()
