from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.features.gps_intervals import (
    build_gps_intervals,
)
from cortex2_engine.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)
from cortex2_engine.features.gps_trajectory_plausibility import (
    add_gps_trajectory_plausibility,
)
from cortex2_engine.features.gps_mobility_features import (
    compute_daily_mobility_features,
)
from cortex2_engine.features.gps_trip_features import (
    compute_smoothed_trip_path_distance,
)


SMOKE_DIR = Path(
    "outputs/trip_distance_validation/geolife/smoke"
)

SUMMARY_FILE = (
    SMOKE_DIR / "smoke_segments.csv"
)

OUT_DIR = Path(
    "outputs/trip_distance_validation/"
    "geolife/degradation_smoke"
)

REFERENCE_MAX_GAP_S = 5.0

SAMPLING_INTERVALS_S = [
    5,
    10,
    30,
    60,
    120,
    300,
]

# Controlled synthetic horizontal-accuracy conditions.
#
# These are experimental noise conditions, not assumed GeoLife
# accuracies.
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

N_REPLICATES = 5

CORTEX_MAX_GAP_MS = 60_000

RANDOM_SEED = 20261116

EARTH_RADIUS_M = 6_371_008.8

ACCURACY68_TO_AXIS_SIGMA = np.sqrt(
    -2.0 * np.log(1.0 - 0.68)
)


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    lat1 = np.deg2rad(
        np.asarray(lat1, dtype=float)
    )
    lon1 = np.deg2rad(
        np.asarray(lon1, dtype=float)
    )
    lat2 = np.deg2rad(
        np.asarray(lat2, dtype=float)
    )
    lon2 = np.deg2rad(
        np.asarray(lon2, dtype=float)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(
        np.sqrt(a),
        np.sqrt(
            np.maximum(
                1.0 - a,
                0.0,
            )
        ),
    )

    return EARTH_RADIUS_M * c


def raw_path_distance_m(
    df: pd.DataFrame,
) -> float:
    if len(df) < 2:
        return 0.0

    return float(
        np.sum(
            haversine_m(
                df["latitude"].to_numpy(
                    dtype=float
                )[:-1],
                df["longitude"].to_numpy(
                    dtype=float
                )[:-1],
                df["latitude"].to_numpy(
                    dtype=float
                )[1:],
                df["longitude"].to_numpy(
                    dtype=float
                )[1:],
            )
        )
    )


def load_segment(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

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


def longest_dense_block(
    df: pd.DataFrame,
    max_gap_s: float,
) -> pd.DataFrame:
    """Return longest naturally continuous high-density block."""

    if len(df) < 2:
        return df.copy()

    dt_s = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    block_id = (
        (
            dt_s.isna()
            |
            (
                dt_s > max_gap_s
            )
        )
        .cumsum()
    )

    x = df.copy()
    x["_block_id"] = block_id

    groups = []

    for _, block in x.groupby(
        "_block_id",
        sort=False,
    ):
        block = (
            block.drop(
                columns="_block_id"
            )
            .reset_index(drop=True)
        )

        if len(block) < 2:
            continue

        duration_s = (
            (
                block["timestamp"].iloc[-1]
                - block["timestamp"].iloc[0]
            )
            .total_seconds()
        )

        groups.append(
            (
                duration_s,
                len(block),
                block,
            )
        )

    if not groups:
        return pd.DataFrame(
            columns=df.columns
        )

    groups.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    return groups[0][2]


def deterministic_downsample(
    df: pd.DataFrame,
    interval_s: int,
) -> pd.DataFrame:
    """Select approximately one fix per fixed elapsed-time interval."""

    if len(df) <= 2:
        return df.copy()

    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    target_times = pd.date_range(
        start=start,
        end=end,
        freq=f"{interval_s}s",
    )

    timestamps_ns = (
        df["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    target_ns = (
        target_times
        .astype("int64")
        .to_numpy()
    )

    indices = np.searchsorted(
        timestamps_ns,
        target_ns,
        side="left",
    )

    indices = np.clip(
        indices,
        0,
        len(df) - 1,
    )

    indices = np.unique(
        np.concatenate(
            [
                np.array(
                    [0],
                    dtype=int,
                ),
                indices,
                np.array(
                    [len(df) - 1],
                    dtype=int,
                ),
            ]
        )
    )

    return (
        df.iloc[indices]
        .copy()
        .reset_index(drop=True)
    )


def latlon_to_xy(
    latitude,
    longitude,
):
    latitude = np.asarray(
        latitude,
        dtype=float,
    )

    longitude = np.asarray(
        longitude,
        dtype=float,
    )

    lat0 = np.deg2rad(
        np.mean(latitude)
    )

    lon0 = np.deg2rad(
        np.mean(longitude)
    )

    lat = np.deg2rad(
        latitude
    )

    lon = np.deg2rad(
        longitude
    )

    x = (
        EARTH_RADIUS_M
        * np.cos(lat0)
        * (
            lon - lon0
        )
    )

    y = (
        EARTH_RADIUS_M
        * (
            lat - lat0
        )
    )

    return (
        x,
        y,
        lat0,
        lon0,
    )


def xy_to_latlon(
    x,
    y,
    lat0,
    lon0,
):
    latitude = (
        lat0
        + np.asarray(
            y,
            dtype=float,
        )
        / EARTH_RADIUS_M
    )

    longitude = (
        lon0
        + np.asarray(
            x,
            dtype=float,
        )
        /
        (
            EARTH_RADIUS_M
            * np.cos(lat0)
        )
    )

    return (
        np.rad2deg(latitude),
        np.rad2deg(longitude),
    )


def add_controlled_noise(
    df: pd.DataFrame,
    accuracy68_m: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add isotropic Gaussian error matching a known 68% radial radius."""

    result = df.copy()

    x, y, lat0, lon0 = (
        latlon_to_xy(
            result["latitude"],
            result["longitude"],
        )
    )

    sigma_axis_m = (
        accuracy68_m
        / ACCURACY68_TO_AXIS_SIGMA
    )

    x_noisy = (
        x
        + rng.normal(
            loc=0.0,
            scale=sigma_axis_m,
            size=len(result),
        )
    )

    y_noisy = (
        y
        + rng.normal(
            loc=0.0,
            scale=sigma_axis_m,
            size=len(result),
        )
    )

    latitude, longitude = (
        xy_to_latlon(
            x_noisy,
            y_noisy,
            lat0,
            lon0,
        )
    )

    result["latitude"] = latitude
    result["longitude"] = longitude
    result["accuracy"] = float(
        accuracy68_m
    )

    return result


def to_cortex_raw(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df[
        [
            "timestamp",
            "latitude",
            "longitude",
            "accuracy",
        ]
    ].copy()

    # Cortex raw GPS expects milliseconds.
    result["timestamp"] = (
        result["timestamp"]
        .astype("int64")
        // 1_000_000
    )

    return result


def build_cortex_edges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    raw = to_cortex_raw(df)

    window_start_ms = int(
        raw["timestamp"].min()
    )

    window_end_ms = (
        int(
            raw["timestamp"].max()
        )
        + 1
    )

    edges = build_gps_intervals(
        raw,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        max_gap_ms=CORTEX_MAX_GAP_MS,
    )

    edges = add_gps_spatial_semantics(
        edges
    )

    edges = add_gps_trajectory_plausibility(
        edges
    )

    return edges


def supported_raw_distance_m(
    edges: pd.DataFrame,
) -> float:
    if edges.empty:
        return 0.0

    valid = (
        edges["supported"]
        .fillna(False)
        .astype(bool)
        &
        pd.to_numeric(
            edges["distance_m"],
            errors="coerce",
        ).notna()
    )

    return float(
        edges.loc[
            valid,
            "distance_m",
        ].sum()
    )


def supported_fraction(
    edges: pd.DataFrame,
) -> float:
    if edges.empty:
        return 0.0

    duration = pd.to_numeric(
        edges["duration_ms"],
        errors="coerce",
    )

    elapsed = float(
        duration[
            np.isfinite(duration)
            &
            (
                duration > 0
            )
        ].sum()
    )

    supported = (
        edges["supported"]
        .fillna(False)
        .astype(bool)
    )

    supported_duration = float(
        duration[
            supported
            &
            np.isfinite(duration)
            &
            (
                duration > 0
            )
        ].sum()
    )

    if elapsed <= 0:
        return np.nan

    return (
        supported_duration
        / elapsed
    )


def legacy_displacement_m(
    edges: pd.DataFrame,
) -> float:
    if edges.empty:
        return 0.0

    features = (
        compute_daily_mobility_features(
            edges
        )
    )

    if isinstance(
        features,
        pd.DataFrame,
    ):
        if features.empty:
            return 0.0

        value = features.iloc[0].get(
            "supported_displacement_distance_m",
            np.nan,
        )

    elif isinstance(
        features,
        pd.Series,
    ):
        value = features.get(
            "supported_displacement_distance_m",
            np.nan,
        )

    elif isinstance(
        features,
        dict,
    ):
        value = features.get(
            "supported_displacement_distance_m",
            np.nan,
        )

    else:
        raise TypeError(
            "Unexpected mobility-feature result type: "
            f"{type(features)}"
        )

    return float(value)


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    smoke = pd.read_csv(
        SUMMARY_FILE
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    reference_rows = []
    benchmark_rows = []

    print("=" * 110)
    print(
        "GEOLIFE CONTROLLED SAMPLING / "
        "POSITION-ERROR SMOKE BENCHMARK"
    )
    print("=" * 110)

    for item in smoke.itertuples(
        index=False
    ):
        full = load_segment(
            Path(
                item.output_file
            )
        )

        reference = (
            longest_dense_block(
                full,
                REFERENCE_MAX_GAP_S,
            )
        )

        if len(reference) < 2:
            print(
                f"Skipping {item.segment_id}: "
                "no usable dense block."
            )
            continue

        reference_distance_m = (
            raw_path_distance_m(
                reference
            )
        )

        duration_s = (
            (
                reference[
                    "timestamp"
                ].iloc[-1]
                -
                reference[
                    "timestamp"
                ].iloc[0]
            )
            .total_seconds()
        )

        reference_rows.append(
            {
                "segment_id":
                    item.segment_id,
                "transportation_mode":
                    item.transportation_mode,
                "reference_n_points":
                    len(reference),
                "reference_duration_s":
                    duration_s,
                "reference_distance_m":
                    reference_distance_m,
            }
        )

        print()
        print(
            f"{item.segment_id} "
            f"({item.transportation_mode})"
        )

        print(
            f"  dense reference block: "
            f"{len(reference):,} points | "
            f"{duration_s / 60.0:.2f} min | "
            f"{reference_distance_m / 1000.0:.3f} km"
        )

        for interval_s in (
            SAMPLING_INTERVALS_S
        ):
            sampled_clean = (
                deterministic_downsample(
                    reference,
                    interval_s,
                )
            )

            for accuracy68_m in (
                ACCURACY68_LEVELS_M
            ):
                for replicate in range(
                    N_REPLICATES
                ):
                    degraded = (
                        add_controlled_noise(
                            sampled_clean,
                            accuracy68_m,
                            rng,
                        )
                    )

                    raw_all_m = (
                        raw_path_distance_m(
                            degraded
                        )
                    )

                    edges = (
                        build_cortex_edges(
                            degraded
                        )
                    )

                    raw_supported_m = (
                        supported_raw_distance_m(
                            edges
                        )
                    )

                    support = (
                        supported_fraction(
                            edges
                        )
                    )

                    legacy_m = (
                        legacy_displacement_m(
                            edges
                        )
                    )

                    start_ms = int(
                        degraded[
                            "timestamp"
                        ]
                        .astype("int64")
                        .iloc[0]
                        // 1_000_000
                    )

                    end_ms = (
                        int(
                            degraded[
                                "timestamp"
                            ]
                            .astype("int64")
                            .iloc[-1]
                            // 1_000_000
                        )
                        + 1
                    )

                    row = {
                        "segment_id":
                            item.segment_id,
                        "transportation_mode":
                            item.transportation_mode,
                        "sampling_interval_s":
                            interval_s,
                        "accuracy68_m":
                            accuracy68_m,
                        "replicate":
                            replicate,
                        "reference_distance_m":
                            reference_distance_m,
                        "n_sampled_points":
                            len(degraded),
                        "supported_fraction":
                            support,
                        "raw_all_distance_m":
                            raw_all_m,
                        "raw_supported_distance_m":
                            raw_supported_m,
                        "legacy_supported_displacement_m":
                            legacy_m,
                    }

                    for acceleration_noise_density in (
                        ACCELERATION_NOISE_DENSITY_GRID
                    ):
                        smooth = (
                            compute_smoothed_trip_path_distance(
                                edges,
                                trip_start_ms=start_ms,
                                trip_end_ms=end_ms,
                                acceleration_noise_density_m_s1p5=(
                                    acceleration_noise_density
                                ),
                            )
                        )

                        key = (
                            "smooth_"
                            f"{acceleration_noise_density:g}_m"
                        )

                        row[key] = smooth[
                            "path_distance_smooth_obs_m"
                        ]

                        row[
                            "smoothing_coverage_"
                            f"{acceleration_noise_density:g}"
                        ] = smooth[
                            "smoothing_coverage_fraction"
                        ]

                    benchmark_rows.append(
                        row
                    )

    references = pd.DataFrame(
        reference_rows
    )

    benchmark = pd.DataFrame(
        benchmark_rows
    )

    references.to_csv(
        OUT_DIR
        / "smoke_reference_blocks.csv",
        index=False,
    )

    benchmark.to_csv(
        OUT_DIR
        / "smoke_degradation_results.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("REFERENCE BLOCKS")
    print("=" * 110)

    if not references.empty:
        x = references.copy()

        x[
            "reference_distance_km"
        ] = (
            x[
                "reference_distance_m"
            ]
            / 1000.0
        )

        print(
            x[
                [
                    "segment_id",
                    "transportation_mode",
                    "reference_n_points",
                    "reference_duration_s",
                    "reference_distance_km",
                ]
            ].to_string(
                index=False
            )
        )

    if benchmark.empty:
        raise RuntimeError(
            "Benchmark produced no rows."
        )

    # --------------------------------------------------------
    # Summarize over synthetic-noise replicates.
    # --------------------------------------------------------

    metric_columns = [
        "raw_all_distance_m",
        "raw_supported_distance_m",
        "legacy_supported_displacement_m",
        "smooth_0.5_m",
        "smooth_1_m",
        "smooth_2_m",
    ]

    summary_rows = []

    group_columns = [
        "segment_id",
        "transportation_mode",
        "sampling_interval_s",
        "accuracy68_m",
    ]

    for keys, group in benchmark.groupby(
        group_columns,
        sort=True,
    ):
        reference_m = float(
            group[
                "reference_distance_m"
            ].iloc[0]
        )

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row[
            "median_supported_fraction"
        ] = float(
            group[
                "supported_fraction"
            ].median()
        )

        for metric in metric_columns:
            estimates = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            valid = (
                estimates.notna()
                &
                np.isfinite(
                    estimates
                )
            )

            if not valid.any():
                row[
                    f"{metric}_median_km"
                ] = np.nan

                row[
                    f"{metric}_median_abs_error_km"
                ] = np.nan

                row[
                    f"{metric}_median_bias_pct"
                ] = np.nan

                continue

            values = (
                estimates.loc[
                    valid
                ]
            )

            error = (
                values
                - reference_m
            )

            row[
                f"{metric}_median_km"
            ] = float(
                values.median()
                / 1000.0
            )

            row[
                f"{metric}_median_abs_error_km"
            ] = float(
                np.median(
                    np.abs(error)
                )
                / 1000.0
            )

            row[
                f"{metric}_median_bias_pct"
            ] = (
                float(
                    np.median(
                        error
                        / reference_m
                        * 100.0
                    )
                )
                if reference_m > 0
                else np.nan
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_DIR
        / "smoke_degradation_summary.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("SMOKE BENCHMARK SUMMARY")
    print("=" * 110)

    display = summary[
        [
            "segment_id",
            "transportation_mode",
            "sampling_interval_s",
            "accuracy68_m",
            "median_supported_fraction",
            "raw_all_distance_m_median_bias_pct",
            "raw_supported_distance_m_median_bias_pct",
            "legacy_supported_displacement_m_median_bias_pct",
            "smooth_0.5_m_median_bias_pct",
            "smooth_1_m_median_bias_pct",
            "smooth_2_m_median_bias_pct",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("Saved benchmark outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
