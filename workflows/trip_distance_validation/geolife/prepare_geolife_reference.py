from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SMOKE_DIR = Path(
    "outputs/trip_distance_validation/geolife/smoke"
)

OUT_DIR = Path(
    "outputs/trip_distance_validation/geolife/reference"
)

SUMMARY_FILE = (
    SMOKE_DIR / "smoke_segments.csv"
)

# We do not silently choose one gap definition yet.
# These values let us test whether the dense-reference distance
# is sensitive to how natural gaps in GeoLife are handled.
GAP_SENSITIVITY_S = [
    5,
    10,
    30,
    60,
]

# Gross sensor-error rejection only.
MAX_PLAUSIBLE_SPEED_KMH = 1000.0

EARTH_RADIUS_M = 6_371_008.8


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

    c = (
        2.0
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(
                np.maximum(
                    1.0 - a,
                    0.0,
                )
            ),
        )
    )

    return (
        EARTH_RADIUS_M * c
    )


def prepare_points(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "timestamp",
        "latitude",
        "longitude",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path}: missing columns "
            f"{sorted(missing)}"
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

    df = (
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

    return df


def build_edges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame()

    left = df.iloc[:-1].reset_index(
        drop=True
    )

    right = df.iloc[1:].reset_index(
        drop=True
    )

    dt_s = (
        (
            right["timestamp"]
            - left["timestamp"]
        )
        .dt.total_seconds()
        .to_numpy(
            dtype=float
        )
    )

    distance_m = haversine_m(
        left["latitude"].to_numpy(
            dtype=float
        ),
        left["longitude"].to_numpy(
            dtype=float
        ),
        right["latitude"].to_numpy(
            dtype=float
        ),
        right["longitude"].to_numpy(
            dtype=float
        ),
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        speed_kmh = (
            distance_m
            / dt_s
            * 3.6
        )

    return pd.DataFrame(
        {
            "start_time":
                left["timestamp"],
            "end_time":
                right["timestamp"],
            "dt_s":
                dt_s,
            "distance_m":
                distance_m,
            "speed_kmh":
                speed_kmh,
        }
    )


def reference_metrics(
    edges: pd.DataFrame,
    gap_threshold_s: float,
) -> dict:
    if edges.empty:
        return {
            "n_edges_total": 0,
            "n_edges_retained": 0,
            "n_gap_edges": 0,
            "n_implausible_speed_edges": 0,
            "reference_distance_m": np.nan,
            "reference_observed_duration_s": 0.0,
        }

    valid_base = (
        np.isfinite(
            edges["dt_s"]
        )
        &
        (
            edges["dt_s"]
            > 0
        )
        &
        np.isfinite(
            edges["distance_m"]
        )
        &
        (
            edges["distance_m"]
            >= 0
        )
    )

    gap_edge = (
        valid_base
        &
        (
            edges["dt_s"]
            > gap_threshold_s
        )
    )

    implausible_speed = (
        valid_base
        &
        np.isfinite(
            edges["speed_kmh"]
        )
        &
        (
            edges["speed_kmh"]
            > MAX_PLAUSIBLE_SPEED_KMH
        )
    )

    retained = (
        valid_base
        &
        (
            edges["dt_s"]
            <= gap_threshold_s
        )
        &
        (
            edges["speed_kmh"].isna()
            |
            (
                edges["speed_kmh"]
                <= MAX_PLAUSIBLE_SPEED_KMH
            )
        )
    )

    return {
        "n_edges_total":
            int(len(edges)),
        "n_edges_retained":
            int(retained.sum()),
        "n_gap_edges":
            int(gap_edge.sum()),
        "n_implausible_speed_edges":
            int(implausible_speed.sum()),
        "reference_distance_m":
            float(
                edges.loc[
                    retained,
                    "distance_m",
                ].sum()
            ),
        "reference_observed_duration_s":
            float(
                edges.loc[
                    retained,
                    "dt_s",
                ].sum()
            ),
    }


def block_count(
    edges: pd.DataFrame,
    gap_threshold_s: float,
) -> int:
    if edges.empty:
        return 0

    usable = (
        np.isfinite(
            edges["dt_s"]
        )
        &
        (
            edges["dt_s"]
            > 0
        )
    )

    if not usable.any():
        return 0

    # Every gap larger than the continuity threshold starts
    # a new high-resolution observed block.
    gaps = (
        edges.loc[
            usable,
            "dt_s",
        ]
        > gap_threshold_s
    )

    return (
        1 + int(gaps.sum())
    )


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing smoke summary: "
            f"{SUMMARY_FILE}"
        )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    rows = []

    print("=" * 100)
    print(
        "GEOLIFE HIGH-RESOLUTION "
        "REFERENCE AUDIT"
    )
    print("=" * 100)

    for item in summary.itertuples(
        index=False
    ):
        path = Path(
            item.output_file
        )

        points = prepare_points(
            path
        )

        edges = build_edges(
            points
        )

        print()
        print(
            f"{item.segment_id} "
            f"({item.transportation_mode})"
        )

        print(
            f"  points={len(points):,}"
        )

        for gap_s in (
            GAP_SENSITIVITY_S
        ):
            metrics = reference_metrics(
                edges,
                gap_s,
            )

            n_blocks = block_count(
                edges,
                gap_s,
            )

            distance_km = (
                metrics[
                    "reference_distance_m"
                ]
                / 1000.0
            )

            print(
                f"  gap<={gap_s:>2}s | "
                f"blocks={n_blocks:>2} | "
                f"distance={distance_km:>9.3f} km | "
                f"gap_edges={metrics['n_gap_edges']:>3} | "
                f"speed_reject={metrics['n_implausible_speed_edges']:>3}"
            )

            rows.append(
                {
                    "segment_id":
                        item.segment_id,
                    "user_id":
                        item.user_id,
                    "transportation_mode":
                        item.transportation_mode,
                    "gap_threshold_s":
                        gap_s,
                    "n_points":
                        len(points),
                    "n_blocks":
                        n_blocks,
                    **metrics,
                }
            )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUT_DIR
        / "reference_gap_sensitivity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summarize distance stability relative to 10-second
    # continuity definition.
    #
    # This does NOT declare 10 s to be ground truth.
    # It is simply the center of the dense-reference audit.
    # --------------------------------------------------------

    pivot = (
        results.pivot(
            index=[
                "segment_id",
                "transportation_mode",
            ],
            columns="gap_threshold_s",
            values="reference_distance_m",
        )
        .reset_index()
    )

    if 10 in pivot.columns:
        for gap_s in (
            GAP_SENSITIVITY_S
        ):
            if gap_s not in pivot.columns:
                continue

            pivot[
                f"distance_change_"
                f"{gap_s}s_vs_10s_pct"
            ] = (
                (
                    pivot[gap_s]
                    - pivot[10]
                )
                /
                pivot[10].replace(
                    0,
                    np.nan,
                )
                * 100.0
            )

    pivot.to_csv(
        OUT_DIR
        / "reference_distance_stability.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("REFERENCE DISTANCE STABILITY")
    print("=" * 100)

    display = pivot.copy()

    for column in [
        c
        for c in display.columns
        if isinstance(c, int)
    ]:
        display[column] = (
            display[column]
            / 1000.0
        )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved reference audit to:"
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()
