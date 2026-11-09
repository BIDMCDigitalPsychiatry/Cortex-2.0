"""
GPS mobility features grounded in the depression digital-phenotyping literature.

Primary methodological sources:
- Saeb et al. JMIR 2015
- Saeb et al. J Med Internet Res / longitudinal replication 2016

This module intentionally preserves literature-compatible definitions separately
from Cortex canonical mobility features.

Implemented:
    location_variance
    n_clusters
    entropy
    normalized_entropy
    homestay_fraction
    transition_time_fraction
    total_distance_m
    speed_mean_kmh
    speed_variance_kmh2
    raw_entropy

Not yet implemented:
    circadian_movement
    life_space_area
    maximum_action_range

Those require additional source-specific implementation QC before inclusion.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except ImportError as exc:
    raise ImportError(
        "gps_literature requires scikit-learn."
    ) from exc


EARTH_RADIUS_M = 6_371_000.0

# Saeb et al. 2015 preprocessing definitions
STATIONARY_SPEED_THRESHOLD_KMH = 1.0
MAX_CLUSTER_RADIUS_M = 500.0


# ---------------------------------------------------------------------
# Column handling
# ---------------------------------------------------------------------

def _resolve_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"None of these columns were found: {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def _standardize_points(
    points: pd.DataFrame,
) -> pd.DataFrame:

    if points.empty:
        return pd.DataFrame(
            columns=["timestamp_ms", "latitude", "longitude"]
        )

    timestamp_col = _resolve_column(
        points,
        [
            "timestamp",
            "timestamp_ms",
            "time",
            "start",
        ],
    )

    lat_col = _resolve_column(
        points,
        [
            "latitude",
            "lat",
        ],
    )

    lon_col = _resolve_column(
        points,
        [
            "longitude",
            "lon",
            "lng",
        ],
    )

    d = points[
        [
            timestamp_col,
            lat_col,
            lon_col,
        ]
    ].copy()

    d.columns = [
        "timestamp_ms",
        "latitude",
        "longitude",
    ]

    d["timestamp_ms"] = pd.to_numeric(
        d["timestamp_ms"],
        errors="coerce",
    )

    d["latitude"] = pd.to_numeric(
        d["latitude"],
        errors="coerce",
    )

    d["longitude"] = pd.to_numeric(
        d["longitude"],
        errors="coerce",
    )

    d = d.dropna(
        subset=[
            "timestamp_ms",
            "latitude",
            "longitude",
        ]
    )

    d = d.loc[
        d["latitude"].between(-90, 90)
        & d["longitude"].between(-180, 180)
    ].copy()

    if d.empty:
        return d

    # Detect seconds instead of milliseconds.
    med_ts = d["timestamp_ms"].median()

    if med_ts < 1e11:
        d["timestamp_ms"] *= 1000.0

    d = (
        d.sort_values("timestamp_ms")
        .drop_duplicates(
            subset=["timestamp_ms"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return d


# ---------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------

def _haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_M
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0.0, 1.0)
            )
        )
    )


def _add_motion_state(
    d: pd.DataFrame,
) -> pd.DataFrame:

    d = d.copy()

    n = len(d)

    d["distance_from_previous_m"] = np.nan
    d["delta_t_s"] = np.nan
    d["speed_kmh"] = np.nan
    d["is_transition"] = False

    if n < 2:
        return d

    distance = _haversine_m(
        d["latitude"].to_numpy()[:-1],
        d["longitude"].to_numpy()[:-1],
        d["latitude"].to_numpy()[1:],
        d["longitude"].to_numpy()[1:],
    )

    dt_s = (
        np.diff(
            d["timestamp_ms"].to_numpy()
        )
        / 1000.0
    )

    valid = dt_s > 0

    speed = np.full(
        len(distance),
        np.nan,
    )

    speed[valid] = (
        distance[valid]
        / dt_s[valid]
        * 3.6
    )

    d.loc[
        d.index[1:],
        "distance_from_previous_m",
    ] = distance

    d.loc[
        d.index[1:],
        "delta_t_s",
    ] = dt_s

    d.loc[
        d.index[1:],
        "speed_kmh",
    ] = speed

    d.loc[
        d.index[1:],
        "is_transition",
    ] = (
        speed
        >= STATIONARY_SPEED_THRESHOLD_KMH
    )

    return d


# ---------------------------------------------------------------------
# Adaptive K-means used by Saeb et al.
# ---------------------------------------------------------------------

def _cluster_max_radius_m(
    coords: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> float:

    center_lat = centers[
        labels,
        0,
    ]

    center_lon = centers[
        labels,
        1,
    ]

    radius = _haversine_m(
        coords[:, 0],
        coords[:, 1],
        center_lat,
        center_lon,
    )

    if len(radius) == 0:
        return np.nan

    return float(
        np.nanmax(radius)
    )


def _adaptive_kmeans(
    stationary: pd.DataFrame,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Literature-compatible adaptive K-means with computational compression.

    Saeb et al. increase K until all stationary locations fall within
    500 m of their assigned cluster center.

    Digital Clinic GPS can contain hundreds of thousands of repeated
    high-frequency coordinates in a single clinical window. To preserve
    the empirical weighting of densely sampled locations without fitting
    KMeans to every raw observation, coordinates are collapsed to unique
    rounded locations and their frequencies are passed as sample weights.

    Returned labels are expanded back to the original stationary rows.
    """

    if stationary.empty:
        return (
            np.array([], dtype=int),
            np.empty((0, 2)),
        )

    coords = stationary[
        ["latitude", "longitude"]
    ].to_numpy(dtype=float)

    # Lossless compression:
    # collapse only EXACT duplicate coordinates and retain their
    # multiplicity as KMeans sample weights.
    #
    # No rounding, spatial snapping, or coordinate modification occurs.
    unique_coords, inverse, counts = np.unique(
        coords,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    n_unique = len(unique_coords)

    if n_unique == 1:
        return (
            np.zeros(len(coords), dtype=int),
            np.array([unique_coords[0]], dtype=float),
        )

    weights = counts.astype(float)

    for k in range(1, n_unique + 1):

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        model.fit(
            unique_coords,
            sample_weight=weights,
        )

        unique_labels = model.labels_.astype(int)
        centers = model.cluster_centers_

        max_radius = _cluster_max_radius_m(
            unique_coords,
            unique_labels,
            centers,
        )

        if (
            np.isfinite(max_radius)
            and max_radius <= MAX_CLUSTER_RADIUS_M
        ):
            expanded_labels = unique_labels[inverse]

            return (
                expanded_labels.astype(int),
                centers,
            )

    # Fallback: each compressed location becomes its own cluster.
    expanded_labels = inverse.astype(int)

    return (
        expanded_labels,
        unique_coords,
    )



# ---------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------

def _shannon_entropy(
    values: np.ndarray,
) -> float:

    values = np.asarray(values)

    if len(values) == 0:
        return np.nan

    _, counts = np.unique(
        values,
        return_counts=True,
    )

    p = (
        counts.astype(float)
        / counts.sum()
    )

    return float(
        -np.sum(
            p * np.log(p)
        )
    )


def _raw_entropy_10_bins(
    stationary: pd.DataFrame,
) -> float:
    """
    Saeb et al. 2016 exploratory raw entropy:
    latitude entropy + longitude entropy,
    each using 10 bins before clustering.
    """

    if len(stationary) < 2:
        return np.nan

    total = 0.0

    for column in [
        "latitude",
        "longitude",
    ]:

        x = stationary[
            column
        ].to_numpy(dtype=float)

        if not np.isfinite(x).any():
            return np.nan

        if np.nanmax(x) == np.nanmin(x):
            entropy = 0.0

        else:
            counts, _ = np.histogram(
                x,
                bins=10,
            )

            counts = counts[
                counts > 0
            ]

            p = (
                counts
                / counts.sum()
            )

            entropy = float(
                -np.sum(
                    p * np.log(p)
                )
            )

        total += entropy

    return float(total)


# ---------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------

def compute_literature_gps_features(
    points: pd.DataFrame,
    *,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Compute GPS features using literature-compatible definitions.

    The feature set is intended for clinical-validation replication
    and should remain separate from Cortex canonical mobility features.
    """

    result = {
        "literature_total_distance_m": np.nan,
        "literature_location_variance": np.nan,
        "literature_n_clusters": np.nan,
        "literature_entropy_nats": np.nan,
        "literature_normalized_entropy": np.nan,
        "literature_homestay_fraction": np.nan,
        "literature_transition_time_fraction": np.nan,
        "literature_speed_mean_kmh": np.nan,
        "literature_speed_variance_kmh2": np.nan,
        "literature_raw_entropy": np.nan,
        "literature_n_points": 0,
        "literature_n_stationary_points": 0,
        "literature_n_transition_points": 0,
    }

    d = _standardize_points(
        points
    )

    result[
        "literature_n_points"
    ] = len(d)

    if len(d) < 2:
        return result

    d = _add_motion_state(
        d
    )

    valid_speed = d[
        "speed_kmh"
    ].dropna()

    transition = d[
        d["is_transition"]
    ].copy()

    stationary = d[
        ~d["is_transition"]
    ].copy()

    # First sample has no speed estimate.
    # It is retained as stationary, consistent with its
    # lack of evidence for transition.

    result[
        "literature_n_stationary_points"
    ] = len(stationary)

    result[
        "literature_n_transition_points"
    ] = len(transition)


    # --------------------------------------------------------------
    # SPEED
    # Saeb 2016: mean and variance of instantaneous speed.
    # --------------------------------------------------------------

    if len(valid_speed):

        result[
            "literature_speed_mean_kmh"
        ] = float(
            valid_speed.mean()
        )

        if len(valid_speed) > 1:

            result[
                "literature_speed_variance_kmh2"
            ] = float(
                valid_speed.var(
                    ddof=1
                )
            )


    # --------------------------------------------------------------
    # TRANSITION TIME
    # Saeb: percentage of time/observations in transition state.
    #
    # Using sample proportion here reproduces the sampled-data
    # representation used by the original feature extraction.
    # --------------------------------------------------------------

    result[
        "literature_transition_time_fraction"
    ] = float(
        transition.shape[0]
        / d.shape[0]
    )


    # --------------------------------------------------------------
    # TOTAL DISTANCE
    # Saeb 2015: distance traveled while in transition state.
    # --------------------------------------------------------------

    result[
        "literature_total_distance_m"
    ] = float(
        transition[
            "distance_from_previous_m"
        ]
        .dropna()
        .sum()
    )


    if stationary.empty:
        return result


    # --------------------------------------------------------------
    # LOCATION VARIANCE
    #
    # log(var(latitude) + var(longitude))
    # stationary states only.
    # --------------------------------------------------------------

    lat_var = stationary[
        "latitude"
    ].var(
        ddof=0
    )

    lon_var = stationary[
        "longitude"
    ].var(
        ddof=0
    )

    variance_sum = (
        lat_var
        + lon_var
    )

    if (
        np.isfinite(variance_sum)
        and variance_sum > 0
    ):
        result[
            "literature_location_variance"
        ] = float(
            np.log(
                variance_sum
            )
        )


    # --------------------------------------------------------------
    # LOCATION CLUSTERS
    # --------------------------------------------------------------

    labels, centers = _adaptive_kmeans(
        stationary,
        random_state=random_state,
    )

    stationary = stationary.copy()

    stationary[
        "literature_cluster"
    ] = labels

    n_clusters = int(
        len(
            np.unique(labels)
        )
    )

    result[
        "literature_n_clusters"
    ] = n_clusters


    # --------------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------------

    entropy = _shannon_entropy(
        labels
    )

    result[
        "literature_entropy_nats"
    ] = entropy


    # --------------------------------------------------------------
    # NORMALIZED ENTROPY
    #
    # H / log(N)
    #
    # For one cluster the theoretical normalized entropy limit is 0.
    # --------------------------------------------------------------

    if n_clusters <= 1:

        result[
            "literature_normalized_entropy"
        ] = 0.0

    elif np.isfinite(entropy):

        result[
            "literature_normalized_entropy"
        ] = float(
            entropy
            / np.log(
                n_clusters
            )
        )


    # --------------------------------------------------------------
    # HOMESTAY
    #
    # Saeb 2015:
    #   1. home is among the 3 most visited clusters
    #   2. among those, identify the cluster most visited 00:00–06:00
    #
    # Time is interpreted from the timestamps in UTC here because
    # the existing Digital Clinic canonical pipeline currently
    # operates in UTC. This is explicitly recorded and should later
    # be sensitivity-tested against participant-local time if local
    # timezone reconstruction is available.
    # --------------------------------------------------------------

    stationary[
        "datetime_utc"
    ] = pd.to_datetime(
        stationary[
            "timestamp_ms"
        ],
        unit="ms",
        utc=True,
    )

    cluster_counts = (
        stationary[
            "literature_cluster"
        ]
        .value_counts()
    )

    top_clusters = (
        cluster_counts
        .head(3)
        .index
        .tolist()
    )

    night = stationary.loc[
        stationary[
            "datetime_utc"
        ].dt.hour.between(
            0,
            5,
        )
        & stationary[
            "literature_cluster"
        ].isin(
            top_clusters
        )
    ]

    if not night.empty:

        home_cluster = int(
            night[
                "literature_cluster"
            ]
            .value_counts()
            .idxmax()
        )

        result[
            "literature_homestay_fraction"
        ] = float(
            (
                stationary[
                    "literature_cluster"
                ]
                == home_cluster
            ).sum()
            / len(d)
        )


    # --------------------------------------------------------------
    # RAW ENTROPY
    # Saeb 2016 exploratory extension.
    # --------------------------------------------------------------

    result[
        "literature_raw_entropy"
    ] = _raw_entropy_10_bins(
        stationary
    )

    return result
