from pathlib import Path
import math

import numpy as np
import pandas as pd


DAY_MS = 86_400_000

INVENTORY_PATH = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "inventory/source_files/gps.parquet"
)

DAILY_ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex_longitudinal/data/"
    "benchmark_5_full/"
    "participant_daily"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "validation/feature_reliability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_OUT = (
    OUT_DIR
    / "gps_feature_replay_reference.csv"
)

SUMMARY_OUT = (
    OUT_DIR
    / "gps_feature_replay_reference_summary.csv"
)


# ============================================================
# CORTEX-EQUIVALENT SIGNIFICANT-LOCATION HELPERS
# ============================================================

def cortex_euclid(gps0, gps1):

    def _euclid(
        lat,
        lng,
        lat0,
        lng0,
    ):
        return (
            110.25
            * (
                (
                    (lat - lat0) ** 2
                    + (
                        (
                            lng - lng0
                        )
                        * np.cos(lat0)
                    ) ** 2
                )
                ** 0.5
            )
        )

    return _euclid(
        gps0[0],
        gps0[1],
        gps1[0],
        gps1[1],
    )


def cortex_distance_m(
    cluster1,
    cluster2,
):

    R = 6378.137

    d_lat = (
        cluster2[0]
        * math.pi
        / 180
        - cluster1[0]
        * math.pi
        / 180
    )

    d_lon = (
        cluster2[1]
        * math.pi
        / 180
        - cluster1[1]
        * math.pi
        / 180
    )

    val1 = (
        math.sin(
            d_lat / 2
        )
        ** 2
    )

    val2 = (
        math.cos(
            cluster1[0]
            * math.pi
            / 180
        )
        * math.cos(
            cluster2[0]
            * math.pi
            / 180
        )
    )

    val3 = (
        math.sin(
            d_lon / 2
        )
        ** 2
    )

    val4 = (
        val1
        + val2
        * val3
    )

    dist = (
        2
        * math.atan2(
            math.sqrt(val4),
            math.sqrt(
                1 - val4
            ),
        )
    )

    return (
        R
        * dist
        * 1000
    )


def cortex_remove_clusters(
    clusters,
    max_dist=300,
):

    clusters = [
        dict(x)
        for x in clusters
    ]

    n = len(
        clusters
    )

    for i in range(
        n - 1
    ):

        big_cluster_prop = (
            clusters[i][
                "proportion"
            ]
        )

        if big_cluster_prop > 0:

            big_cluster_latlon = (
                clusters[i][
                    "latitude"
                ],
                clusters[i][
                    "longitude"
                ],
            )

            for j in range(
                i + 1,
                n,
            ):

                small_cluster_latlon = (
                    clusters[j][
                        "latitude"
                    ],
                    clusters[j][
                        "longitude"
                    ],
                )

                dist = (
                    cortex_distance_m(
                        big_cluster_latlon,
                        small_cluster_latlon,
                    )
                )

                if (
                    dist
                    < max_dist
                ):

                    clusters[i][
                        "proportion"
                    ] += (
                        clusters[j][
                            "proportion"
                        ]
                    )

                    clusters[j][
                        "proportion"
                    ] = 0

                    clusters[i][
                        "duration"
                    ] += (
                        clusters[j][
                            "duration"
                        ]
                    )

    clusters = [
        cl
        for cl in clusters
        if (
            cl[
                "proportion"
            ]
            > 0
        )
    ]

    for i, cluster in enumerate(
        clusters
    ):
        cluster[
            "rank"
        ] = i

    return clusters


def cortex_location_duration(
    df,
    cluster,
):

    # Cortex explicitly reverses here.
    df = (
        df[::-1]
        .reset_index()
    )

    arr_ext = np.r_[
        False,
        (
            df[
                "cluster"
            ]
            == cluster
        ).to_numpy(),
        False,
    ]

    idx = np.flatnonzero(
        arr_ext[:-1]
        != arr_ext[1:]
    )

    idx_list = list(
        zip(
            idx[:-1:2],
            idx[1::2]
            - 1,
        )
    )

    durations = []

    for start_i, end_i in idx_list:

        if (
            start_i
            == end_i
        ):
            continue

        duration = (
            df[
                "timestamp"
            ].iloc[
                end_i
            ]
            - df[
                "timestamp"
            ].iloc[
                start_i
            ]
        )

        durations.append(
            duration
        )

    return sum(
        durations
    )


def cortex_significant_locations_mode(
    gps_desc,
    min_cluster_size=0.01,
    max_clusters=-1,
    max_dist=300,
):

    if len(
        gps_desc
    ) == 0:

        return []


    # Same consecutive-timestamp filtering
    # performed by Cortex.
    df_original = (
        gps_desc.copy()
    )

    df_original = (
        df_original[
            df_original[
                "timestamp"
            ]
            != df_original[
                "timestamp"
            ].shift()
        ]
        .copy()
    )


    if len(
        df_original
    ) == 0:

        return []


    df_clusters = (
        df_original.copy(
            deep=True
        )
    )

    df_clusters[
        "cluster"
    ] = (
        [-1]
        * len(
            df_clusters
        )
    )


    # Cortex mode implementation rounds to
    # 3 decimal places.
    df_clusters[
        "latitude"
    ] = (
        df_clusters[
            "latitude"
        ]
        .apply(
            lambda x:
                round(
                    x,
                    3,
                )
        )
    )

    df_clusters[
        "longitude"
    ] = (
        df_clusters[
            "longitude"
        ]
        .apply(
            lambda x:
                round(
                    x,
                    3,
                )
        )
    )


    top_counts = (
        df_clusters[
            [
                "latitude",
                "longitude",
            ]
        ]
        .value_counts()
    )

    top_points = (
        top_counts
        .index
        .tolist()
    )

    min_cluster_points = int(
        min_cluster_size
        * len(
            df_original
        )
    )

    cluster_locs = []


    if (
        max_clusters
        != -1
    ):

        for k in range(
            max_clusters
        ):

            if (
                k
                < len(
                    top_points
                )
            ):

                point = (
                    top_points[k]
                )

                mask = (
                    (
                        df_clusters[
                            "latitude"
                        ]
                        == point[0]
                    )
                    & (
                        df_clusters[
                            "longitude"
                        ]
                        == point[1]
                    )
                )

                df_clusters.loc[
                    mask,
                    "cluster",
                ] = k

                cluster_locs.append(
                    point
                )

    else:

        k = 0

        while (
            k
            < len(
                top_counts
            )
            and (
                top_counts.iloc[k]
                > min_cluster_points
            )
            and (
                k
                < len(
                    df_original
                )
                - 1
            )
        ):

            point = (
                top_points[k]
            )

            mask = (
                (
                    df_clusters[
                        "latitude"
                    ]
                    == point[0]
                )
                & (
                    df_clusters[
                        "longitude"
                    ]
                    == point[1]
                )
            )

            df_clusters.loc[
                mask,
                "cluster",
            ] = k

            cluster_locs.append(
                point
            )

            k += 1


    denominator = (
        df_clusters[
            df_clusters[
                "cluster"
            ]
            != -1
        ].size
    )


    clusters = []

    for idx, center in enumerate(
        cluster_locs
    ):

        cluster_rows = (
            df_clusters[
                df_clusters[
                    "cluster"
                ]
                == idx
            ]
        )

        proportion = (
            cluster_rows.size
            / denominator
            if denominator
            else 0
        )

        duration = (
            cortex_location_duration(
                df_clusters,
                idx,
            )
        )

        clusters.append({
            "latitude":
                center[0],

            "longitude":
                center[1],

            "rank":
                idx,

            "proportion":
                proportion,

            "duration":
                duration,
        })


    return cortex_remove_clusters(
        clusters,
        max_dist=max_dist,
    )


# ============================================================
# CORTEX-EQUIVALENT TRIP HELPERS
# ============================================================

def cortex_haversine_np(
    lon1,
    lat1,
    lon2,
    lat2,
):

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [
            lon1,
            lat1,
            lon2,
            lat2,
        ],
    )

    dlon = (
        lon2
        - lon1
    )

    dlat = (
        lat2
        - lat1
    )

    val = (
        np.sin(
            dlat / 2.0
        )
        ** 2
        + np.cos(
            lat1
        )
        * np.cos(
            lat2
        )
        * np.sin(
            dlon / 2.0
        )
        ** 2
    )

    val2 = (
        2
        * np.arcsin(
            np.sqrt(
                val
            )
        )
    )

    return (
        6367
        * val2
    )


def cortex_trips(
    gps_desc,
):

    if len(
        gps_desc
    ) == 0:

        return []


    # Cortex trips.py:
    # pd.DataFrame.from_dict(
    #     list(reversed(gps()['data']))
    # )
    gps_data = (
        gps_desc.iloc[::-1]
        .reset_index(
            drop=True
        )
        .copy()
    )


    gps_data.loc[
        :,
        "dt",
    ] = (
        (
            gps_data[
                "timestamp"
            ]
            - gps_data[
                "timestamp"
            ].shift()
        )
        / (
            1000
            * 3600
        )
    )


    # Preserve Cortex's exact argument order.
    gps_data.loc[
        :,
        "dx",
    ] = cortex_haversine_np(
        gps_data[
            "latitude"
        ].shift(
            fill_value=0
        ),

        gps_data[
            "longitude"
        ].shift(
            fill_value=0
        ),

        gps_data.loc[
            1:,
            "latitude",
        ],

        gps_data.loc[
            1:,
            "longitude",
        ],
    )


    gps_data.loc[
        :,
        "speed",
    ] = (
        gps_data[
            "dx"
        ]
        / gps_data[
            "dt"
        ]
    )


    gps_data.loc[
        :,
        "stationary",
    ] = (
        (
            gps_data[
                "speed"
            ]
            < 10.0
        )
        | (
            gps_data[
                "dt"
            ]
            > 600
        )
    )


    gps_data.loc[
        :,
        "stationary_1",
    ] = (
        gps_data[
            "stationary"
        ].shift()
    )

    gps_data.loc[
        :,
        "idx",
    ] = (
        gps_data.index
    )


    new = (
        gps_data[
            gps_data[
                "stationary"
            ]
            != gps_data[
                "stationary_1"
            ]
        ]
        .copy()
    )


    new.loc[
        :,
        "idx_shift",
    ] = (
        new[
            "idx"
        ]
        .shift(
            -1,
            fill_value=0,
        )
    )


    new = (
        new[
            ~new[
                "stationary"
            ]
        ]
        .copy()
    )

    new = (
        new[
            new[
                "idx_shift"
            ]
            != 0
        ]
        .copy()
    )


    trips = []

    for row in new.itertuples(
        index=False
    ):

        idx1 = int(
            row.idx
        )

        idx2 = int(
            row.idx_shift
        )

        distance = (
            gps_data[
                "dx"
            ][
                idx1:idx2
            ]
            .sum()
        )

        end = (
            gps_data[
                "timestamp"
            ][
                idx2
                - 1
            ]
        )

        trips.append({
            "start":
                row.timestamp,

            "end":
                end,

            "latitude":
                row.latitude,

            "longitude":
                row.longitude,

            "distance":
                distance,
        })

    return trips


# ============================================================
# FEATURE COMPUTATION
# ============================================================

def compute_features(
    gps_desc,
):

    if len(
        gps_desc
    ) == 0:

        return {
            "has_raw_data":
                0,

            "n_raw_points":
                0,

            "n_significant_locations":
                0,

            "n_trips":
                0,

            "hometime":
                np.nan,

            "entropy":
                np.nan,

            "trip_distance":
                np.nan,

            "trip_duration":
                np.nan,
        }


    siglocs = (
        cortex_significant_locations_mode(
            gps_desc
        )
    )

    trips = (
        cortex_trips(
            gps_desc
        )
    )


    if siglocs:

        home_rows = [
            x
            for x in siglocs
            if (
                x[
                    "rank"
                ]
                == 0
            )
        ]

        hometime = (
            home_rows[0][
                "duration"
            ]
            if home_rows
            else np.nan
        )

        entropy = -sum(
            (
                x[
                    "proportion"
                ]
                * math.log(
                    x[
                        "proportion"
                    ]
                )
            )
            for x in siglocs
            if (
                0
                < x[
                    "proportion"
                ]
                <= 1
            )
        )

    else:

        hometime = np.nan
        entropy = np.nan


    trip_distance = sum(
        x[
            "distance"
        ]
        for x in trips
    )

    trip_duration = sum(
        (
            x[
                "end"
            ]
            - x[
                "start"
            ]
        )
        for x in trips
    )


    return {
        "has_raw_data":
            1,

        "n_raw_points":
            len(
                gps_desc
            ),

        "n_significant_locations":
            len(
                siglocs
            ),

        "n_trips":
            len(
                trips
            ),

        "hometime":
            hometime,

        "entropy":
            entropy,

        "trip_distance":
            trip_distance,

        "trip_duration":
            trip_duration,
    }


# ============================================================
# LOAD REFERENCE DAYS
# ============================================================

daily_frames = []

for path in sorted(
    DAILY_ROOT.glob(
        "*.csv"
    )
):

    df = pd.read_csv(
        path
    )

    daily_frames.append(
        df
    )


if not daily_frames:

    raise RuntimeError(
        "No benchmark participant_daily files found."
    )


daily = pd.concat(
    daily_frames,
    ignore_index=True,
)


for col in [
    "timestamp",
    "gps_data_quality",
    "hometime",
    "entropy",
    "trip_distance",
    "trip_duration",
]:

    daily[
        col
    ] = pd.to_numeric(
        daily[
            col
        ],
        errors="coerce",
    )


reference = (
    daily[
        daily[
            "gps_data_quality"
        ]
        == 1.0
    ]
    .copy()
)


reference = (
    reference.sort_values(
        [
            "mindlamp_uid",
            "timestamp",
        ]
    )
)


print(
    "=" * 80
)

print(
    "CORTEX GPS FEATURE REPLAY VALIDATION"
)

print(
    "=" * 80
)

print(
    "Reference days:",
    len(
        reference
    ),
)

print(
    "Participants:",
    reference[
        "mindlamp_uid"
    ].nunique(),
)


# ============================================================
# LOAD INVENTORY
# ============================================================

inventory = pd.read_parquet(
    INVENTORY_PATH
)

inventory = (
    inventory[
        inventory[
            "scan_status"
        ]
        == "ok"
    ]
    .copy()
)


# ============================================================
# REPLAY BY PARTICIPANT
# ============================================================

rows = []


for uid, refs in reference.groupby(
    "mindlamp_uid",
    sort=True,
):

    print()

    print(
        "-" * 80
    )

    print(
        uid,
        f"reference_days={len(refs)}",
    )


    inv = (
        inventory[
            inventory[
                "source_participant_id"
            ]
            .astype(str)
            == str(uid)
        ]
        .copy()
    )


    if len(
        inv
    ) == 0:

        print(
            "  ERROR: no inventory rows"
        )

        continue


    # Select only raw files overlapping at least
    # one DQ=1 reference day.
    selected_paths = set()

    for ref in refs.itertuples(
        index=False
    ):

        start = int(
            ref.timestamp
        )

        end = (
            start
            + DAY_MS
        )

        overlap = inv[
            (
                inv[
                    "max_timestamp_ms"
                ]
                >= start
            )
            & (
                inv[
                    "min_timestamp_ms"
                ]
                <= end
            )
        ]

        selected_paths.update(
            overlap[
                "source_file"
            ]
            .astype(str)
            .tolist()
        )


    print(
        "  raw files:",
        len(
            selected_paths
        ),
    )


    raw_parts = []

    failed_files = 0


    for file_name in sorted(
        selected_paths
    ):

        path = Path(
            file_name
        )

        try:

            x = pd.read_csv(
                path,
                usecols=lambda c:
                    c
                    in {
                        "timestamp",
                        "latitude",
                        "longitude",
                        "altitude",
                        "accuracy",
                    },
            )

        except Exception as e:

            failed_files += 1

            print(
                "  read failure:",
                path.name,
                type(e).__name__,
            )

            continue


        required = {
            "timestamp",
            "latitude",
            "longitude",
        }

        if not required.issubset(
            x.columns
        ):

            failed_files += 1

            print(
                "  missing GPS columns:",
                path.name,
            )

            continue


        raw_parts.append(
            x
        )


    if not raw_parts:

        print(
            "  ERROR: no raw GPS loaded"
        )

        continue


    raw = pd.concat(
        raw_parts,
        ignore_index=True,
    )


    for col in [
        "timestamp",
        "latitude",
        "longitude",
    ]:

        raw[
            col
        ] = pd.to_numeric(
            raw[
                col
            ],
            errors="coerce",
        )


    raw = (
        raw.dropna(
            subset=[
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .copy()
    )


    raw[
        "timestamp"
    ] = (
        raw[
            "timestamp"
        ]
        .astype(
            "int64"
        )
    )


    print(
        "  loaded raw rows:",
        f"{len(raw):,}",
        f"(failed files={failed_files})",
    )


    # LAMP SensorEvent results are consumed by
    # Cortex in reverse chronological order.
    raw = (
        raw.sort_values(
            "timestamp",
            ascending=False,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )


    for ref in refs.itertuples(
        index=False
    ):

        start = int(
            ref.timestamp
        )

        end = (
            start
            + DAY_MS
        )


        # Cortex raw wrapper keeps:
        # timestamp >= start AND timestamp <= end
        day_gps = (
            raw[
                (
                    raw[
                        "timestamp"
                    ]
                    >= start
                )
                & (
                    raw[
                        "timestamp"
                    ]
                    <= end
                )
            ]
            .copy()
        )


        computed = (
            compute_features(
                day_gps
            )
        )


        row = {
            "participant_id":
                ref.participant_id,

            "mindlamp_uid":
                uid,

            "date":
                ref.date,

            "timestamp":
                start,

            "gps_data_quality":
                ref.gps_data_quality,

            **computed,
        }


        for feature in [
            "hometime",
            "entropy",
            "trip_distance",
            "trip_duration",
        ]:

            reference_value = getattr(
                ref,
                feature,
            )

            replay_value = (
                computed[
                    feature
                ]
            )


            row[
                f"{feature}_reference"
            ] = reference_value

            row[
                f"{feature}_replay"
            ] = replay_value


            if (
                pd.notna(
                    reference_value
                )
                and pd.notna(
                    replay_value
                )
            ):

                abs_error = abs(
                    float(
                        replay_value
                    )
                    - float(
                        reference_value
                    )
                )

            else:

                abs_error = np.nan


            row[
                f"{feature}_abs_error"
            ] = abs_error


        rows.append(
            row
        )


# ============================================================
# SAVE DETAIL
# ============================================================

result = pd.DataFrame(
    rows
)


result.to_csv(
    DETAIL_OUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

tolerances = {
    "hometime":
        1.0,

    "entropy":
        1e-9,

    "trip_distance":
        1e-6,

    "trip_duration":
        1.0,
}


summary_rows = []


for feature, tolerance in tolerances.items():

    ref_col = (
        f"{feature}_reference"
    )

    replay_col = (
        f"{feature}_replay"
    )

    error_col = (
        f"{feature}_abs_error"
    )


    valid = result[
        result[
            ref_col
        ].notna()
        & result[
            replay_col
        ].notna()
    ].copy()


    both_missing = (
        result[
            ref_col
        ].isna()
        & result[
            replay_col
        ].isna()
    )


    mismatch_missingness = (
        result[
            ref_col
        ].isna()
        != result[
            replay_col
        ].isna()
    )


    if len(
        valid
    ):

        median_error = (
            valid[
                error_col
            ].median()
        )

        max_error = (
            valid[
                error_col
            ].max()
        )

        exact_n = int(
            (
                valid[
                    error_col
                ]
                <= tolerance
            ).sum()
        )

    else:

        median_error = np.nan
        max_error = np.nan
        exact_n = 0


    summary_rows.append({
        "feature":
            feature,

        "days_total":
            len(
                result
            ),

        "days_numeric_compared":
            len(
                valid
            ),

        "both_missing":
            int(
                both_missing.sum()
            ),

        "missingness_mismatch":
            int(
                mismatch_missingness.sum()
            ),

        "tolerance":
            tolerance,

        "within_tolerance_n":
            exact_n,

        "within_tolerance_fraction":
            (
                exact_n
                / len(
                    valid
                )
                if len(
                    valid
                )
                else np.nan
            ),

        "median_abs_error":
            median_error,

        "max_abs_error":
            max_error,
    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


print()
print(
    "=" * 80
)

print(
    "REPLAY SUMMARY"
)

print(
    "=" * 80
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.10g}",
    )
)


print()
print(
    "=" * 80
)

print(
    "RAW-DAY DIAGNOSTICS"
)

print(
    "=" * 80
)

if len(
    result
):

    diagnostics = (
        result.groupby(
            "participant_id"
        )
        .agg(
            reference_days=(
                "date",
                "size",
            ),

            raw_points_median=(
                "n_raw_points",
                "median",
            ),

            raw_points_min=(
                "n_raw_points",
                "min",
            ),

            raw_points_max=(
                "n_raw_points",
                "max",
            ),

            siglocs_median=(
                "n_significant_locations",
                "median",
            ),

            trips_median=(
                "n_trips",
                "median",
            ),
        )
        .reset_index()
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )


print()
print(
    "Detailed output:"
)

print(
    DETAIL_OUT
)

print(
    "Summary output:"
)

print(
    SUMMARY_OUT
)

print()
print(
    "=" * 80
)

print(
    "DONE"
)

print(
    "=" * 80
)
