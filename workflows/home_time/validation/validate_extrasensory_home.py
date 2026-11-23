from pathlib import Path

import numpy as np
import pandas as pd

from timezonefinder import TimezoneFinder

from cortex2_engine.features.gps_pipeline_qc import (
    GPSQCConfig,
    clean_gps_fixes,
)
from cortex2_engine.features.gps_intervals import (
    build_gps_intervals,
)
from cortex2_engine.features.gps_spatial_semantics import (
    add_gps_spatial_semantics,
)
from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)
from cortex2_engine.features.gps_place_candidates import (
    build_place_candidates,
)
from cortex2_engine.features.gps_home_shared import (
    HomeConfig,
    calculate_home_time,
)


ROOT = Path("data/external/extrasensory")

LOCATION_DIR = ROOT / "absolute_location"
FEATURE_DIR = ROOT / "features_labels"

REFERENCE_FILE = (
    ROOT / "extrasensory_home_dominant_cluster.csv"
)

OUTPUT_DIR = Path(
    "outputs/home_time_validation/extrasensory"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MAX_GAP_VALUES_MS = [60_000]
COMPATIBILITY_MULTIPLIER = 1.0

MIN_REFERENCE_POINTS = 30

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(
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

    return float(
        2
        * EARTH_RADIUS_M
        * np.arcsin(np.sqrt(a))
    )


def infer_timezone(
    fixes: pd.DataFrame,
    tf: TimezoneFinder,
):
    valid = fixes[
        fixes["latitude"].notna()
        & fixes["longitude"].notna()
    ]

    if valid.empty:
        return None

    lat = float(
        valid["latitude"].median()
    )

    lon = float(
        valid["longitude"].median()
    )

    timezone = tf.timezone_at(
        lat=lat,
        lng=lon,
    )

    if timezone is None:
        timezone = tf.closest_timezone_at(
            lat=lat,
            lng=lon,
        )

    return timezone


def build_participant_places(
    fixes: pd.DataFrame,
    timezone: str,
    max_gap_ms: int,
):
    qc_config = GPSQCConfig(
        max_accuracy_m=None,
        max_speed_kmh=1000.0,
        max_gap_ms=max_gap_ms,
        min_coverage_fraction=None,
    )

    fixes = clean_gps_fixes(
        fixes,
        config=qc_config,
    )

    if len(fixes) < 2:
        return (
            fixes,
            pd.DataFrame(),
            pd.DataFrame(),
            0.0,
        )

    utc_time = pd.to_datetime(
        fixes["timestamp"],
        unit="ms",
        utc=True,
    )

    fixes = fixes.copy()

    fixes["local_date"] = (
        utc_time
        .dt.tz_convert(timezone)
        .dt.date
    )

    all_microsegments = []

    total_supported_ms = 0.0

    for local_date, day_raw in fixes.groupby(
        "local_date",
        sort=True,
    ):
        day_raw = day_raw[
            [
                "timestamp",
                "latitude",
                "longitude",
                "accuracy",
            ]
        ].copy()

        day_start = pd.Timestamp(
            f"{local_date} 00:00:00"
        ).tz_localize(
            timezone
        )

        day_end = (
            day_start
            + pd.DateOffset(days=1)
        )

        start_ms = int(
            day_start
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )

        end_ms = int(
            day_end
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )

        intervals = build_gps_intervals(
            day_raw,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            max_gap_ms=max_gap_ms,
        )

        if intervals.empty:
            continue

        supported = intervals[
            intervals["supported"]
            .fillna(False)
            .astype(bool)
        ].copy()

        if supported.empty:
            continue

        total_supported_ms += float(
            pd.to_numeric(
                supported["duration_ms"],
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

        spatial = (
            add_gps_spatial_semantics(
                supported
            )
        )

        microsegments = (
            build_anchor_compatible_location_episodes(
                spatial
            )
        )

        if microsegments.empty:
            continue

        microsegments = (
            microsegments.copy()
        )

        microsegments["date"] = (
            str(local_date)
        )

        all_microsegments.append(
            microsegments
        )

    if not all_microsegments:
        return (
            fixes,
            pd.DataFrame(),
            pd.DataFrame(),
            total_supported_ms,
        )

    participant_segments = pd.concat(
        all_microsegments,
        ignore_index=True,
    )

    assignments, places = (
        build_place_candidates(
            participant_segments,
            compatibility_multiplier=(
                COMPATIBILITY_MULTIPLIER
            ),
        )
    )

    return (
        fixes,
        assignments,
        places,
        total_supported_ms,
    )


def infer_home(
    fixes,
    assignments,
    places,
    total_supported_ms,
    timezone,
    max_gap_ms,
):
    result = {
        "home_place_candidate_id": np.nan,
        "home_confidence": np.nan,
        "home_valid_nights": 0,
        "home_status": (
            "insufficient_observability"
        ),
        "inferred_latitude": np.nan,
        "inferred_longitude": np.nan,
    }

    if (
        assignments.empty
        or places.empty
        or fixes.empty
    ):
        return result

    start_ms = int(
        fixes["timestamp"].min()
    )

    end_ms = int(
        fixes["timestamp"].max()
        + max_gap_ms
    )

    home_config = HomeConfig(
        timezone=timezone,
        min_home_confidence=0.50,
        min_daily_supported_ms=(
            6 * 60 * 60 * 1000
        ),
    )

    home = calculate_home_time(
        assignments,
        supported_duration_ms=(
            total_supported_ms
        ),
        start_ms=start_ms,
        end_ms=end_ms,
        config=home_config,
    )

    result.update(
        {
            "home_place_candidate_id":
                home[
                    "home_place_candidate_id"
                ],
            "home_confidence":
                home["home_confidence"],
            "home_valid_nights":
                home["home_valid_nights"],
            "home_status":
                home["home_status"],
        }
    )

    home_id = home[
        "home_place_candidate_id"
    ]

    if not np.isfinite(home_id):
        return result

    place = places[
        pd.to_numeric(
            places["place_candidate_id"],
            errors="coerce",
        )
        == int(home_id)
    ]

    if len(place) != 1:
        return result

    result[
        "inferred_latitude"
    ] = float(
        place.iloc[0][
            "representative_latitude"
        ]
    )

    result[
        "inferred_longitude"
    ] = float(
        place.iloc[0][
            "representative_longitude"
        ]
    )

    return result


def main():
    reference = pd.read_csv(
        REFERENCE_FILE
    )

    reference = reference[
        reference["n_home_gps"]
        >= MIN_REFERENCE_POINTS
    ].copy()

    reference = reference[
        reference["home_ref_lat"].notna()
        & reference["home_ref_lon"].notna()
    ].copy()

    reference = reference.set_index(
        "uuid"
    )

    tf = TimezoneFinder()

    rows = []

    uuids = sorted(
        reference.index.tolist()
    )

    print(
        "ExtraSensory Cortex home validation"
    )

    print(
        "Reference participants:",
        len(uuids),
    )

    total_runs = len(uuids) * len(MAX_GAP_VALUES_MS)
    run_i = 0

    for max_gap_ms in MAX_GAP_VALUES_MS:
        for uuid in uuids:
            run_i += 1
            loc_path = (
                LOCATION_DIR
                / f"{uuid}.absolute_locations.csv.gz"
            )

            feature_path = (
                FEATURE_DIR
                / f"{uuid}.features_labels.csv.gz"
            )

            loc = pd.read_csv(
                loc_path
            )

            features = pd.read_csv(
                feature_path,
                usecols=[
                    "timestamp",
                    "location:best_horizontal_accuracy",
                ],
            )

            data = loc.merge(
                features,
                on="timestamp",
                how="left",
                validate="one_to_one",
            )

            fixes = data.rename(
                columns={
                    "location:best_horizontal_accuracy":
                        "accuracy",
                }
            )[
                [
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            ].copy()

            # ExtraSensory timestamps are Unix seconds.
            # Cortex uses Unix milliseconds.
            fixes["timestamp"] = (
                pd.to_numeric(
                    fixes["timestamp"],
                    errors="coerce",
                )
                * 1000
            )

            timezone = infer_timezone(
                fixes,
                tf,
            )

            row = {
                "max_gap_s": max_gap_ms / 1000.0,
                "uuid": uuid,
                "timezone": timezone,
                "n_input_rows": len(fixes),
                "n_reference_home_gps": (
                    reference.loc[
                        uuid,
                        "n_home_gps",
                    ]
                ),
                "reference_dominant_fraction": (
                    reference.loc[
                        uuid,
                        "dominant_fraction",
                    ]
                ),
                "reference_home_days": (
                    reference.loc[
                        uuid,
                        "dominant_home_days",
                    ]
                ),
                "reference_p90_m": (
                    reference.loc[
                        uuid,
                        "p90_distance_m",
                    ]
                ),
                "reference_latitude": (
                    reference.loc[
                        uuid,
                        "home_ref_lat",
                    ]
                ),
                "reference_longitude": (
                    reference.loc[
                        uuid,
                        "home_ref_lon",
                    ]
                ),
            }

            if timezone is None:
                row.update(
                    {
                        "n_clean_fixes": 0,
                        "n_places": 0,
                        "home_place_candidate_id":
                            np.nan,
                        "home_confidence": np.nan,
                        "home_valid_nights": 0,
                        "home_status":
                            "timezone_unresolved",
                        "inferred_latitude":
                            np.nan,
                        "inferred_longitude":
                            np.nan,
                        "home_error_m":
                            np.nan,
                    }
                )

                rows.append(row)
                continue

            (
                clean_fixes,
                assignments,
                places,
                total_supported_ms,
            ) = build_participant_places(
                fixes,
                timezone,
                max_gap_ms,
            )

            home = infer_home(
                clean_fixes,
                assignments,
                places,
                total_supported_ms,
                timezone,
                max_gap_ms,
            )

            row.update(
                {
                    "n_clean_fixes":
                        len(clean_fixes),
                    "n_microsegments":
                        len(assignments),
                    "n_places":
                        len(places),
                    "supported_hours":
                        (
                            total_supported_ms
                            / 3_600_000
                        ),
                    **home,
                }
            )

            if (
                np.isfinite(
                    home[
                        "inferred_latitude"
                    ]
                )
                and np.isfinite(
                    home[
                        "inferred_longitude"
                    ]
                )
            ):
                row["home_error_m"] = (
                    haversine_m(
                        home[
                            "inferred_latitude"
                        ],
                        home[
                            "inferred_longitude"
                        ],
                        row[
                            "reference_latitude"
                        ],
                        row[
                            "reference_longitude"
                        ],
                    )
                )
            else:
                row["home_error_m"] = np.nan

            rows.append(row)

            print(
                f"\r{run_i:3d}/{total_runs}",
                end="",
                flush=True,
            )
    print()

    results = pd.DataFrame(
        rows
    )

    result_path = (
        OUTPUT_DIR
        / "extrasensory_home_validation.csv"
    )

    results.to_csv(
        result_path,
        index=False,
    )

    print()

    print("=" * 90)
    print("EXTRASENSORY HOME VALIDATION — GAP SENSITIVITY")
    print("=" * 90)

    summary_rows = []

    for gap_s, g in results.groupby(
        "max_gap_s",
        sort=True,
    ):
        successful = g[
            g["home_error_m"].notna()
        ].copy()

        error = successful[
            "home_error_m"
        ]

        summary_rows.append(
            {
                "max_gap_s": gap_s,
                "n_reference": len(g),
                "n_inferred": len(successful),
                "inference_fraction":
                    len(successful) / len(g),
                "median_error_m":
                    error.median()
                    if len(error) else np.nan,
                "p75_error_m":
                    error.quantile(0.75)
                    if len(error) else np.nan,
                "p90_error_m":
                    error.quantile(0.90)
                    if len(error) else np.nan,
                "within_50m":
                    (error <= 50).mean()
                    if len(error) else np.nan,
                "within_100m":
                    (error <= 100).mean()
                    if len(error) else np.nan,
                "within_200m":
                    (error <= 200).mean()
                    if len(error) else np.nan,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "extrasensory_home_validation_gap_sensitivity.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("Saved:")
    print(result_path)
    print(summary_path)

if __name__ == "__main__":
    main()
