"""Personalized GPS degradation -> clinical prediction degradation.

Primary exploratory candidates:
- LPE511: displacement distance -> depression
- BPH745: dominant-place fraction -> depression
- VTE817: displacement distance -> depression

For every complete 7/7 clinical week:
1. fit the participant's one-feature Ridge model using all OTHER
   undegraded reference weeks;
2. freeze that model;
3. artificially degrade raw GPS for the held-out week;
4. rebuild the complete Cortex 2.0 GPS feature pipeline;
5. predict depression using the degraded feature;
6. quantify feature and prediction degradation.

Missingness design is inherited from the frozen Cortex 2.0 GPS
reliability experiment:
- 5-minute temporal bins
- 90%, 75%, 50%, 25% target coverage
- distributed and contiguous missingness
- 5 independent replicates
- base seed 20261019

This workflow does NOT refit clinical models on degraded data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cortex2_engine.features.gps_degradation import (
    make_missing_bin_mask,
    apply_missing_bin_mask_to_points,
)

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

from cortex2_engine.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)

from cortex2_engine.features.gps_place_candidates import (
    build_place_candidates,
)

from cortex2_engine.features.gps_location_features import (
    compute_daily_location_features,
)


# ============================================================
# FROZEN DESIGN
# ============================================================

DAY_MS = 86_400_000
MAX_GAP_MS = 60_000
BIN_SIZE_MS = 5 * 60_000
BASE_SEED = 20261019

COMPATIBILITY_MULTIPLIER = 2.0

COVERAGES = [
    0.90,
    0.75,
    0.50,
    0.25,
]

PATTERNS = [
    "distributed",
    "contiguous",
]

N_REPLICATES = 5


CANDIDATES = {
    "LPE511": {
        "feature":
            "displacement_distance_mean_m",
        "feature_name":
            "displacement_distance",
    },

    "BPH745": {
        "feature":
            "dominant_place_fraction_mean",
        "feature_name":
            "dominant_place_fraction",
    },

    "VTE817": {
        "feature":
            "displacement_distance_mean_m",
        "feature_name":
            "displacement_distance",
    },
}


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

WEEKLY_PATH = (
    ROOT
    / "gps_clinical_weekly_features.parquet"
)

DAILY_PATH = (
    ROOT
    / "gps_clinical_weekly_daily_features.parquet"
)

RAW_PATH = (
    ROOT
    / "gps_clinical_weekly_raw_points.parquet"
)

OUT_DIR = (
    ROOT
    / "personalized_clinical_degradation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXACT FROZEN SEED FUNCTION
# ============================================================
#
# Rather than duplicating or guessing the historical seed
# implementation, extract the already committed stable_seed()
# function from the frozen degradation workflow.
# ============================================================

def load_frozen_stable_seed():
    path = Path(
        "workflows/"
        "run_gps_location_degradation_condition.py"
    )

    source = path.read_text()

    tree = ast.parse(
        source
    )

    function_node = None

    for node in tree.body:

        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "stable_seed"
        ):
            function_node = node
            break

    if function_node is None:
        raise RuntimeError(
            "Could not locate frozen stable_seed()."
        )

    module = ast.Module(
        body=[
            function_node
        ],
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    namespace = {
        "hashlib":
            hashlib,
    }

    exec(
        compile(
            module,
            str(path),
            "exec",
        ),
        namespace,
    )

    return namespace[
        "stable_seed"
    ]


stable_seed = (
    load_frozen_stable_seed()
)


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--participant",
    choices=sorted(
        CANDIDATES
    ),
    required=True,
)

args = parser.parse_args()

PARTICIPANT_ID = (
    args.participant
)

candidate = (
    CANDIDATES[
        PARTICIPANT_ID
    ]
)

FEATURE_COLUMN = (
    candidate[
        "feature"
    ]
)

FEATURE_NAME = (
    candidate[
        "feature_name"
    ]
)


OUT_PATH = (
    OUT_DIR
    / (
        f"{PARTICIPANT_ID}_"
        "gps_clinical_degradation.parquet"
    )
)


# ============================================================
# HELPERS
# ============================================================

def fit_reference_model(
    train_feature,
    train_outcome,
):
    """Fit the exact one-feature personalized Ridge model."""

    X = np.asarray(
        train_feature,
        dtype=float,
    ).reshape(
        -1,
        1,
    )

    y = np.asarray(
        train_outcome,
        dtype=float,
    )

    model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=1.0
                ),
            ),
        ]
    )

    model.fit(
        X,
        y,
    )

    return model


def predict_one(
    model,
    value,
):
    return float(
        model.predict(
            np.asarray(
                [[value]],
                dtype=float,
            )
        )[0]
    )


def process_degraded_day(
    day_raw,
    *,
    participant_id,
    date,
    day_start_ms,
    day_end_ms,
    coverage,
    pattern,
    replicate,
):
    """Degrade one day and compute Cortex 2.0 daily evidence."""

    seed = stable_seed(
        BASE_SEED,
        participant_id,
        date,
        coverage,
        pattern,
        replicate,
    )

    missing_mask, mask_info = (
        make_missing_bin_mask(
            window_start_ms=(
                day_start_ms
            ),
            window_end_ms=(
                day_end_ms
            ),
            target_coverage=(
                coverage
            ),
            pattern=pattern,
            bin_size_ms=(
                BIN_SIZE_MS
            ),
            seed=seed,
        )
    )

    degraded_raw = (
        apply_missing_bin_mask_to_points(
            day_raw,
            window_start_ms=(
                day_start_ms
            ),
            window_end_ms=(
                day_end_ms
            ),
            bin_size_ms=(
                BIN_SIZE_MS
            ),
            missing_mask=(
                missing_mask
            ),
        )
    )

    n_original = len(
        day_raw
    )

    n_retained = len(
        degraded_raw
    )

    raw_fraction_retained = (
        n_retained
        / n_original
        if n_original > 0
        else np.nan
    )

    if degraded_raw.empty:

        return {
            "microsegments":
                pd.DataFrame(),

            "supported_duration_ms":
                0.0,

            "supported_fraction_of_day":
                0.0,

            "displacement_distance_m":
                np.nan,

            "raw_fraction_retained":
                raw_fraction_retained,

            "planned_coverage":
                mask_info[
                    "planned_coverage"
                ],

            "n_raw_original":
                n_original,

            "n_raw_retained":
                n_retained,
        }

    intervals = (
        build_gps_intervals(
            degraded_raw[
                [
                    "timestamp",
                    "latitude",
                    "longitude",
                    "accuracy",
                ]
            ],
            window_start_ms=(
                day_start_ms
            ),
            window_end_ms=(
                day_end_ms
            ),
            max_gap_ms=(
                MAX_GAP_MS
            ),
        )
    )

    supported = (
        intervals[
            intervals[
                "supported"
            ]
        ]
        .copy()
    )

    supported_duration_ms = float(
        supported[
            "duration_ms"
        ].sum()
    )

    supported_fraction = (
        supported_duration_ms
        / (
            day_end_ms
            - day_start_ms
        )
    )

    if supported.empty:

        return {
            "microsegments":
                pd.DataFrame(),

            "supported_duration_ms":
                supported_duration_ms,

            "supported_fraction_of_day":
                supported_fraction,

            "displacement_distance_m":
                np.nan,

            "raw_fraction_retained":
                raw_fraction_retained,

            "planned_coverage":
                mask_info[
                    "planned_coverage"
                ],

            "n_raw_original":
                n_original,

            "n_raw_retained":
                n_retained,
        }

    spatial = (
        add_gps_spatial_semantics(
            supported
        )
    )

    trajectory = (
        add_gps_trajectory_plausibility(
            spatial,
            max_plausible_speed_kmh=(
                1000.0
            ),
        )
    )

    mobility = (
        compute_daily_mobility_features(
            trajectory
        )
    )

    microsegments = (
        build_anchor_compatible_location_episodes(
            spatial
        )
    )

    if not microsegments.empty:

        microsegments = (
            microsegments.copy()
        )

        microsegments[
            "participant_id"
        ] = participant_id

        microsegments[
            "date"
        ] = date

    return {
        "microsegments":
            microsegments,

        "supported_duration_ms":
            supported_duration_ms,

        "supported_fraction_of_day":
            supported_fraction,

        "displacement_distance_m":
            mobility[
                "supported_displacement_distance_m"
            ],

        "raw_fraction_retained":
            raw_fraction_retained,

        "planned_coverage":
            mask_info[
                "planned_coverage"
            ],

        "n_raw_original":
            n_original,

        "n_raw_retained":
            n_retained,
    }


def rebuild_degraded_week(
    *,
    participant_id,
    window_id,
    window_daily,
    participant_raw,
    coverage,
    pattern,
    replicate,
    feature_name,
):
    """Rebuild one degraded clinical week."""

    day_results = {}

    segment_frames = []

    for _, day in (
        window_daily
        .sort_values(
            "window_day"
        )
        .iterrows()
    ):

        date = str(
            day[
                "date"
            ]
        )

        day_raw = (
            participant_raw[
                participant_raw[
                    "date"
                ].eq(date)
            ]
            .copy()
        )

        if day_raw.empty:
            raise RuntimeError(
                f"{participant_id} {window_id} "
                f"{date}: complete reference week "
                "unexpectedly has no raw GPS."
            )

        starts = (
            day_raw[
                "day_start_ms"
            ]
            .dropna()
            .astype(
                np.int64
            )
            .unique()
        )

        ends = (
            day_raw[
                "day_end_ms"
            ]
            .dropna()
            .astype(
                np.int64
            )
            .unique()
        )

        if (
            len(starts) != 1
            or len(ends) != 1
        ):
            raise RuntimeError(
                f"{participant_id} {date}: "
                "unexpected day boundaries."
            )

        result = (
            process_degraded_day(
                day_raw,
                participant_id=(
                    participant_id
                ),
                date=date,
                day_start_ms=int(
                    starts[0]
                ),
                day_end_ms=int(
                    ends[0]
                ),
                coverage=coverage,
                pattern=pattern,
                replicate=replicate,
            )
        )

        day_results[
            date
        ] = result

        seg = (
            result[
                "microsegments"
            ]
        )

        if (
            seg is not None
            and not seg.empty
        ):
            segment_frames.append(
                seg
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # Place candidates are rebuilt across the complete degraded
    # 7-day clinical window, exactly as in the reference builder.
    # --------------------------------------------------------

    if segment_frames:

        window_segments = (
            pd.concat(
                segment_frames,
                ignore_index=True,
            )
        )

        assignments, places = (
            build_place_candidates(
                window_segments,
                compatibility_multiplier=(
                    COMPATIBILITY_MULTIPLIER
                ),
            )
        )

    else:

        assignments = pd.DataFrame(
            columns=[
                "date",
                "place_candidate_id",
                "duration_ms",
            ]
        )

        places = pd.DataFrame()

    daily_feature_values = []

    supported_fractions = []
    retained_fractions = []
    planned_coverages = []

    for _, day in (
        window_daily
        .sort_values(
            "window_day"
        )
        .iterrows()
    ):

        date = str(
            day[
                "date"
            ]
        )

        meta = (
            day_results[
                date
            ]
        )

        supported_fractions.append(
            meta[
                "supported_fraction_of_day"
            ]
        )

        retained_fractions.append(
            meta[
                "raw_fraction_retained"
            ]
        )

        planned_coverages.append(
            meta[
                "planned_coverage"
            ]
        )

        if (
            feature_name
            == "displacement_distance"
        ):

            value = (
                meta[
                    "displacement_distance_m"
                ]
            )

        elif (
            feature_name
            == "dominant_place_fraction"
        ):

            if assignments.empty:

                day_assignments = (
                    pd.DataFrame(
                        columns=[
                            "place_candidate_id",
                            "duration_ms",
                        ]
                    )
                )

            else:

                day_assignments = (
                    assignments[
                        assignments[
                            "date"
                        ]
                        .astype(str)
                        .eq(date)
                    ]
                )

            location = (
                compute_daily_location_features(
                    day_assignments,
                    supported_duration_ms=(
                        meta[
                            "supported_duration_ms"
                        ]
                    ),
                )
            )

            value = (
                location[
                    "dominant_place_fraction"
                ]
            )

        else:
            raise ValueError(
                feature_name
            )

        daily_feature_values.append(
            value
        )

    feature_array = pd.to_numeric(
        pd.Series(
            daily_feature_values
        ),
        errors="coerce",
    )

    n_feature_days = int(
        feature_array.notna().sum()
    )

    weekly_feature = (
        float(
            feature_array.mean()
        )
        if n_feature_days > 0
        else np.nan
    )

    return {
        "degraded_feature":
            weekly_feature,

        "degraded_feature_days":
            n_feature_days,

        "mean_supported_fraction":
            float(
                np.nanmean(
                    supported_fractions
                )
            ),

        "mean_raw_fraction_retained":
            float(
                np.nanmean(
                    retained_fractions
                )
            ),

        "mean_planned_coverage":
            float(
                np.nanmean(
                    planned_coverages
                )
            ),

        "n_window_place_candidates":
            (
                len(places)
                if places is not None
                else 0
            ),
    }


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

weekly = pd.read_parquet(
    WEEKLY_PATH
)

daily = pd.read_parquet(
    DAILY_PATH
)

weekly = weekly[
    weekly[
        "participant_id"
    ].eq(
        PARTICIPANT_ID
    )
    & weekly[
        "usable_feature_days"
    ].ge(7)
    & weekly[
        "depression_score"
    ].notna()
].copy()

weekly = (
    weekly.sort_values(
        "assessment_date"
    )
    .reset_index(
        drop=True
    )
)

daily = daily[
    daily[
        "participant_id"
    ].eq(
        PARTICIPANT_ID
    )
].copy()

daily[
    "date"
] = daily[
    "date"
].astype(str)


table = pq.read_table(
    RAW_PATH,
    filters=[
        (
            "participant_id",
            "=",
            PARTICIPANT_ID,
        )
    ],
)

participant_raw = (
    table.to_pandas()
)

participant_raw[
    "date"
] = participant_raw[
    "date"
].astype(str)


if len(
    weekly
) < 5:
    raise RuntimeError(
        f"{PARTICIPANT_ID}: fewer than five "
        "eligible reference assessments."
    )


print("=" * 100)
print(
    "PERSONALIZED GPS -> CLINICAL DEGRADATION"
)
print("=" * 100)

print(
    "Participant:",
    PARTICIPANT_ID,
)

print(
    "Feature:",
    FEATURE_NAME,
)

print(
    "Reference assessments:",
    len(weekly),
)

print(
    "Coverage levels:",
    COVERAGES,
)

print(
    "Patterns:",
    PATTERNS,
)

print(
    "Replicates:",
    N_REPLICATES,
)

print(
    "Conditions per assessment:",
    len(COVERAGES)
    * len(PATTERNS)
    * N_REPLICATES,
)

print()


# ============================================================
# EXPERIMENT
# ============================================================

rows = []


for test_index, test_row in (
    weekly.iterrows()
):

    window_id = (
        test_row[
            "clinical_window_id"
        ]
    )

    reference_feature = float(
        test_row[
            FEATURE_COLUMN
        ]
    )

    actual_score = float(
        test_row[
            "depression_score"
        ]
    )

    train = (
        weekly.drop(
            index=test_index
        )
    )

    train = train[
        train[
            FEATURE_COLUMN
        ].notna()
        & train[
            "depression_score"
        ].notna()
    ]

    model = (
        fit_reference_model(
            train[
                FEATURE_COLUMN
            ].to_numpy(
                dtype=float
            ),
            train[
                "depression_score"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    reference_prediction = (
        predict_one(
            model,
            reference_feature,
        )
    )

    reference_absolute_error = abs(
        actual_score
        - reference_prediction
    )

    window_daily = (
        daily[
            daily[
                "clinical_window_id"
            ].eq(
                window_id
            )
        ]
        .drop_duplicates(
            [
                "date",
                "window_day",
            ]
        )
        .sort_values(
            "window_day"
        )
    )

    if len(
        window_daily
    ) != 7:
        raise RuntimeError(
            f"{window_id}: expected 7 days, "
            f"found {len(window_daily)}."
        )

    print(
        "-" * 100
    )

    print(
        f"{PARTICIPANT_ID} | "
        f"{window_id} | "
        f"reference feature="
        f"{reference_feature:.4f} | "
        f"PHQ={actual_score:.1f}",
        flush=True,
    )

    for coverage in COVERAGES:

        for pattern in PATTERNS:

            for replicate in range(
                N_REPLICATES
            ):

                degraded = (
                    rebuild_degraded_week(
                        participant_id=(
                            PARTICIPANT_ID
                        ),
                        window_id=(
                            window_id
                        ),
                        window_daily=(
                            window_daily
                        ),
                        participant_raw=(
                            participant_raw
                        ),
                        coverage=coverage,
                        pattern=pattern,
                        replicate=replicate,
                        feature_name=(
                            FEATURE_NAME
                        ),
                    )
                )

                degraded_feature = (
                    degraded[
                        "degraded_feature"
                    ]
                )

                if np.isfinite(
                    degraded_feature
                ):

                    degraded_prediction = (
                        predict_one(
                            model,
                            degraded_feature,
                        )
                    )

                    degraded_absolute_error = abs(
                        actual_score
                        - degraded_prediction
                    )

                    prediction_perturbation = abs(
                        degraded_prediction
                        - reference_prediction
                    )

                    delta_mae = (
                        degraded_absolute_error
                        - reference_absolute_error
                    )

                    feature_delta = (
                        degraded_feature
                        - reference_feature
                    )

                    feature_absolute_error = abs(
                        feature_delta
                    )

                    if (
                        reference_feature
                        != 0
                    ):
                        feature_relative_error = (
                            feature_absolute_error
                            / abs(
                                reference_feature
                            )
                        )
                    else:
                        feature_relative_error = (
                            np.nan
                        )

                else:

                    degraded_prediction = (
                        np.nan
                    )

                    degraded_absolute_error = (
                        np.nan
                    )

                    prediction_perturbation = (
                        np.nan
                    )

                    delta_mae = (
                        np.nan
                    )

                    feature_delta = (
                        np.nan
                    )

                    feature_absolute_error = (
                        np.nan
                    )

                    feature_relative_error = (
                        np.nan
                    )

                rows.append(
                    {
                        "participant_id":
                            PARTICIPANT_ID,

                        "clinical_window_id":
                            window_id,

                        "assessment_date":
                            test_row[
                                "assessment_date"
                            ],

                        "outcome":
                            "depression_score",

                        "clinical_feature":
                            FEATURE_NAME,

                        "coverage":
                            coverage,

                        "pattern":
                            pattern,

                        "replicate":
                            replicate,

                        "actual_score":
                            actual_score,

                        "reference_feature":
                            reference_feature,

                        "degraded_feature":
                            degraded_feature,

                        "feature_delta":
                            feature_delta,

                        "feature_absolute_error":
                            feature_absolute_error,

                        "feature_relative_error":
                            feature_relative_error,

                        "reference_prediction":
                            reference_prediction,

                        "degraded_prediction":
                            degraded_prediction,

                        "reference_absolute_error":
                            reference_absolute_error,

                        "degraded_absolute_error":
                            degraded_absolute_error,

                        "prediction_perturbation":
                            prediction_perturbation,

                        "delta_mae":
                            delta_mae,

                        "degraded_feature_days":
                            degraded[
                                "degraded_feature_days"
                            ],

                        "mean_supported_fraction":
                            degraded[
                                "mean_supported_fraction"
                            ],

                        "mean_raw_fraction_retained":
                            degraded[
                                "mean_raw_fraction_retained"
                            ],

                        "mean_planned_coverage":
                            degraded[
                                "mean_planned_coverage"
                            ],

                        "n_window_place_candidates":
                            degraded[
                                "n_window_place_candidates"
                            ],

                        "n_reference_training_weeks":
                            len(train),
                    }
                )

            condition_rows = (
                pd.DataFrame(
                    rows
                )
            )

            current = (
                condition_rows[
                    condition_rows[
                        "clinical_window_id"
                    ].eq(
                        window_id
                    )
                    & condition_rows[
                        "coverage"
                    ].eq(
                        coverage
                    )
                    & condition_rows[
                        "pattern"
                    ].eq(
                        pattern
                    )
                ]
            )

            print(
                f"  coverage={coverage:.2f} "
                f"{pattern:11s} | "
                f"mean feature error="
                f"{current['feature_absolute_error'].mean():.4f} | "
                f"mean prediction perturbation="
                f"{current['prediction_perturbation'].mean():.4f} | "
                f"mean ΔMAE="
                f"{current['delta_mae'].mean():+.4f}",
                flush=True,
            )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(
    rows
)

result.to_parquet(
    OUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    result.groupby(
        [
            "participant_id",
            "clinical_feature",
            "coverage",
            "pattern",
        ],
        as_index=False,
    )
    .agg(
        n_conditions=(
            "delta_mae",
            "size",
        ),

        n_valid=(
            "delta_mae",
            "count",
        ),

        mean_feature_absolute_error=(
            "feature_absolute_error",
            "mean",
        ),

        median_feature_absolute_error=(
            "feature_absolute_error",
            "median",
        ),

        mean_prediction_perturbation=(
            "prediction_perturbation",
            "mean",
        ),

        mean_delta_mae=(
            "delta_mae",
            "mean",
        ),

        median_delta_mae=(
            "delta_mae",
            "median",
        ),

        fraction_prediction_worsened=(
            "delta_mae",
            lambda x:
                float(
                    (
                        x.dropna()
                        > 0
                    ).mean()
                )
                if x.notna().any()
                else np.nan,
        ),

        mean_supported_fraction=(
            "mean_supported_fraction",
            "mean",
        ),

        mean_raw_fraction_retained=(
            "mean_raw_fraction_retained",
            "mean",
        ),
    )
)


SUMMARY_PATH = (
    OUT_DIR
    / (
        f"{PARTICIPANT_ID}_"
        "gps_clinical_degradation_summary.csv"
    )
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "Saved conditions:",
    OUT_PATH,
)

print(
    "Saved summary:",
    SUMMARY_PATH,
)
