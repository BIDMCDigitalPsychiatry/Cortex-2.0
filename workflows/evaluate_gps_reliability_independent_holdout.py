"""Evaluate frozen RAPHE GPS reliability models on the independent holdout.

IMPORTANT
---------
The reliability model specification and coefficients were frozen before
holdout degradation outcomes were examined.

Primary comparison:
    coverage-only model
vs
    coverage + missingness-geometry model

Calibration analysis unit:
    participant x coverage x missingness design

The independent holdout is summarized identically:
    median across 7 reference days x 5 degradation replicates.

JSC434 remains in all absolute-error outputs. Its personalized natural
variability is zero for all three features, so normalized outcomes are
mathematically undefined and excluded only from normalized-error metrics.

No coefficient fitting or tuning occurs here.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from raphe.features.gps_degradation import (
    make_missing_bin_mask,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/"
    "raphe_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

EXTREME_DIR = (
    ROOT
    / "reliability_holdout_degradation_runs"
)

INTERMEDIATE_DIR = (
    ROOT
    / "reliability_holdout_intermediate_missingness_runs"
)

LOCKED_DAYS_PATH = (
    ROOT
    / "gps_reliability_holdout_locked_days.parquet"
)

VARIABILITY_PATH = (
    ROOT
    / "gps_reliability_holdout_reference_feature_variability_by_participant.csv"
)

COEFFICIENT_PATH = (
    ROOT
    / "gps_reliability_frozen_model_coefficients.csv"
)

DESIGN_PATH = (
    ROOT
    / "gps_reliability_frozen_model_design.json"
)


COMBINED_OUT = (
    ROOT
    / "gps_reliability_holdout_continuum_normalized.parquet"
)

PARTICIPANT_DESIGN_OUT = (
    ROOT
    / "gps_reliability_holdout_participant_design.csv"
)

PREDICTION_OUT = (
    ROOT
    / "gps_reliability_holdout_frozen_model_predictions.csv"
)

PERFORMANCE_OUT = (
    ROOT
    / "gps_reliability_holdout_frozen_model_performance.csv"
)

BOOTSTRAP_OUT = (
    ROOT
    / "gps_reliability_holdout_frozen_model_bootstrap.csv"
)


# ============================================================
# FROZEN DESIGN
# ============================================================

MAX_GAP_MS = 60_000
BIN_SIZE_MS = 5 * 60_000
COMPATIBILITY_MULTIPLIER = 2.0

EXTREME_BASE_SEED = 20261020
EXTREME_SEED_NAMESPACE = (
    "reliability_holdout_validation"
)

N_BOOT = 5000
BOOTSTRAP_SEED = 20261023


EXPECTED_COEFFICIENT_SHA256 = (
    "6c1e420c382176be9cb8ec6c6e19048a9f429853f25fdc10e95e514dfda8a2a7"
)

EXPECTED_DESIGN_SHA256 = (
    "fad14bd9dfeb911c41bf077d0e52a1e5df3b2ae0fd745217e784339e7e2f28cb"
)


FEATURES = {
    "entropy": {
        "absolute_error":
            "abs_entropy_error",

        "denominator":
            "entropy_pairwise_median_abs_diff",
    },

    "dominant_fraction": {
        "absolute_error":
            "abs_dominant_fraction_error",

        "denominator":
            "dominant_fraction_pairwise_median_abs_diff",
    },

    "effective_places": {
        "absolute_error":
            "abs_effective_places_error",

        "denominator":
            "effective_places_pairwise_median_abs_diff",
    },
}


DESIGN_ORDER = {
    "k1_contiguous": 1,
    "k2": 2,
    "k4": 4,
    "k8": 8,
    "k16": 16,
    "random_distributed": 999,
}


# ============================================================
# HELPERS
# ============================================================

def sha256(
    path: Path,
) -> str:

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def stable_seed(
    *parts,
) -> int:

    text = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return (
        int.from_bytes(
            digest,
            byteorder="little",
            signed=False,
        )
        % (2 ** 32)
    )


def missing_run_concentration(
    mask,
) -> float:

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    n_missing = int(
        mask.sum()
    )

    if n_missing == 0:
        return 0.0

    padded = np.concatenate(
        [
            np.array(
                [False]
            ),
            mask,
            np.array(
                [False]
            ),
        ]
    )

    starts = np.flatnonzero(
        (~padded[:-1])
        & padded[1:]
    )

    ends = np.flatnonzero(
        padded[:-1]
        & (~padded[1:])
    )

    lengths = (
        ends
        - starts
    ).astype(
        float
    )

    proportions = (
        lengths
        / n_missing
    )

    return float(
        np.sum(
            proportions ** 2
        )
    )


def prediction_metrics(
    observed,
    predicted,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(
            observed
        )
        & np.isfinite(
            predicted
        )
    )

    observed = observed[
        valid
    ]

    predicted = predicted[
        valid
    ]

    residual = (
        observed
        - predicted
    )

    mae = float(
        np.mean(
            np.abs(
                residual
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    denominator = float(
        np.sum(
            (
                observed
                - np.mean(
                    observed
                )
            )
            ** 2
        )
    )

    if denominator > 0:

        r2 = float(
            1.0
            - (
                np.sum(
                    residual ** 2
                )
                / denominator
            )
        )

    else:

        r2 = np.nan

    return {
        "n":
            len(
                observed
            ),

        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,
    }


def percentile_ci(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    return (
        float(
            np.quantile(
                values,
                0.025,
            )
        ),
        float(
            np.quantile(
                values,
                0.975,
            )
        ),
    )


# ============================================================
# VERIFY FROZEN MODEL FILES BEFORE READING HOLDOUT OUTCOMES
# ============================================================

coefficient_hash = sha256(
    COEFFICIENT_PATH
)

design_hash = sha256(
    DESIGN_PATH
)


if (
    coefficient_hash
    != EXPECTED_COEFFICIENT_SHA256
):

    raise RuntimeError(
        "Frozen coefficient file hash mismatch.\n"
        f"Expected: {EXPECTED_COEFFICIENT_SHA256}\n"
        f"Found:    {coefficient_hash}"
    )


if (
    design_hash
    != EXPECTED_DESIGN_SHA256
):

    raise RuntimeError(
        "Frozen design file hash mismatch.\n"
        f"Expected: {EXPECTED_DESIGN_SHA256}\n"
        f"Found:    {design_hash}"
    )


print("=" * 80)
print("FROZEN MODEL HASH CHECK")
print("=" * 80)

print(
    "Coefficient SHA256:",
    coefficient_hash,
)

print(
    "Design SHA256:",
    design_hash,
)

print(
    "Frozen model integrity: PASS"
)


# ============================================================
# LOCKED DAY WINDOWS
# ============================================================

locked_days = pd.read_parquet(
    LOCKED_DAYS_PATH
)

locked_days[
    "participant_id"
] = locked_days[
    "participant_id"
].astype(str)

locked_days[
    "date"
] = locked_days[
    "date"
].astype(str)


if (
    "window_start_ms"
    in locked_days.columns
):

    start_column = (
        "window_start_ms"
    )

else:

    start_column = (
        "start_ms"
    )


if (
    "window_end_ms"
    in locked_days.columns
):

    end_column = (
        "window_end_ms"
    )

else:

    end_column = (
        "end_ms"
    )


window_lookup = {
    (
        str(
            row[
                "participant_id"
            ]
        ),
        str(
            row[
                "date"
            ]
        ),
    ): (
        int(
            row[
                start_column
            ]
        ),
        int(
            row[
                end_column
            ]
        ),
    )
    for _, row
    in locked_days.iterrows()
}


if len(
    window_lookup
) != 112:

    raise RuntimeError(
        "Expected 112 unique locked participant-day windows."
    )


# ============================================================
# LOAD EXTREME RUNS
# ============================================================

extreme_files = sorted(
    EXTREME_DIR.glob(
        "coverage_*_*_rep_*.parquet"
    )
)


if len(
    extreme_files
) != 40:

    raise RuntimeError(
        f"Expected 40 extreme files, found {len(extreme_files)}."
    )


extreme_frames = []


for path in extreme_files:

    x = pd.read_parquet(
        path
    )

    x[
        "participant_id"
    ] = x[
        "participant_id"
    ].astype(str)

    x[
        "date"
    ] = x[
        "date"
    ].astype(str)


    if len(x) != 112:

        raise RuntimeError(
            f"{path.name}: expected 112 rows."
        )


    pattern_values = (
        x[
            "missingness_pattern"
        ]
        .astype(str)
        .unique()
    )

    if len(
        pattern_values
    ) != 1:

        raise RuntimeError(
            f"{path.name}: expected one pattern."
        )


    pattern = str(
        pattern_values[0]
    )


    if pattern == "contiguous":

        x[
            "missingness_design"
        ] = "k1_contiguous"

    elif pattern == "distributed":

        x[
            "missingness_design"
        ] = "random_distributed"

    else:

        raise RuntimeError(
            f"{path.name}: unknown pattern {pattern}."
        )


    concentrations = []


    for _, row in x.iterrows():

        participant_id = str(
            row[
                "participant_id"
            ]
        )

        date = str(
            row[
                "date"
            ]
        )

        key = (
            participant_id,
            date,
        )


        if key not in window_lookup:

            raise RuntimeError(
                f"Locked day not found: {key}"
            )


        start_ms, end_ms = (
            window_lookup[
                key
            ]
        )


        coverage = float(
            row[
                "target_coverage"
            ]
        )

        replicate = int(
            row[
                "replicate"
            ]
        )


        seed = stable_seed(
            EXTREME_BASE_SEED,
            EXTREME_SEED_NAMESPACE,
            participant_id,
            date,
            coverage,
            pattern,
            replicate,
        )


        missing_mask, _ = (
            make_missing_bin_mask(
                window_start_ms=
                    start_ms,

                window_end_ms=
                    end_ms,

                target_coverage=
                    coverage,

                pattern=
                    pattern,

                bin_size_ms=
                    BIN_SIZE_MS,

                seed=
                    seed,
            )
        )


        concentrations.append(
            missing_run_concentration(
                missing_mask
            )
        )


    x[
        "missing_run_concentration"
    ] = concentrations


    if (
        "mask_planned_coverage"
        in x.columns
    ):

        x[
            "planned_coverage_canonical"
        ] = pd.to_numeric(
            x[
                "mask_planned_coverage"
            ],
            errors="coerce",
        )

    elif (
        "planned_coverage"
        in x.columns
    ):

        x[
            "planned_coverage_canonical"
        ] = pd.to_numeric(
            x[
                "planned_coverage"
            ],
            errors="coerce",
        )

    else:

        x[
            "planned_coverage_canonical"
        ] = pd.to_numeric(
            x[
                "target_coverage"
            ],
            errors="coerce",
        )


    extreme_frames.append(
        x
    )


extreme = pd.concat(
    extreme_frames,
    ignore_index=True,
)


# ============================================================
# LOAD INTERMEDIATE RUNS
# ============================================================

intermediate_files = sorted(
    INTERMEDIATE_DIR.glob(
        "coverage_*_k*_rep_*.parquet"
    )
)


if len(
    intermediate_files
) != 80:

    raise RuntimeError(
        f"Expected 80 intermediate files, found {len(intermediate_files)}."
    )


intermediate_frames = []


for path in intermediate_files:

    x = pd.read_parquet(
        path
    )

    x[
        "participant_id"
    ] = x[
        "participant_id"
    ].astype(str)

    x[
        "date"
    ] = x[
        "date"
    ].astype(str)


    if len(x) != 112:

        raise RuntimeError(
            f"{path.name}: expected 112 rows."
        )


    n_run_column = None


    for candidate in [
        "n_runs_design",
        "n_runs",
        "missing_run_count_design",
    ]:

        if candidate in x.columns:

            n_run_column = candidate
            break


    if n_run_column is None:

        raise RuntimeError(
            f"{path.name}: no k-run design column found."
        )


    k_values = (
        pd.to_numeric(
            x[
                n_run_column
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
    )


    if len(
        k_values
    ) != 1:

        raise RuntimeError(
            f"{path.name}: expected one k value."
        )


    k = int(
        k_values[0]
    )


    x[
        "missingness_design"
    ] = f"k{k}"


    if (
        "missing_run_concentration"
        not in x.columns
    ):

        raise RuntimeError(
            f"{path.name}: missing concentration descriptor."
        )


    if (
        "planned_coverage"
        in x.columns
    ):

        x[
            "planned_coverage_canonical"
        ] = pd.to_numeric(
            x[
                "planned_coverage"
            ],
            errors="coerce",
        )

    else:

        x[
            "planned_coverage_canonical"
        ] = pd.to_numeric(
            x[
                "target_coverage"
            ],
            errors="coerce",
        )


    intermediate_frames.append(
        x
    )


intermediate = pd.concat(
    intermediate_frames,
    ignore_index=True,
)


# ============================================================
# COMBINE FULL HOLDOUT MATRIX
# ============================================================

combined = pd.concat(
    [
        extreme,
        intermediate,
    ],
    ignore_index=True,
)


if len(
    combined
) != 13_440:

    raise RuntimeError(
        f"Expected 13,440 rows, found {len(combined):,}."
    )


combined[
    "missing_fraction"
] = (
    1.0
    - pd.to_numeric(
        combined[
            "planned_coverage_canonical"
        ],
        errors="coerce",
    )
)


combined[
    "design_order"
] = combined[
    "missingness_design"
].map(
    DESIGN_ORDER
)


if combined[
    "design_order"
].isna().any():

    raise RuntimeError(
        "Unknown missingness design after combination."
    )


# ============================================================
# ATTACH PERSONALIZED NATURAL-VARIABILITY SCALES
# ============================================================

variability = pd.read_csv(
    VARIABILITY_PATH
)

variability[
    "participant_id"
] = variability[
    "participant_id"
].astype(str)


if (
    variability[
        "participant_id"
    ].nunique()
    != 16
):

    raise RuntimeError(
        "Expected 16 holdout variability rows."
    )


denominator_columns = [
    specification[
        "denominator"
    ]
    for specification
    in FEATURES.values()
]


combined = combined.merge(
    variability[
        [
            "participant_id",
        ]
        + denominator_columns
    ],
    on="participant_id",
    how="left",
    validate="many_to_one",
)


for feature, specification in FEATURES.items():

    absolute_column = (
        specification[
            "absolute_error"
        ]
    )

    denominator_column = (
        specification[
            "denominator"
        ]
    )

    normalized_column = (
        f"{feature}_normalized_error"
    )


    if absolute_column not in combined.columns:

        raise RuntimeError(
            f"Missing absolute-error column: {absolute_column}"
        )


    denominator = pd.to_numeric(
        combined[
            denominator_column
        ],
        errors="coerce",
    )


    absolute_error = pd.to_numeric(
        combined[
            absolute_column
        ],
        errors="coerce",
    )


    valid = (
        denominator.notna()
        & np.isfinite(
            denominator
        )
        & (
            denominator
            > 0
        )
    )


    combined[
        f"{feature}_normalization_valid"
    ] = valid


    combined[
        normalized_column
    ] = np.where(
        valid,
        absolute_error
        / denominator,
        np.nan,
    )


combined.to_parquet(
    COMBINED_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT-BALANCED ANALYSIS UNIT
#
# 7 days x 5 replicates = 35 observations per
# participant x coverage x structural design.
# ============================================================

participant_rows = []


for (
    participant_id,
    coverage,
    design,
), x in combined.groupby(
    [
        "participant_id",
        "target_coverage",
        "missingness_design",
    ],
    sort=True,
):

    if len(
        x
    ) != 35:

        raise RuntimeError(
            f"{participant_id}, {coverage}, {design}: "
            f"expected 35 rows, found {len(x)}."
        )


    row = {
        "participant_id":
            participant_id,

        "target_coverage":
            float(
                coverage
            ),

        "missingness_design":
            design,

        "design_order":
            DESIGN_ORDER[
                design
            ],

        "observations":
            len(x),

        "median_missing_fraction":
            float(
                pd.to_numeric(
                    x[
                        "missing_fraction"
                    ],
                    errors="coerce",
                ).median()
            ),

        "median_missing_run_concentration":
            float(
                pd.to_numeric(
                    x[
                        "missing_run_concentration"
                    ],
                    errors="coerce",
                ).median()
            ),
    }


    for feature, specification in FEATURES.items():

        absolute_column = (
            specification[
                "absolute_error"
            ]
        )

        normalized_column = (
            f"{feature}_normalized_error"
        )


        row[
            f"median_{feature}_absolute_error"
        ] = float(
            pd.to_numeric(
                x[
                    absolute_column
                ],
                errors="coerce",
            ).median()
        )


        normalized_values = pd.to_numeric(
            x[
                normalized_column
            ],
            errors="coerce",
        )


        if normalized_values.notna().any():

            row[
                f"median_{feature}_normalized_error"
            ] = float(
                normalized_values.median()
            )

            row[
                f"{feature}_normalization_valid"
            ] = True

        else:

            row[
                f"median_{feature}_normalized_error"
            ] = np.nan

            row[
                f"{feature}_normalization_valid"
            ] = False


    participant_rows.append(
        row
    )


participant_design = pd.DataFrame(
    participant_rows
)


if len(
    participant_design
) != 384:

    raise RuntimeError(
        f"Expected 384 participant-design rows, "
        f"found {len(participant_design)}."
    )


participant_design.to_csv(
    PARTICIPANT_DESIGN_OUT,
    index=False,
)


# ============================================================
# LOAD FROZEN COEFFICIENTS
# ============================================================

coefficients = pd.read_csv(
    COEFFICIENT_PATH
)


coefficient_lookup = {
    (
        str(
            row[
                "feature"
            ]
        ),
        str(
            row[
                "model"
            ]
        ),
        str(
            row[
                "term"
            ]
        ),
    ):
        float(
            row[
                "coefficient"
            ]
        )

    for _, row
    in coefficients.iterrows()
}


def frozen_prediction(
    feature,
    model,
    missing_fraction,
    concentration,
):

    values = {
        "intercept":
            1.0,

        "missing_fraction":
            missing_fraction,

        "missing_fraction_sq":
            missing_fraction ** 2,

        "missing_run_concentration":
            concentration,

        "missing_run_concentration_sq":
            concentration ** 2,

        "missing_fraction_x_concentration":
            missing_fraction
            * concentration,
    }


    if model == "coverage_only":

        terms = [
            "intercept",
            "missing_fraction",
            "missing_fraction_sq",
        ]

    elif model == "coverage_geometry":

        terms = [
            "intercept",
            "missing_fraction",
            "missing_fraction_sq",
            "missing_run_concentration",
            "missing_run_concentration_sq",
            "missing_fraction_x_concentration",
        ]

    else:

        raise ValueError(
            model
        )


    return float(
        sum(
            coefficient_lookup[
                (
                    feature,
                    model,
                    term,
                )
            ]
            * values[
                term
            ]

            for term in terms
        )
    )


# ============================================================
# APPLY FROZEN MODELS
# ============================================================

prediction_rows = []


for _, row in participant_design.iterrows():

    m = float(
        row[
            "median_missing_fraction"
        ]
    )

    c = float(
        row[
            "median_missing_run_concentration"
        ]
    )


    for feature in FEATURES:

        observed = row[
            f"median_{feature}_normalized_error"
        ]


        prediction_rows.append({
            "participant_id":
                row[
                    "participant_id"
                ],

            "target_coverage":
                row[
                    "target_coverage"
                ],

            "missingness_design":
                row[
                    "missingness_design"
                ],

            "feature":
                feature,

            "normalization_valid":
                bool(
                    row[
                        f"{feature}_normalization_valid"
                    ]
                ),

            "missing_fraction":
                m,

            "missing_run_concentration":
                c,

            "observed_normalized_error":
                observed,

            "predicted_coverage_only":
                frozen_prediction(
                    feature,
                    "coverage_only",
                    m,
                    c,
                ),

            "predicted_coverage_geometry":
                frozen_prediction(
                    feature,
                    "coverage_geometry",
                    m,
                    c,
                ),
        })


predictions = pd.DataFrame(
    prediction_rows
)


predictions.to_csv(
    PREDICTION_OUT,
    index=False,
)


# ============================================================
# PRIMARY INDEPENDENT-HOLDOUT PERFORMANCE
# ============================================================

performance_rows = []


for feature, x in predictions.groupby(
    "feature",
    sort=True,
):

    valid = x[
        x[
            "normalization_valid"
        ]
    ].copy()


    observed = (
        valid[
            "observed_normalized_error"
        ]
        .to_numpy(
            dtype=float
        )
    )


    coverage_prediction = (
        valid[
            "predicted_coverage_only"
        ]
        .to_numpy(
            dtype=float
        )
    )


    geometry_prediction = (
        valid[
            "predicted_coverage_geometry"
        ]
        .to_numpy(
            dtype=float
        )
    )


    coverage_metrics = prediction_metrics(
        observed,
        coverage_prediction,
    )

    geometry_metrics = prediction_metrics(
        observed,
        geometry_prediction,
    )


    performance_rows.append({
        "feature":
            feature,

        "participants_total":
            x[
                "participant_id"
            ].nunique(),

        "participants_normalized":
            valid[
                "participant_id"
            ].nunique(),

        "predictions":
            len(valid),

        "coverage_only_mae":
            coverage_metrics[
                "mae"
            ],

        "coverage_geometry_mae":
            geometry_metrics[
                "mae"
            ],

        "mae_reduction_fraction":
            (
                coverage_metrics[
                    "mae"
                ]
                - geometry_metrics[
                    "mae"
                ]
            )
            / coverage_metrics[
                "mae"
            ],

        "coverage_only_rmse":
            coverage_metrics[
                "rmse"
            ],

        "coverage_geometry_rmse":
            geometry_metrics[
                "rmse"
            ],

        "rmse_reduction_fraction":
            (
                coverage_metrics[
                    "rmse"
                ]
                - geometry_metrics[
                    "rmse"
                ]
            )
            / coverage_metrics[
                "rmse"
            ],

        "coverage_only_r2":
            coverage_metrics[
                "r2"
            ],

        "coverage_geometry_r2":
            geometry_metrics[
                "r2"
            ],

        "r2_gain":
            (
                geometry_metrics[
                    "r2"
                ]
                - coverage_metrics[
                    "r2"
                ]
            ),
    })


performance = pd.DataFrame(
    performance_rows
)


performance.to_csv(
    PERFORMANCE_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT-CLUSTER BOOTSTRAP
#
# Predictions are already frozen. Resample independent-holdout
# participants as paired clusters.
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


bootstrap_rows = []


for feature, x in predictions.groupby(
    "feature",
    sort=True,
):

    x = x[
        x[
            "normalization_valid"
        ]
    ].copy()


    participants = sorted(
        x[
            "participant_id"
        ].unique()
    )


    blocks = []


    for participant_id in participants:

        block = (
            x[
                x[
                    "participant_id"
                ]
                == participant_id
            ]
            .sort_values(
                [
                    "target_coverage",
                    "missingness_design",
                ]
            )
        )


        if len(
            block
        ) != 24:

            raise RuntimeError(
                f"{feature} {participant_id}: "
                f"expected 24 predictions, found {len(block)}."
            )


        blocks.append(
            block[
                [
                    "observed_normalized_error",
                    "predicted_coverage_only",
                    "predicted_coverage_geometry",
                ]
            ].to_numpy(
                dtype=float
            )
        )


    blocks = np.stack(
        blocks,
        axis=0,
    )


    n_participants = (
        blocks.shape[0]
    )


    mae_reductions = np.empty(
        N_BOOT,
        dtype=float,
    )

    rmse_reductions = np.empty(
        N_BOOT,
        dtype=float,
    )

    r2_gains = np.empty(
        N_BOOT,
        dtype=float,
    )


    for bootstrap_index in range(
        N_BOOT
    ):

        sampled_indices = rng.integers(
            0,
            n_participants,
            size=n_participants,
        )


        sampled = blocks[
            sampled_indices
        ]


        observed = sampled[
            :,
            :,
            0,
        ].reshape(
            -1
        )

        coverage_prediction = sampled[
            :,
            :,
            1,
        ].reshape(
            -1
        )

        geometry_prediction = sampled[
            :,
            :,
            2,
        ].reshape(
            -1
        )


        coverage_metrics = prediction_metrics(
            observed,
            coverage_prediction,
        )

        geometry_metrics = prediction_metrics(
            observed,
            geometry_prediction,
        )


        mae_reductions[
            bootstrap_index
        ] = (
            coverage_metrics[
                "mae"
            ]
            - geometry_metrics[
                "mae"
            ]
        ) / coverage_metrics[
            "mae"
        ]


        rmse_reductions[
            bootstrap_index
        ] = (
            coverage_metrics[
                "rmse"
            ]
            - geometry_metrics[
                "rmse"
            ]
        ) / coverage_metrics[
            "rmse"
        ]


        r2_gains[
            bootstrap_index
        ] = (
            geometry_metrics[
                "r2"
            ]
            - coverage_metrics[
                "r2"
            ]
        )


    mae_lower, mae_upper = percentile_ci(
        mae_reductions
    )

    rmse_lower, rmse_upper = percentile_ci(
        rmse_reductions
    )

    r2_lower, r2_upper = percentile_ci(
        r2_gains
    )


    bootstrap_rows.append({
        "feature":
            feature,

        "participants":
            n_participants,

        "bootstrap_resamples":
            N_BOOT,

        "mae_reduction_ci95_lower":
            mae_lower,

        "mae_reduction_ci95_upper":
            mae_upper,

        "bootstrap_probability_mae_improvement":
            float(
                np.mean(
                    mae_reductions
                    > 0
                )
            ),

        "rmse_reduction_ci95_lower":
            rmse_lower,

        "rmse_reduction_ci95_upper":
            rmse_upper,

        "bootstrap_probability_rmse_improvement":
            float(
                np.mean(
                    rmse_reductions
                    > 0
                )
            ),

        "r2_gain_ci95_lower":
            r2_lower,

        "r2_gain_ci95_upper":
            r2_upper,

        "bootstrap_probability_r2_gain":
            float(
                np.mean(
                    r2_gains
                    > 0
                )
            ),
    })


bootstrap = pd.DataFrame(
    bootstrap_rows
)


bootstrap.to_csv(
    BOOTSTRAP_OUT,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

report = performance.merge(
    bootstrap,
    on="feature",
    how="left",
    validate="one_to_one",
)


print()
print("=" * 80)
print("INDEPENDENT HOLDOUT — FROZEN MODEL PERFORMANCE")
print("=" * 80)

print(
    report[
        [
            "feature",
            "participants_total",
            "participants_normalized",
            "predictions",
            "coverage_only_mae",
            "coverage_geometry_mae",
            "mae_reduction_fraction",
            "mae_reduction_ci95_lower",
            "mae_reduction_ci95_upper",
            "coverage_only_rmse",
            "coverage_geometry_rmse",
            "rmse_reduction_fraction",
            "rmse_reduction_ci95_lower",
            "rmse_reduction_ci95_upper",
            "coverage_only_r2",
            "coverage_geometry_r2",
            "r2_gain",
            "r2_gain_ci95_lower",
            "r2_gain_ci95_upper",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print()
print("=" * 80)
print("BOOTSTRAP SUPPORT")
print("=" * 80)

print(
    report[
        [
            "feature",
            "bootstrap_probability_mae_improvement",
            "bootstrap_probability_rmse_improvement",
            "bootstrap_probability_r2_gain",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print()
print("=" * 80)
print("NORMALIZATION AUDIT")
print("=" * 80)

normalization_audit = (
    predictions[
        [
            "participant_id",
            "feature",
            "normalization_valid",
        ]
    ]
    .drop_duplicates()
    .groupby(
        [
            "participant_id",
            "feature",
        ],
        as_index=False,
    )[
        "normalization_valid"
    ]
    .first()
)


invalid = normalization_audit[
    ~normalization_audit[
        "normalization_valid"
    ]
]


print(
    "Participants retained in absolute-error matrix:",
    combined[
        "participant_id"
    ].nunique(),
)

print(
    "Invalid participant-feature normalization pairs:",
    len(
        invalid
    ),
)

print(
    invalid.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(COMBINED_OUT)
print(PARTICIPANT_DESIGN_OUT)
print(PREDICTION_OUT)
print(PERFORMANCE_OUT)
print(BOOTSTRAP_OUT)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
