"""Freeze Cortex 2.0 GPS reliability calibration models.

Fits the exact models previously evaluated by LOPO CV using all 16
calibration participants.

No independent-holdout data are accessed here.

Models
------
Coverage only:
    error ~ 1 + m + m^2

Coverage + geometry:
    error ~ 1 + m + m^2 + c + c^2 + m*c

where:
    m = missing fraction
    c = missing-run concentration

Outcome:
    participant-normalized GPS feature degradation error

Features:
    entropy
    dominant_fraction
    effective_places

These coefficients are frozen before examining degradation outcomes
in the independent holdout cohort.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/"
    "gps_observability"
)

CALIBRATION_PATH = (
    ROOT
    / "gps_missingness_continuum_participant_design.csv"
)

COEFFICIENT_OUT = (
    ROOT
    / "gps_reliability_frozen_model_coefficients.csv"
)

DESIGN_OUT = (
    ROOT
    / "gps_reliability_frozen_model_design.json"
)


FEATURES = {
    "entropy":
        "median_entropy_normalized_error",

    "dominant_fraction":
        "median_dominant_fraction_normalized_error",

    "effective_places":
        "median_effective_places_normalized_error",
}


COVERAGE_TERMS = [
    "intercept",
    "missing_fraction",
    "missing_fraction_sq",
]


GEOMETRY_TERMS = [
    "intercept",
    "missing_fraction",
    "missing_fraction_sq",
    "missing_run_concentration",
    "missing_run_concentration_sq",
    "missing_fraction_x_concentration",
]


def coverage_matrix(
    missing_fraction,
):
    m = np.asarray(
        missing_fraction,
        dtype=float,
    )

    return np.column_stack(
        [
            np.ones(len(m)),
            m,
            m ** 2,
        ]
    )


def geometry_matrix(
    missing_fraction,
    concentration,
):
    m = np.asarray(
        missing_fraction,
        dtype=float,
    )

    c = np.asarray(
        concentration,
        dtype=float,
    )

    return np.column_stack(
        [
            np.ones(len(m)),
            m,
            m ** 2,
            c,
            c ** 2,
            m * c,
        ]
    )


def fit(
    x,
    y,
):
    coefficients, _, _, _ = np.linalg.lstsq(
        x,
        y,
        rcond=None,
    )

    return coefficients


# ============================================================
# LOAD CALIBRATION DATA ONLY
# ============================================================

df = pd.read_csv(
    CALIBRATION_PATH
)


if len(df) != 384:
    raise RuntimeError(
        f"Expected 384 calibration rows, found {len(df)}."
    )


if (
    df["participant_id"].nunique()
    != 16
):
    raise RuntimeError(
        "Expected 16 calibration participants."
    )


m = pd.to_numeric(
    df[
        "median_missing_fraction"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)


c = pd.to_numeric(
    df[
        "median_missing_run_concentration"
    ],
    errors="raise",
).to_numpy(
    dtype=float
)


x_coverage = coverage_matrix(
    m
)

x_geometry = geometry_matrix(
    m,
    c,
)


coefficient_rows = []


for feature, outcome_column in FEATURES.items():

    y = pd.to_numeric(
        df[
            outcome_column
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )


    if not np.isfinite(
        y
    ).all():
        raise RuntimeError(
            f"Non-finite outcome for {feature}."
        )


    beta_coverage = fit(
        x_coverage,
        y,
    )


    beta_geometry = fit(
        x_geometry,
        y,
    )


    for term, value in zip(
        COVERAGE_TERMS,
        beta_coverage,
    ):

        coefficient_rows.append({
            "feature":
                feature,

            "model":
                "coverage_only",

            "term":
                term,

            "coefficient":
                float(value),
        })


    for term, value in zip(
        GEOMETRY_TERMS,
        beta_geometry,
    ):

        coefficient_rows.append({
            "feature":
                feature,

            "model":
                "coverage_geometry",

            "term":
                term,

            "coefficient":
                float(value),
        })


coefficients = pd.DataFrame(
    coefficient_rows
)


coefficients.to_csv(
    COEFFICIENT_OUT,
    index=False,
)


# ============================================================
# FREEZE DESIGN METADATA
# ============================================================

source_hash = hashlib.sha256(
    CALIBRATION_PATH.read_bytes()
).hexdigest()


design = {
    "status":
        "frozen_before_independent_holdout_degradation_analysis",

    "frozen_on":
        "2026-10-23",

    "calibration_source":
        str(
            CALIBRATION_PATH
        ),

    "calibration_source_sha256":
        source_hash,

    "calibration_participants":
        sorted(
            df[
                "participant_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),

    "calibration_participant_count":
        16,

    "calibration_rows":
        384,

    "analysis_unit":
        "participant_x_coverage_x_missingness_design",

    "features": [
        "entropy",
        "dominant_fraction",
        "effective_places",
    ],

    "outcome":
        "participant_normalized_absolute_feature_degradation_error",

    "normalization":
        (
            "participant median absolute difference across all "
            "pairs of seven well-observed reference days"
        ),

    "coverage_only_model": {
        "terms":
            COVERAGE_TERMS,
    },

    "coverage_geometry_model": {
        "terms":
            GEOMETRY_TERMS,

        "geometry_descriptor":
            "missing_run_concentration",
    },

    "prediction_clipping":
        "none",

    "model_fitting":
        "ordinary least squares using numpy.linalg.lstsq",

    "selection_note":
        (
            "Model specification was selected and evaluated using "
            "the 16-participant calibration cohort before independent "
            "holdout degradation outcomes were examined."
        ),
}


DESIGN_OUT.write_text(
    json.dumps(
        design,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


# ============================================================
# REPORT
# ============================================================

print("=" * 80)
print("Cortex 2.0 GPS RELIABILITY MODELS — FROZEN")
print("=" * 80)

print(
    "Calibration participants:",
    df[
        "participant_id"
    ].nunique(),
)

print(
    "Calibration rows:",
    len(df),
)

print(
    "Input SHA256:",
    source_hash,
)

print()
print("Frozen coefficients:")

print(
    coefficients.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.10f}",
    )
)

print()
print("Outputs:")
print(COEFFICIENT_OUT)
print(DESIGN_OUT)

print()
print("=" * 80)
print("FROZEN BEFORE HOLDOUT OUTCOME ANALYSIS")
print("=" * 80)
