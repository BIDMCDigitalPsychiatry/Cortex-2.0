"""Exploratory participant-level GPS-symptom profiling.

For participants with >=5 complete 7-day assessments:

1. quantify symptom variability;
2. compute within-person Spearman association for each GPS feature;
3. evaluate one-feature-at-a-time personalized prediction using
   leave-one-assessment-out cross-validation;
4. compare each GPS feature against the participant's own training-mean
   symptom baseline.

This is exploratory individualized profiling, not confirmatory inference.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DC = Path("/shared/Digital Clinic")

ROOT = (
    DC
    / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

INFILE = (
    ROOT
    / "gps_clinical_weekly_features.parquet"
)

LONG_OUT = (
    ROOT
    / "personalized_gps_symptom_feature_profiles.csv"
)

SUMMARY_OUT = (
    ROOT
    / "personalized_gps_symptom_participant_summary.csv"
)


FEATURES = {
    "location_entropy_mean":
        "location_entropy",

    "effective_places_mean":
        "effective_places",

    "dominant_place_fraction_mean":
        "dominant_place_fraction",

    "observed_places_mean":
        "observed_places",

    "displacement_distance_mean_m":
        "displacement_distance",
}


OUTCOMES = [
    "depression_score",
    "anxiety_score",
]


MIN_ASSESSMENTS = 5


def loo_univariate_prediction(
    x,
    y,
):
    """Leave-one-out personalized prediction for one feature."""

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    baseline_predictions = []
    feature_predictions = []
    observed = []

    n = len(y)

    for test_i in range(n):

        train_mask = np.ones(
            n,
            dtype=bool,
        )

        train_mask[
            test_i
        ] = False

        X_train = (
            x[
                train_mask
            ]
            .reshape(-1, 1)
        )

        X_test = (
            x[
                [test_i]
            ]
            .reshape(-1, 1)
        )

        y_train = (
            y[
                train_mask
            ]
        )

        y_test = float(
            y[
                test_i
            ]
        )

        baseline = float(
            np.mean(
                y_train
            )
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
            X_train,
            y_train,
        )

        prediction = float(
            model.predict(
                X_test
            )[0]
        )

        baseline_predictions.append(
            baseline
        )

        feature_predictions.append(
            prediction
        )

        observed.append(
            y_test
        )

    observed = np.asarray(
        observed
    )

    baseline_predictions = np.asarray(
        baseline_predictions
    )

    feature_predictions = np.asarray(
        feature_predictions
    )

    baseline_mae = (
        mean_absolute_error(
            observed,
            baseline_predictions,
        )
    )

    feature_mae = (
        mean_absolute_error(
            observed,
            feature_predictions,
        )
    )

    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                observed,
                baseline_predictions,
            )
        )
    )

    feature_rmse = float(
        np.sqrt(
            mean_squared_error(
                observed,
                feature_predictions,
            )
        )
    )

    return {
        "baseline_mae":
            baseline_mae,

        "feature_mae":
            feature_mae,

        "mae_improvement":
            baseline_mae
            - feature_mae,

        "relative_mae_improvement":
            (
                (
                    baseline_mae
                    - feature_mae
                )
                / baseline_mae
                if baseline_mae > 0
                else np.nan
            ),

        "baseline_rmse":
            baseline_rmse,

        "feature_rmse":
            feature_rmse,

        "rmse_improvement":
            baseline_rmse
            - feature_rmse,
    }


df = pd.read_parquet(
    INFILE
)

# Primary personalized analysis:
# complete 7/7 GPS weeks only.
df = df[
    df[
        "usable_feature_days"
    ]
    >= 7
].copy()


print("=" * 90)
print("PERSONALIZED GPS-SYMPTOM PROFILING")
print("=" * 90)

print(
    "Complete clinical windows:",
    len(df),
)

print(
    "Participants:",
    df[
        "participant_id"
    ].nunique(),
)


rows = []


for outcome in OUTCOMES:

    for participant_id, g in (
        df.groupby(
            "participant_id"
        )
    ):

        g = (
            g[
                g[outcome].notna()
            ]
            .sort_values(
                "assessment_date"
            )
            .copy()
        )

        n_assessments = len(
            g
        )

        if (
            n_assessments
            < MIN_ASSESSMENTS
        ):
            continue

        symptom_mean = float(
            g[
                outcome
            ].mean()
        )

        symptom_sd = float(
            g[
                outcome
            ].std()
        )

        symptom_range = float(
            g[
                outcome
            ].max()
            - g[
                outcome
            ].min()
        )

        symptom_variable = (
            np.isfinite(
                symptom_sd
            )
            and symptom_sd > 0
        )

        for feature_col, feature_name in (
            FEATURES.items()
        ):

            x = pd.to_numeric(
                g[
                    feature_col
                ],
                errors="coerce",
            )

            y = pd.to_numeric(
                g[
                    outcome
                ],
                errors="coerce",
            )

            valid = (
                x.notna()
                & y.notna()
            )

            xv = (
                x[
                    valid
                ]
                .to_numpy(
                    dtype=float
                )
            )

            yv = (
                y[
                    valid
                ]
                .to_numpy(
                    dtype=float
                )
            )

            n_valid = len(
                yv
            )

            feature_sd = (
                float(
                    np.std(
                        xv,
                        ddof=1,
                    )
                )
                if n_valid > 1
                else np.nan
            )

            if (
                n_valid
                < MIN_ASSESSMENTS
                or not symptom_variable
                or not np.isfinite(
                    feature_sd
                )
                or feature_sd == 0
            ):

                rho = np.nan
                pvalue = np.nan

                prediction = {
                    "baseline_mae":
                        np.nan,
                    "feature_mae":
                        np.nan,
                    "mae_improvement":
                        np.nan,
                    "relative_mae_improvement":
                        np.nan,
                    "baseline_rmse":
                        np.nan,
                    "feature_rmse":
                        np.nan,
                    "rmse_improvement":
                        np.nan,
                }

            else:

                rho_result = (
                    spearmanr(
                        xv,
                        yv,
                    )
                )

                rho = float(
                    rho_result.statistic
                )

                pvalue = float(
                    rho_result.pvalue
                )

                prediction = (
                    loo_univariate_prediction(
                        xv,
                        yv,
                    )
                )

            rows.append(
                {
                    "participant_id":
                        participant_id,

                    "outcome":
                        outcome,

                    "feature":
                        feature_name,

                    "feature_column":
                        feature_col,

                    "n_assessments":
                        n_assessments,

                    "n_valid":
                        n_valid,

                    "symptom_mean":
                        symptom_mean,

                    "symptom_sd":
                        symptom_sd,

                    "symptom_range":
                        symptom_range,

                    "symptom_variable":
                        symptom_variable,

                    "feature_sd":
                        feature_sd,

                    "spearman_rho":
                        rho,

                    # Descriptive only.
                    # Do not interpret as confirmatory significance.
                    "spearman_pvalue":
                        pvalue,

                    **prediction,
                }
            )


profiles = pd.DataFrame(
    rows
)

profiles.to_csv(
    LONG_OUT,
    index=False,
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

eligible = profiles[
    profiles[
        "mae_improvement"
    ].notna()
].copy()


summary_rows = []


for (
    participant_id,
    outcome,
), g in eligible.groupby(
    [
        "participant_id",
        "outcome",
    ]
):

    best = (
        g.sort_values(
            [
                "mae_improvement",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    strongest_association = (
        g.assign(
            abs_rho=
                g[
                    "spearman_rho"
                ].abs()
        )
        .sort_values(
            [
                "abs_rho",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    n_helpful = int(
        (
            g[
                "mae_improvement"
            ]
            > 0
        ).sum()
    )

    summary_rows.append(
        {
            "participant_id":
                participant_id,

            "outcome":
                outcome,

            "n_assessments":
                int(
                    best[
                        "n_assessments"
                    ]
                ),

            "symptom_mean":
                best[
                    "symptom_mean"
                ],

            "symptom_sd":
                best[
                    "symptom_sd"
                ],

            "symptom_range":
                best[
                    "symptom_range"
                ],

            "best_predictive_feature":
                best[
                    "feature"
                ],

            "best_feature_mae_improvement":
                best[
                    "mae_improvement"
                ],

            "best_feature_relative_mae_improvement":
                best[
                    "relative_mae_improvement"
                ],

            "best_feature_rho":
                best[
                    "spearman_rho"
                ],

            "strongest_association_feature":
                strongest_association[
                    "feature"
                ],

            "strongest_abs_rho":
                abs(
                    strongest_association[
                        "spearman_rho"
                    ]
                ),

            "strongest_rho_direction":
                (
                    "positive"
                    if strongest_association[
                        "spearman_rho"
                    ] > 0
                    else (
                        "negative"
                        if strongest_association[
                            "spearman_rho"
                        ] < 0
                        else "zero"
                    )
                ),

            "n_features_improving_mae":
                n_helpful,
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 90)
print("ELIGIBILITY")
print("=" * 90)

for outcome in OUTCOMES:

    x = summary[
        summary[
            "outcome"
        ].eq(outcome)
    ]

    print(
        outcome,
        ":",
        len(x),
        "participants",
    )


print()
print("=" * 90)
print("PARTICIPANTS WITH AT LEAST ONE GPS FEATURE IMPROVING PERSONALIZED MAE")
print("=" * 90)

for outcome in OUTCOMES:

    x = summary[
        summary[
            "outcome"
        ].eq(outcome)
    ]

    n = (
        x[
            "n_features_improving_mae"
        ]
        > 0
    ).sum()

    print(
        outcome,
        ":",
        n,
        "/",
        len(x),
    )


print()
print("=" * 90)
print("BEST PERSONALIZED RESULTS")
print("=" * 90)

display_cols = [
    "participant_id",
    "outcome",
    "n_assessments",
    "symptom_sd",
    "best_predictive_feature",
    "best_feature_mae_improvement",
    "best_feature_relative_mae_improvement",
    "best_feature_rho",
    "n_features_improving_mae",
]

print(
    summary[
        display_cols
    ]
    .sort_values(
        [
            "outcome",
            "best_feature_mae_improvement",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 90)
print("FEATURE-LEVEL COUNTS")
print("=" * 90)

feature_summary = (
    eligible.groupby(
        [
            "outcome",
            "feature",
        ]
    )
    .agg(
        participants=(
            "participant_id",
            "nunique",
        ),

        participants_improved=(
            "mae_improvement",
            lambda x:
                int(
                    (x > 0).sum()
                ),
        ),

        median_mae_improvement=(
            "mae_improvement",
            "median",
        ),

        median_spearman_rho=(
            "spearman_rho",
            "median",
        ),
    )
    .reset_index()
)

print(
    feature_summary.to_string(
        index=False
    )
)

print()
print(
    "Saved feature profiles:",
    LONG_OUT,
)

print(
    "Saved participant summary:",
    SUMMARY_OUT,
)
