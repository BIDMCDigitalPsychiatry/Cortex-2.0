"""Compare personalized PHQ regression algorithms for LPE511.

Same participant.
Same weekly GPS feature.
Same leave-one-assessment-out evaluation.

Only the prediction algorithm changes.

Models:
- Ridge regression
- Gaussian Process Regression
- Random Forest regression

Mixed-effects/hierarchical models are intentionally excluded because
a single participant provides no between-participant hierarchy.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

INPUT_PATH = (
    ROOT
    / "gps_clinical_weekly_features.parquet"
)

OUT_DIR = (
    ROOT
    / "personalized_model_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARTICIPANT_ID = "LPE511"

FEATURE = (
    "displacement_distance_mean_m"
)

OUTCOME = (
    "depression_score"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    INPUT_PATH
)

df = df[
    df[
        "participant_id"
    ].eq(
        PARTICIPANT_ID
    )
    & df[
        "usable_feature_days"
    ].ge(7)
    & df[
        FEATURE
    ].notna()
    & df[
        OUTCOME
    ].notna()
].copy()

df = (
    df.sort_values(
        "assessment_date"
    )
    .reset_index(
        drop=True
    )
)


print("=" * 90)
print(
    "LPE511 PERSONALIZED REGRESSION MODEL COMPARISON"
)
print("=" * 90)

print(
    "Eligible assessments:",
    len(df),
)

print(
    "Feature:",
    FEATURE,
)

print(
    "Outcome:",
    OUTCOME,
)

print()


# ============================================================
# MODELS
# ============================================================

def make_ridge():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=1.0
                ),
            ),
        ]
    )


def make_gpr():

    # Conservative kernel for extremely small n.
    #
    # Input gets standardized inside each training fold.
    # normalize_y=True centers/scales PHQ using TRAINING data only.
    #
    # Hyperparameter optimization is allowed only on the training
    # observations of each fold; the held-out week remains unseen.

    kernel = (
        ConstantKernel(
            1.0,
            (
                1e-2,
                1e2,
            ),
        )
        * RBF(
            length_scale=1.0,
            length_scale_bounds=(
                1e-2,
                1e2,
            ),
        )
        + WhiteKernel(
            noise_level=0.5,
            noise_level_bounds=(
                1e-4,
                1e1,
            ),
        )
    )

    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=10,
                    random_state=20261030,
                ),
            ),
        ]
    )


def make_random_forest():

    # Deliberately constrained because each fold has only
    # 9 training observations.
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=2,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=20261030,
        n_jobs=-1,
    )


MODEL_FACTORIES = {
    "ridge":
        make_ridge,

    "gaussian_process":
        make_gpr,

    "random_forest":
        make_random_forest,
}


# ============================================================
# LEAVE-ONE-ASSESSMENT-OUT
# ============================================================

prediction_rows = []


for test_i in range(
    len(df)
):

    train = (
        df.drop(
            index=test_i
        )
    )

    test = (
        df.iloc[
            test_i
        ]
    )

    X_train = (
        train[
            FEATURE
        ]
        .to_numpy(
            dtype=float
        )
        .reshape(
            -1,
            1,
        )
    )

    y_train = (
        train[
            OUTCOME
        ]
        .to_numpy(
            dtype=float
        )
    )

    X_test = np.asarray(
        [
            [
                float(
                    test[
                        FEATURE
                    ]
                )
            ]
        ],
        dtype=float,
    )

    y_test = float(
        test[
            OUTCOME
        ]
    )

    print(
        f"Test week "
        f"{test_i + 1}/{len(df)} | "
        f"{test['assessment_date']} | "
        f"PHQ={y_test:.1f}"
    )

    for (
        model_name,
        factory
    ) in MODEL_FACTORIES.items():

        model = factory()

        model.fit(
            X_train,
            y_train,
        )

        prediction = float(
            model.predict(
                X_test
            )[0]
        )

        absolute_error = abs(
            y_test
            - prediction
        )

        row = {
            "participant_id":
                PARTICIPANT_ID,

            "assessment_date":
                test[
                    "assessment_date"
                ],

            "clinical_window_id":
                test[
                    "clinical_window_id"
                ],

            "model":
                model_name,

            "reference_feature":
                float(
                    test[
                        FEATURE
                    ]
                ),

            "actual_phq":
                y_test,

            "predicted_phq":
                prediction,

            "absolute_error":
                absolute_error,

            "n_training_weeks":
                len(train),
        }

        # GPR also gives predictive uncertainty.
        if (
            model_name
            == "gaussian_process"
        ):

            scaled_X = (
                model.named_steps[
                    "scale"
                ]
                .transform(
                    X_test
                )
            )

            gpr = (
                model.named_steps[
                    "model"
                ]
            )

            (
                gpr_prediction,
                gpr_std,
            ) = gpr.predict(
                scaled_X,
                return_std=True,
            )

            row[
                "predictive_std"
            ] = float(
                gpr_std[0]
            )

        else:

            row[
                "predictive_std"
            ] = np.nan

        prediction_rows.append(
            row
        )


predictions = pd.DataFrame(
    prediction_rows
)


# ============================================================
# BASELINE
#
# Training-set mean PHQ for each held-out fold.
# ============================================================

baseline_rows = []

for test_i in range(
    len(df)
):

    train = (
        df.drop(
            index=test_i
        )
    )

    test = (
        df.iloc[
            test_i
        ]
    )

    prediction = float(
        train[
            OUTCOME
        ].mean()
    )

    baseline_rows.append(
        {
            "participant_id":
                PARTICIPANT_ID,

            "assessment_date":
                test[
                    "assessment_date"
                ],

            "clinical_window_id":
                test[
                    "clinical_window_id"
                ],

            "model":
                "personal_mean_baseline",

            "reference_feature":
                float(
                    test[
                        FEATURE
                    ]
                ),

            "actual_phq":
                float(
                    test[
                        OUTCOME
                    ]
                ),

            "predicted_phq":
                prediction,

            "absolute_error":
                abs(
                    float(
                        test[
                            OUTCOME
                        ]
                    )
                    - prediction
                ),

            "predictive_std":
                np.nan,

            "n_training_weeks":
                len(train),
        }
    )


predictions = pd.concat(
    [
        predictions,
        pd.DataFrame(
            baseline_rows
        ),
    ],
    ignore_index=True,
)


# ============================================================
# METRICS
# ============================================================

metric_rows = []


for model_name, group in (
    predictions.groupby(
        "model"
    )
):

    y_true = (
        group[
            "actual_phq"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_pred = (
        group[
            "predicted_phq"
        ]
        .to_numpy(
            dtype=float
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    if (
        np.std(
            y_pred
        ) > 0
        and np.std(
            y_true
        ) > 0
    ):

        correlation = float(
            np.corrcoef(
                y_true,
                y_pred,
            )[0, 1]
        )

    else:

        correlation = np.nan

    metric_rows.append(
        {
            "model":
                model_name,

            "n_test_weeks":
                len(group),

            "mae":
                mae,

            "rmse":
                rmse,

            "r2":
                r2,

            "pearson_correlation":
                correlation,
        }
    )


metrics = (
    pd.DataFrame(
        metric_rows
    )
    .sort_values(
        "mae"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# COMPARE AGAINST RIDGE
# ============================================================

ridge_mae = float(
    metrics.loc[
        metrics[
            "model"
        ].eq(
            "ridge"
        ),
        "mae",
    ].iloc[0]
)

metrics[
    "mae_change_vs_ridge"
] = (
    metrics[
        "mae"
    ]
    - ridge_mae
)

metrics[
    "mae_improvement_vs_ridge_pct"
] = (
    100.0
    * (
        ridge_mae
        - metrics[
            "mae"
        ]
    )
    / ridge_mae
)


# ============================================================
# SAVE
# ============================================================

PREDICTION_PATH = (
    OUT_DIR
    / "LPE511_model_predictions.csv"
)

METRIC_PATH = (
    OUT_DIR
    / "LPE511_model_comparison.csv"
)

predictions.to_csv(
    PREDICTION_PATH,
    index=False,
)

metrics.to_csv(
    METRIC_PATH,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 90)
print("MODEL COMPARISON")
print("=" * 90)

print(
    metrics.to_string(
        index=False
    )
)


print()
print("=" * 90)
print("WEEK-BY-WEEK PREDICTIONS")
print("=" * 90)

display = (
    predictions[
        [
            "assessment_date",
            "model",
            "actual_phq",
            "predicted_phq",
            "absolute_error",
            "predictive_std",
        ]
    ]
    .sort_values(
        [
            "assessment_date",
            "model",
        ]
    )
)

print(
    display.to_string(
        index=False
    )
)


print()
print("Saved:")
print(
    " ",
    PREDICTION_PATH,
)
print(
    " ",
    METRIC_PATH,
)
