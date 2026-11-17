"""Compare LPE511 personalized Ridge vs hierarchical mixed-effects regression.

Goal
----
Predict each held-out LPE511 PHQ assessment using:

1. Personalized Ridge:
   LPE511's other weeks only.

2. Mixed-effects regression:
   all eligible participants' training weeks, with
   participant-specific random intercepts and random slopes.

Common predictor:
    weekly mean displacement distance

Outcome:
    PHQ-9 / depression_score

Evaluation:
    leave-one-assessment-out for LPE511.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import statsmodels.formula.api as smf


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

TARGET = "LPE511"

FEATURE = "displacement_distance_mean_m"
OUTCOME = "depression_score"

MIN_WEEKS = 5


# ============================================================
# LOAD COMPLETE 7/7 WINDOWS
# ============================================================

df = pd.read_parquet(
    INPUT_PATH
)

df = df[
    df[
        "usable_feature_days"
    ].ge(7)
    & df[
        FEATURE
    ].notna()
    & df[
        OUTCOME
    ].notna()
].copy()

df[
    "participant_id"
] = df[
    "participant_id"
].astype(str)


# ============================================================
# ELIGIBLE PARTICIPANTS
#
# Match the prior personalized analysis:
# >=5 assessments and nonzero PHQ variability.
# ============================================================

participant_stats = (
    df.groupby(
        "participant_id"
    )
    .agg(
        n_weeks=(
            OUTCOME,
            "size",
        ),
        symptom_sd=(
            OUTCOME,
            "std",
        ),
    )
    .reset_index()
)

eligible = (
    participant_stats[
        participant_stats[
            "n_weeks"
        ].ge(MIN_WEEKS)
        & participant_stats[
            "symptom_sd"
        ].gt(0)
    ][
        "participant_id"
    ]
    .tolist()
)

df = (
    df[
        df[
            "participant_id"
        ].isin(
            eligible
        )
    ]
    .copy()
)

target_df = (
    df[
        df[
            "participant_id"
        ].eq(TARGET)
    ]
    .sort_values(
        "assessment_date"
    )
    .reset_index(
        drop=True
    )
)

if TARGET not in eligible:
    raise RuntimeError(
        f"{TARGET} is not eligible."
    )

print("=" * 100)
print(
    "LPE511 RIDGE VS MIXED-EFFECTS COMPARISON"
)
print("=" * 100)

print(
    "Eligible participants:",
    len(eligible),
)

print(
    "Total eligible weekly observations:",
    len(df),
)

print(
    "LPE511 assessments:",
    len(target_df),
)

print()


# ============================================================
# WITHIN-PARTICIPANT STANDARDIZATION
#
# Crucially, for each LPE511 test fold:
# - held-out LPE511 week is excluded
# - scaling for LPE511 comes only from its training weeks
#
# For all other participants, all their weeks are training data.
# ============================================================

def add_training_standardized_feature(
    train_df,
    test_feature,
):
    train_df = (
        train_df.copy()
    )

    scale_info = {}

    z_values = np.empty(
        len(train_df),
        dtype=float,
    )

    for participant_id, idx in (
        train_df.groupby(
            "participant_id"
        ).groups.items()
    ):

        values = (
            train_df.loc[
                idx,
                FEATURE,
            ]
            .to_numpy(
                dtype=float
            )
        )

        mean = float(
            np.mean(values)
        )

        sd = float(
            np.std(
                values,
                ddof=0,
            )
        )

        if (
            not np.isfinite(sd)
            or sd == 0
        ):
            sd = 1.0

        z_values[
            train_df.index.get_indexer(
                idx
            )
        ] = (
            values
            - mean
        ) / sd

        scale_info[
            participant_id
        ] = (
            mean,
            sd,
        )

    train_df[
        "feature_z"
    ] = z_values

    target_mean, target_sd = (
        scale_info[
            TARGET
        ]
    )

    test_z = (
        float(test_feature)
        - target_mean
    ) / target_sd

    return (
        train_df,
        float(test_z),
    )


# ============================================================
# LPE511 LEAVE-ONE-ASSESSMENT-OUT
# ============================================================

rows = []


for test_i in range(
    len(target_df)
):

    test_row = (
        target_df.iloc[
            test_i
        ]
    )

    test_window_id = (
        test_row[
            "clinical_window_id"
        ]
    )

    # All eligible data except the held-out LPE511 week.
    training = (
        df[
            ~(
                df[
                    "participant_id"
                ].eq(TARGET)
                & df[
                    "clinical_window_id"
                ].eq(
                    test_window_id
                )
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # RIDGE:
    # LPE511 training weeks only
    # --------------------------------------------------------

    lpe_train = (
        training[
            training[
                "participant_id"
            ].eq(TARGET)
        ]
    )

    X_train = (
        lpe_train[
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
        lpe_train[
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
                    test_row[
                        FEATURE
                    ]
                )
            ]
        ],
        dtype=float,
    )

    ridge = Pipeline(
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

    ridge.fit(
        X_train,
        y_train,
    )

    ridge_prediction = float(
        ridge.predict(
            X_test
        )[0]
    )

    # --------------------------------------------------------
    # MIXED EFFECTS
    #
    # PHQ = fixed population intercept
    #     + fixed population displacement effect
    #     + participant-specific intercept
    #     + participant-specific displacement effect
    # --------------------------------------------------------

    (
        mixed_train,
        test_z,
    ) = add_training_standardized_feature(
        training,
        test_row[
            FEATURE
        ],
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        mixed = smf.mixedlm(
            formula=(
                "depression_score ~ feature_z"
            ),
            data=mixed_train,
            groups=mixed_train[
                "participant_id"
            ],
            re_formula=(
                "~feature_z"
            ),
        )

        mixed_result = mixed.fit(
            reml=True,
            method="lbfgs",
            maxiter=2000,
            disp=False,
        )

    # Fixed population part.
    fixed_prediction = (
        float(
            mixed_result.fe_params[
                "Intercept"
            ]
        )
        + float(
            mixed_result.fe_params[
                "feature_z"
            ]
        )
        * test_z
    )

    # LPE511 random effects learned only from its
    # non-held-out training weeks.
    random_effects = (
        mixed_result.random_effects[
            TARGET
        ]
    )

    random_intercept = float(
        random_effects.iloc[
            0
        ]
    )

    random_slope = (
        float(
            random_effects.iloc[
                1
            ]
        )
        if len(
            random_effects
        ) > 1
        else 0.0
    )

    mixed_prediction = (
        fixed_prediction
        + random_intercept
        + random_slope
        * test_z
    )

    actual = float(
        test_row[
            OUTCOME
        ]
    )

    rows.extend(
        [
            {
                "assessment_date":
                    test_row[
                        "assessment_date"
                    ],

                "clinical_window_id":
                    test_window_id,

                "model":
                    "ridge_personalized",

                "actual_phq":
                    actual,

                "predicted_phq":
                    ridge_prediction,

                "absolute_error":
                    abs(
                        actual
                        - ridge_prediction
                    ),

                "n_lpe_training_weeks":
                    len(
                        lpe_train
                    ),

                "n_total_training_rows":
                    len(
                        training
                    ),
            },

            {
                "assessment_date":
                    test_row[
                        "assessment_date"
                    ],

                "clinical_window_id":
                    test_window_id,

                "model":
                    "mixed_effects",

                "actual_phq":
                    actual,

                "predicted_phq":
                    mixed_prediction,

                "absolute_error":
                    abs(
                        actual
                        - mixed_prediction
                    ),

                "n_lpe_training_weeks":
                    len(
                        lpe_train
                    ),

                "n_total_training_rows":
                    len(
                        training
                    ),
            },
        ]
    )

    print(
        f"{test_i + 1:2d}/{len(target_df)} | "
        f"{test_row['assessment_date']} | "
        f"PHQ={actual:.1f} | "
        f"Ridge={ridge_prediction:.2f} | "
        f"Mixed={mixed_prediction:.2f}",
        flush=True,
    )


predictions = pd.DataFrame(
    rows
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

    correlation = float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )

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


ridge_mae = float(
    metrics.loc[
        metrics[
            "model"
        ].eq(
            "ridge_personalized"
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
    100
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

PRED_PATH = (
    OUT_DIR
    / "LPE511_ridge_vs_mixed_effects_predictions.csv"
)

METRIC_PATH = (
    OUT_DIR
    / "LPE511_ridge_vs_mixed_effects_metrics.csv"
)

predictions.to_csv(
    PRED_PATH,
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
print("=" * 100)
print(
    "RIDGE VS MIXED-EFFECTS"
)
print("=" * 100)

print(
    metrics.to_string(
        index=False
    )
)

print()
print("=" * 100)
print(
    "WEEK-BY-WEEK"
)
print("=" * 100)

print(
    predictions[
        [
            "assessment_date",
            "model",
            "actual_phq",
            "predicted_phq",
            "absolute_error",
        ]
    ]
    .sort_values(
        [
            "assessment_date",
            "model",
        ]
    )
    .to_string(
        index=False
    )
)

print()
print("Saved:")
print(" ", PRED_PATH)
print(" ", METRIC_PATH)
