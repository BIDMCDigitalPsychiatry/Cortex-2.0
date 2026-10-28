from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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

OUTFILE = (
    ROOT
    / "weekly_gps_clinical_prediction_performance.csv"
)

PRED_OUT = (
    ROOT
    / "weekly_gps_clinical_prediction_predictions.csv"
)

FEATURES = [
    "location_entropy_mean",
    "effective_places_mean",
    "dominant_place_fraction_mean",
    "observed_places_mean",
    "displacement_distance_mean_m",
]


def metrics(y, p):
    return {
        "mae":
            mean_absolute_error(y, p),

        "rmse":
            np.sqrt(
                mean_squared_error(y, p)
            ),

        "r2":
            r2_score(y, p),

        "correlation":
            (
                np.corrcoef(y, p)[0, 1]
                if np.std(p) > 0
                else np.nan
            ),
    }


def run_lopo(
    df,
    score_col,
    mode,
):
    rows = []

    participants = sorted(
        df["participant_id"].unique()
    )

    for pid in participants:

        train = df[
            df["participant_id"] != pid
        ].copy()

        test = df[
            df["participant_id"] == pid
        ].copy()

        if mode == "absolute":

            X_train = train[FEATURES].copy()
            X_test = test[FEATURES].copy()

            y_train = train[score_col].copy()
            y_test = test[score_col].copy()

            baseline_value = (
                y_train.mean()
            )

            baseline_pred = np.repeat(
                baseline_value,
                len(test),
            )

        elif mode == "within_person":

            # Center features using participant-specific means.
            # For held-out participant, centering uses only that
            # participant's GPS values, not symptom scores.
            train_means = (
                train.groupby(
                    "participant_id"
                )[FEATURES]
                .transform("mean")
            )

            X_train = (
                train[FEATURES]
                - train_means
            )

            test_means = (
                test[FEATURES]
                .mean()
            )

            X_test = (
                test[FEATURES]
                - test_means
            )

            # Center symptom score within participant
            # for training participants only.
            train_y_mean = (
                train.groupby(
                    "participant_id"
                )[score_col]
                .transform("mean")
            )

            y_train = (
                train[score_col]
                - train_y_mean
            )

            # For evaluation, use held-out participant's own
            # symptom mean only to define observed deviation.
            # It is NOT used in model fitting.
            test_y_mean = (
                test[score_col]
                .mean()
            )

            y_test = (
                test[score_col]
                - test_y_mean
            )

            baseline_pred = np.zeros(
                len(test)
            )

        else:
            raise ValueError(mode)

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

        gps_pred = model.predict(
            X_test
        )

        for i, (_, r) in enumerate(
            test.iterrows()
        ):
            rows.append(
                {
                    "outcome":
                        score_col,

                    "mode":
                        mode,

                    "participant_id":
                        pid,

                    "assessment_date":
                        r[
                            "assessment_date"
                        ],

                    "observed":
                        float(
                            y_test.iloc[i]
                        ),

                    "baseline_prediction":
                        float(
                            baseline_pred[i]
                        ),

                    "gps_prediction":
                        float(
                            gps_pred[i]
                        ),
                }
            )

    pred = pd.DataFrame(rows)

    y = pred[
        "observed"
    ].to_numpy()

    b = pred[
        "baseline_prediction"
    ].to_numpy()

    g = pred[
        "gps_prediction"
    ].to_numpy()

    bm = metrics(
        y,
        b,
    )

    gm = metrics(
        y,
        g,
    )

    return pred, [
        {
            "outcome":
                score_col,

            "mode":
                mode,

            "model":
                "baseline",

            "n_assessments":
                len(pred),

            "n_participants":
                pred[
                    "participant_id"
                ].nunique(),

            **bm,
        },
        {
            "outcome":
                score_col,

            "mode":
                mode,

            "model":
                "weekly_gps_ridge",

            "n_assessments":
                len(pred),

            "n_participants":
                pred[
                    "participant_id"
                ].nunique(),

            **gm,
        },
    ]


df = pd.read_parquet(
    INFILE
)

# Primary analysis:
# require all 7 usable GPS feature days.
df = df[
    df[
        "usable_feature_days"
    ]
    >= 7
].copy()


print("=" * 80)
print("WEEKLY GPS CLINICAL PREDICTION")
print("=" * 80)

print(
    "Complete 7/7 windows:",
    len(df),
)

print(
    "Participants:",
    df[
        "participant_id"
    ].nunique(),
)


all_predictions = []
all_metrics = []


for outcome in [
    "depression_score",
    "anxiety_score",
]:

    x = df[
        df[outcome].notna()
    ].copy()

    print()
    print("-" * 80)
    print(outcome)

    print(
        "Assessments:",
        len(x),
    )

    print(
        "Participants:",
        x[
            "participant_id"
        ].nunique(),
    )

    print(
        "Score range:",
        (
            x[outcome].min(),
            x[outcome].max(),
        ),
    )

    for mode in [
        "absolute",
        "within_person",
    ]:

        pred, result = run_lopo(
            x,
            outcome,
            mode,
        )

        all_predictions.append(
            pred
        )

        all_metrics.extend(
            result
        )


predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

performance = pd.DataFrame(
    all_metrics
)

predictions.to_csv(
    PRED_OUT,
    index=False,
)

performance.to_csv(
    OUTFILE,
    index=False,
)


print()
print("=" * 80)
print("PERFORMANCE")
print("=" * 80)

print(
    performance.to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    OUTFILE,
)
