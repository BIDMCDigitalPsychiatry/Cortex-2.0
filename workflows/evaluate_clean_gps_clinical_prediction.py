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

INFILE = (
    DC / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_reference_days_with_clinical_scores.csv"
)

OUT = (
    DC / "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "location_entropy_nats",
    "effective_number_places",
    "dominant_place_fraction",
]


def build_assessment_level(df, score_col):
    date_col = f"{score_col}_assessment_date"
    distance_col = f"{score_col}_distance_days"

    x = df[
        df[score_col].notna()
        & df[date_col].notna()
    ].copy()

    x[date_col] = pd.to_datetime(
        x[date_col],
        errors="coerce",
    )

    # Multiple selected GPS reference days can map to the
    # same weekly clinical assessment.
    # Collapse these to one observation per assessment.
    agg = {
        score_col: "first",
        "cohort": "first",
        distance_col: "mean",
    }

    for f in FEATURES:
        agg[f] = "mean"

    out = (
        x.groupby(
            ["participant_id", date_col],
            as_index=False,
        )
        .agg(agg)
    )

    return out


def evaluate_lopo(x, score_col):
    participants = sorted(
        x["participant_id"].unique()
    )

    predictions = []

    # Fixed modest regularization.
    # No tuning on the held-out participant.
    alpha = 1.0

    for pid in participants:
        train = x[
            x["participant_id"] != pid
        ].copy()

        test = x[
            x["participant_id"] == pid
        ].copy()

        X_train = train[FEATURES]
        y_train = train[score_col]

        X_test = test[FEATURES]
        y_test = test[score_col]

        # Baseline = training-set mean symptom score.
        baseline_pred = np.repeat(
            y_train.mean(),
            len(test),
        )

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "ridge",
                    Ridge(alpha=alpha),
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

        for i, (_, row) in enumerate(
            test.iterrows()
        ):
            predictions.append(
                {
                    "participant_id": pid,
                    "cohort": row["cohort"],
                    "assessment_date": row[
                        f"{score_col}_assessment_date"
                    ],
                    "observed_score": y_test.iloc[i],
                    "baseline_prediction": baseline_pred[i],
                    "gps_prediction": gps_pred[i],
                }
            )

    pred = pd.DataFrame(predictions)

    y = pred["observed_score"].to_numpy()

    metrics = []

    for model_name, col in [
        (
            "mean_baseline",
            "baseline_prediction",
        ),
        (
            "clean_gps_ridge",
            "gps_prediction",
        ),
    ]:
        p = pred[col].to_numpy()

        metrics.append(
            {
                "outcome": score_col,
                "model": model_name,
                "n_assessments": len(pred),
                "n_participants": (
                    pred["participant_id"]
                    .nunique()
                ),
                "mae": mean_absolute_error(
                    y,
                    p,
                ),
                "rmse": np.sqrt(
                    mean_squared_error(
                        y,
                        p,
                    )
                ),
                "r2": r2_score(
                    y,
                    p,
                ),
                "correlation": (
                    np.corrcoef(
                        y,
                        p,
                    )[0, 1]
                    if np.std(p) > 0
                    else np.nan
                ),
            }
        )

    return pred, pd.DataFrame(metrics)


df = pd.read_csv(
    INFILE,
    low_memory=False,
)

all_metrics = []

for score_col in [
    "depression_score",
    "anxiety_score",
]:
    x = build_assessment_level(
        df,
        score_col,
    )

    print("\n" + "=" * 80)
    print(score_col.upper())
    print("=" * 80)

    print(
        "Unique assessments:",
        len(x),
    )
    print(
        "Participants:",
        x["participant_id"].nunique(),
    )

    print(
        "\nAssessments per participant:"
    )
    print(
        x.groupby("participant_id")
        .size()
        .describe()
        .to_string()
    )

    pred, metrics = evaluate_lopo(
        x,
        score_col,
    )

    pred.to_csv(
        OUT
        / f"{score_col}_clean_gps_lopo_predictions.csv",
        index=False,
    )

    all_metrics.append(metrics)

    print("\nPrediction performance:")
    print(
        metrics.to_string(
            index=False
        )
    )

metrics = pd.concat(
    all_metrics,
    ignore_index=True,
)

metrics.to_csv(
    OUT
    / "clean_gps_clinical_prediction_performance.csv",
    index=False,
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(metrics.to_string(index=False))

print(
    "\nSaved to:",
    OUT,
)
