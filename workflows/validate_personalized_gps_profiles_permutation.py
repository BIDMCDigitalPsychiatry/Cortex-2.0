"""Permutation validation of personalized GPS-symptom profiles.

For each participant/outcome:
- keep the observed GPS feature series fixed;
- randomly permute symptom scores across that participant's assessment weeks;
- recompute leave-one-assessment-out prediction for every GPS feature;
- retain the BEST feature improvement for each permutation.

This tests whether the participant's observed best GPS feature performs
better than expected simply from searching across multiple GPS features.

Exploratory analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

INFILE = (
    ROOT
    / "gps_clinical_weekly_features.parquet"
)

OUTFILE = (
    ROOT
    / "personalized_gps_symptom_permutation_validation.csv"
)


FEATURES = [
    "location_entropy_mean",
    "effective_places_mean",
    "dominant_place_fraction_mean",
    "observed_places_mean",
    "displacement_distance_mean_m",
]

FEATURE_NAMES = {
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
N_PERMUTATIONS = 5000
RANDOM_SEED = 20261027


def loo_improvement(
    x,
    y,
):
    """Fast analytical equivalent of one-feature StandardScaler + Ridge LOO."""

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(y)

    baseline_errors = []
    feature_errors = []

    alpha = 1.0

    for test_i in range(n):

        mask = np.ones(
            n,
            dtype=bool,
        )
        mask[test_i] = False

        x_train = x[mask]
        y_train = y[mask]

        x_test = x[test_i]
        y_test = y[test_i]

        x_mean = float(
            np.mean(x_train)
        )

        x_std = float(
            np.std(
                x_train,
                ddof=0,
            )
        )

        y_mean = float(
            np.mean(y_train)
        )

        baseline_prediction = y_mean

        if (
            not np.isfinite(x_std)
            or x_std == 0
        ):
            feature_prediction = y_mean

        else:
            z_train = (
                x_train
                - x_mean
            ) / x_std

            z_test = (
                x_test
                - x_mean
            ) / x_std

            y_centered = (
                y_train
                - y_mean
            )

            coefficient = (
                np.sum(
                    z_train
                    * y_centered
                )
                / (
                    np.sum(
                        z_train ** 2
                    )
                    + alpha
                )
            )

            feature_prediction = (
                y_mean
                + coefficient
                * z_test
            )

        baseline_errors.append(
            abs(
                y_test
                - baseline_prediction
            )
        )

        feature_errors.append(
            abs(
                y_test
                - feature_prediction
            )
        )

    baseline_mae = float(
        np.mean(
            baseline_errors
        )
    )

    feature_mae = float(
        np.mean(
            feature_errors
        )
    )

    return (
        baseline_mae
        - feature_mae
    )


def bh_fdr(
    pvalues,
):
    pvalues = np.asarray(
        pvalues,
        dtype=float,
    )

    n = len(pvalues)

    order = np.argsort(
        pvalues
    )

    ranked = (
        pvalues[order]
    )

    adjusted = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    result = np.empty(
        n,
        dtype=float,
    )

    result[order] = adjusted

    return result


df = pd.read_parquet(
    INFILE
)

df = df[
    df[
        "usable_feature_days"
    ]
    >= 7
].copy()


rng = np.random.default_rng(
    RANDOM_SEED
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

        if len(g) < MIN_ASSESSMENTS:
            continue

        y = (
            g[outcome]
            .to_numpy(
                dtype=float
            )
        )

        if (
            not np.isfinite(y).all()
            or np.std(
                y,
                ddof=1,
            ) == 0
        ):
            continue

        feature_data = {}

        observed_results = []

        for feature in FEATURES:

            x = (
                pd.to_numeric(
                    g[feature],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            if (
                not np.isfinite(x).all()
                or np.std(
                    x,
                    ddof=1,
                ) == 0
            ):
                continue

            feature_data[
                feature
            ] = x

            improvement = (
                loo_improvement(
                    x,
                    y,
                )
            )

            observed_results.append(
                (
                    feature,
                    improvement,
                )
            )

        if not observed_results:
            continue

        best_feature, observed_best = max(
            observed_results,
            key=lambda z: z[1],
        )

        null_best = np.empty(
            N_PERMUTATIONS,
            dtype=float,
        )

        for b in range(
            N_PERMUTATIONS
        ):

            permuted_y = rng.permutation(
                y
            )

            best = -np.inf

            for feature, x in (
                feature_data.items()
            ):

                improvement = (
                    loo_improvement(
                        x,
                        permuted_y,
                    )
                )

                if improvement > best:
                    best = improvement

            null_best[b] = best

        permutation_p = (
            1
            + np.sum(
                null_best
                >= observed_best
            )
        ) / (
            N_PERMUTATIONS
            + 1
        )

        rows.append(
            {
                "participant_id":
                    participant_id,

                "outcome":
                    outcome,

                "n_assessments":
                    len(g),

                "symptom_sd":
                    float(
                        np.std(
                            y,
                            ddof=1,
                        )
                    ),

                "best_feature":
                    FEATURE_NAMES[
                        best_feature
                    ],

                "observed_best_mae_improvement":
                    observed_best,

                "null_best_mean":
                    float(
                        np.mean(
                            null_best
                        )
                    ),

                "null_best_95th_percentile":
                    float(
                        np.quantile(
                            null_best,
                            0.95,
                        )
                    ),

                "permutation_p":
                    permutation_p,
            }
        )

        print(
            f"{outcome} | {participant_id} | "
            f"n={len(g)} | "
            f"best={FEATURE_NAMES[best_feature]} | "
            f"improvement={observed_best:.4f} | "
            f"p={permutation_p:.4f}",
            flush=True,
        )


result = pd.DataFrame(
    rows
)


# ============================================================
# MULTIPLE-TESTING ADJUSTMENT WITHIN OUTCOME
# ============================================================

result["fdr_q"] = np.nan

for outcome in OUTCOMES:

    mask = (
        result[
            "outcome"
        ]
        == outcome
    )

    result.loc[
        mask,
        "fdr_q",
    ] = bh_fdr(
        result.loc[
            mask,
            "permutation_p",
        ].to_numpy()
    )


# Descriptive profile categories.
# Do not treat "candidate" as confirmed biomarker status.

result[
    "profile_status"
] = "not_supported"

result.loc[
    (
        result[
            "observed_best_mae_improvement"
        ]
        > 0
    )
    & (
        result[
            "permutation_p"
        ]
        < 0.05
    ),
    "profile_status",
] = "candidate_gps_sensitive"

result.loc[
    (
        result[
            "observed_best_mae_improvement"
        ]
        > 0
    )
    & (
        result[
            "permutation_p"
        ]
        >= 0.05
    ),
    "profile_status",
] = "positive_but_uncertain"


result.to_csv(
    OUTFILE,
    index=False,
)


print("=" * 100)
print("PERSONALIZED GPS PROFILE PERMUTATION VALIDATION")
print("=" * 100)

print(
    "Permutation repetitions:",
    N_PERMUTATIONS,
)

print()

for outcome in OUTCOMES:

    x = result[
        result[
            "outcome"
        ]
        == outcome
    ]

    print("-" * 100)
    print(outcome)

    print(
        "Participants:",
        len(x),
    )

    print(
        "Observed positive best feature:",
        (
            x[
                "observed_best_mae_improvement"
            ]
            > 0
        ).sum(),
    )

    print(
        "Permutation p < .05:",
        (
            x[
                "permutation_p"
            ]
            < 0.05
        ).sum(),
    )

    print(
        "FDR q < .05:",
        (
            x[
                "fdr_q"
            ]
            < 0.05
        ).sum(),
    )


print()
print("=" * 100)
print("PARTICIPANT RESULTS")
print("=" * 100)

print(
    result.sort_values(
        [
            "outcome",
            "permutation_p",
            "observed_best_mae_improvement",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    OUTFILE,
)
