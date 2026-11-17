"""Personalized daily GPS -> EMA relationships.

For each participant separately:
- 5 GPS features
- mood and anxiety
- same-day and exact next-day relationships
- Spearman correlation
- simple standardized linear effect
- BH-FDR across all personalized tests

This is exploratory personalized phenotyping, not a population-average model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "clinical/daily_ema"
)

INPUT = ROOT / "daily_ema_gps_features.parquet"

OUT_DIR = ROOT / "personalized_daily_associations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_OUT = OUT_DIR / "personalized_daily_gps_ema_results.csv"
SUMMARY_OUT = OUT_DIR / "personalized_daily_gps_ema_summary.csv"


MIN_OBSERVATIONS = 15


FEATURES = {
    "displacement":
        "supported_displacement_distance_m",

    "location_entropy":
        "location_entropy_nats",

    "effective_places":
        "effective_number_places",

    "dominant_place_fraction":
        "dominant_place_fraction",

    "observed_places":
        "n_observed_places",
}


OUTCOMES = [
    "mood",
    "anxiety",
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(INPUT)

df["participant_id"] = (
    df["participant_id"].astype(str)
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df = df[
    df["gps_feature_status"].eq("ok")
].copy()


# Skewed GPS features
df["displacement"] = np.log1p(
    pd.to_numeric(
        df[
            "supported_displacement_distance_m"
        ],
        errors="coerce",
    )
)

df["observed_places"] = np.log1p(
    pd.to_numeric(
        df["n_observed_places"],
        errors="coerce",
    )
)

df["location_entropy"] = pd.to_numeric(
    df["location_entropy_nats"],
    errors="coerce",
)

df["effective_places"] = pd.to_numeric(
    df["effective_number_places"],
    errors="coerce",
)

df["dominant_place_fraction"] = pd.to_numeric(
    df["dominant_place_fraction"],
    errors="coerce",
)


# ============================================================
# EXACT NEXT-DAY EMA
# ============================================================

future = df[
    [
        "participant_id",
        "date",
        "mood",
        "anxiety",
    ]
].copy()

# Move tomorrow's outcome backward to today's GPS row.
future["date"] = (
    future["date"]
    - pd.Timedelta(days=1)
)

future = future.rename(
    columns={
        "mood": "mood_next_day",
        "anxiety": "anxiety_next_day",
    }
)

df = df.merge(
    future,
    on=[
        "participant_id",
        "date",
    ],
    how="left",
)


# ============================================================
# ANALYSIS
# ============================================================

rows = []


for participant_id, g in df.groupby(
    "participant_id"
):

    g = g.sort_values("date").copy()

    for feature in FEATURES:

        for outcome in OUTCOMES:

            for timing in [
                "same_day",
                "next_day",
            ]:

                if timing == "same_day":
                    outcome_col = outcome
                else:
                    outcome_col = (
                        f"{outcome}_next_day"
                    )

                x = pd.to_numeric(
                    g[feature],
                    errors="coerce",
                )

                y = pd.to_numeric(
                    g[outcome_col],
                    errors="coerce",
                )

                valid = (
                    x.notna()
                    & y.notna()
                )

                x = x[valid].to_numpy(
                    dtype=float
                )

                y = y[valid].to_numpy(
                    dtype=float
                )

                n = len(x)

                if n < MIN_OBSERVATIONS:
                    continue

                # Need actual variation.
                if (
                    np.std(x) == 0
                    or np.std(y) == 0
                ):
                    continue


                # --------------------------------------------
                # SPEARMAN
                # --------------------------------------------

                rho, spearman_p = (
                    spearmanr(
                        x,
                        y,
                    )
                )


                # --------------------------------------------
                # STANDARDIZED LINEAR EFFECT
                #
                # Beta = EMA-point difference associated
                # with a 1-SD difference in this person's
                # GPS feature.
                # --------------------------------------------

                x_z = (
                    x - np.mean(x)
                ) / np.std(x)

                X = sm.add_constant(
                    x_z
                )

                model = sm.OLS(
                    y,
                    X,
                ).fit(
                    cov_type="HC3"
                )

                beta = float(
                    model.params[1]
                )

                se = float(
                    model.bse[1]
                )

                p_linear = float(
                    model.pvalues[1]
                )

                ci = model.conf_int(
                    alpha=0.05
                )

                ci_low = float(
                    ci[1, 0]
                )

                ci_high = float(
                    ci[1, 1]
                )


                rows.append(
                    {
                        "participant_id":
                            participant_id,

                        "feature":
                            feature,

                        "outcome":
                            outcome,

                        "timing":
                            timing,

                        "n":
                            n,

                        "spearman_rho":
                            float(rho),

                        "spearman_p":
                            float(
                                spearman_p
                            ),

                        "beta_per_person_sd":
                            beta,

                        "beta_se":
                            se,

                        "beta_ci_low":
                            ci_low,

                        "beta_ci_high":
                            ci_high,

                        "linear_p":
                            p_linear,
                    }
                )


results = pd.DataFrame(rows)


# ============================================================
# FDR
#
# Correct separately for:
# outcome × timing × statistical test.
#
# This still accounts for all participants and all features
# within each clinical question.
# ============================================================

results["spearman_q"] = np.nan
results["linear_q"] = np.nan


for (
    outcome,
    timing,
), idx in results.groupby(
    [
        "outcome",
        "timing",
    ]
).groups.items():

    idx = list(idx)

    _, q, _, _ = multipletests(
        results.loc[
            idx,
            "spearman_p",
        ],
        method="fdr_bh",
    )

    results.loc[
        idx,
        "spearman_q",
    ] = q


    _, q, _, _ = multipletests(
        results.loc[
            idx,
            "linear_p",
        ],
        method="fdr_bh",
    )

    results.loc[
        idx,
        "linear_q",
    ] = q


results[
    "spearman_fdr_significant"
] = (
    results[
        "spearman_q"
    ] < 0.05
)

results[
    "linear_fdr_significant"
] = (
    results[
        "linear_q"
    ] < 0.05
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

summary_rows = []


for participant_id, g in results.groupby(
    "participant_id"
):

    best = g.sort_values(
        "spearman_p"
    ).iloc[0]

    summary_rows.append(
        {
            "participant_id":
                participant_id,

            "n_tests":
                len(g),

            "n_nominal_spearman_p_lt_05":
                int(
                    (
                        g[
                            "spearman_p"
                        ] < 0.05
                    ).sum()
                ),

            "n_fdr_spearman_q_lt_05":
                int(
                    (
                        g[
                            "spearman_q"
                        ] < 0.05
                    ).sum()
                ),

            "best_feature":
                best[
                    "feature"
                ],

            "best_outcome":
                best[
                    "outcome"
                ],

            "best_timing":
                best[
                    "timing"
                ],

            "best_n":
                int(
                    best["n"]
                ),

            "best_rho":
                float(
                    best[
                        "spearman_rho"
                    ]
                ),

            "best_p":
                float(
                    best[
                        "spearman_p"
                    ]
                ),

            "best_q":
                float(
                    best[
                        "spearman_q"
                    ]
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
).sort_values(
    [
        "n_fdr_spearman_q_lt_05",
        "best_p",
    ],
    ascending=[
        False,
        True,
    ],
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    RESULT_OUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("PERSONALIZED DAILY GPS -> EMA")
print("=" * 110)

print(
    "Participants analyzed:",
    results[
        "participant_id"
    ].nunique(),
)

print(
    "Total personalized tests:",
    len(results),
)

print(
    "Nominal Spearman p < .05:",
    int(
        (
            results[
                "spearman_p"
            ] < .05
        ).sum()
    ),
)

print(
    "FDR Spearman q < .05:",
    int(
        (
            results[
                "spearman_q"
            ] < .05
        ).sum()
    ),
)

print(
    "Nominal linear p < .05:",
    int(
        (
            results[
                "linear_p"
            ] < .05
        ).sum()
    ),
)

print(
    "FDR linear q < .05:",
    int(
        (
            results[
                "linear_q"
            ] < .05
        ).sum()
    ),
)


print()
print("=" * 110)
print("TOP 30 PERSONALIZED RELATIONSHIPS")
print("=" * 110)

top = (
    results.sort_values(
        [
            "spearman_q",
            "spearman_p",
        ]
    )
    .head(30)
)

print(
    top[
        [
            "participant_id",
            "feature",
            "outcome",
            "timing",
            "n",
            "spearman_rho",
            "spearman_p",
            "spearman_q",
            "beta_per_person_sd",
            "linear_p",
            "linear_q",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("FDR-SIGNIFICANT PERSONALIZED RELATIONSHIPS")
print("=" * 110)

sig = results[
    (
        results[
            "spearman_q"
        ] < 0.05
    )
    | (
        results[
            "linear_q"
        ] < 0.05
    )
].sort_values(
    [
        "spearman_q",
        "linear_q",
    ]
)

if sig.empty:

    print("None.")

else:

    print(
        sig[
            [
                "participant_id",
                "feature",
                "outcome",
                "timing",
                "n",
                "spearman_rho",
                "spearman_p",
                "spearman_q",
                "beta_per_person_sd",
                "linear_p",
                "linear_q",
            ]
        ]
        .to_string(
            index=False
        )
    )


print()
print("=" * 110)
print("PARTICIPANT SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False
    )
)


print()
print("Saved:")
print(" ", RESULT_OUT)
print(" ", SUMMARY_OUT)
