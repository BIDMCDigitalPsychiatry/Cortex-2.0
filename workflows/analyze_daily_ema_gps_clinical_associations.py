"""Daily GPS -> EMA repeated-measures clinical associations.

Primary outcomes:
- mood
- anxiety

Primary GPS features:
- displacement
- location entropy
- effective number of places
- dominant-place fraction
- number of observed places

Analyses:
1. Same-day GPS -> same-day EMA
2. Today's GPS -> next-day EMA

For each GPS feature, decompose into:
- within-person deviation
- between-person mean

Repeated measures are modeled using Gaussian GEE with participant clusters.

Supported GPS coverage is included continuously, also decomposed into
within-person and between-person components.

No arbitrary GPS-quality threshold is imposed.

Multiple testing:
BH-FDR across primary within-person tests and separately across
between-person tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "clinical/daily_ema"
)

INPUT = (
    ROOT
    / "daily_ema_gps_features.parquet"
)

OUT_DIR = (
    ROOT
    / "clinical_associations"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_OUT = (
    OUT_DIR
    / "daily_ema_gps_gee_results.csv"
)

SAMPLE_OUT = (
    OUT_DIR
    / "daily_ema_gps_gee_sample_summary.csv"
)


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

df = pd.read_parquet(
    INPUT
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df["participant_id"] = (
    df[
        "participant_id"
    ].astype(str)
)


# Keep days where Cortex successfully computed GPS features.
df = df[
    df[
        "gps_feature_status"
    ].eq("ok")
].copy()


print("=" * 100)
print("DAILY GPS -> EMA CLINICAL ASSOCIATIONS")
print("=" * 100)

print(
    "Usable GPS+EMA rows:",
    len(df),
)

print(
    "Participants:",
    df[
        "participant_id"
    ].nunique(),
)


# ============================================================
# TRANSFORMS
# ============================================================

# Highly skewed positive features.
df[
    "feature_displacement"
] = np.log1p(
    pd.to_numeric(
        df[
            FEATURES[
                "displacement"
            ]
        ],
        errors="coerce",
    )
)

df[
    "feature_observed_places"
] = np.log1p(
    pd.to_numeric(
        df[
            FEATURES[
                "observed_places"
            ]
        ],
        errors="coerce",
    )
)


# Other features remain on their natural scale.
df[
    "feature_location_entropy"
] = pd.to_numeric(
    df[
        FEATURES[
            "location_entropy"
        ]
    ],
    errors="coerce",
)

df[
    "feature_effective_places"
] = pd.to_numeric(
    df[
        FEATURES[
            "effective_places"
        ]
    ],
    errors="coerce",
)

df[
    "feature_dominant_place_fraction"
] = pd.to_numeric(
    df[
        FEATURES[
            "dominant_place_fraction"
        ]
    ],
    errors="coerce",
)


# Continuous GPS observability covariate.
df[
    "coverage"
] = pd.to_numeric(
    df[
        "supported_fraction_of_day"
    ],
    errors="coerce",
)


# ============================================================
# EXACT NEXT-DAY OUTCOMES
# ============================================================

next_outcomes = (
    df[
        [
            "participant_id",
            "date",
            "mood",
            "anxiety",
        ]
    ]
    .copy()
)

next_outcomes[
    "date"
] = (
    next_outcomes[
        "date"
    ]
    - pd.Timedelta(
        days=1
    )
)

next_outcomes = (
    next_outcomes.rename(
        columns={
            "mood":
                "mood_next_day",

            "anxiety":
                "anxiety_next_day",
        }
    )
)

df = df.merge(
    next_outcomes,
    on=[
        "participant_id",
        "date",
    ],
    how="left",
)


# ============================================================
# HELPER: PERSON-MEAN DECOMPOSITION
# ============================================================

def decompose(
    x,
    participants,
):
    """Return within-person deviation and participant mean."""

    temp = pd.DataFrame(
        {
            "participant_id":
                participants,

            "x":
                x,
        }
    )

    person_mean = (
        temp.groupby(
            "participant_id"
        )[
            "x"
        ]
        .transform("mean")
    )

    within = (
        temp["x"]
        - person_mean
    )

    return (
        within,
        person_mean,
    )


def standardize(
    x,
):
    """Global z-score, preserving NaN."""

    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    mean = x.mean()
    sd = x.std(
        ddof=0
    )

    if (
        not np.isfinite(sd)
        or sd == 0
    ):
        return x * np.nan

    return (
        x
        - mean
    ) / sd


# ============================================================
# COVERAGE DECOMPOSITION
# ============================================================

(
    df[
        "coverage_within"
    ],
    df[
        "coverage_between"
    ],
) = decompose(
    df[
        "coverage"
    ],
    df[
        "participant_id"
    ],
)

df[
    "coverage_within_z"
] = standardize(
    df[
        "coverage_within"
    ]
)

df[
    "coverage_between_z"
] = standardize(
    df[
        "coverage_between"
    ]
)


# ============================================================
# RUN ANALYSES
# ============================================================

rows = []
sample_rows = []


for feature_name in FEATURES:

    feature_col = (
        f"feature_{feature_name}"
    )

    (
        df[
            "feature_within"
        ],
        df[
            "feature_between"
        ],
    ) = decompose(
        df[
            feature_col
        ],
        df[
            "participant_id"
        ],
    )

    df[
        "feature_within_z"
    ] = standardize(
        df[
            "feature_within"
        ]
    )

    df[
        "feature_between_z"
    ] = standardize(
        df[
            "feature_between"
        ]
    )


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


            analysis = df[
                [
                    "participant_id",
                    "date",
                    outcome_col,
                    "feature_within_z",
                    "feature_between_z",
                    "coverage_within_z",
                    "coverage_between_z",
                ]
            ].dropna().copy()


            n_rows = len(
                analysis
            )

            n_participants = (
                analysis[
                    "participant_id"
                ].nunique()
            )


            sample_rows.append(
                {
                    "feature":
                        feature_name,

                    "outcome":
                        outcome,

                    "timing":
                        timing,

                    "n_rows":
                        n_rows,

                    "n_participants":
                        n_participants,
                }
            )


            if (
                n_rows < 30
                or n_participants < 5
            ):

                continue


            formula = (
                f"{outcome_col} ~ "
                "feature_within_z + "
                "feature_between_z + "
                "coverage_within_z + "
                "coverage_between_z"
            )


            try:

                model = smf.gee(
                    formula=formula,
                    groups="participant_id",
                    data=analysis,
                    family=(
                        sm.families.Gaussian()
                    ),
                    cov_struct=(
                        Exchangeable()
                    ),
                )

                result = model.fit()

            except Exception as exc:

                print(
                    "FAILED:",
                    feature_name,
                    outcome,
                    timing,
                    repr(exc),
                )

                continue


            for effect in [
                "feature_within_z",
                "feature_between_z",
            ]:

                beta = float(
                    result.params[
                        effect
                    ]
                )

                se = float(
                    result.bse[
                        effect
                    ]
                )

                p = float(
                    result.pvalues[
                        effect
                    ]
                )

                ci_low = float(
                    result.conf_int().loc[
                        effect,
                        0,
                    ]
                )

                ci_high = float(
                    result.conf_int().loc[
                        effect,
                        1,
                    ]
                )


                rows.append(
                    {
                        "feature":
                            feature_name,

                        "outcome":
                            outcome,

                        "timing":
                            timing,

                        "effect":
                            (
                                "within_person"
                                if effect
                                == "feature_within_z"
                                else "between_person"
                            ),

                        "beta":
                            beta,

                        "se":
                            se,

                        "ci_low":
                            ci_low,

                        "ci_high":
                            ci_high,

                        "p_value":
                            p,

                        "n_rows":
                            n_rows,

                        "n_participants":
                            n_participants,
                    }
                )


results = pd.DataFrame(
    rows
)

samples = pd.DataFrame(
    sample_rows
)


# ============================================================
# FDR
# ============================================================

results[
    "q_value"
] = np.nan


for effect in [
    "within_person",
    "between_person",
]:

    idx = (
        results[
            "effect"
        ]
        .eq(
            effect
        )
    )

    p = (
        results.loc[
            idx,
            "p_value",
        ]
        .to_numpy()
    )

    if len(p):

        _, q, _, _ = multipletests(
            p,
            alpha=0.05,
            method="fdr_bh",
        )

        results.loc[
            idx,
            "q_value",
        ] = q


results[
    "fdr_significant"
] = (
    results[
        "q_value"
    ] < 0.05
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    RESULT_OUT,
    index=False,
)

samples.to_csv(
    SAMPLE_OUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 100)
print("SAMPLE SIZE")
print("=" * 100)

print(
    samples.sort_values(
        [
            "timing",
            "outcome",
            "feature",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("WITHIN-PERSON RESULTS")
print("=" * 100)

within = (
    results[
        results[
            "effect"
        ].eq(
            "within_person"
        )
    ]
    .sort_values(
        [
            "q_value",
            "p_value",
        ]
    )
)

print(
    within[
        [
            "feature",
            "outcome",
            "timing",
            "beta",
            "ci_low",
            "ci_high",
            "p_value",
            "q_value",
            "n_rows",
            "n_participants",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("BETWEEN-PERSON RESULTS")
print("=" * 100)

between = (
    results[
        results[
            "effect"
        ].eq(
            "between_person"
        )
    ]
    .sort_values(
        [
            "q_value",
            "p_value",
        ]
    )
)

print(
    between[
        [
            "feature",
            "outcome",
            "timing",
            "beta",
            "ci_low",
            "ci_high",
            "p_value",
            "q_value",
            "n_rows",
            "n_participants",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("FDR-SIGNIFICANT RESULTS")
print("=" * 100)

sig = (
    results[
        results[
            "fdr_significant"
        ]
    ]
    .sort_values(
        "q_value"
    )
)

if sig.empty:

    print(
        "None at q < 0.05."
    )

else:

    print(
        sig[
            [
                "feature",
                "outcome",
                "timing",
                "effect",
                "beta",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
                "n_rows",
                "n_participants",
            ]
        ]
        .to_string(
            index=False
        )
    )


print()
print("Saved:")
print(" ", RESULT_OUT)
print(" ", SAMPLE_OUT)
