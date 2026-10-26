from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

x = pd.read_parquet(PATH)

x["prev_nearby_evidence"] = (
    x["ctx_prev_nearby_observed"]
    .astype(int)
)

x["prev_screen_evidence"] = (
    x["ctx_prev_battery_observed"]
    .astype(int)
)

x["history"] = np.log1p(
    x["prior_gps_specific_onsets"]
)

x["episode_day_scaled"] = (
    x["episode_day"] / 30
)

# Remove competing shared-onset hours.
x = x[
    ~x["shared_onset"]
].copy()


PREDICTORS = [
    "prev_acc_dq",
    "prev_screen_evidence",
    "prev_nearby_evidence",
    "history",
    "episode_day_scaled",
]


counts = (
    x[
        x["gps_specific_onset"]
    ]
    .groupby("episode_id")
    .size()
    .sort_values(
        ascending=False
    )
)


for remove_n in [
    0,
    1,
    4,
    10,
]:

    exclude = set(
        counts.head(
            remove_n
        ).index
    )

    z = x[
        ~x["episode_id"]
        .isin(exclude)
    ].copy()

    z = z[
        [
            "episode_id",
            "gps_specific_onset",
        ]
        + PREDICTORS
    ].dropna()

    variation = (
        z.groupby(
            "episode_id"
        )[
            "gps_specific_onset"
        ]
        .agg(
            ["min", "max"]
        )
    )

    valid = variation[
        variation["min"]
        != variation["max"]
    ].index

    z = z[
        z["episode_id"]
        .isin(valid)
    ].copy()

    design = z[
        PREDICTORS
    ].copy()

    binary = {
        "prev_screen_evidence",
        "prev_nearby_evidence",
    }

    for col in PREDICTORS:

        design[col] = (
            pd.to_numeric(
                design[col],
                errors="coerce",
            )
            .astype(float)
        )

        if col in binary:
            continue

        sd = design[col].std()

        if sd > 0:
            design[col] = (
                design[col]
                - design[col].mean()
            ) / sd

    dummies = pd.get_dummies(
        z["episode_id"],
        drop_first=True,
        dtype=float,
    )

    design = pd.concat(
        [
            design.reset_index(
                drop=True
            ),
            dummies.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    design = sm.add_constant(
        design,
        has_constant="add",
    )

    y = (
        z["gps_specific_onset"]
        .astype(int)
        .reset_index(drop=True)
    )

    groups = (
        z["episode_id"]
        .reset_index(drop=True)
    )

    result = sm.GLM(
        y,
        design,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
        },
        maxiter=200,
    )

    beta = result.params[
        "prev_nearby_evidence"
    ]

    ci = result.conf_int().loc[
        "prev_nearby_evidence"
    ]

    print(
        "\nExclude top",
        remove_n,
        "episodes"
    )

    print(
        "Episodes:",
        z["episode_id"].nunique()
    )

    print(
        "Onsets:",
        int(
            z[
                "gps_specific_onset"
            ].sum()
        )
    )

    print(
        "Nearby OR:",
        round(
            np.exp(beta),
            3,
        )
    )

    print(
        "95% CI:",
        (
            round(
                np.exp(ci.iloc[0]),
                3,
            ),
            round(
                np.exp(ci.iloc[1]),
                3,
            ),
        )
    )

    print(
        "p:",
        result.pvalues[
            "prev_nearby_evidence"
        ]
    )
