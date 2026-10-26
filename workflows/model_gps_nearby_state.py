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


# ============================================================
# PREVIOUS-HOUR MODALITY STATE
# ============================================================

wifi = (
    pd.to_numeric(
        x["ctx_prev_unique_wifi_id_count"],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
)

bt = (
    pd.to_numeric(
        x["ctx_prev_unique_bluetooth_id_count"],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
)


x["nearby_state"] = np.select(
    [
        (~wifi) & (~bt),
        wifi & (~bt),
        (~wifi) & bt,
        wifi & bt,
    ],
    [
        "none",
        "wifi_only",
        "bluetooth_only",
        "wifi_and_bluetooth",
    ],
    default="none",
)


for state in [
    "wifi_only",
    "bluetooth_only",
    "wifi_and_bluetooth",
]:
    x[
        f"state_{state}"
    ] = (
        x["nearby_state"]
        == state
    ).astype(int)


# ============================================================
# OTHER COVARIATES
# ============================================================

x["prev_screen_evidence"] = (
    x["ctx_prev_battery_observed"]
    .astype(int)
)

x["log1p_prior_gps_specific"] = np.log1p(
    x["prior_gps_specific_onsets"]
)

x["log1p_prior_shared"] = np.log1p(
    x["prior_shared_onsets"]
)

x["episode_day_scaled"] = (
    x["episode_day"] / 30.0
)

x["utc_hour_sin"] = np.sin(
    2 * np.pi
    * x["utc_hour"]
    / 24.0
)

x["utc_hour_cos"] = np.cos(
    2 * np.pi
    * x["utc_hour"]
    / 24.0
)

x["utc_dow_sin"] = np.sin(
    2 * np.pi
    * x["utc_day_of_week"]
    / 7.0
)

x["utc_dow_cos"] = np.cos(
    2 * np.pi
    * x["utc_day_of_week"]
    / 7.0
)


PREDICTORS = [
    "prev_acc_dq",
    "prev_screen_evidence",
    "state_wifi_only",
    "state_bluetooth_only",
    "state_wifi_and_bluetooth",
    "history",
    "episode_day_scaled",
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_dow_sin",
    "utc_dow_cos",
]


BINARY = {
    "prev_screen_evidence",
    "state_wifi_only",
    "state_bluetooth_only",
    "state_wifi_and_bluetooth",
}


def descriptive(data, outcome):

    print()
    print("=" * 72)
    print(outcome.upper(), "STATE DISTRIBUTION")
    print("=" * 72)

    case = data[
        data[outcome]
    ]

    control = data[
        ~data[
            "gps_specific_onset"
        ]
        & ~data[
            "shared_onset"
        ]
    ]

    for label, z in [
        ("onset", case),
        ("no_onset", control),
    ]:

        print()
        print(label)

        counts = (
            z["nearby_state"]
            .value_counts()
            .reindex(
                [
                    "none",
                    "wifi_only",
                    "bluetooth_only",
                    "wifi_and_bluetooth",
                ],
                fill_value=0,
            )
        )

        for state, n in counts.items():

            pct = (
                100 * n / len(z)
                if len(z)
                else np.nan
            )

            print(
                f"{state:22s}"
                f"{n:8,d} "
                f"({pct:6.2f}%)"
            )


def fit(
    outcome_name,
    outcome,
    competing,
    history_col,
):

    d = x[
        ~x[competing]
    ].copy()

    d["history"] = (
        d[history_col]
    )

    cols = (
        [
            "episode_id",
            outcome,
        ]
        + PREDICTORS
    )

    z = (
        d[cols]
        .dropna()
        .copy()
    )

    variation = (
        z.groupby(
            "episode_id"
        )[outcome]
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

    for col in PREDICTORS:

        design[col] = (
            pd.to_numeric(
                design[col],
                errors="coerce",
            )
            .astype(float)
        )

        if col in BINARY:
            continue

        sd = design[col].std()

        if (
            np.isfinite(sd)
            and sd > 0
        ):
            design[col] = (
                design[col]
                - design[col].mean()
            ) / sd

    episode_dummies = pd.get_dummies(
        z["episode_id"],
        drop_first=True,
        dtype=float,
    )

    design = pd.concat(
        [
            design.reset_index(
                drop=True
            ),
            episode_dummies.reset_index(
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
        z[outcome]
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

    print()
    print("=" * 72)
    print(outcome_name.upper(), "| NEARBY STATE MODEL")
    print("=" * 72)

    print(
        "Reference state: NO nearby evidence"
    )

    print(
        "Rows:",
        f"{len(z):,}",
    )

    print(
        "Episodes:",
        z["episode_id"].nunique(),
    )

    print(
        "Onsets:",
        int(z[outcome].sum()),
    )

    rows = []

    for predictor in [
        "state_wifi_only",
        "state_bluetooth_only",
        "state_wifi_and_bluetooth",
    ]:

        beta = result.params[
            predictor
        ]

        ci = result.conf_int().loc[
            predictor
        ]

        rows.append({
            "state":
                predictor.replace(
                    "state_",
                    "",
                ),

            "odds_ratio":
                np.exp(beta),

            "ci_low":
                np.exp(
                    ci.iloc[0]
                ),

            "ci_high":
                np.exp(
                    ci.iloc[1]
                ),

            "p":
                result.pvalues[
                    predictor
                ],
        })

    table = pd.DataFrame(rows)

    print()
    print(
        table.to_string(
            index=False,
            float_format=lambda v:
                f"{v:.4f}",
        )
    )


descriptive(
    x,
    "gps_specific_onset",
)

descriptive(
    x,
    "shared_onset",
)


fit(
    "gps_specific",
    "gps_specific_onset",
    "shared_onset",
    "log1p_prior_gps_specific",
)

fit(
    "shared",
    "shared_onset",
    "gps_specific_onset",
    "log1p_prior_shared",
)
