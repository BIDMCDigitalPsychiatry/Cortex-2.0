from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

RISK_PATH = (
    ROOT
    / "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

OS_PATH = (
    ROOT
    / "inventory/"
    "risk_episode_ios_sensorkit_evidence_timeaware.csv"
)

OUTDIR = (
    ROOT
    / "quality/hourly_missingness/"
    "onset_models/os_interaction"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

x = pd.read_parquet(
    RISK_PATH
)

os_map = pd.read_csv(
    OS_PATH
)

x = x.merge(
    os_map[
        [
            "episode_id",
            "sensorkit_ios_evidence_in_episode",
        ]
    ],
    on="episode_id",
    how="left",
    validate="many_to_one",
)


x["confirmed_ios"] = (
    x[
        "sensorkit_ios_evidence_in_episode"
    ]
    .fillna(False)
    .astype(int)
)


# ============================================================
# DERIVED VARIABLES
# ============================================================

x["prev_screen_evidence"] = (
    x[
        "ctx_prev_battery_observed"
    ]
    .astype(int)
)

x["prev_wifi_evidence"] = (
    pd.to_numeric(
        x[
            "ctx_prev_unique_wifi_id_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
    .astype(int)
)

x["prev_bluetooth_evidence"] = (
    pd.to_numeric(
        x[
            "ctx_prev_unique_bluetooth_id_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
    .astype(int)
)


x[
    "ios_x_prev_wifi"
] = (
    x["confirmed_ios"]
    * x["prev_wifi_evidence"]
)


x[
    "log1p_prior_gps_specific"
] = np.log1p(
    x[
        "prior_gps_specific_onsets"
    ]
)

x[
    "log1p_prior_shared"
] = np.log1p(
    x[
        "prior_shared_onsets"
    ]
)


x[
    "episode_day_scaled"
] = (
    x["episode_day"]
    / 30.0
)


x["utc_hour_sin"] = np.sin(
    2
    * np.pi
    * x["utc_hour"]
    / 24.0
)

x["utc_hour_cos"] = np.cos(
    2
    * np.pi
    * x["utc_hour"]
    / 24.0
)

x["utc_dow_sin"] = np.sin(
    2
    * np.pi
    * x["utc_day_of_week"]
    / 7.0
)

x["utc_dow_cos"] = np.cos(
    2
    * np.pi
    * x["utc_day_of_week"]
    / 7.0
)


BASE_PREDICTORS = [
    "prev_acc_dq",
    "prev_screen_evidence",
    "prev_wifi_evidence",
    "prev_bluetooth_evidence",
    "history",
    "episode_day_scaled",
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_dow_sin",
    "utc_dow_cos",
    "ios_x_prev_wifi",
]


BINARY = {
    "prev_screen_evidence",
    "prev_wifi_evidence",
    "prev_bluetooth_evidence",
    "ios_x_prev_wifi",
}


OUTCOMES = {
    "gps_specific": {
        "outcome":
            "gps_specific_onset",

        "competing":
            "shared_onset",

        "history":
            "log1p_prior_gps_specific",
    },

    "shared": {
        "outcome":
            "shared_onset",

        "competing":
            "gps_specific_onset",

        "history":
            "log1p_prior_shared",
    },
}


# ============================================================
# FIT
# ============================================================

def fit_interaction(
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
            "confirmed_ios",
            outcome,
        ]
        + BASE_PREDICTORS
    )

    z = (
        d[cols]
        .dropna()
        .copy()
    )


    # Episode fixed-effects models only use episodes
    # containing both outcome states.
    variation = (
        z.groupby(
            "episode_id"
        )[outcome]
        .agg(
            ["min", "max"]
        )
    )

    valid_episodes = variation[
        variation["min"]
        != variation["max"]
    ].index

    z = z[
        z["episode_id"]
        .isin(valid_episodes)
    ].copy()


    print()
    print("=" * 80)
    print(
        f"{outcome_name.upper()} | "
        "IOS × WIFI INTERACTION"
    )
    print("=" * 80)

    print(
        "Rows:",
        f"{len(z):,}",
    )

    print(
        "Episodes:",
        z[
            "episode_id"
        ].nunique(),
    )

    print(
        "Confirmed-iOS episodes:",
        z.loc[
            z["confirmed_ios"] == 1,
            "episode_id",
        ].nunique(),
    )

    print(
        "OS-unresolved episodes:",
        z.loc[
            z["confirmed_ios"] == 0,
            "episode_id",
        ].nunique(),
    )

    print(
        "Onsets:",
        int(
            z[outcome].sum()
        ),
    )


    design = z[
        BASE_PREDICTORS
    ].copy()


    # Standardize continuous predictors exactly as in
    # previous onset models.
    for col in BASE_PREDICTORS:

        design[col] = pd.to_numeric(
            design[col],
            errors="coerce",
        ).astype(float)

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
        .reset_index(
            drop=True
        )
    )

    groups = (
        z["episode_id"]
        .reset_index(
            drop=True
        )
    )


    result = sm.GLM(
        y,
        design,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                groups,
        },
        maxiter=200,
    )


    # --------------------------------------------------------
    # Extract Wi-Fi effects
    # --------------------------------------------------------

    b_wifi = result.params[
        "prev_wifi_evidence"
    ]

    b_int = result.params[
        "ios_x_prev_wifi"
    ]

    cov = result.cov_params()


    # OS-unresolved Wi-Fi effect
    se_unresolved = result.bse[
        "prev_wifi_evidence"
    ]

    unresolved_low = (
        b_wifi
        - 1.96
        * se_unresolved
    )

    unresolved_high = (
        b_wifi
        + 1.96
        * se_unresolved
    )


    # Confirmed-iOS Wi-Fi effect = beta_wifi + beta_interaction
    b_ios = (
        b_wifi
        + b_int
    )

    var_ios = (
        cov.loc[
            "prev_wifi_evidence",
            "prev_wifi_evidence",
        ]
        + cov.loc[
            "ios_x_prev_wifi",
            "ios_x_prev_wifi",
        ]
        + 2
        * cov.loc[
            "prev_wifi_evidence",
            "ios_x_prev_wifi",
        ]
    )

    se_ios = np.sqrt(
        var_ios
    )

    ios_low = (
        b_ios
        - 1.96
        * se_ios
    )

    ios_high = (
        b_ios
        + 1.96
        * se_ios
    )


    interaction_or = np.exp(
        b_int
    )

    interaction_ci_low = np.exp(
        b_int
        - 1.96
        * result.bse[
            "ios_x_prev_wifi"
        ]
    )

    interaction_ci_high = np.exp(
        b_int
        + 1.96
        * result.bse[
            "ios_x_prev_wifi"
        ]
    )

    interaction_p = result.pvalues[
        "ios_x_prev_wifi"
    ]


    summary = pd.DataFrame(
        [
            {
                "outcome":
                    outcome_name,

                "group":
                    "OS_unresolved",

                "wifi_OR":
                    np.exp(
                        b_wifi
                    ),

                "ci_low":
                    np.exp(
                        unresolved_low
                    ),

                "ci_high":
                    np.exp(
                        unresolved_high
                    ),
            },

            {
                "outcome":
                    outcome_name,

                "group":
                    "confirmed_iOS",

                "wifi_OR":
                    np.exp(
                        b_ios
                    ),

                "ci_low":
                    np.exp(
                        ios_low
                    ),

                "ci_high":
                    np.exp(
                        ios_high
                    ),
            },
        ]
    )


    print()
    print(
        "Adjusted previous-hour Wi-Fi effects:"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda v:
                f"{v:.4f}",
        )
    )


    print()
    print(
        "Interaction OR ratio "
        "(confirmed iOS / unresolved):"
    )

    print(
        f"{interaction_or:.4f} "
        f"[{interaction_ci_low:.4f}, "
        f"{interaction_ci_high:.4f}]"
    )

    print(
        "Interaction p:",
        f"{interaction_p:.4f}",
    )


    interaction_table = pd.DataFrame(
        [
            {
                "outcome":
                    outcome_name,

                "interaction_beta":
                    b_int,

                "interaction_se":
                    result.bse[
                        "ios_x_prev_wifi"
                    ],

                "interaction_OR_ratio":
                    interaction_or,

                "ci_low":
                    interaction_ci_low,

                "ci_high":
                    interaction_ci_high,

                "p":
                    interaction_p,

                "rows":
                    len(z),

                "episodes":
                    z[
                        "episode_id"
                    ].nunique(),

                "onsets":
                    int(
                        z[outcome].sum()
                    ),
            }
        ]
    )


    summary.to_csv(
        OUTDIR
        / (
            f"{outcome_name}"
            "__wifi_group_effects.csv"
        ),
        index=False,
    )

    interaction_table.to_csv(
        OUTDIR
        / (
            f"{outcome_name}"
            "__wifi_interaction.csv"
        ),
        index=False,
    )


    return (
        summary,
        interaction_table,
    )


# ============================================================
# RUN
# ============================================================

all_effects = []
all_interactions = []


for outcome_name, cfg in (
    OUTCOMES.items()
):

    effects, interaction = (
        fit_interaction(
            outcome_name,
            cfg["outcome"],
            cfg["competing"],
            cfg["history"],
        )
    )

    all_effects.append(
        effects
    )

    all_interactions.append(
        interaction
    )


pd.concat(
    all_effects,
    ignore_index=True,
).to_csv(
    OUTDIR
    / "wifi_group_effects.csv",
    index=False,
)


pd.concat(
    all_interactions,
    ignore_index=True,
).to_csv(
    OUTDIR
    / "wifi_interactions.csv",
    index=False,
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(
    OUTDIR
)
