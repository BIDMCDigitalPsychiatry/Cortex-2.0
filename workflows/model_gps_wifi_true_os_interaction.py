from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(
    "../raphe_outputs/digital_clinic"
)

RISK_PATH = (
    ROOT
    / "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

OS_PATH = (
    ROOT
    / "inventory/"
    "risk_episode_lamp_analytics_os.csv"
)

OUTDIR = (
    ROOT
    / "quality/hourly_missingness/"
    "onset_models/true_os_interaction"
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
            "phone_os",
        ]
    ],
    on="episode_id",
    how="left",
    validate="many_to_one",
)


# Keep only explicitly identified phone OS.
x = x[
    x["phone_os"].isin(
        [
            "iOS",
            "Android",
        ]
    )
].copy()


x["is_android"] = (
    x["phone_os"]
    .eq("Android")
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
    "android_x_prev_wifi"
] = (
    x["is_android"]
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


PREDICTORS = [
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
    "android_x_prev_wifi",
]


BINARY = {
    "prev_screen_evidence",
    "prev_wifi_evidence",
    "prev_bluetooth_evidence",
    "android_x_prev_wifi",
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
# MODEL
# ============================================================

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
            "phone_os",
            outcome,
        ]
        + PREDICTORS
    )


    z = (
        d[cols]
        .dropna()
        .copy()
    )


    # Fixed-effect logistic model only uses episodes
    # containing both outcome states.
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


    print()
    print("=" * 80)
    print(
        f"{outcome_name.upper()} | "
        "TRUE OS × WIFI INTERACTION"
    )
    print("=" * 80)

    print(
        "Rows:",
        f"{len(z):,}",
    )

    print(
        "Episodes:",
        z["episode_id"].nunique(),
    )

    print(
        "iOS episodes:",
        z.loc[
            z["phone_os"] == "iOS",
            "episode_id",
        ].nunique(),
    )

    print(
        "Android episodes:",
        z.loc[
            z["phone_os"] == "Android",
            "episode_id",
        ].nunique(),
    )

    print(
        "Onsets:",
        int(
            z[outcome].sum()
        ),
    )


    print()
    print(
        "Onsets by OS:"
    )

    onset_by_os = (
        z.groupby(
            "phone_os"
        )[outcome]
        .agg(
            episodes="count",
            onsets="sum",
        )
    )

    print(
        onset_by_os.to_string()
    )


    design = z[
        PREDICTORS
    ].copy()


    for col in PREDICTORS:

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


    cov = result.cov_params()


    # --------------------------------------------------------
    # iOS Wi-Fi effect
    # --------------------------------------------------------

    b_ios = result.params[
        "prev_wifi_evidence"
    ]

    se_ios = result.bse[
        "prev_wifi_evidence"
    ]


    # --------------------------------------------------------
    # Android Wi-Fi effect
    # beta_wifi + beta_interaction
    # --------------------------------------------------------

    b_int = result.params[
        "android_x_prev_wifi"
    ]

    b_android = (
        b_ios
        + b_int
    )


    var_android = (
        cov.loc[
            "prev_wifi_evidence",
            "prev_wifi_evidence",
        ]
        + cov.loc[
            "android_x_prev_wifi",
            "android_x_prev_wifi",
        ]
        + 2
        * cov.loc[
            "prev_wifi_evidence",
            "android_x_prev_wifi",
        ]
    )

    se_android = np.sqrt(
        var_android
    )


    group_effects = pd.DataFrame(
        [
            {
                "outcome":
                    outcome_name,

                "phone_os":
                    "iOS",

                "wifi_OR":
                    np.exp(
                        b_ios
                    ),

                "ci_low":
                    np.exp(
                        b_ios
                        - 1.96
                        * se_ios
                    ),

                "ci_high":
                    np.exp(
                        b_ios
                        + 1.96
                        * se_ios
                    ),
            },

            {
                "outcome":
                    outcome_name,

                "phone_os":
                    "Android",

                "wifi_OR":
                    np.exp(
                        b_android
                    ),

                "ci_low":
                    np.exp(
                        b_android
                        - 1.96
                        * se_android
                    ),

                "ci_high":
                    np.exp(
                        b_android
                        + 1.96
                        * se_android
                    ),
            },
        ]
    )


    interaction_se = result.bse[
        "android_x_prev_wifi"
    ]

    interaction_or = np.exp(
        b_int
    )

    interaction_low = np.exp(
        b_int
        - 1.96
        * interaction_se
    )

    interaction_high = np.exp(
        b_int
        + 1.96
        * interaction_se
    )

    interaction_p = result.pvalues[
        "android_x_prev_wifi"
    ]


    print()
    print(
        "Adjusted previous-hour Wi-Fi effects:"
    )

    print(
        group_effects.to_string(
            index=False,
            float_format=lambda v:
                f"{v:.4f}",
        )
    )


    print()
    print(
        "Interaction OR ratio "
        "(Android / iOS):"
    )

    print(
        f"{interaction_or:.4f} "
        f"[{interaction_low:.4f}, "
        f"{interaction_high:.4f}]"
    )

    print(
        "Interaction p:",
        f"{interaction_p:.4f}",
    )


    interaction = pd.DataFrame(
        [
            {
                "outcome":
                    outcome_name,

                "interaction":
                    "Android × previous Wi-Fi evidence",

                "OR_ratio_android_vs_ios":
                    interaction_or,

                "ci_low":
                    interaction_low,

                "ci_high":
                    interaction_high,

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


    group_effects.to_csv(
        OUTDIR
        / (
            f"{outcome_name}"
            "__wifi_effects_by_os.csv"
        ),
        index=False,
    )

    interaction.to_csv(
        OUTDIR
        / (
            f"{outcome_name}"
            "__os_wifi_interaction.csv"
        ),
        index=False,
    )


    return (
        group_effects,
        interaction,
    )


# ============================================================
# RUN
# ============================================================

print("=" * 80)
print("TRUE PHONE OS DISTRIBUTION")
print("=" * 80)

episode_os = (
    x[
        [
            "episode_id",
            "phone_os",
        ]
    ]
    .drop_duplicates()
)

print(
    episode_os[
        "phone_os"
    ]
    .value_counts()
    .to_string()
)


all_effects = []
all_interactions = []


for outcome_name, cfg in (
    OUTCOMES.items()
):

    effects, interaction = fit(
        outcome_name,
        cfg["outcome"],
        cfg["competing"],
        cfg["history"],
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
    / "wifi_effects_by_true_os.csv",
    index=False,
)


pd.concat(
    all_interactions,
    ignore_index=True,
).to_csv(
    OUTDIR
    / "true_os_wifi_interactions.csv",
    index=False,
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)
print(OUTDIR)
