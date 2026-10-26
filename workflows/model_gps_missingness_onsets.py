from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PATH = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)

OUT = Path(
    "../cortex2_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "onset_models"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


x = pd.read_parquet(PATH)


# ============================================================
# DERIVED VARIABLES
# ============================================================

x["prev_nearby_evidence"] = (
    x["ctx_prev_nearby_observed"]
    .astype(int)
)

# Nearby-device modality decomposition.
# These represent positive detection evidence,
# not connection to a network/device.
x["prev_wifi_evidence"] = (
    pd.to_numeric(
        x["ctx_prev_unique_wifi_id_count"],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
    .astype(int)
)

x["prev_bluetooth_evidence"] = (
    pd.to_numeric(
        x["ctx_prev_unique_bluetooth_id_count"],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
    .astype(int)
)

x["prev_screen_evidence"] = (
    x["ctx_prev_battery_observed"]
    .astype(int)
)

x["prev_device_usage_evidence"] = (
    x["ctx_prev_device_usage_observed"]
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

# Cyclic UTC time controls.
# These are nuisance temporal covariates only;
# UTC is not interpreted as participant-local behavior time.
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


MODELS = {
    "core": [
        "prev_acc_dq",
        "prev_screen_evidence",
        "prev_nearby_evidence",
        "history",
        "episode_day_scaled",
        "utc_hour_sin",
        "utc_hour_cos",
        "utc_dow_sin",
        "utc_dow_cos",
    ],

    "battery": [
        "prev_acc_dq",
        "prev_nearby_evidence",
        "ctx_prev_battery_last",
        "history",
        "episode_day_scaled",
        "utc_hour_sin",
        "utc_hour_cos",
        "utc_dow_sin",
        "utc_dow_cos",
    ],

    "nearby_modalities": [
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
    ],

    "device_usage": [
        "prev_acc_dq",
        "prev_screen_evidence",
        "prev_nearby_evidence",
        "prev_device_usage_evidence",
        "history",
        "episode_day_scaled",
        "utc_hour_sin",
        "utc_hour_cos",
        "utc_dow_sin",
        "utc_dow_cos",
    ],

    "device_activity": [
        "prev_acc_dq",
        "prev_screen_evidence",
        "prev_nearby_evidence",
        "ctx_prev_unlock_count",
        "ctx_prev_screen_wake_count",
        "history",
        "episode_day_scaled",
        "utc_hour_sin",
        "utc_hour_cos",
        "utc_dow_sin",
        "utc_dow_cos",
    ],
}


OUTCOMES = {
    "gps_specific": {
        "outcome": "gps_specific_onset",
        "competing": "shared_onset",
        "history": "log1p_prior_gps_specific",
    },

    "shared": {
        "outcome": "shared_onset",
        "competing": "gps_specific_onset",
        "history": "log1p_prior_shared",
    },
}


BINARY = {
    "prev_screen_evidence",
    "prev_nearby_evidence",
    "prev_wifi_evidence",
    "prev_bluetooth_evidence",
    "prev_device_usage_evidence",
}


def fit_model(
    data,
    *,
    outcome,
    predictors,
    model_name,
    outcome_name,
):

    cols = (
        [outcome, "episode_id"]
        + predictors
    )

    z = (
        data[cols]
        .dropna()
        .copy()
    )

    # Keep episodes containing both an onset and
    # at least one no-onset risk hour.
    variation = (
        z.groupby("episode_id")[outcome]
        .agg(["min", "max"])
    )

    valid = variation[
        variation["min"]
        != variation["max"]
    ].index

    z = z[
        z["episode_id"].isin(valid)
    ].copy()

    print()
    print("=" * 72)
    print(
        outcome_name.upper(),
        "|",
        model_name.upper(),
    )
    print("=" * 72)

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

    # --------------------------------------------------------
    # Standardize continuous predictors
    # --------------------------------------------------------

    design = z[
        predictors
    ].copy()

    scaling = {}

    for col in predictors:

        design[col] = pd.to_numeric(
            design[col],
            errors="coerce",
        ).astype(float)

        if col in BINARY:
            continue

        mean = float(
            design[col].mean()
        )

        sd = float(
            design[col].std()
        )

        scaling[col] = {
            "mean": mean,
            "sd": sd,
        }

        if np.isfinite(sd) and sd > 0:
            design[col] = (
                design[col] - mean
            ) / sd

    # --------------------------------------------------------
    # Episode fixed effects
    # --------------------------------------------------------

    episode_dummies = pd.get_dummies(
        z["episode_id"],
        prefix="episode",
        drop_first=True,
        dtype=float,
    )

    design = pd.concat(
        [
            design.reset_index(drop=True),
            episode_dummies.reset_index(drop=True),
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

    # --------------------------------------------------------
    # Logistic regression with episode fixed effects
    # and episode-clustered robust SEs.
    # --------------------------------------------------------

    model = sm.GLM(
        y,
        design,
        family=sm.families.Binomial(),
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
        },
        maxiter=200,
    )

    # Only report substantive predictors,
    # not dozens of episode intercepts.
    params = result.params[
        predictors
    ]

    se = result.bse[
        predictors
    ]

    p = result.pvalues[
        predictors
    ]

    conf = result.conf_int().loc[
        predictors
    ]

    table = pd.DataFrame({
        "predictor":
            predictors,

        "beta":
            params.values,

        "se":
            se.values,

        "p":
            p.values,

        "odds_ratio":
            np.exp(
                params.values
            ),

        "ci_low":
            np.exp(
                conf.iloc[:, 0]
                .values
            ),

        "ci_high":
            np.exp(
                conf.iloc[:, 1]
                .values
            ),
    })

    print()

    print(
        table.to_string(
            index=False,
            float_format=lambda v:
                f"{v:.4f}",
        )
    )

    table.to_csv(
        OUT
        / f"{outcome_name}__{model_name}.csv",
        index=False,
    )

    # Save scaling information so OR interpretation
    # remains reproducible.
    if scaling:

        scaling_df = pd.DataFrame(
            [
                {
                    "predictor": k,
                    **v,
                }
                for k, v
                in scaling.items()
            ]
        )

        scaling_df.to_csv(
            OUT
            / (
                f"{outcome_name}"
                f"__{model_name}"
                "__scaling.csv"
            ),
            index=False,
        )


# ============================================================
# FIT CAUSE-SPECIFIC MODELS
# ============================================================

for outcome_name, cfg in (
    OUTCOMES.items()
):

    outcome = cfg["outcome"]
    competing = cfg["competing"]
    history_col = cfg["history"]

    # Remove competing onset hours.
    d = x[
        ~x[competing]
    ].copy()

    d["history"] = (
        d[history_col]
    )

    for model_name, predictors in (
        MODELS.items()
    ):

        fit_model(
            d,
            outcome=outcome,
            predictors=predictors,
            model_name=model_name,
            outcome_name=outcome_name,
        )


print()
print("Saved model tables:")
print(OUT)
