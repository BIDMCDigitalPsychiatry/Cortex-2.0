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

OS_MAP = (
    ROOT
    / "inventory/"
    "risk_episode_ios_sensorkit_evidence_timeaware.csv"
)

OUTDIR = (
    ROOT
    / "quality/hourly_missingness/"
    "onset_models/os_stratified"
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
    OS_MAP
)


keep = [
    "episode_id",
    "sensorkit_ios_evidence_in_episode",
]

x = x.merge(
    os_map[keep],
    on="episode_id",
    how="left",
    validate="many_to_one",
)


x["os_group"] = np.where(
    x["sensorkit_ios_evidence_in_episode"]
    .fillna(False),
    "definite_iOS",
    "OS_unresolved",
)


# ============================================================
# DERIVED VARIABLES
# ============================================================

x["prev_screen_evidence"] = (
    x["ctx_prev_battery_observed"]
    .astype(int)
)

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
        x[
            "ctx_prev_unique_bluetooth_id_count"
        ],
        errors="coerce",
    )
    .fillna(0)
    .gt(0)
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
    "prev_wifi_evidence",
    "prev_bluetooth_evidence",
    "history",
    "episode_day_scaled",
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_dow_sin",
    "utc_dow_cos",
]


BINARY = {
    "prev_screen_evidence",
    "prev_wifi_evidence",
    "prev_bluetooth_evidence",
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
# MODEL FUNCTION
# ============================================================

def fit_model(
    data,
    *,
    outcome_name,
    outcome,
    competing,
    history_col,
    os_group,
):

    d = data[
        data["os_group"]
        == os_group
    ].copy()

    d = d[
        ~d[competing]
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

    print()
    print("=" * 80)
    print(
        f"{os_group} | "
        f"{outcome_name.upper()}"
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
        "Onsets:",
        int(
            z[outcome].sum()
        ),
    )

    if (
        len(z) == 0
        or z[outcome].sum() == 0
    ):
        print(
            "Insufficient data."
        )
        return None

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

    conf = result.conf_int()

    table = pd.DataFrame({
        "predictor":
            PREDICTORS,

        "beta":
            [
                result.params[p]
                for p in PREDICTORS
            ],

        "se":
            [
                result.bse[p]
                for p in PREDICTORS
            ],

        "p":
            [
                result.pvalues[p]
                for p in PREDICTORS
            ],

        "odds_ratio":
            [
                np.exp(
                    result.params[p]
                )
                for p in PREDICTORS
            ],

        "ci_low":
            [
                np.exp(
                    conf.loc[p, 0]
                )
                for p in PREDICTORS
            ],

        "ci_high":
            [
                np.exp(
                    conf.loc[p, 1]
                )
                for p in PREDICTORS
            ],
    })

    print()
    print(
        table.to_string(
            index=False,
            float_format=lambda v:
                f"{v:.4f}",
        )
    )

    table[
        "os_group"
    ] = os_group

    table[
        "outcome"
    ] = outcome_name

    table[
        "rows"
    ] = len(z)

    table[
        "episodes"
    ] = (
        z["episode_id"]
        .nunique()
    )

    table[
        "onsets"
    ] = int(
        z[outcome].sum()
    )

    path = (
        OUTDIR
        / (
            f"{os_group}"
            f"__{outcome_name}.csv"
        )
    )

    table.to_csv(
        path,
        index=False,
    )

    return table


# ============================================================
# DESCRIPTIVE OS GROUP COUNTS
# ============================================================

print("=" * 80)
print("OS EVIDENCE GROUPS")
print("=" * 80)

episode_counts = (
    x[
        [
            "episode_id",
            "os_group",
        ]
    ]
    .drop_duplicates()
    ["os_group"]
    .value_counts()
)

print(
    episode_counts.to_string()
)


# ============================================================
# FIT
# ============================================================

all_tables = []

for os_group in [
    "definite_iOS",
    "OS_unresolved",
]:

    for outcome_name, cfg in (
        OUTCOMES.items()
    ):

        table = fit_model(
            x,
            outcome_name=outcome_name,
            outcome=cfg["outcome"],
            competing=cfg["competing"],
            history_col=cfg["history"],
            os_group=os_group,
        )

        if table is not None:
            all_tables.append(
                table
            )


if all_tables:

    combined = pd.concat(
        all_tables,
        ignore_index=True,
    )

    combined.to_csv(
        OUTDIR
        / "os_stratified_models.csv",
        index=False,
    )


print()
print("Saved:")
print(OUTDIR)
