from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(
    "../raphe_outputs/digital_clinic"
)

RISK_PATH = (
    ROOT
    / "quality/hourly_missingness"
    / "gps_onset_contextual_risk_set.parquet"
)

CONTEXT_DIR = (
    ROOT
    / "quality/hourly_context/episodes"
)

OUT = (
    ROOT
    / "quality/hourly_missingness"
    / "onset_models"
    / "gps_nearby_lag_profile.csv"
)


# ============================================================
# RISK SET
# ============================================================

x = pd.read_parquet(
    RISK_PATH
)

# GPS-specific onset vs no-onset only.
x = x[
    ~x["shared_onset"]
].copy()


# ============================================================
# COMPLETE HOURLY NEARBY CONTEXT
# ============================================================

frames = []

for path in sorted(
    CONTEXT_DIR.glob("*.parquet")
):

    z = pd.read_parquet(
        path,
        columns=[
            "episode_id",
            "hour_index",
            "nearby_observation_present",
        ],
    )

    frames.append(z)


ctx = pd.concat(
    frames,
    ignore_index=True,
)

ctx = ctx.sort_values(
    [
        "episode_id",
        "hour_index",
    ]
).reset_index(drop=True)


# Build lags on COMPLETE hourly grids.
for lag in [
    1,
    2,
    3,
    6,
]:

    ctx[
        f"nearby_lag{lag}"
    ] = (
        ctx
        .groupby(
            "episode_id",
            sort=False,
        )[
            "nearby_observation_present"
        ]
        .shift(lag)
    )


keep = [
    "episode_id",
    "hour_index",
    "nearby_lag1",
    "nearby_lag2",
    "nearby_lag3",
    "nearby_lag6",
]

x = x.merge(
    ctx[keep],
    on=[
        "episode_id",
        "hour_index",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# OTHER CORE COVARIATES
# ============================================================

x[
    "prev_screen_evidence"
] = (
    x[
        "ctx_prev_battery_observed"
    ]
    .astype(int)
)

x[
    "history"
] = np.log1p(
    x[
        "prior_gps_specific_onsets"
    ]
)

x[
    "episode_day_scaled"
] = (
    x["episode_day"]
    / 30.0
)

x[
    "utc_hour_sin"
] = np.sin(
    2 * np.pi
    * x["utc_hour"]
    / 24.0
)

x[
    "utc_hour_cos"
] = np.cos(
    2 * np.pi
    * x["utc_hour"]
    / 24.0
)

x[
    "utc_dow_sin"
] = np.sin(
    2 * np.pi
    * x["utc_day_of_week"]
    / 7.0
)

x[
    "utc_dow_cos"
] = np.cos(
    2 * np.pi
    * x["utc_day_of_week"]
    / 7.0
)


BASE = [
    "prev_acc_dq",
    "prev_screen_evidence",
    "history",
    "episode_day_scaled",
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_dow_sin",
    "utc_dow_cos",
]


BINARY = {
    "prev_screen_evidence",
    "nearby_lag1",
    "nearby_lag2",
    "nearby_lag3",
    "nearby_lag6",
}


def fit(
    predictors,
    label,
):

    z = x[
        [
            "episode_id",
            "gps_specific_onset",
        ]
        + predictors
    ].dropna().copy()

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
        predictors
    ].copy()

    for col in predictors:

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
        z[
            "gps_specific_onset"
        ]
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

    rows = []

    lag_predictors = [
        p for p in predictors
        if p.startswith(
            "nearby_lag"
        )
    ]

    for p in lag_predictors:

        beta = result.params[p]

        ci = result.conf_int().loc[
            p
        ]

        rows.append({
            "model":
                label,

            "predictor":
                p,

            "rows":
                len(z),

            "episodes":
                z[
                    "episode_id"
                ].nunique(),

            "onsets":
                int(
                    z[
                        "gps_specific_onset"
                    ].sum()
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
                result.pvalues[p],
        })

    return rows


results = []


# ============================================================
# ONE LAG AT A TIME
# ============================================================

for lag in [
    1,
    2,
    3,
    6,
]:

    results += fit(
        BASE
        + [
            f"nearby_lag{lag}"
        ],
        f"lag_{lag}h",
    )


# ============================================================
# JOINT MODEL
# ============================================================

results += fit(
    BASE
    + [
        "nearby_lag1",
        "nearby_lag2",
        "nearby_lag3",
        "nearby_lag6",
    ],
    "joint_lags",
)


out = pd.DataFrame(
    results
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)


print()
print("=" * 76)
print("GPS-SPECIFIC NEARBY-DEVICE LAG PROFILE")
print("=" * 76)

print(
    out.to_string(
        index=False,
        float_format=lambda v:
            f"{v:.4f}",
    )
)

print()
print("Saved:")
print(OUT)
