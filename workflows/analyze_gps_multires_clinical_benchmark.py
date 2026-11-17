from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


BASE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "multires_clinical_benchmark"
)

DAILY_FEATURES = (
    BASE
    / "gps_multires_daily_features_resolution_aware.parquet"
)

WEEKLY = (
    BASE
    / "gps_multires_weekly_alignment_resolution_aware.parquet"
)

DAILY_EMA = (
    BASE
    / "gps_multires_daily_ema_alignment_resolution_aware.parquet"
)

OUT_DIR = BASE / "analysis"
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ORDER = [
    "native",
    "1min",
    "5min",
    "10min",
    "15min",
    "30min",
    "60min",
    "120min",
    "240min",
]


DAILY_FEATURE_MAP = {
    "displacement":
        "supported_displacement_distance_m",

    "radius_gyration":
        "supported_radius_of_gyration_m",

    "entropy":
        "location_entropy_nats",

    "effective_places":
        "effective_number_places",

    "dominant_place_fraction":
        "dominant_place_fraction",

    "observed_places":
        "n_observed_places",
}


WEEKLY_FEATURE_MAP = {
    name: f"{col}_mean"
    for name, col in DAILY_FEATURE_MAP.items()
}


# ============================================================
# HELPERS
# ============================================================

def zscore(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    sd = series.std()

    if (
        not np.isfinite(sd)
        or sd == 0
    ):
        return pd.Series(
            np.nan,
            index=series.index,
        )

    return (
        series - series.mean()
    ) / sd


def fit_within_between(
    data,
    *,
    outcome,
    feature,
):

    d = data[
        [
            "participant_id",
            outcome,
            feature,
        ]
    ].copy()

    d[outcome] = pd.to_numeric(
        d[outcome],
        errors="coerce",
    )

    d[feature] = pd.to_numeric(
        d[feature],
        errors="coerce",
    )

    d = d.dropna()

    if (
        len(d) < 20
        or d["participant_id"].nunique() < 3
        or d[feature].nunique() < 3
    ):
        return None

    # Person-level mean:
    # stable between-person differences.
    d["feature_person_mean"] = (
        d.groupby("participant_id")[feature]
        .transform("mean")
    )

    # Within-person deviation:
    # whether being above/below one's own typical level
    # relates to symptom severity.
    d["feature_within"] = (
        d[feature]
        - d["feature_person_mean"]
    )

    d["feature_between_z"] = zscore(
        d["feature_person_mean"]
    )

    d["feature_within_z"] = zscore(
        d["feature_within"]
    )

    d = d.dropna(
        subset=[
            "feature_between_z",
            "feature_within_z",
        ]
    )

    if len(d) < 20:
        return None

    formula = (
        f"{outcome} ~ "
        "feature_between_z + "
        "feature_within_z"
    )

    fit = None
    used_method = None

    for method in [
        "lbfgs",
        "powell",
    ]:

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore",
                    ConvergenceWarning,
                )

                warnings.simplefilter(
                    "ignore",
                    UserWarning,
                )

                model = smf.mixedlm(
                    formula,
                    data=d,
                    groups=d["participant_id"],
                )

                candidate = model.fit(
                    reml=False,
                    method=method,
                    disp=False,
                )

            fit = candidate
            used_method = method

            if getattr(
                candidate,
                "converged",
                True,
            ):
                break

        except Exception:
            continue

    if fit is None:
        return None

    result = {
        "n_observations":
            len(d),

        "n_participants":
            d["participant_id"].nunique(),

        "converged":
            bool(
                getattr(
                    fit,
                    "converged",
                    False,
                )
            ),

        "optimizer":
            used_method,
    }

    for component in [
        "between",
        "within",
    ]:

        term = (
            f"feature_{component}_z"
        )

        beta = fit.params.get(
            term,
            np.nan,
        )

        se = fit.bse.get(
            term,
            np.nan,
        )

        p = fit.pvalues.get(
            term,
            np.nan,
        )

        result[
            f"{component}_beta"
        ] = beta

        result[
            f"{component}_ci_low"
        ] = (
            beta - 1.96 * se
            if np.isfinite(beta)
            and np.isfinite(se)
            else np.nan
        )

        result[
            f"{component}_ci_high"
        ] = (
            beta + 1.96 * se
            if np.isfinite(beta)
            and np.isfinite(se)
            else np.nan
        )

        result[
            f"{component}_p"
        ] = p

    return result


# ============================================================
# TECHNICAL FIDELITY
# ============================================================

print()
print("=" * 90)
print("TECHNICAL FIDELITY")
print("=" * 90)

daily = pd.read_parquet(
    DAILY_FEATURES
)

# Remove fidelity columns that were already generated
# by the benchmark workflow.
drop_cols = [
    c
    for c in daily.columns
    if (
        c.endswith("__native")
        or c.endswith("__absolute_error")
        or c.endswith("__relative_error")
    )
]

daily = daily.drop(
    columns=drop_cols,
    errors="ignore",
)

native = (
    daily.loc[
        daily["resolution"].eq("native"),
        [
            "participant_id",
            "date",
        ]
        + list(
            DAILY_FEATURE_MAP.values()
        ),
    ]
    .copy()
)

native = native.rename(
    columns={
        col: f"{col}__native"
        for col
        in DAILY_FEATURE_MAP.values()
    }
)

merged = daily.merge(
    native,
    on=[
        "participant_id",
        "date",
    ],
    how="left",
    validate="many_to_one",
)

technical_rows = []

for resolution in ORDER:

    g = merged.loc[
        merged["resolution"].eq(
            resolution
        )
    ]

    if g.empty:
        continue

    for feature_name, feature in (
        DAILY_FEATURE_MAP.items()
    ):

        x = pd.to_numeric(
            g[feature],
            errors="coerce",
        )

        y = pd.to_numeric(
            g[f"{feature}__native"],
            errors="coerce",
        )

        valid = (
            x.notna()
            & y.notna()
        )

        xv = x[valid]
        yv = y[valid]

        pearson = np.nan
        spearman = np.nan

        if (
            len(xv) >= 3
            and xv.nunique() > 1
            and yv.nunique() > 1
        ):

            pearson = xv.corr(
                yv,
                method="pearson",
            )

            spearman = xv.corr(
                yv,
                method="spearman",
            )

        abs_error = (
            xv - yv
        ).abs()

        denominator = (
            yv.abs()
            .replace(0, np.nan)
        )

        relative_error = (
            abs_error
            / denominator
        )

        technical_rows.append({
            "resolution":
                resolution,

            "feature":
                feature_name,

            "n_total_days":
                len(g),

            "n_valid_pairs":
                len(xv),

            "availability":
                len(xv) / len(g)
                if len(g)
                else np.nan,

            "pearson":
                pearson,

            "spearman":
                spearman,

            "median_absolute_error":
                abs_error.median(),

            "median_relative_error":
                relative_error.median(),

            "median_sampled_points":
                pd.to_numeric(
                    g["sampled_n_points"],
                    errors="coerce",
                ).median(),
        })


technical = pd.DataFrame(
    technical_rows
)

technical.to_csv(
    OUT_DIR
    / "technical_fidelity_summary.csv",
    index=False,
)


# ============================================================
# WEEKLY PHQ-9 / GAD-7
# ============================================================

print()
print("=" * 90)
print("WEEKLY CLINICAL ALIGNMENT")
print("=" * 90)

weekly = pd.read_parquet(
    WEEKLY
)

weekly_rows = []

for outcome in [
    "depression_score",
    "anxiety_score",
]:

    if outcome not in weekly.columns:
        continue

    for feature_name, feature in (
        WEEKLY_FEATURE_MAP.items()
    ):

        if feature not in weekly.columns:
            continue

        for resolution in ORDER:

            g = weekly.loc[
                weekly["resolution"].eq(
                    resolution
                )
            ].copy()

            result = fit_within_between(
                g,
                outcome=outcome,
                feature=feature,
            )

            if result is None:
                continue

            result.update({
                "outcome":
                    outcome,

                "feature":
                    feature_name,

                "resolution":
                    resolution,
            })

            weekly_rows.append(
                result
            )


weekly_models = pd.DataFrame(
    weekly_rows
)


# Effect retention relative to native
for component in [
    "between",
    "within",
]:

    native_effects = (
        weekly_models.loc[
            weekly_models[
                "resolution"
            ].eq("native"),
            [
                "outcome",
                "feature",
                f"{component}_beta",
            ],
        ]
        .rename(
            columns={
                f"{component}_beta":
                    f"{component}_beta_native"
            }
        )
    )

    weekly_models = (
        weekly_models.merge(
            native_effects,
            on=[
                "outcome",
                "feature",
            ],
            how="left",
        )
    )

    weekly_models[
        f"{component}_effect_retention"
    ] = np.where(
        weekly_models[
            f"{component}_beta_native"
        ].abs() >= 0.10,
        weekly_models[
            f"{component}_beta"
        ]
        / weekly_models[
            f"{component}_beta_native"
        ],
        np.nan,
    )

    weekly_models[
        f"{component}_absolute_beta_change"
    ] = (
        weekly_models[
            f"{component}_beta"
        ]
        - weekly_models[
            f"{component}_beta_native"
        ]
    ).abs()


weekly_models.to_csv(
    OUT_DIR
    / "weekly_clinical_multires_models.csv",
    index=False,
)


# ============================================================
# DAILY EMA
# ============================================================

print()
print("=" * 90)
print("DAILY EMA CLINICAL ALIGNMENT")
print("=" * 90)

ema = pd.read_parquet(
    DAILY_EMA
)

ema_rows = []

for outcome in [
    "anxiety",
    "mood",
    "daily_function",
]:

    if outcome not in ema.columns:
        continue

    for feature_name, feature in (
        DAILY_FEATURE_MAP.items()
    ):

        if feature not in ema.columns:
            continue

        for resolution in ORDER:

            g = ema.loc[
                ema["resolution"].eq(
                    resolution
                )
            ].copy()

            result = fit_within_between(
                g,
                outcome=outcome,
                feature=feature,
            )

            if result is None:
                continue

            result.update({
                "outcome":
                    outcome,

                "feature":
                    feature_name,

                "resolution":
                    resolution,
            })

            ema_rows.append(
                result
            )


ema_models = pd.DataFrame(
    ema_rows
)


for component in [
    "between",
    "within",
]:

    native_effects = (
        ema_models.loc[
            ema_models[
                "resolution"
            ].eq("native"),
            [
                "outcome",
                "feature",
                f"{component}_beta",
            ],
        ]
        .rename(
            columns={
                f"{component}_beta":
                    f"{component}_beta_native"
            }
        )
    )

    ema_models = (
        ema_models.merge(
            native_effects,
            on=[
                "outcome",
                "feature",
            ],
            how="left",
        )
    )

    ema_models[
        f"{component}_effect_retention"
    ] = np.where(
        ema_models[
            f"{component}_beta_native"
        ].abs() >= 0.10,
        ema_models[
            f"{component}_beta"
        ]
        / ema_models[
            f"{component}_beta_native"
        ],
        np.nan,
    )

    ema_models[
        f"{component}_absolute_beta_change"
    ] = (
        ema_models[
            f"{component}_beta"
        ]
        - ema_models[
            f"{component}_beta_native"
        ]
    ).abs()


ema_models.to_csv(
    OUT_DIR
    / "daily_ema_multires_models.csv",
    index=False,
)


# ============================================================
# COMPACT OUTPUT
# ============================================================

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    f"Participant-days: "
    f"{daily[daily['resolution'].eq('native')].shape[0]:,}"
)

print(
    f"Participants: "
    f"{daily['participant_id'].nunique():,}"
)

print(
    f"Weekly models: "
    f"{len(weekly_models):,}"
)

print(
    f"Daily EMA models: "
    f"{len(ema_models):,}"
)


print()
print("Technical fidelity: Pearson vs native")
print()

tech_print = (
    technical.pivot(
        index="feature",
        columns="resolution",
        values="pearson",
    )
    .reindex(
        columns=ORDER
    )
)

print(
    tech_print.round(3).to_string()
)


print()
print("PHQ-9 WITHIN-PERSON beta")
print()

phq = weekly_models.loc[
    weekly_models[
        "outcome"
    ].eq("depression_score")
]

if len(phq):

    phq_table = (
        phq.pivot(
            index="feature",
            columns="resolution",
            values="within_beta",
        )
        .reindex(
            columns=ORDER
        )
    )

    print(
        phq_table.round(3).to_string()
    )


print()
print("GAD-7 WITHIN-PERSON beta")
print()

gad = weekly_models.loc[
    weekly_models[
        "outcome"
    ].eq("anxiety_score")
]

if len(gad):

    gad_table = (
        gad.pivot(
            index="feature",
            columns="resolution",
            values="within_beta",
        )
        .reindex(
            columns=ORDER
        )
    )

    print(
        gad_table.round(3).to_string()
    )


print()
print("Saved:")
print(
    OUT_DIR
    / "technical_fidelity_summary.csv"
)
print(
    OUT_DIR
    / "weekly_clinical_multires_models.csv"
)
print(
    OUT_DIR
    / "daily_ema_multires_models.csv"
)
