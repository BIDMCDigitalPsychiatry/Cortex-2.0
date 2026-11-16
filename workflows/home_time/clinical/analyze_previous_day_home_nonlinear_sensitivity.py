from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import dmatrix, build_design_matrices

warnings.filterwarnings("ignore")

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_longitudinal"
)

DATA_FILE = ROOT / "home_t_function_t1_analysis_dataset.csv"

OUT_RESULTS = ROOT / "previous_day_home_nonlinear_sensitivity_stats.csv"
OUT_CURVES = ROOT / "previous_day_home_nonlinear_sensitivity_curves.csv"
OUT_FIG = ROOT / "previous_day_home_nonlinear_sensitivity.png"

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "participant_id",
        "home_dev",
        "functioning_t1",
        "weekend",
        "study_day_10",
    ]
).copy()

# Central 95% of exposure distribution
# Same restriction for every model.
lo = df["home_dev"].quantile(0.025)
hi = df["home_dev"].quantile(0.975)

df = df[
    df["home_dev"].between(lo, hi)
].copy()

print()
print("=" * 100)
print("NONLINEAR PREVIOUS-DAY HOME-TIME SENSITIVITY")
print("=" * 100)

print("Observations:", len(df))
print("Participants:", df["participant_id"].nunique())
print(
    "Exposure range:",
    f"{lo:.1f} to {hi:.1f} pp relative to personal usual"
)

# ============================================================
# ONE COMMON SPLINE BASIS
#
# Crucial: all models use exactly the same knots/basis.
# ============================================================

basis = dmatrix(
    "cr(home_dev, df=3) - 1",
    data=df,
    return_type="dataframe"
)

basis.columns = [
    "spline1",
    "spline2",
    "spline3",
]

basis_info = basis.design_info

model_df = pd.concat(
    [
        df.reset_index(drop=True),
        basis.reset_index(drop=True),
    ],
    axis=1
)

# prior-day functioning decomposition already exists
# from the earlier audited analysis file
if "functioning_dev_t" not in model_df.columns:

    person_mean_func = (
        model_df.groupby("participant_id")["functioning_t"]
        .transform("mean")
    )

    model_df["functioning_dev_t"] = (
        model_df["functioning_t"]
        - person_mean_func
    )

# ============================================================
# HELPERS
# ============================================================

def clustered_fit(formula, data):

    return smf.ols(
        formula=formula,
        data=data
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": data["participant_id"]
        }
    )


def global_spline_p(fit, interaction=False):

    if not interaction:

        test = fit.wald_test(
            "spline1 = 0, spline2 = 0, spline3 = 0",
            scalar=True
        )

    else:

        test = fit.wald_test(
            "spline1:weekend = 0, "
            "spline2:weekend = 0, "
            "spline3:weekend = 0",
            scalar=True
        )

    return float(test.pvalue)


def make_basis(values):

    b = build_design_matrices(
        [basis_info],
        {"home_dev": np.asarray(values)},
        return_type="dataframe"
    )[0]

    b.columns = [
        "spline1",
        "spline2",
        "spline3",
    ]

    return b


def relative_curve(
    fit,
    grid,
    weekend_value=0,
    prior_adjusted=False
):

    gb = make_basis(grid)
    rb = make_basis([0])

    pred = gb.copy()
    ref = rb.copy()

    pred["weekend"] = weekend_value
    ref["weekend"] = weekend_value

    pred["study_day_10"] = (
        model_df["study_day_10"].median()
    )

    ref["study_day_10"] = (
        model_df["study_day_10"].median()
    )

    reference_pid = (
        model_df["participant_id"].iloc[0]
    )

    pred["participant_id"] = reference_pid
    ref["participant_id"] = reference_pid

    if prior_adjusted:
        # person-centered prior functioning = 0 means
        # functioning is at that person's usual level
        pred["functioning_dev_t"] = 0
        ref["functioning_dev_t"] = 0

    design_info = fit.model.data.design_info

    Xg = build_design_matrices(
        [design_info],
        pred,
        return_type="dataframe"
    )[0]

    Xr = build_design_matrices(
        [design_info],
        ref,
        return_type="dataframe"
    )[0]

    beta = fit.params.values
    cov = fit.cov_params().values

    refvec = Xr.iloc[0].values

    effects = []
    ses = []

    for i in range(len(Xg)):

        diff = (
            Xg.iloc[i].values
            - refvec
        )

        effect = diff @ beta

        var = (
            diff
            @ cov
            @ diff
        )

        effects.append(effect)

        ses.append(
            np.sqrt(max(var, 0))
        )

    out = pd.DataFrame({
        "home_dev": grid,
        "effect": effects,
        "se": ses,
    })

    out["ci_low"] = (
        out["effect"]
        - 1.96 * out["se"]
    )

    out["ci_high"] = (
        out["effect"]
        + 1.96 * out["se"]
    )

    return out


# ============================================================
# MODEL 1
# NONLINEAR WITHOUT PRIOR FUNCTIONING
# ============================================================

m1_df = model_df.dropna(
    subset=[
        "functioning_t1",
        "spline1",
        "spline2",
        "spline3",
        "weekend",
        "study_day_10",
    ]
).copy()

m1 = clustered_fit(
    (
        "functioning_t1 ~ "
        "spline1 + spline2 + spline3 + "
        "weekend + study_day_10 + "
        "C(participant_id)"
    ),
    m1_df
)

m1_p = global_spline_p(m1)

# ============================================================
# MODEL 2
# NONLINEAR + PRIOR-DAY FUNCTIONING
# ============================================================

m2_df = model_df.dropna(
    subset=[
        "functioning_t1",
        "functioning_dev_t",
        "spline1",
        "spline2",
        "spline3",
        "weekend",
        "study_day_10",
    ]
).copy()

m2 = clustered_fit(
    (
        "functioning_t1 ~ "
        "spline1 + spline2 + spline3 + "
        "functioning_dev_t + "
        "weekend + study_day_10 + "
        "C(participant_id)"
    ),
    m2_df
)

m2_p = global_spline_p(m2)

# ============================================================
# MODEL 3
# WEEKDAY/WEEKEND MODERATION
#
# spline(home deviation) × weekend
# ============================================================

m3_df = m1_df.copy()

m3 = clustered_fit(
    (
        "functioning_t1 ~ "
        "(spline1 + spline2 + spline3) * weekend + "
        "study_day_10 + "
        "C(participant_id)"
    ),
    m3_df
)

m3_home_p = global_spline_p(m3)

try:
    m3_interaction_p = global_spline_p(
        m3,
        interaction=True
    )
except Exception:
    m3_interaction_p = np.nan

# ============================================================
# PREDICTION CURVES
# ============================================================

grid = np.linspace(
    lo,
    hi,
    250
)

curve1 = relative_curve(
    m1,
    grid,
    weekend_value=0,
    prior_adjusted=False
)

curve1["model"] = "without_prior_function"

curve2 = relative_curve(
    m2,
    grid,
    weekend_value=0,
    prior_adjusted=True
)

curve2["model"] = "with_prior_function"

weekday_curve = relative_curve(
    m3,
    grid,
    weekend_value=0,
    prior_adjusted=False
)

weekday_curve["model"] = "weekday"

weekend_curve = relative_curve(
    m3,
    grid,
    weekend_value=1,
    prior_adjusted=False
)

weekend_curve["model"] = "weekend"

all_curves = pd.concat(
    [
        curve1,
        curve2,
        weekday_curve,
        weekend_curve,
    ],
    ignore_index=True
)

all_curves.to_csv(
    OUT_CURVES,
    index=False
)

# ============================================================
# CONCRETE CONTRASTS
# ============================================================

contrast_values = [
    -30,
    -20,
    -10,
    10,
    20,
    30,
]

def print_contrasts(name, curve):

    print()
    print(name)

    for v in contrast_values:

        if v < lo or v > hi:
            continue

        idx = (
            curve["home_dev"] - v
        ).abs().idxmin()

        r = curve.loc[idx]

        print(
            f"{v:+.0f} pp: "
            f"{r['effect']:+.4f} "
            f"(95% CI "
            f"{r['ci_low']:+.4f}, "
            f"{r['ci_high']:+.4f})"
        )


# ============================================================
# RESULTS
# ============================================================

stats = pd.DataFrame(
    [
        {
            "model": "nonlinear_without_prior_function",
            "n": len(m1_df),
            "n_participants":
                m1_df["participant_id"].nunique(),
            "global_home_p": m1_p,
            "interaction_p": np.nan,
        },
        {
            "model": "nonlinear_with_prior_function",
            "n": len(m2_df),
            "n_participants":
                m2_df["participant_id"].nunique(),
            "global_home_p": m2_p,
            "interaction_p": np.nan,
        },
        {
            "model": "weekday_weekend_interaction",
            "n": len(m3_df),
            "n_participants":
                m3_df["participant_id"].nunique(),
            "global_home_p": m3_home_p,
            "interaction_p":
                m3_interaction_p,
        },
    ]
)

stats.to_csv(
    OUT_RESULTS,
    index=False
)

print()
print("=" * 100)
print("MODEL RESULTS")
print("=" * 100)

print(
    stats.to_string(
        index=False,
        float_format=lambda z: f"{z:.5f}"
    )
)

print_contrasts(
    "WITHOUT PRIOR FUNCTIONING",
    curve1
)

print_contrasts(
    "WITH PRIOR FUNCTIONING",
    curve2
)

print_contrasts(
    "WEEKDAY",
    weekday_curve
)

print_contrasts(
    "WEEKEND",
    weekend_curve
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.2)
)

# ------------------------------------------------------------
# A. WITH VS WITHOUT PRIOR FUNCTIONING
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    curve1["home_dev"],
    curve1["effect"],
    linewidth=2.2,
    label="Without prior functioning"
)

ax.fill_between(
    curve1["home_dev"],
    curve1["ci_low"],
    curve1["ci_high"],
    alpha=0.12
)

ax.plot(
    curve2["home_dev"],
    curve2["effect"],
    linewidth=2.2,
    linestyle="--",
    label="Adjusted for prior functioning"
)

ax.fill_between(
    curve2["home_dev"],
    curve2["ci_low"],
    curve2["ci_high"],
    alpha=0.10
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.6"
)

ax.axvline(
    0,
    linestyle=":",
    linewidth=1,
    color="0.6"
)

ax.set_xlabel(
    "Previous-day home time relative to personal usual\n"
    "(percentage points)"
)

ax.set_ylabel(
    "Difference in next-day functioning\n"
    "relative to usual home time"
)

ax.set_title(
    "A. Sensitivity to prior functioning adjustment"
)

p1_text = (
    "p<0.001"
    if m1_p < 0.001
    else f"p={m1_p:.3f}"
)

p2_text = (
    "p<0.001"
    if m2_p < 0.001
    else f"p={m2_p:.3f}"
)

ax.text(
    0.03,
    0.05,
    (
        f"Without prior functioning: {p1_text}\n"
        f"Adjusted for prior functioning: {p2_text}"
    ),
    transform=ax.transAxes,
    fontsize=9
)

ax.legend(
    frameon=False,
    fontsize=9
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# B. WEEKDAY VS WEEKEND
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    weekday_curve["home_dev"],
    weekday_curve["effect"],
    linewidth=2.2,
    label="Weekday"
)

ax.fill_between(
    weekday_curve["home_dev"],
    weekday_curve["ci_low"],
    weekday_curve["ci_high"],
    alpha=0.12
)

ax.plot(
    weekend_curve["home_dev"],
    weekend_curve["effect"],
    linewidth=2.2,
    linestyle="--",
    label="Weekend"
)

ax.fill_between(
    weekend_curve["home_dev"],
    weekend_curve["ci_low"],
    weekend_curve["ci_high"],
    alpha=0.10
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.6"
)

ax.axvline(
    0,
    linestyle=":",
    linewidth=1,
    color="0.6"
)

ax.set_xlabel(
    "Previous-day home time relative to personal usual\n"
    "(percentage points)"
)

ax.set_ylabel(
    "Difference in next-day functioning\n"
    "relative to usual home time"
)

ax.set_title(
    "B. Weekday vs weekend context"
)

# Panel B focuses on whether the nonlinear relationship
# differs between weekdays and weekends.
interaction_text = (
    "Interaction p<0.001"
    if m3_interaction_p < 0.001
    else f"Interaction p={m3_interaction_p:.3f}"
)

ax.text(
    0.97,
    0.95,
    interaction_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9
)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="upper left"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Nonlinear association between previous-day home time "
    "and next-day functioning",
    fontsize=14
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("Saved:")
print(OUT_RESULTS)
print(OUT_CURVES)
print(OUT_FIG)
