from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import dmatrix

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_longitudinal"
)

DATA_FILE = ROOT / "home_t_function_t1_analysis_dataset.csv"

OUT_FIG = ROOT / "previous_day_home_nonlinear_functioning.png"
OUT_CSV = ROOT / "previous_day_home_nonlinear_functioning_curve.csv"

df = pd.read_csv(DATA_FILE, low_memory=False)

# ============================================================
# CLEAN
# ============================================================

df = df.dropna(
    subset=[
        "participant_id",
        "home_dev",
        "functioning_t1",
        "weekend",
        "study_day_10",
    ]
).copy()

# restrict visualization/model to observed central range
# so tails are not dominated by sparse extreme values
lo = df["home_dev"].quantile(0.025)
hi = df["home_dev"].quantile(0.975)

x = df[
    df["home_dev"].between(lo, hi)
].copy()

print()
print("=" * 100)
print("PREVIOUS-DAY NONLINEAR HOME-TIME ANALYSIS")
print("=" * 100)

print("Observations:", len(x))
print("Participants:", x["participant_id"].nunique())
print(
    "Exposure range used:",
    f"{lo:.1f} to {hi:.1f} percentage points"
)

# ============================================================
# NATURAL CUBIC SPLINE
# 3 df = restrained nonlinear shape
# ============================================================

basis = dmatrix(
    "cr(home_dev, df=3) - 1",
    data=x,
    return_type="dataframe"
)

basis.columns = [
    "spline1",
    "spline2",
    "spline3"
]

model_df = pd.concat(
    [
        x.reset_index(drop=True),
        basis.reset_index(drop=True)
    ],
    axis=1
)

formula = (
    "functioning_t1 ~ "
    "spline1 + spline2 + spline3 + "
    "weekend + study_day_10 + "
    "C(participant_id)"
)

fit = smf.ols(
    formula,
    data=model_df
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": model_df["participant_id"]
    }
)

# ============================================================
# GLOBAL TEST FOR HOME-TIME SPLINE
# ============================================================

try:
    global_test = fit.wald_test(
        "spline1 = 0, spline2 = 0, spline3 = 0",
        scalar=True
    )
    global_p = float(global_test.pvalue)
except Exception:
    global_p = np.nan

print("Global spline p =", f"{global_p:.4f}")

# ============================================================
# PREDICTION GRID
#
# Build predictions relative to a reference exposure = 0,
# so the y-axis shows change from the person's usual home time.
# ============================================================

grid = np.linspace(
    lo,
    hi,
    250
)

from patsy import build_design_matrices

# IMPORTANT: reuse the knots learned from the observed data
grid_basis = build_design_matrices(
    [basis.design_info],
    {"home_dev": grid},
    return_type="dataframe"
)[0]

grid_basis.columns = [
    "spline1",
    "spline2",
    "spline3"
]

# prediction covariates held constant
pred = grid_basis.copy()
pred["weekend"] = 0
pred["study_day_10"] = model_df["study_day_10"].median()

# use one participant as reference dummy level;
# relative prediction removes that intercept anyway
reference_pid = model_df["participant_id"].iloc[0]
pred["participant_id"] = reference_pid

# reference at home_dev = 0
# Use exactly the same learned spline basis for home_dev = 0
ref_basis = build_design_matrices(
    [basis.design_info],
    {"home_dev": [0]},
    return_type="dataframe"
)[0]

ref_basis.columns = [
    "spline1",
    "spline2",
    "spline3"
]

ref = ref_basis.copy()
ref["weekend"] = 0
ref["study_day_10"] = model_df["study_day_10"].median()
ref["participant_id"] = reference_pid

# ============================================================
# DESIGN MATRICES FROM MODEL FORMULA
# ============================================================

design_info = fit.model.data.design_info

Xgrid = build_design_matrices(
    [design_info],
    pred,
    return_type="dataframe"
)[0]

Xref = build_design_matrices(
    [design_info],
    ref,
    return_type="dataframe"
)[0]

beta = fit.params.values
cov = fit.cov_params().values

ref_vec = Xref.iloc[0].values

effects = []
ses = []

for i in range(len(Xgrid)):

    diff = (
        Xgrid.iloc[i].values
        - ref_vec
    )

    effect = diff @ beta

    var = (
        diff
        @ cov
        @ diff
    )

    se = np.sqrt(
        max(var, 0)
    )

    effects.append(effect)
    ses.append(se)

curve = pd.DataFrame(
    {
        "home_dev_pct_points": grid,
        "effect_vs_usual": effects,
        "se": ses,
    }
)

curve["ci_low"] = (
    curve["effect_vs_usual"]
    - 1.96 * curve["se"]
)

curve["ci_high"] = (
    curve["effect_vs_usual"]
    + 1.96 * curve["se"]
)

curve.to_csv(
    OUT_CSV,
    index=False
)

# ============================================================
# USEFUL CONTRASTS
# ============================================================

for value in [-30, -20, -10, 10, 20, 30]:

    if value < lo or value > hi:
        continue

    idx = (
        curve["home_dev_pct_points"]
        - value
    ).abs().idxmin()

    r = curve.loc[idx]

    print(
        f"{value:+.0f} pp vs personal usual: "
        f"{r['effect_vs_usual']:+.4f} "
        f"(95% CI {r['ci_low']:+.4f}, "
        f"{r['ci_high']:+.4f})"
    )

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig = plt.figure(
    figsize=(8.8, 6.2)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[4.4, 1],
    hspace=0.08
)

# ------------------------------------------------------------
# TOP: nonlinear dose-response
# ------------------------------------------------------------

ax = fig.add_subplot(gs[0])

ax.plot(
    curve["home_dev_pct_points"],
    curve["effect_vs_usual"],
    linewidth=2.4
)

ax.fill_between(
    curve["home_dev_pct_points"],
    curve["ci_low"],
    curve["ci_high"],
    alpha=0.18
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.5"
)

ax.axvline(
    0,
    linestyle=":",
    linewidth=1,
    color="0.55"
)

ax.set_ylabel(
    "Difference in next-day functioning\n"
    "relative to usual home time"
)

ax.set_title(
    "Previous-day home time and next-day functioning"
)

ax.text(
    0.02,
    0.05,
    f"Global nonlinear association: p={global_p:.3f}",
    transform=ax.transAxes,
    fontsize=9
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.setp(
    ax.get_xticklabels(),
    visible=False
)

# ------------------------------------------------------------
# BOTTOM: exposure distribution
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    gs[1],
    sharex=ax
)

ax2.hist(
    x["home_dev"],
    bins=35,
    density=True,
    alpha=0.30
)

ax2.axvline(
    0,
    linestyle=":",
    linewidth=1,
    color="0.55"
)

ax2.set_xlabel(
    "Previous-day home time relative to personal usual "
    "(percentage points)"
)

ax2.set_ylabel(
    "Density"
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("Saved:")
print(OUT_FIG)
print(OUT_CSV)
