from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from patsy import dmatrix

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)

WEEKEND_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "temporal_functioning"
    / "weekend_home_following_week_functioning.csv"
)

CLINICAL_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "paper_replication"
    / "home_time_age_gad7_analysis_dataset.csv"
)

OUT = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "temporal_functioning"
)

OUT.mkdir(parents=True, exist_ok=True)

FIG_OUT = (
    OUT
    / "weekend_home_functioning_real_multipanel.png"
)

THRESHOLD_OUT = (
    OUT
    / "weekend_home_functioning_phq_threshold_sensitivity.csv"
)

SURFACE_OUT = (
    OUT
    / "weekend_home_functioning_phq_surface.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    WEEKEND_FILE,
    low_memory=False
)

clin = pd.read_csv(
    CLINICAL_FILE,
    low_memory=False
)

clin["phq9_total"] = pd.to_numeric(
    clin["phq9_total"],
    errors="coerce"
)

# Participant-level mean PHQ-9
phq = (
    clin.groupby("participant_id")
    .agg(
        mean_phq9=("phq9_total", "mean"),
        n_phq=("phq9_total", "count"),
    )
    .reset_index()
)

df = df.merge(
    phq,
    on="participant_id",
    how="left"
)

df = df.dropna(
    subset=[
        "weekend_home_pct",
        "following_week_functioning",
        "mean_phq9",
    ]
).copy()

# scale for interpretable coefficients
df["weekend_home_10"] = (
    df["weekend_home_pct"] / 10.0
)

print()
print("=" * 100)
print("REAL MULTIPANEL ANALYSIS")
print("=" * 100)
print("Total windows:", len(df))
print("Participants:", df["participant_id"].nunique())

# ============================================================
# PANEL A DATA
# Elevated depression = PHQ-9 >= 10
# Spline dose-response using real data
# ============================================================

a = df[
    df["mean_phq9"] >= 10
].copy()

print()
print("Panel A participants:", a["participant_id"].nunique())
print("Panel A windows:", len(a))

# Use natural cubic spline with 4 df
spline_basis = dmatrix(
    "cr(x, df=4) - 1",
    {"x": a["weekend_home_pct"]},
    return_type="dataframe",
)

spline_cols = [
    f"spline_{i}"
    for i in range(spline_basis.shape[1])
]

spline_basis.columns = spline_cols

a_model = pd.concat(
    [
        a.reset_index(drop=True),
        spline_basis.reset_index(drop=True),
    ],
    axis=1,
)

formula_a = (
    "following_week_functioning ~ "
    + " + ".join(spline_cols)
    + " + study_time_weeks"
)

gee_a = smf.gee(
    formula=formula_a,
    groups="participant_id",
    data=a_model,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

# Prediction grid only over observed central range
xmin = np.nanpercentile(
    a["weekend_home_pct"],
    2.5
)

xmax = np.nanpercentile(
    a["weekend_home_pct"],
    97.5
)

xgrid = np.linspace(
    xmin,
    xmax,
    200
)

grid_basis = dmatrix(
    "cr(x, df=4) - 1",
    {"x": xgrid},
    return_type="dataframe",
)

grid_basis.columns = spline_cols

pred_a = grid_basis.copy()

pred_a["study_time_weeks"] = (
    a["study_time_weeks"].median()
)

# manually construct design matrix in model order
design_cols = gee_a.params.index.tolist()

Xpred = pd.DataFrame(
    index=np.arange(len(xgrid))
)

for col in design_cols:

    if col == "Intercept":
        Xpred[col] = 1.0

    elif col == "study_time_weeks":
        Xpred[col] = pred_a[
            "study_time_weeks"
        ].values

    elif col in spline_cols:
        Xpred[col] = pred_a[col].values

Xpred = Xpred[
    design_cols
]

beta = gee_a.params.values
cov = gee_a.cov_params().values

mean_pred = (
    Xpred.values @ beta
)

pred_var = np.einsum(
    "ij,jk,ik->i",
    Xpred.values,
    cov,
    Xpred.values
)

pred_se = np.sqrt(
    np.clip(
        pred_var,
        0,
        None
    )
)

lower = mean_pred - 1.96 * pred_se
upper = mean_pred + 1.96 * pred_se

# ============================================================
# PANEL B DATA
# Continuous PHQ x home-stay interaction surface
# ============================================================

b = df.copy()

# center to improve numerical stability
home_center = b["weekend_home_pct"].mean()
phq_center = b["mean_phq9"].mean()

b["home_c"] = (
    b["weekend_home_pct"]
    - home_center
) / 10.0

b["phq_c"] = (
    b["mean_phq9"]
    - phq_center
) / 5.0

gee_b = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "home_c * phq_c + study_time_weeks"
    ),
    groups="participant_id",
    data=b,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

home_grid = np.linspace(
    np.nanpercentile(
        b["weekend_home_pct"],
        5
    ),
    np.nanpercentile(
        b["weekend_home_pct"],
        95
    ),
    80
)

phq_grid = np.linspace(
    np.nanpercentile(
        b["mean_phq9"],
        5
    ),
    np.nanpercentile(
        b["mean_phq9"],
        95
    ),
    80
)

HH, PP = np.meshgrid(
    home_grid,
    phq_grid
)

surface = pd.DataFrame(
    {
        "weekend_home_pct": HH.ravel(),
        "mean_phq9": PP.ravel(),
    }
)

surface["home_c"] = (
    surface["weekend_home_pct"]
    - home_center
) / 10.0

surface["phq_c"] = (
    surface["mean_phq9"]
    - phq_center
) / 5.0

surface["study_time_weeks"] = (
    b["study_time_weeks"].median()
)

surface["predicted_functioning"] = (
    gee_b.predict(surface)
)

surface.to_csv(
    SURFACE_OUT,
    index=False
)

ZZ = (
    surface["predicted_functioning"]
    .to_numpy()
    .reshape(HH.shape)
)

# ============================================================
# PANEL C
# PHQ threshold sensitivity
# ============================================================

threshold_rows = []

for threshold in range(5, 19):

    z = df[
        df["mean_phq9"] >= threshold
    ].copy()

    # require enough data to fit meaningfully
    n_participants = (
        z["participant_id"].nunique()
    )

    n_windows = len(z)

    if (
        n_participants < 8
        or n_windows < 20
    ):
        continue

    counts = (
        z.groupby("participant_id")
        .size()
    )

    keep = counts[
        counts >= 2
    ].index

    z = z[
        z["participant_id"].isin(keep)
    ].copy()

    if z["participant_id"].nunique() < 5:
        continue

    z["weekend_home_10"] = (
        z["weekend_home_pct"] / 10.0
    )

    try:

        m = smf.gee(
            formula=(
                "following_week_functioning ~ "
                "weekend_home_10 + study_time_weeks"
            ),
            groups="participant_id",
            data=z,
            family=sm.families.Gaussian(),
            cov_struct=Exchangeable(),
        ).fit()

        beta = float(
            m.params["weekend_home_10"]
        )

        se = float(
            m.bse["weekend_home_10"]
        )

        p = float(
            m.pvalues["weekend_home_10"]
        )

        threshold_rows.append(
            {
                "phq_threshold": threshold,
                "beta_per_10pp_home": beta,
                "ci_low": beta - 1.96 * se,
                "ci_high": beta + 1.96 * se,
                "p_value": p,
                "n_participants": z[
                    "participant_id"
                ].nunique(),
                "n_windows": len(z),
            }
        )

    except Exception:
        continue

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    THRESHOLD_OUT,
    index=False
)

print()
print("=" * 100)
print("PHQ THRESHOLD SENSITIVITY")
print("=" * 100)

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9.5

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15.5, 5.2)
)

# ------------------------------------------------------------
# A — dose response
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    xgrid,
    mean_pred,
    linewidth=2.2,
)

ax.fill_between(
    xgrid,
    lower,
    upper,
    alpha=0.18,
)

# Rug showing real data density
ymin = np.nanmin(lower)
yrange = (
    np.nanmax(upper)
    - np.nanmin(lower)
)

rug_y0 = ymin - 0.06 * yrange
rug_y1 = ymin - 0.025 * yrange

for val in a["weekend_home_pct"]:
    ax.vlines(
        val,
        rug_y0,
        rug_y1,
        linewidth=0.5,
        alpha=0.15,
    )

ax.set_xlabel(
    "Weekend home stay (%)"
)

ax.set_ylabel(
    "Predicted following-week functioning"
)

ax.set_title(
    "A. Continuous dose-response\nPHQ-9 ≥10"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# B — interaction surface
# ------------------------------------------------------------

ax = axes[1]

contour = ax.contourf(
    HH,
    PP,
    ZZ,
    levels=14,
)

lines = ax.contour(
    HH,
    PP,
    ZZ,
    levels=6,
    linewidths=0.7,
)

ax.clabel(
    lines,
    inline=True,
    fontsize=7,
    fmt="%.2f"
)

ax.scatter(
    b["weekend_home_pct"],
    b["mean_phq9"],
    s=7,
    alpha=0.12,
)

ax.set_xlabel(
    "Weekend home stay (%)"
)

ax.set_ylabel(
    "Mean PHQ-9"
)

ax.set_title(
    "B. Depression severity × weekend home stay"
)

cbar = fig.colorbar(
    contour,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

cbar.set_label(
    "Predicted functioning"
)

# ------------------------------------------------------------
# C — threshold sensitivity
# ------------------------------------------------------------

ax = axes[2]

if not threshold_df.empty:

    ax.plot(
        threshold_df["phq_threshold"],
        threshold_df["beta_per_10pp_home"],
        marker="o",
        linewidth=2,
    )

    ax.fill_between(
        threshold_df["phq_threshold"],
        threshold_df["ci_low"],
        threshold_df["ci_high"],
        alpha=0.18,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="0.5",
    )

    for _, r in threshold_df.iterrows():

        ax.text(
            r["phq_threshold"],
            r["ci_low"] - 0.015,
            f'n={int(r["n_participants"])}',
            ha="center",
            va="top",
            fontsize=6.5,
            rotation=90,
        )

ax.set_xlabel(
    "Minimum mean PHQ-9 used for subgroup"
)

ax.set_ylabel(
    "Change in functioning per +10 pp\nweekend home stay"
)

ax.set_title(
    "C. Depression-threshold sensitivity"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Weekend home stay and following-week functioning",
    fontsize=14
)

fig.text(
    0.5,
    0.015,
    (
        "All panels use observed Digital Clinic data. "
        "Panel C reports participant count under each PHQ-9 threshold."
    ),
    ha="center",
    fontsize=8.5
)

fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        0.93,
    ]
)

fig.savefig(
    FIG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print()
print("Saved:")
print(FIG_OUT)
print(THRESHOLD_OUT)
print(SURFACE_OUT)
