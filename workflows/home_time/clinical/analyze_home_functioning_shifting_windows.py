from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_longitudinal"
)

DATA_FILE = ROOT / "home_t_function_t1_analysis_dataset.csv"

OUT_RESULTS = ROOT / "home_functioning_shifting_window_results.csv"
OUT_DATA = ROOT / "home_functioning_shifting_window_dataset.csv"
OUT_FIG = ROOT / "home_functioning_shifting_windows.png"

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# ============================================================
# RECONSTRUCT DAILY HOME + FUNCTIONING
# ============================================================

daily = (
    df[
        [
            "participant_id",
            "date",
            "home_pct",
            "gps_hours",
            "functioning_t",
        ]
    ]
    .drop_duplicates(
        subset=[
            "participant_id",
            "date",
        ]
    )
    .sort_values(
        [
            "participant_id",
            "date",
        ]
    )
    .copy()
)

daily = daily.rename(
    columns={
        "functioning_t": "functioning"
    }
)

# Person-centered home time
daily["home_mean"] = (
    daily.groupby(
        "participant_id"
    )["home_pct"]
    .transform("mean")
)

daily["home_dev"] = (
    daily["home_pct"]
    - daily["home_mean"]
)

# Person-centered functioning for visualization
daily["function_mean"] = (
    daily.groupby(
        "participant_id"
    )["functioning"]
    .transform("mean")
)

daily["function_dev"] = (
    daily["functioning"]
    - daily["function_mean"]
)

# ============================================================
# BUILD EXACT PREVIOUS-DAY WINDOWS
#
# For functioning assessed on date t:
#
# 1-day  = t-1
# 3-day  = mean(t-1:t-3)
# 7-day  = mean(t-1:t-7)
#
# Only observed calendar days are used.
# Require enough observed days within each window:
#   1-day: 1/1
#   3-day: >=2/3
#   7-day: >=4/7
# ============================================================

rows = []

for pid, g in daily.groupby("participant_id"):

    g = g.sort_values("date").copy()

    lookup = (
        g.set_index("date")
    )

    for r in g.itertuples():

        if pd.isna(r.functioning):
            continue

        date = r.date

        out = {
            "participant_id": pid,
            "date": date,
            "functioning": r.functioning,
            "function_dev": r.function_dev,
        }

        for window, min_days in [
            (1, 1),
            (3, 2),
            (7, 4),
        ]:

            dates = [
                date - pd.Timedelta(days=k)
                for k in range(1, window + 1)
            ]

            vals = []

            for d in dates:
                if d in lookup.index:
                    v = lookup.loc[d, "home_dev"]

                    if isinstance(v, pd.Series):
                        v = v.iloc[0]

                    if pd.notna(v):
                        vals.append(float(v))

            out[f"home_dev_{window}d"] = (
                np.mean(vals)
                if len(vals) >= min_days
                else np.nan
            )

            out[f"n_home_days_{window}d"] = len(vals)

        rows.append(out)

analysis = pd.DataFrame(rows)

# ============================================================
# COVARIATES
# ============================================================

analysis["weekend"] = (
    analysis["date"].dt.dayofweek >= 5
).astype(int)

first_date = (
    analysis.groupby(
        "participant_id"
    )["date"]
    .transform("min")
)

analysis["study_day_10"] = (
    (
        analysis["date"]
        - first_date
    ).dt.days / 10
)

for w in [1, 3, 7]:
    analysis[f"home_dev_{w}d_10"] = (
        analysis[f"home_dev_{w}d"] / 10
    )

analysis.to_csv(
    OUT_DATA,
    index=False
)

# ============================================================
# SEPARATE FIXED-EFFECTS MODELS
#
# Each window answers its own question.
# No simultaneous inclusion of correlated windows.
# ============================================================

results = []

fits = {}

for window in [1, 3, 7]:

    term = f"home_dev_{window}d_10"

    mdf = analysis.dropna(
        subset=[
            "participant_id",
            "functioning",
            term,
            "weekend",
            "study_day_10",
        ]
    ).copy()

    # Need repeated outcome observations
    counts = (
        mdf.groupby("participant_id")
        .size()
    )

    keep = counts[
        counts >= 3
    ].index

    mdf = mdf[
        mdf["participant_id"].isin(keep)
    ].copy()

    fit = smf.ols(
        formula=(
            f"functioning ~ {term} "
            "+ weekend "
            "+ study_day_10 "
            "+ C(participant_id)"
        ),
        data=mdf
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": mdf["participant_id"]
        }
    )

    fits[window] = (
        fit,
        mdf
    )

    beta = float(
        fit.params[term]
    )

    se = float(
        fit.bse[term]
    )

    results.append(
        {
            "window_days": window,
            "n_observations": len(mdf),
            "n_participants": mdf["participant_id"].nunique(),
            "beta_per_10pp": beta,
            "se": se,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "p_value": float(
                fit.pvalues[term]
            ),
        }
    )

results = pd.DataFrame(results)

results.to_csv(
    OUT_RESULTS,
    index=False
)

print()
print("=" * 100)
print("SHIFTING-WINDOW WITHIN-PERSON RESULTS")
print("=" * 100)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# PANEL A:
# 7-DAY WITHIN-PERSON DOSE RESPONSE
#
# Bin observed 7-day home deviations into deciles solely for
# visualization. The statistical model above remains continuous.
# ============================================================

fit7, d7 = fits[7]

vis = d7[
    [
        "home_dev_7d",
        "function_dev",
    ]
].dropna().copy()

# Remove only exact duplicate quantile edges if needed
vis["bin"] = pd.qcut(
    vis["home_dev_7d"],
    q=10,
    duplicates="drop"
)

binned = (
    vis.groupby(
        "bin",
        observed=True
    )
    .agg(
        home_dev=("home_dev_7d", "mean"),
        function_dev=("function_dev", "mean"),
        n=("function_dev", "size"),
    )
    .reset_index()
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5)
)

# ------------------------------------------------------------
# A. Within-person dose response
# ------------------------------------------------------------

ax = axes[0]

ax.scatter(
    binned["home_dev"],
    binned["function_dev"],
    s=55,
)

# Simple continuous visual fit to the binned points
coef = np.polyfit(
    binned["home_dev"],
    binned["function_dev"],
    deg=1
)

xgrid = np.linspace(
    binned["home_dev"].min(),
    binned["home_dev"].max(),
    100
)

ygrid = (
    coef[0] * xgrid
    + coef[1]
)

ax.plot(
    xgrid,
    ygrid,
    linewidth=2
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.6"
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.6"
)

ax.set_xlabel(
    "Previous 7-day home time\n"
    "relative to personal usual (percentage points)"
)

ax.set_ylabel(
    "Daily functioning\n"
    "relative to personal usual"
)

r7 = results.loc[
    results["window_days"] == 7
].iloc[0]

ax.set_title(
    "A. Within-person dose-response\n"
    f"continuous model p = {r7['p_value']:.3f}"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# B. Temporal window comparison
# ------------------------------------------------------------

ax = axes[1]

order = results.copy()

y = np.arange(
    len(order)
)

ax.errorbar(
    order["beta_per_10pp"],
    y,
    xerr=[
        order["beta_per_10pp"]
        - order["ci_low"],

        order["ci_high"]
        - order["beta_per_10pp"],
    ],
    fmt="o",
    markersize=7,
    capsize=4,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.6"
)

ax.set_yticks(y)

ax.set_yticklabels(
    [
        "Previous day",
        "Previous 3 days",
        "Previous 7 days",
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Change in functioning per +10 pp\n"
    "home time above personal usual"
)

ax.set_title(
    "B. Temporal horizon"
)

# Add compact result labels
for i, r in order.reset_index(drop=True).iterrows():

    ax.text(
        r["ci_high"] + 0.001,
        i,
        (
            f"β={r['beta_per_10pp']:.3f}, "
            f"p={r['p_value']:.3f}"
        ),
        va="center",
        fontsize=8.5,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

fig.suptitle(
    "Within-person association between recent home time and daily functioning",
    fontsize=14
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.93
    ]
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
print(OUT_DATA)
print(OUT_FIG)
