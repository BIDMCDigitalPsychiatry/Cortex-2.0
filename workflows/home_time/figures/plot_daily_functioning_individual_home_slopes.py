from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_functioning"
)

DATA_PATH = (
    ROOT / "daily_functioning_home_time_analysis_dataset.csv"
)

RESULTS_PATH = (
    ROOT / "daily_functioning_home_time_gee_results.csv"
)

PNG_OUT = (
    ROOT / "daily_functioning_individual_home_slopes.png"
)

PDF_OUT = (
    ROOT / "daily_functioning_individual_home_slopes.pdf"
)

SLOPES_OUT = (
    ROOT / "daily_functioning_individual_home_slopes.csv"
)

# ============================================================
# LOAD PRIMARY ANALYSIS DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

df = df[
    df["analysis_definition"].eq(
        "daily_mean_primary"
    )
].copy()

df["functioning_score"] = pd.to_numeric(
    df["functioning_score"],
    errors="coerce",
)

df["home_time_pct"] = pd.to_numeric(
    df["home_time_pct"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "participant_id",
        "functioning_score",
        "home_time_pct",
    ]
).copy()

# ============================================================
# PARTICIPANT-SPECIFIC WITHIN-PERSON SLOPES
#
# For each participant:
#
#   x_it = (home_it - mean_home_i) / 10
#   y_it = functioning_it - mean_functioning_i
#
# Then:
#
#   slope_i = sum(x*y) / sum(x^2)
#
# This is the participant's descriptive within-person association:
# change in functioning score per +10 percentage points home time.
#
# No arbitrary N cutoff is used.
# A slope is retained whenever:
#   - participant has >=2 observations
#   - home time varies within participant
# ============================================================

rows = []

for pid, g in df.groupby("participant_id"):

    g = g.dropna(
        subset=[
            "functioning_score",
            "home_time_pct",
        ]
    ).copy()

    n = len(g)

    home_mean = (
        g["home_time_pct"].mean()
    )

    func_mean = (
        g["functioning_score"].mean()
    )

    x = (
        g["home_time_pct"].to_numpy()
        - home_mean
    ) / 10.0

    y = (
        g["functioning_score"].to_numpy()
        - func_mean
    )

    denominator = np.sum(
        x ** 2
    )

    functioning_unique = (
        g["functioning_score"].nunique()
    )

    home_unique = (
        g["home_time_pct"].nunique()
    )

    if (
        n >= 2
        and denominator > 0
    ):
        slope = (
            np.sum(x * y)
            / denominator
        )
        estimable = True
    else:
        slope = np.nan
        estimable = False

    rows.append(
        {
            "participant_id": pid,
            "n_days": n,
            "functioning_mean": func_mean,
            "functioning_unique": functioning_unique,
            "home_mean_pct": home_mean,
            "home_unique": home_unique,
            "within_slope_per_10pp": slope,
            "estimable": estimable,
        }
    )

slopes = pd.DataFrame(rows)

slopes.to_csv(
    SLOPES_OUT,
    index=False,
)

estimable = slopes[
    slopes["estimable"]
    & slopes["within_slope_per_10pp"].notna()
].copy()

# ============================================================
# POOLED GEE REFERENCE
# ============================================================

results = pd.read_csv(
    RESULTS_PATH,
    low_memory=False,
)

pooled_row = results[
    results["analysis_definition"].eq(
        "daily_mean_primary"
    )
    &
    results["term"].eq(
        "home_within_10pp"
    )
]

if len(pooled_row) != 1:
    raise RuntimeError(
        "Could not identify pooled within-person GEE coefficient."
    )

pooled_beta = float(
    pooled_row.iloc[0]["beta"]
)

# ============================================================
# SUMMARY
# ============================================================

n_total = slopes["participant_id"].nunique()
n_estimable = len(estimable)
n_not_estimable = n_total - n_estimable

n_negative = int(
    (
        estimable["within_slope_per_10pp"]
        < 0
    ).sum()
)

n_positive = int(
    (
        estimable["within_slope_per_10pp"]
        > 0
    ).sum()
)

n_zero = int(
    (
        estimable["within_slope_per_10pp"]
        == 0
    ).sum()
)

print("=" * 100)
print("PARTICIPANT-SPECIFIC WITHIN-PERSON HOME-TIME SLOPES")
print("=" * 100)

print("\nTotal participants:", n_total)
print("Estimable participant slopes:", n_estimable)
print("Not estimable:", n_not_estimable)

print("\nDirection:")
print("Negative:", n_negative)
print("Positive:", n_positive)
print("Exactly zero:", n_zero)

print("\nSlope distribution:")
print(
    estimable[
        "within_slope_per_10pp"
    ]
    .describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
    .to_string()
)

print(
    "\nPooled GEE within-person beta:",
    round(
        pooled_beta,
        4,
    ),
)

print("\nDays contributing to estimable slopes:")
print(
    estimable["n_days"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
    .to_string()
)

# ============================================================
# FIGURE STYLE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig = plt.figure(
    figsize=(8.5, 8.0)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[
        1,
        1.35,
    ],
    hspace=0.34,
)

# ============================================================
# PANEL A: HISTOGRAM
# ============================================================

ax1 = fig.add_subplot(
    gs[0, 0]
)

vals = estimable[
    "within_slope_per_10pp"
].to_numpy()

ax1.hist(
    vals,
    bins=25,
    edgecolor="black",
    linewidth=0.6,
)

ax1.axvline(
    0,
    linestyle="--",
    linewidth=1.0,
)

ax1.axvline(
    pooled_beta,
    linestyle="-",
    linewidth=1.5,
)

ax1.set_xlabel(
    "Participant-specific within-person slope\n"
    "(functioning points per +10 percentage points home time)"
)

ax1.set_ylabel(
    "Number of participants"
)

ax1.set_title(
    "Distribution of participant-specific associations"
)

ax1.text(
    0.01,
    0.97,
    "(A)",
    transform=ax1.transAxes,
    va="top",
    ha="left",
    fontweight="bold",
    fontsize=12,
)

ax1.text(
    0.99,
    0.97,
    (
        f"N = {n_estimable} estimable participants\n"
        f"Pooled GEE β = {pooled_beta:.3f}"
    ),
    transform=ax1.transAxes,
    va="top",
    ha="right",
    fontsize=9,
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ============================================================
# PANEL B: RANKED PARTICIPANT SLOPES
# ============================================================

ax2 = fig.add_subplot(
    gs[1, 0]
)

ranked = (
    estimable
    .sort_values(
        "within_slope_per_10pp"
    )
    .reset_index(drop=True)
)

ranked["rank"] = (
    np.arange(
        1,
        len(ranked) + 1
    )
)

ax2.scatter(
    ranked["rank"],
    ranked[
        "within_slope_per_10pp"
    ],
    s=18,
    alpha=0.7,
)

ax2.axhline(
    0,
    linestyle="--",
    linewidth=1.0,
)

ax2.axhline(
    pooled_beta,
    linestyle="-",
    linewidth=1.5,
)

ax2.set_xlabel(
    "Participants ranked by within-person slope"
)

ax2.set_ylabel(
    "Within-person slope\n"
    "(functioning points per +10 pp home time)"
)

ax2.set_title(
    "Heterogeneity across participants"
)

ax2.text(
    0.01,
    0.97,
    "(B)",
    transform=ax2.transAxes,
    va="top",
    ha="left",
    fontweight="bold",
    fontsize=12,
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ============================================================
# SAVE
# ============================================================

fig.subplots_adjust(
    left=0.14,
    right=0.97,
    top=0.96,
    bottom=0.09,
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

plt.close(fig)

print("\nSaved:")
print(" ", PNG_OUT)
print(" ", PDF_OUT)
print(" ", SLOPES_OUT)
