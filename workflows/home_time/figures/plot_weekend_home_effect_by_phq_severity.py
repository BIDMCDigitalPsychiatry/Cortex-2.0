from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

warnings.filterwarnings("ignore")

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

FIG_OUT = OUT / "weekend_home_effect_by_phq_severity.png"
CSV_OUT = OUT / "weekend_home_effect_by_phq_severity.csv"
COUNTS_OUT = OUT / "weekend_home_phq_support_counts.csv"

# ============================================================
# LOAD
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

# participant-level mean PHQ-9
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

df["weekend_home_10"] = (
    df["weekend_home_pct"] / 10.0
)

# center PHQ for interaction model
phq_center = df["mean_phq9"].mean()

df["phq_c"] = (
    df["mean_phq9"] - phq_center
)

# ============================================================
# SUPPORT COUNTS BY PHQ THRESHOLD
# ============================================================

count_rows = []

for threshold in range(
    int(np.floor(df["mean_phq9"].min())),
    int(np.ceil(df["mean_phq9"].max())) + 1
):

    z = df[
        df["mean_phq9"] >= threshold
    ]

    count_rows.append(
        {
            "phq_threshold": threshold,
            "n_participants": z[
                "participant_id"
            ].nunique(),
            "n_windows": len(z),
        }
    )

counts = pd.DataFrame(
    count_rows
)

counts.to_csv(
    COUNTS_OUT,
    index=False
)

print()
print("=" * 90)
print("PHQ SUPPORT COUNTS")
print("=" * 90)

print(
    counts.to_string(
        index=False
    )
)

# ============================================================
# INTERACTION MODEL
#
# functioning ~ weekend_home * PHQ
#
# This directly tests whether the effect of home stay changes
# as depression severity increases.
# ============================================================

model = smf.gee(
    formula=(
        "following_week_functioning ~ "
        "weekend_home_10 * phq_c + "
        "study_time_weeks"
    ),
    groups="participant_id",
    data=df,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable(),
).fit()

print()
print("=" * 90)
print("INTERACTION MODEL")
print("=" * 90)
print(model.summary())

interaction_p = float(
    model.pvalues[
        "weekend_home_10:phq_c"
    ]
)

print()
print(
    "Interaction p =",
    f"{interaction_p:.4f}"
)

# ============================================================
# MARGINAL EFFECT OF +10PP HOME TIME ACROSS PHQ
#
# slope(PHQ) =
# beta_home + beta_interaction * (PHQ - center)
# ============================================================

phq_min = max(
    0,
    np.nanpercentile(
        df["mean_phq9"],
        2.5
    )
)

phq_max = np.nanpercentile(
    df["mean_phq9"],
    97.5
)

phq_grid = np.linspace(
    phq_min,
    phq_max,
    150
)

params = model.params
cov = model.cov_params()

b_home = float(
    params["weekend_home_10"]
)

b_int = float(
    params["weekend_home_10:phq_c"]
)

var_home = float(
    cov.loc[
        "weekend_home_10",
        "weekend_home_10"
    ]
)

var_int = float(
    cov.loc[
        "weekend_home_10:phq_c",
        "weekend_home_10:phq_c"
    ]
)

cov_hi = float(
    cov.loc[
        "weekend_home_10",
        "weekend_home_10:phq_c"
    ]
)

rows = []

for phq_value in phq_grid:

    pc = (
        phq_value - phq_center
    )

    effect = (
        b_home
        + b_int * pc
    )

    var_effect = (
        var_home
        + (pc ** 2) * var_int
        + 2 * pc * cov_hi
    )

    se_effect = np.sqrt(
        max(
            var_effect,
            0
        )
    )

    rows.append(
        {
            "phq9": phq_value,
            "effect_per_10pp_home": effect,
            "ci_low": effect - 1.96 * se_effect,
            "ci_high": effect + 1.96 * se_effect,
        }
    )

effects = pd.DataFrame(
    rows
)

effects.to_csv(
    CSV_OUT,
    index=False
)

# ============================================================
# SAMPLE SUPPORT ACROSS PHQ
#
# Count participants around each PHQ value using a ±1-point
# window. This shows where the effect curve is data-supported.
# ============================================================

support_rows = []

for phq_value in phq_grid:

    nearby = df[
        np.abs(
            df["mean_phq9"]
            - phq_value
        ) <= 1.0
    ]

    support_rows.append(
        {
            "phq9": phq_value,
            "n_participants_nearby": nearby[
                "participant_id"
            ].nunique(),
            "n_windows_nearby": len(nearby),
        }
    )

support = pd.DataFrame(
    support_rows
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig = plt.figure(
    figsize=(9.5, 6.4)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[
        4,
        1.2
    ],
    hspace=0.08
)

# ------------------------------------------------------------
# TOP — marginal effect curve
# ------------------------------------------------------------

ax = fig.add_subplot(
    gs[0]
)

ax.plot(
    effects["phq9"],
    effects["effect_per_10pp_home"],
    linewidth=2.3,
)

ax.fill_between(
    effects["phq9"],
    effects["ci_low"],
    effects["ci_high"],
    alpha=0.18,
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="0.5"
)

# Clinical reference markers
for value, label in [
    (5, "Mild"),
    (10, "Moderate"),
    (15, "Moderately severe"),
]:

    if (
        effects["phq9"].min()
        <= value
        <= effects["phq9"].max()
    ):

        ax.axvline(
            value,
            linestyle=":",
            linewidth=0.8,
            color="0.65"
        )

        ax.text(
            value,
            ax.get_ylim()[1],
            label,
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.set_ylabel(
    "Change in following-week functioning\n"
    "per +10 percentage points weekend home stay"
)

ax.set_title(
    "Does weekend home stay become more informative as depression severity increases?"
)

ax.text(
    0.02,
    0.04,
    f"PHQ × weekend-home interaction p = {interaction_p:.3f}",
    transform=ax.transAxes,
    fontsize=9
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ------------------------------------------------------------
# BOTTOM — support
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    gs[1],
    sharex=ax
)

ax2.fill_between(
    support["phq9"],
    0,
    support["n_participants_nearby"],
    alpha=0.25
)

ax2.plot(
    support["phq9"],
    support["n_participants_nearby"],
    linewidth=1.5
)

ax2.set_xlabel(
    "Mean PHQ-9"
)

ax2.set_ylabel(
    "Participants\nwithin ±1 PHQ point"
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.setp(
    ax.get_xticklabels(),
    visible=False
)

fig.suptitle(
    "Weekend home stay × depression severity → following-week functioning",
    fontsize=14,
    y=0.98
)

fig.savefig(
    FIG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("Saved:")
print(FIG_OUT)
print(CSV_OUT)
print(COUNTS_OUT)
