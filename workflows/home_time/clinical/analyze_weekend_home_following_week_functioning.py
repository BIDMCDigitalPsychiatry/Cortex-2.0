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

HOME_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_home_time.csv"
)

EMA_FILE = (
    ROOT
    / "clinical/daily_ema"
    / "daily_ema_all_participants.csv"
)

MAP_FILE = (
    ROOT
    / "clinical/daily_ema"
    / "daily_ema_gps_participants.csv"
)

OUT = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "temporal_functioning"
)

OUT.mkdir(parents=True, exist_ok=True)

DATA_OUT = OUT / "weekend_home_following_week_functioning.csv"
STATS_OUT = OUT / "weekend_home_following_week_functioning_stats.csv"
FIG_OUT = OUT / "weekend_home_following_week_functioning_quantiles.png"

# ============================================================
# LOAD HOME TIME
# ============================================================

home = pd.read_csv(
    HOME_FILE,
    low_memory=False
)

home["date"] = pd.to_datetime(
    home["date"],
    errors="coerce"
).dt.normalize()

home["supported_duration_ms"] = pd.to_numeric(
    home["supported_duration_ms"],
    errors="coerce"
)

home["raw_home_duration_ms"] = pd.to_numeric(
    home["raw_home_duration_ms"],
    errors="coerce"
)

home["supported_hours"] = (
    home["supported_duration_ms"] / 3_600_000
)

home["home_time_pct"] = (
    100
    * home["raw_home_duration_ms"]
    / home["supported_duration_ms"]
)

# Current operational daily quality criterion
home = home[
    (home["supported_hours"] >= 6)
    &
    home["home_time_pct"].notna()
].copy()

# Weekend only
home["weekday"] = home["date"].dt.weekday

weekend = home[
    home["weekday"].isin([5, 6])
].copy()

# Map both Saturday and Sunday to weekend Saturday
weekend["weekend_start"] = weekend["date"]

sunday = weekend["weekday"] == 6

weekend.loc[
    sunday,
    "weekend_start"
] = (
    weekend.loc[sunday, "date"]
    - pd.Timedelta(days=1)
)

# Weighted weekend home fraction:
# total observed time at home / total supported GPS time
weekend_summary = (
    weekend
    .groupby(
        ["participant_id", "weekend_start"],
        as_index=False
    )
    .agg(
        weekend_home_ms=(
            "raw_home_duration_ms",
            "sum"
        ),
        weekend_supported_ms=(
            "supported_duration_ms",
            "sum"
        ),
        n_valid_weekend_days=(
            "date",
            "nunique"
        ),
        mean_weekend_supported_hours=(
            "supported_hours",
            "mean"
        ),
    )
)

weekend_summary["weekend_home_pct"] = (
    100
    * weekend_summary["weekend_home_ms"]
    / weekend_summary["weekend_supported_ms"]
)

# ============================================================
# LOAD + MAP EMA IDS
# ============================================================

ema = pd.read_csv(
    EMA_FILE,
    low_memory=False
)

mapping = pd.read_csv(
    MAP_FILE,
    low_memory=False
)

mapping = (
    mapping[
        [
            "source_participant_id",
            "cortex_participant_id",
        ]
    ]
    .dropna()
    .drop_duplicates()
)

# Ensure one-to-one mapping
nmap = (
    mapping.groupby("source_participant_id")
    ["cortex_participant_id"]
    .nunique()
)

bad = nmap[nmap > 1]

if len(bad):
    raise RuntimeError(
        f"Found {len(bad)} ambiguous source->Cortex mappings."
    )

mapping = mapping.drop_duplicates(
    subset=["source_participant_id"]
)

ema["source_participant_id"] = (
    ema["participant_id"].astype(str)
)

ema = ema.merge(
    mapping,
    on="source_participant_id",
    how="left"
)

ema = ema.dropna(
    subset=["cortex_participant_id"]
).copy()

ema["participant_id"] = (
    ema["cortex_participant_id"]
)

ema["date"] = pd.to_datetime(
    ema["date"],
    errors="coerce"
).dt.normalize()

ema["daily_function"] = pd.to_numeric(
    ema["daily_function"],
    errors="coerce"
)

ema = ema.dropna(
    subset=[
        "participant_id",
        "date",
        "daily_function",
    ]
).copy()

# Collapse duplicate EMA responses within participant-day
ema_daily = (
    ema
    .groupby(
        ["participant_id", "date"],
        as_index=False
    )
    .agg(
        daily_function=(
            "daily_function",
            "mean"
        ),
        n_function_responses=(
            "daily_function",
            "size"
        ),
    )
)

# Keep following Monday-Friday only
ema_daily["weekday"] = (
    ema_daily["date"].dt.weekday
)

ema_weekday = ema_daily[
    ema_daily["weekday"].between(0, 4)
].copy()

# Assign each Mon-Fri to the immediately preceding Saturday
ema_weekday["weekend_start"] = (
    ema_weekday["date"]
    - pd.to_timedelta(
        ema_weekday["weekday"] + 2,
        unit="D"
    )
)

following_week = (
    ema_weekday
    .groupby(
        ["participant_id", "weekend_start"],
        as_index=False
    )
    .agg(
        following_week_functioning=(
            "daily_function",
            "mean"
        ),
        functioning_sd=(
            "daily_function",
            "std"
        ),
        n_functioning_days=(
            "date",
            "nunique"
        ),
    )
)

# ============================================================
# MERGE WEEKEND EXPOSURE -> FOLLOWING WEEK OUTCOME
# ============================================================

df = weekend_summary.merge(
    following_week,
    on=[
        "participant_id",
        "weekend_start",
    ],
    how="inner"
)

df = df.dropna(
    subset=[
        "weekend_home_pct",
        "following_week_functioning",
    ]
).copy()

# Study-time covariate
first_weekend = (
    df.groupby("participant_id")
    ["weekend_start"]
    .transform("min")
)

df["study_time_weeks"] = (
    (df["weekend_start"] - first_weekend)
    .dt.days
    / 7
)

# Scale continuous home time per +10 percentage points
df["weekend_home_10"] = (
    df["weekend_home_pct"] / 10
)

df.to_csv(
    DATA_OUT,
    index=False
)

print()
print("=" * 100)
print("WEEKEND HOME TIME -> FOLLOWING-WEEK FUNCTIONING")
print("=" * 100)

print("Weekend windows:", len(df))
print(
    "Participants:",
    df["participant_id"].nunique()
)
print(
    "Median weekends/participant:",
    df.groupby("participant_id").size().median()
)
print(
    "Windows with both Sat+Sun valid:",
    int((df["n_valid_weekend_days"] == 2).sum())
)
print(
    "Windows with >=3 following functioning days:",
    int((df["n_functioning_days"] >= 3).sum())
)

# ============================================================
# PRIMARY CONTINUOUS GEE
# ============================================================

model_df = df.copy()

counts = (
    model_df.groupby("participant_id")
    .size()
)

keep_ids = counts[
    counts >= 2
].index

model_df = model_df[
    model_df["participant_id"].isin(keep_ids)
].copy()

if model_df["participant_id"].nunique() >= 2:

    gee = smf.gee(
        formula=(
            "following_week_functioning ~ "
            "weekend_home_10 + study_time_weeks"
        ),
        groups="participant_id",
        data=model_df,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()

    beta = float(
        gee.params["weekend_home_10"]
    )

    se = float(
        gee.bse["weekend_home_10"]
    )

    p = float(
        gee.pvalues["weekend_home_10"]
    )

    print()
    print("PRIMARY CONTINUOUS GEE")
    print(
        "Functioning change per +10pp weekend home time:",
        f"{beta:.4f}"
    )
    print(
        "95% CI:",
        f"{beta - 1.96*se:.4f}",
        "to",
        f"{beta + 1.96*se:.4f}"
    )
    print("p =", f"{p:.4f}")

else:
    beta = se = p = np.nan

# ============================================================
# SENSITIVITY:
# BOTH WEEKEND DAYS + >=3 FUNCTIONING DAYS
# ============================================================

strict = df[
    (df["n_valid_weekend_days"] == 2)
    &
    (df["n_functioning_days"] >= 3)
].copy()

strict_counts = (
    strict.groupby("participant_id")
    .size()
)

strict_ids = strict_counts[
    strict_counts >= 2
].index

strict = strict[
    strict["participant_id"].isin(strict_ids)
].copy()

strict_beta = strict_se = strict_p = np.nan

if strict["participant_id"].nunique() >= 2:

    gee_strict = smf.gee(
        formula=(
            "following_week_functioning ~ "
            "weekend_home_10 + study_time_weeks"
        ),
        groups="participant_id",
        data=strict,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()

    strict_beta = float(
        gee_strict.params["weekend_home_10"]
    )

    strict_se = float(
        gee_strict.bse["weekend_home_10"]
    )

    strict_p = float(
        gee_strict.pvalues["weekend_home_10"]
    )

    print()
    print("STRICT SENSITIVITY")
    print(
        "Both weekend days + >=3 following functioning days"
    )
    print(
        "Functioning change per +10pp weekend home time:",
        f"{strict_beta:.4f}"
    )
    print(
        "95% CI:",
        f"{strict_beta - 1.96*strict_se:.4f}",
        "to",
        f"{strict_beta + 1.96*strict_se:.4f}"
    )
    print("p =", f"{strict_p:.4f}")

# ============================================================
# COWORKER-STYLE QUANTILE FIGURE
#
# Repeated weekend windows are grouped by weekend home time.
# Trend p-values use GEE clustered by participant.
# ============================================================

SPLITS = [
    (2, "Median split"),
    (3, "Tertiles"),
    (4, "Quartiles"),
    (5, "Quintiles"),
]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

fig, axes = plt.subplots(
    1,
    4,
    figsize=(14, 4.2)
)

for ax, (q, title) in zip(
    axes,
    SPLITS
):

    x = df[
        [
            "participant_id",
            "weekend_start",
            "weekend_home_pct",
            "following_week_functioning",
            "study_time_weeks",
        ]
    ].dropna().copy()

    try:
        x["group"] = pd.qcut(
            x["weekend_home_pct"],
            q=q,
            labels=False,
            duplicates="drop"
        )
    except Exception:
        ax.set_axis_off()
        continue

    if x["group"].nunique() != q:
        ax.set_axis_off()
        continue

    # Bar summary
    summary = (
        x.groupby("group")
        .agg(
            mean_functioning=(
                "following_week_functioning",
                "mean"
            ),
            sd_functioning=(
                "following_week_functioning",
                "std"
            ),
            n_windows=(
                "following_week_functioning",
                "size"
            ),
            n_participants=(
                "participant_id",
                "nunique"
            ),
        )
        .reset_index()
    )

    summary["se"] = (
        summary["sd_functioning"]
        / np.sqrt(summary["n_windows"])
    )

    summary["ci95"] = (
        1.96 * summary["se"]
    )

    # Clustered trend test
    x["group_numeric"] = (
        x["group"].astype(float)
    )

    trend_p = np.nan

    try:
        trend = smf.gee(
            formula=(
                "following_week_functioning ~ "
                "group_numeric + study_time_weeks"
            ),
            groups="participant_id",
            data=x,
            family=sm.families.Gaussian(),
            cov_struct=Exchangeable(),
        ).fit()

        trend_p = float(
            trend.pvalues["group_numeric"]
        )

    except Exception:
        pass

    pos = np.arange(q)

    ax.bar(
        pos,
        summary["mean_functioning"],
        yerr=summary["ci95"],
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )

    for i, row in summary.iterrows():

        ax.text(
            i,
            row["mean_functioning"]
            + row["ci95"],
            f'{row["mean_functioning"]:.2f}',
            ha="center",
            va="bottom",
            fontsize=8,
        )

        ax.text(
            i,
            ax.get_ylim()[0],
            (
                f"\nn={int(row['n_windows'])}"
                f"\np={int(row['n_participants'])}"
            ),
            ha="center",
            va="top",
            fontsize=6.5,
        )

    if q == 2:
        labels = [
            "Low",
            "High",
        ]
    else:
        labels = [
            f"Q{i+1}"
            for i in range(q)
        ]

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)

    ax.set_title(
        f"{title}\ntrend p = {trend_p:.4f}",
        fontsize=9
    )

    ax.set_xlabel(
        "Weekend home time"
    )

    if ax is axes[0]:
        ax.set_ylabel(
            "Mean functioning\nfollowing Mon-Fri"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Following-week functioning across increasing weekend home stay",
    fontsize=13
)

fig.tight_layout(
    rect=[0, 0, 1, 0.92]
)

fig.savefig(
    FIG_OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

# ============================================================
# SAVE STATS
# ============================================================

stats = pd.DataFrame(
    [
        {
            "analysis": "primary",
            "criterion": ">=1 valid weekend day; >=1 functioning day",
            "n_windows": len(model_df),
            "n_participants": model_df[
                "participant_id"
            ].nunique(),
            "beta_functioning_per_10pp_home": beta,
            "ci_low": beta - 1.96*se
            if np.isfinite(beta) else np.nan,
            "ci_high": beta + 1.96*se
            if np.isfinite(beta) else np.nan,
            "p_value": p,
        },
        {
            "analysis": "strict",
            "criterion": "2 valid weekend days; >=3 functioning days",
            "n_windows": len(strict),
            "n_participants": strict[
                "participant_id"
            ].nunique(),
            "beta_functioning_per_10pp_home": strict_beta,
            "ci_low": strict_beta - 1.96*strict_se
            if np.isfinite(strict_beta) else np.nan,
            "ci_high": strict_beta + 1.96*strict_se
            if np.isfinite(strict_beta) else np.nan,
            "p_value": strict_p,
        },
    ]
)

stats.to_csv(
    STATS_OUT,
    index=False
)

print()
print("Saved:")
print(DATA_OUT)
print(STATS_OUT)
print(FIG_OUT)
