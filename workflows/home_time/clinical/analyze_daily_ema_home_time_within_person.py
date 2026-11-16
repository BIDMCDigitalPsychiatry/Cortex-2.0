from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

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

OUT = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_ema_alignment"
)

OUT.mkdir(parents=True, exist_ok=True)

DATA_OUT = OUT / "daily_ema_home_time_master.csv"
RESULTS_OUT = OUT / "daily_ema_home_time_within_between_results.csv"
FIG_OUT = OUT / "daily_ema_home_time_within_person_forest.png"


# ============================================================
# LOAD
# ============================================================

home = pd.read_csv(
    HOME_FILE,
    low_memory=False
)

ema = pd.read_csv(
    EMA_FILE,
    low_memory=False
)

# ============================================================
# MAP EMA SOURCE IDS -> CORTEX PARTICIPANT IDS
# ============================================================

MAPPING_FILE = (
    ROOT
    / "clinical/daily_ema"
    / "daily_ema_gps_participants.csv"
)

mapping = pd.read_csv(
    MAPPING_FILE,
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

# Guard against ambiguous mappings
ambiguous = (
    mapping.groupby("source_participant_id")
    ["cortex_participant_id"]
    .nunique()
)

ambiguous = ambiguous[
    ambiguous > 1
]

if len(ambiguous):
    raise RuntimeError(
        f"Ambiguous source-to-Cortex mappings: {len(ambiguous)}"
    )

mapping = mapping.drop_duplicates(
    subset=["source_participant_id"]
)

ema["source_participant_id"] = (
    ema["participant_id"]
    .astype(str)
)

ema = ema.merge(
    mapping,
    on="source_participant_id",
    how="left",
)

print()
print(
    "EMA source participants:",
    ema["source_participant_id"].nunique()
)

print(
    "EMA participants mapped to Cortex:",
    ema["cortex_participant_id"].nunique()
)

print(
    "EMA rows with Cortex mapping:",
    int(ema["cortex_participant_id"].notna().sum())
)

# Replace source ID with Cortex ID for downstream matching
ema["participant_id"] = (
    ema["cortex_participant_id"]
)

ema = ema.dropna(
    subset=["participant_id"]
).copy()

print()
print("HOME COLUMNS:")
print(home.columns.tolist())

print()
print("EMA COLUMNS:")
print(ema.columns.tolist())


# ============================================================
# DATES
# ============================================================

home["date"] = pd.to_datetime(
    home["date"],
    errors="coerce"
).dt.normalize()

ema["date"] = pd.to_datetime(
    ema["date"],
    errors="coerce"
).dt.normalize()


# ============================================================
# HOME TIME
# ============================================================

if {
    "raw_home_duration_ms",
    "supported_duration_ms"
}.issubset(home.columns):

    home["supported_hours"] = (
        pd.to_numeric(
            home["supported_duration_ms"],
            errors="coerce"
        )
        / 3_600_000
    )

    home["home_time_pct"] = (
        100
        * pd.to_numeric(
            home["raw_home_duration_ms"],
            errors="coerce"
        )
        / pd.to_numeric(
            home["supported_duration_ms"],
            errors="coerce"
        )
    )

elif "home_time_pct" in home.columns:

    home["home_time_pct"] = pd.to_numeric(
        home["home_time_pct"],
        errors="coerce"
    )

    if "supported_duration_ms" in home.columns:
        home["supported_hours"] = (
            pd.to_numeric(
                home["supported_duration_ms"],
                errors="coerce"
            )
            / 3_600_000
        )
    else:
        home["supported_hours"] = np.nan

else:
    raise RuntimeError(
        "Could not identify daily home-time columns."
    )


# ============================================================
# CURRENT PRIMARY QUALITY RULE
#
# 6 h/day is kept as the current operational primary analysis.
# We will test its sensitivity separately rather than pretending
# that it is a universally validated threshold.
# ============================================================

QUALITY_HOURS = 6.0

if home["supported_hours"].notna().any():
    home = home[
        home["supported_hours"] >= QUALITY_HOURS
    ].copy()


# ============================================================
# ONE ROW PER PARTICIPANT-DAY
# ============================================================

home_daily = (
    home[
        [
            "participant_id",
            "date",
            "home_time_pct",
            "supported_hours",
        ]
    ]
    .dropna(
        subset=[
            "participant_id",
            "date",
            "home_time_pct",
        ]
    )
    .groupby(
        ["participant_id", "date"],
        as_index=False
    )
    .agg(
        home_time_pct=("home_time_pct", "mean"),
        supported_hours=("supported_hours", "mean"),
    )
)


# EMA can have duplicates on a date.
# Use daily mean rather than arbitrarily choosing one response.

OUTCOMES = [
    "anxiety",
    "mood",
    "daily_function",
]

for col in OUTCOMES:
    ema[col] = pd.to_numeric(
        ema[col],
        errors="coerce"
    )

ema_daily = (
    ema[
        [
            "participant_id",
            "date",
            *OUTCOMES,
        ]
    ]
    .groupby(
        ["participant_id", "date"],
        as_index=False
    )
    .agg(
        {
            "anxiety": "mean",
            "mood": "mean",
            "daily_function": "mean",
        }
    )
)


# ============================================================
# MERGE
# ============================================================

df = home_daily.merge(
    ema_daily,
    on=[
        "participant_id",
        "date",
    ],
    how="inner",
)

print()
print("=" * 100)
print("DAILY HOME-TIME / EMA ALIGNMENT")
print("=" * 100)
print("Participant-days:", len(df))
print("Participants:", df["participant_id"].nunique())

for outcome in OUTCOMES:
    print(
        f"{outcome:20s}",
        "observations=",
        int(df[outcome].notna().sum()),
        "participants=",
        int(
            df.loc[
                df[outcome].notna(),
                "participant_id"
            ].nunique()
        ),
    )


# ============================================================
# PERSONAL HOME-TIME BASELINE
# ============================================================

df["participant_mean_home_pct"] = (
    df.groupby("participant_id")["home_time_pct"]
    .transform("mean")
)

df["home_within_pct"] = (
    df["home_time_pct"]
    - df["participant_mean_home_pct"]
)

# Interpret coefficients per 10 percentage points.
df["home_within_10"] = (
    df["home_within_pct"] / 10.0
)

df["home_between_10"] = (
    df["participant_mean_home_pct"] / 10.0
)


# ============================================================
# STUDY TIME
# ============================================================

first_date = (
    df.groupby("participant_id")["date"]
    .transform("min")
)

df["study_day"] = (
    df["date"] - first_date
).dt.days

df["study_time_30d"] = (
    df["study_day"] / 30
)


# ============================================================
# SAVE MASTER TABLE
# ============================================================

df.to_csv(
    DATA_OUT,
    index=False
)


# ============================================================
# MODELS
# ============================================================

rows = []

LABELS = {
    "anxiety": "Daily anxiety",
    "mood": "Daily mood",
    "daily_function": "Daily functioning",
}

for outcome in OUTCOMES:

    x = df[
        [
            "participant_id",
            "date",
            "home_time_pct",
            "home_within_10",
            "home_between_10",
            "study_time_30d",
            outcome,
        ]
    ].dropna().copy()

    # Need repeated observations for a within-person estimate.
    counts = (
        x.groupby("participant_id")
        .size()
    )

    valid_ids = counts[
        counts >= 2
    ].index

    x = x[
        x["participant_id"].isin(valid_ids)
    ].copy()

    # Also require actual within-person variation in home time.
    variation = (
        x.groupby("participant_id")["home_time_pct"]
        .std()
    )

    varying_ids = variation[
        variation > 0
    ].index

    x = x[
        x["participant_id"].isin(varying_ids)
    ].copy()

    if x.empty:
        continue

    # --------------------------------------------------------
    # Standardized outcome
    #
    # This makes anxiety, mood and functioning directly
    # comparable on one forest plot.
    # --------------------------------------------------------

    sd = x[outcome].std()

    if not np.isfinite(sd) or sd == 0:
        continue

    x["outcome_z"] = (
        x[outcome] - x[outcome].mean()
    ) / sd

    model = smf.gee(
        formula=(
            "outcome_z ~ "
            "home_within_10 + "
            "home_between_10 + "
            "study_time_30d"
        ),
        groups="participant_id",
        data=x,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()

    for term, effect_type in [
        ("home_within_10", "within"),
        ("home_between_10", "between"),
    ]:

        beta = float(model.params[term])
        se = float(model.bse[term])
        p = float(model.pvalues[term])

        rows.append(
            {
                "outcome": outcome,
                "outcome_label": LABELS[outcome],
                "effect_type": effect_type,
                "beta_sd_per_10pp_home": beta,
                "ci_low": beta - 1.96 * se,
                "ci_high": beta + 1.96 * se,
                "p_value": p,
                "n_observations": len(x),
                "n_participants": x[
                    "participant_id"
                ].nunique(),
                "median_days_per_participant": (
                    x.groupby("participant_id")
                    .size()
                    .median()
                ),
            }
        )


results = pd.DataFrame(rows)

results.to_csv(
    RESULTS_OUT,
    index=False
)


# ============================================================
# PRINT WITHIN-PERSON RESULTS
# ============================================================

within = results[
    results["effect_type"] == "within"
].copy()

print()
print("=" * 105)
print("WITHIN-PERSON HOME TIME -> DAILY ACTIVE MEASURES")
print("=" * 105)

print(
    within[
        [
            "outcome_label",
            "beta_sd_per_10pp_home",
            "ci_low",
            "ci_high",
            "p_value",
            "n_observations",
            "n_participants",
            "median_days_per_participant",
        ]
    ].to_string(
        index=False,
        float_format=lambda z: f"{z:.4f}",
    )
)

print()
print(
    "Beta = standardized change in the daily clinical measure "
    "when that person spends 10 percentage points more time "
    "at home than their own usual level."
)


# ============================================================
# FOREST PLOT
# ============================================================

if not within.empty:

    within = within.reset_index(drop=True)

    y = np.arange(len(within))

    beta = within[
        "beta_sd_per_10pp_home"
    ].to_numpy()

    low = within["ci_low"].to_numpy()
    high = within["ci_high"].to_numpy()

    xerr = np.vstack(
        [
            beta - low,
            high - beta,
        ]
    )

    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(
        figsize=(8.0, 4.7)
    )

    ax.errorbar(
        beta,
        y,
        xerr=xerr,
        fmt="o",
        markersize=7,
        linewidth=1.8,
        capsize=4,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        color="0.5",
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        within["outcome_label"]
    )

    ax.set_xlabel(
        "Change in clinical measure (SD)\n"
        "per +10 percentage points home time above personal usual level"
    )

    ax.set_title(
        "Within-person association between daily home time "
        "and active clinical measures"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, row in within.iterrows():

        ax.text(
            row["ci_high"] + 0.01,
            yi,
            f'p={row["p_value"]:.3f}',
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        FIG_OUT,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


print()
print("Saved:")
print(DATA_OUT)
print(RESULTS_OUT)
print(FIG_OUT)
