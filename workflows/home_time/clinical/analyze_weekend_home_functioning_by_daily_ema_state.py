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

WEEKEND_FILE = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "temporal_functioning"
    / "weekend_home_following_week_functioning.csv"
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

DATA_OUT = (
    OUT
    / "weekend_home_following_week_functioning_daily_ema_states.csv"
)

STATS_OUT = (
    OUT
    / "weekend_home_functioning_daily_ema_state_stats.csv"
)

FIG_OUT = (
    OUT
    / "weekend_home_functioning_daily_ema_states.png"
)

# ============================================================
# LOAD WEEKEND -> FOLLOWING WEEK DATA
# ============================================================

week = pd.read_csv(
    WEEKEND_FILE,
    low_memory=False
)

week["weekend_start"] = pd.to_datetime(
    week["weekend_start"],
    errors="coerce"
).dt.normalize()

week["weekend_home_pct"] = pd.to_numeric(
    week["weekend_home_pct"],
    errors="coerce"
)

week["following_week_functioning"] = pd.to_numeric(
    week["following_week_functioning"],
    errors="coerce"
)

# ============================================================
# LOAD EMA + MAP SOURCE IDS -> CORTEX IDS
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

# reject ambiguous mappings
nmap = (
    mapping.groupby("source_participant_id")
    ["cortex_participant_id"]
    .nunique()
)

ambiguous = nmap[nmap > 1]

if len(ambiguous):
    raise RuntimeError(
        f"{len(ambiguous)} ambiguous source->Cortex mappings found."
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

for c in [
    "anxiety",
    "mood",
    "daily_function",
]:
    ema[c] = pd.to_numeric(
        ema[c],
        errors="coerce"
    )

# one row per participant-day
ema_daily = (
    ema.groupby(
        ["participant_id", "date"],
        as_index=False
    )
    .agg(
        anxiety=("anxiety", "mean"),
        mood=("mood", "mean"),
        daily_function=("daily_function", "mean"),
    )
)

# ============================================================
# PRECEDING-WEEK EMA FOR EACH WEEKEND
#
# Weekend starts Saturday.
# Preceding EMA window = prior Monday-Friday:
#
# Saturday -5d = Monday
# Saturday -1d = Friday
# ============================================================

rows = []

for r in week.itertuples():

    pid = r.participant_id
    weekend_start = r.weekend_start

    if pd.isna(weekend_start):
        continue

    start = weekend_start - pd.Timedelta(days=5)
    end = weekend_start - pd.Timedelta(days=1)

    d = ema_daily[
        (ema_daily["participant_id"] == pid)
        &
        (ema_daily["date"] >= start)
        &
        (ema_daily["date"] <= end)
    ]

    anxiety_vals = d["anxiety"].dropna()
    mood_vals = d["mood"].dropna()

    rows.append(
        {
            "participant_id": pid,
            "weekend_start": weekend_start,

            "preceding_week_anxiety": (
                anxiety_vals.mean()
                if len(anxiety_vals)
                else np.nan
            ),

            "preceding_week_mood": (
                mood_vals.mean()
                if len(mood_vals)
                else np.nan
            ),

            "n_preceding_anxiety_days": len(anxiety_vals),
            "n_preceding_mood_days": len(mood_vals),
        }
    )

preceding = pd.DataFrame(rows)

week = week.merge(
    preceding,
    on=[
        "participant_id",
        "weekend_start",
    ],
    how="left"
)

# ============================================================
# PERSONAL EMA BASELINES
#
# Use all mapped daily EMA observations for each participant.
# ============================================================

personal = (
    ema_daily.groupby("participant_id")
    .agg(
        personal_median_anxiety=("anxiety", "median"),
        personal_median_mood=("mood", "median"),
        personal_mean_anxiety=("anxiety", "mean"),
        personal_mean_mood=("mood", "mean"),
    )
    .reset_index()
)

week = week.merge(
    personal,
    on="participant_id",
    how="left"
)

# ============================================================
# PERSONALIZED STATES
# ============================================================

week["high_anxiety_state"] = np.where(
    week["preceding_week_anxiety"].notna()
    &
    week["personal_median_anxiety"].notna(),
    (
        week["preceding_week_anxiety"]
        >
        week["personal_median_anxiety"]
    ).astype(int),
    np.nan
)

week["low_mood_state"] = np.where(
    week["preceding_week_mood"].notna()
    &
    week["personal_median_mood"].notna(),
    (
        week["preceding_week_mood"]
        <
        week["personal_median_mood"]
    ).astype(int),
    np.nan
)

week["weekend_home_10"] = (
    week["weekend_home_pct"] / 10.0
)

week.to_csv(
    DATA_OUT,
    index=False
)

# ============================================================
# MODEL HELPER
# ============================================================

def run_interaction(data, state_col, label):

    x = data[
        [
            "participant_id",
            "weekend_home_10",
            "following_week_functioning",
            "study_time_weeks",
            state_col,
        ]
    ].dropna().copy()

    x[state_col] = x[state_col].astype(int)

    # require repeated observations
    counts = (
        x.groupby("participant_id")
        .size()
    )

    ids = counts[counts >= 2].index

    x = x[
        x["participant_id"].isin(ids)
    ].copy()

    if (
        len(x) < 20
        or x["participant_id"].nunique() < 5
        or x[state_col].nunique() < 2
    ):
        return None, x

    formula = (
        "following_week_functioning ~ "
        f"weekend_home_10 * {state_col} + "
        "study_time_weeks"
    )

    model = smf.gee(
        formula=formula,
        groups="participant_id",
        data=x,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()

    return model, x


# ============================================================
# HIGH ANXIETY
# ============================================================

anx_model, anx = run_interaction(
    week,
    "high_anxiety_state",
    "High anxiety"
)

# ============================================================
# LOW MOOD
# ============================================================

mood_model, mood = run_interaction(
    week,
    "low_mood_state",
    "Low mood"
)

# ============================================================
# RESULTS TABLE
# ============================================================

stats = []

for state_name, state_col, model, x in [
    (
        "High anxiety",
        "high_anxiety_state",
        anx_model,
        anx,
    ),
    (
        "Low mood",
        "low_mood_state",
        mood_model,
        mood,
    ),
]:

    if model is None:
        continue

    interaction_term = (
        f"weekend_home_10:{state_col}"
    )

    stats.append(
        {
            "state": state_name,
            "n_windows": len(x),
            "n_participants": x["participant_id"].nunique(),

            "baseline_home_beta": float(
                model.params["weekend_home_10"]
            ),

            "baseline_home_p": float(
                model.pvalues["weekend_home_10"]
            ),

            "interaction_beta": float(
                model.params[interaction_term]
            ),

            "interaction_p": float(
                model.pvalues[interaction_term]
            ),
        }
    )

stats = pd.DataFrame(stats)

stats.to_csv(
    STATS_OUT,
    index=False
)

print()
print("=" * 100)
print("DAILY EMA STATE × WEEKEND HOME STAY")
print("=" * 100)

print()
print(
    "Total weekend windows:",
    len(week)
)

print(
    "Participants:",
    week["participant_id"].nunique()
)

print()
print(
    "High-anxiety usable windows:",
    len(anx)
)

print(
    "High-anxiety participants:",
    anx["participant_id"].nunique()
)

print()
print(
    "Low-mood usable windows:",
    len(mood)
)

print(
    "Low-mood participants:",
    mood["participant_id"].nunique()
)

print()
print(stats.to_string(
    index=False,
    float_format=lambda z: f"{z:.4f}"
))

# ============================================================
# PREDICTED RELATIONSHIPS
#
# Plot weekend home stay continuously, separately for
# symptom-state vs non-symptom-state.
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.8)
)

for ax, state_name, state_col, model, x in [
    (
        axes[0],
        "Anxiety",
        "high_anxiety_state",
        anx_model,
        anx,
    ),
    (
        axes[1],
        "Mood",
        "low_mood_state",
        mood_model,
        mood,
    ),
]:

    if model is None:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        continue

    xmin = np.percentile(
        x["weekend_home_10"] * 10,
        5
    )

    xmax = np.percentile(
        x["weekend_home_10"] * 10,
        95
    )

    grid = np.linspace(
        xmin,
        xmax,
        100
    )

    for state_value, label in [
        (0, "Typical/lower symptom state"),
        (1, "Elevated symptom state"),
    ]:

        pred = pd.DataFrame(
            {
                "weekend_home_10": grid / 10,
                state_col: state_value,
                "study_time_weeks":
                    x["study_time_weeks"].median(),
            }
        )

        y = model.predict(pred)

        ax.plot(
            grid,
            y,
            linewidth=2.2,
            label=label,
        )

    interaction_term = (
        f"weekend_home_10:{state_col}"
    )

    pint = float(
        model.pvalues[interaction_term]
    )

    ax.set_xlabel(
        "Weekend home stay (%)"
    )

    ax.set_ylabel(
        "Predicted following-week functioning"
    )

    ax.set_title(
        f"{state_name} state × weekend home stay\n"
        f"interaction p = {pint:.3f}"
    )

    ax.legend(
        frameon=False,
        fontsize=8
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Does weekend home stay become more informative during adverse daily EMA states?",
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

print()
print("Saved:")
print(DATA_OUT)
print(STATS_OUT)
print(FIG_OUT)
