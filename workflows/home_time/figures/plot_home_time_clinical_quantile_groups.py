from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)

HOME = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "daily_home_time.csv"
)

PHQGAD = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "paper_replication"
    / "home_time_age_gad7_analysis_dataset.csv"
)

EMA = (
    ROOT
    / "clinical/daily_ema"
    / "daily_ema_all_participants.csv"
)

MAP = (
    ROOT
    / "clinical/daily_ema"
    / "daily_ema_gps_participants.csv"
)

OUT = (
    ROOT
    / "feature_reliability/gps_clinical/home_time"
    / "clinical_quantile_figures"
)

OUT.mkdir(parents=True, exist_ok=True)

PNG = OUT / "home_time_clinical_quantile_groups.png"
CSV = OUT / "home_time_clinical_quantile_participant_data.csv"


# ============================================================
# ROBUST PARTICIPANT-LEVEL HOME TIME
# ============================================================

home = pd.read_csv(HOME, low_memory=False)

home["supported_hours"] = (
    pd.to_numeric(
        home["supported_duration_ms"],
        errors="coerce"
    )
    / 3_600_000
)

home["home_time_pct"] = pd.to_numeric(
    home["time_at_home_pct"],
    errors="coerce"
)

# Current operational daily quality threshold
home = home[
    (home["supported_hours"] >= 6)
    &
    home["home_time_pct"].notna()
].copy()

home_person = (
    home.groupby("participant_id")
    .agg(
        mean_home_time_pct=("home_time_pct", "mean"),
        median_home_time_pct=("home_time_pct", "median"),
        n_home_days=("home_time_pct", "size"),
    )
    .reset_index()
)


# ============================================================
# PHQ-9 / GAD-7 PARTICIPANT MEANS
# ============================================================

clin = pd.read_csv(
    PHQGAD,
    low_memory=False
)

clin["phq9_total"] = pd.to_numeric(
    clin["phq9_total"],
    errors="coerce"
)

clin["gad7_total"] = pd.to_numeric(
    clin["gad7_total"],
    errors="coerce"
)

phqgad_person = (
    clin.groupby("participant_id")
    .agg(
        phq9=("phq9_total", "mean"),
        gad7=("gad7_total", "mean"),
        n_phq=("phq9_total", "count"),
        n_gad=("gad7_total", "count"),
    )
    .reset_index()
)


# ============================================================
# DAILY EMA -> CORTEX IDS
# ============================================================

ema = pd.read_csv(
    EMA,
    low_memory=False
)

mapping = pd.read_csv(
    MAP,
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
    .drop_duplicates(
        subset=["source_participant_id"]
    )
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

for c in [
    "mood",
    "daily_function",
]:
    ema[c] = pd.to_numeric(
        ema[c],
        errors="coerce"
    )

ema_person = (
    ema.groupby("participant_id")
    .agg(
        mood=("mood", "mean"),
        functioning=("daily_function", "mean"),
        n_mood=("mood", "count"),
        n_function=("daily_function", "count"),
    )
    .reset_index()
)


# ============================================================
# MASTER PARTICIPANT TABLE
# ============================================================

df = (
    home_person
    .merge(
        phqgad_person,
        on="participant_id",
        how="left"
    )
    .merge(
        ema_person,
        on="participant_id",
        how="left"
    )
)

df.to_csv(
    CSV,
    index=False
)

print()
print("=" * 100)
print("PARTICIPANT-LEVEL DATA")
print("=" * 100)

print("Home time participants:", df["mean_home_time_pct"].notna().sum())

for c in [
    "phq9",
    "gad7",
    "mood",
    "functioning",
]:
    print(
        f"{c:15s}",
        "n =",
        int(df[c].notna().sum())
    )


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8.5

OUTCOMES = [
    ("phq9", "PHQ-9"),
    ("gad7", "GAD-7"),
    ("mood", "Daily mood"),
    ("functioning", "Daily functioning"),
]

SPLITS = [
    (2, "Median split"),
    (3, "Tertiles"),
    (4, "Quartiles"),
    (5, "Quintiles"),
]

fig, axes = plt.subplots(
    len(OUTCOMES),
    len(SPLITS),
    figsize=(13.5, 10.5)
)


# ============================================================
# PANELS
# ============================================================

for row_i, (outcome, ylabel) in enumerate(OUTCOMES):

    base = df[
        [
            "participant_id",
            "mean_home_time_pct",
            outcome,
        ]
    ].dropna().copy()

    for col_i, (q, title) in enumerate(SPLITS):

        ax = axes[row_i, col_i]

        if len(base) < q * 3:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        x = base.copy()

        # Quantile grouping by participant-level mean home time
        try:
            x["group"] = pd.qcut(
                x["mean_home_time_pct"],
                q=q,
                labels=False,
                duplicates="drop"
            )
        except Exception:
            ax.set_axis_off()
            continue

        ngroups = x["group"].nunique()

        if ngroups != q:
            ax.set_axis_off()
            continue

        summary = (
            x.groupby("group")
            .agg(
                mean=(outcome, "mean"),
                sd=(outcome, "std"),
                n=(outcome, "count"),
            )
            .reset_index()
        )

        summary["se"] = (
            summary["sd"]
            / np.sqrt(summary["n"])
        )

        summary["ci95"] = (
            1.96
            * summary["se"]
        )

        # ----------------------------------------------------
        # Ordered trend test
        # ----------------------------------------------------

        X = sm.add_constant(
            x["group"].astype(float)
        )

        trend = sm.OLS(
            x[outcome],
            X
        ).fit()

        ptrend = float(
            trend.pvalues["group"]
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        if q == 2:
            labels = [
                "Low",
                "High",
            ]

        else:
            prefix = {
                3: "T",
                4: "Q",
                5: "Q",
            }[q]

            labels = [
                f"{prefix}{i+1}"
                for i in range(q)
            ]

        positions = np.arange(q)

        bars = ax.bar(
            positions,
            summary["mean"],
            yerr=summary["ci95"],
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

        # Color progression without manually assigning
        # custom colors; matplotlib defaults handle palette.

        for i, r in summary.iterrows():

            ax.text(
                i,
                r["mean"] + r["ci95"],
                f'{r["mean"]:.2f}',
                ha="center",
                va="bottom",
                fontsize=7
            )

            ax.text(
                i,
                ax.get_ylim()[0],
                f"\nn={int(r['n'])}",
                ha="center",
                va="top",
                fontsize=6.5
            )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            labels
        )

        if row_i == len(OUTCOMES) - 1:
            ax.set_xlabel(
                "Participant mean home time"
            )

        if col_i == 0:
            ax.set_ylabel(
                ylabel
            )

        ax.set_title(
            f"{title}   trend p = {ptrend:.4f}",
            fontsize=8.5
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


fig.suptitle(
    "Clinical measures across increasing levels of participant home time",
    fontsize=14,
    y=0.995
)

fig.tight_layout(
    rect=[0, 0, 1, 0.975]
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("Saved:")
print(PNG)
print(CSV)
