#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

INPUT = BASE / "gps_clinical_weekly_features_with_home.parquet"

OUT_DIR = BASE / "home_time" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
})


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(INPUT)

df = df.rename(columns={
    "depression_score": "phq9",
    "anxiety_score": "gad7",
})

for col in [
    "time_at_home_pct",
    "phq9",
    "gad7",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


df = df[
    df["time_at_home_pct"].between(0, 100)
].copy()


# ============================================================
# WITHIN-PERSON FUNCTION
# ============================================================

def participant_state_summary(df, score_col):

    d = df[
        [
            "participant_id",
            "time_at_home_pct",
            score_col,
        ]
    ].dropna().copy()

    d["symptom_state"] = np.where(
        d[score_col] >= 10,
        "≥10",
        "<10",
    )

    # Median home time for each participant
    # within each symptom state
    summary = (
        d.groupby(
            [
                "participant_id",
                "symptom_state",
            ]
        )["time_at_home_pct"]
        .median()
        .unstack()
    )

    # Keep only people observed in BOTH states
    paired = summary.dropna(
        subset=[
            "<10",
            "≥10",
        ]
    ).copy()

    paired["difference"] = (
        paired["≥10"]
        - paired["<10"]
    )

    return paired


phq = participant_state_summary(
    df,
    "phq9",
)

gad = participant_state_summary(
    df,
    "gad7",
)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_summary(name, paired):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Participants with both symptom states:",
        len(paired),
    )

    print(
        "Median home time <10:",
        f"{paired['<10'].median():.2f}%"
    )

    print(
        "Median home time ≥10:",
        f"{paired['≥10'].median():.2f}%"
    )

    print(
        "Median within-person difference:",
        f"{paired['difference'].median():+.2f} percentage points"
    )

    print(
        "Mean within-person difference:",
        f"{paired['difference'].mean():+.2f} percentage points"
    )

    print(
        "Participants with higher home time in ≥10 state:",
        f"{(paired['difference'] > 0).sum()} / {len(paired)}"
    )


print_summary(
    "PHQ-9",
    phq,
)

print_summary(
    "GAD-7",
    gad,
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 5.8),
    sharey=True,
)


for ax, paired, title in [
    (
        axes[0],
        phq,
        "PHQ-9",
    ),
    (
        axes[1],
        gad,
        "GAD-7",
    ),
]:

    # Individual participant lines
    for _, row in paired.iterrows():

        ax.plot(
            [0, 1],
            [
                row["<10"],
                row["≥10"],
            ],
            alpha=0.20,
            linewidth=0.9,
        )

        ax.scatter(
            [0, 1],
            [
                row["<10"],
                row["≥10"],
            ],
            s=13,
            alpha=0.25,
        )

    # Cohort medians
    med_low = paired["<10"].median()
    med_high = paired["≥10"].median()

    ax.scatter(
        [0, 1],
        [
            med_low,
            med_high,
        ],
        s=100,
        marker="D",
        zorder=10,
        label="Cohort median",
    )

    ax.plot(
        [0, 1],
        [
            med_low,
            med_high,
        ],
        linewidth=3,
        zorder=9,
    )

    ax.set_xticks(
        [0, 1],
    )

    ax.set_xticklabels(
        [
            "<10",
            "≥10",
        ]
    )

    ax.set_xlim(
        -0.35,
        1.35,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.set_xlabel(
        f"{title} score"
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    ax.text(
        0.5,
        0.04,
        f"N = {len(paired)} participants with both states",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )


axes[0].set_ylabel(
    "Median weekly home time (%)"
)


fig.suptitle(
    "Within-person home time by symptom burden",
    fontsize=19,
    fontweight="bold",
)


fig.text(
    0.5,
    0.01,
    (
        "Each line represents one participant observed "
        "during both lower- and higher-symptom clinical windows."
    ),
    ha="center",
    fontsize=10,
)


fig.tight_layout(
    rect=[
        0,
        0.055,
        1,
        0.94,
    ]
)


png = (
    OUT_DIR /
    "home_time_symptoms_within_person.png"
)

pdf = (
    OUT_DIR /
    "home_time_symptoms_within_person.pdf"
)


fig.savefig(
    png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved:")
print(png)
print(pdf)
