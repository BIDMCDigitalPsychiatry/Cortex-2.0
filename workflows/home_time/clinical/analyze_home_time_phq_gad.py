#!/usr/bin/env python3

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

INPUT_PATH = BASE / "gps_clinical_weekly_features_with_home.parquet"

OUT_DIR = BASE / "home_time" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE / "home_time" / "clinical_comparisons"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
})


# ============================================================
# HELPERS
# ============================================================

def normalize(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_column(df, exact_candidates, contains_candidates):
    """
    Prefer exact normalized matches.
    Fall back to columns containing requested strings.
    """

    normalized = {
        normalize(c): c
        for c in df.columns
    }

    for candidate in exact_candidates:
        key = normalize(candidate)

        if key in normalized:
            return normalized[key]

    matches = []

    for col in df.columns:
        n = normalize(col)

        if any(
            normalize(token) in n
            for token in contains_candidates
        ):
            matches.append(col)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(
            "\nPossible matches for",
            exact_candidates,
            ":"
        )

        for m in matches:
            print("  ", m)

    return None



def fit_repeated_model(df, outcome, group_col):
    """
    GEE accounting for repeated clinical windows within participant.

    Model:
        home time ~ symptom group

    Correlation:
        exchangeable within participant
    """

    model_df = df[
        ["participant_id", outcome, group_col]
    ].dropna().copy()

    model_df[group_col] = (
        model_df[group_col]
        .astype(int)
    )

    model = smf.gee(
        formula=f"{outcome} ~ {group_col}",
        groups="participant_id",
        data=model_df,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )

    return model.fit()



def summarize_groups(df, score_col, label):
    temp = df[
        [
            "participant_id",
            "time_at_home_pct",
            score_col,
        ]
    ].dropna().copy()

    temp["elevated"] = (
        temp[score_col] >= 10
    )

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    print(
        f"Score column: {score_col}"
    )

    for elevated_value, group_name in [
        (False, "<10"),
        (True, "≥10"),
    ]:

        g = temp[
            temp["elevated"] == elevated_value
        ]

        vals = g[
            "time_at_home_pct"
        ]

        print()
        print(
            f"{label} {group_name}"
        )

        print(
            f"  Windows:      {len(g):,}"
        )

        print(
            "  Participants:",
            g["participant_id"].nunique(),
        )

        print(
            f"  Mean home time:   "
            f"{vals.mean():.2f}%"
        )

        print(
            f"  Median home time: "
            f"{vals.median():.2f}%"
        )

        print(
            f"  SD:               "
            f"{vals.std():.2f}"
        )

        print(
            f"  Q1–Q3:            "
            f"{vals.quantile(.25):.2f}% – "
            f"{vals.quantile(.75):.2f}%"
        )

    return temp


def make_boxplot(
    df,
    score_col,
    symptom_name,
    output_stem,
):

    temp = df[
        [
            "participant_id",
            "time_at_home_pct",
            score_col,
        ]
    ].dropna().copy()

    low = (
        temp.loc[
            temp[score_col] < 10,
            "time_at_home_pct",
        ]
        .to_numpy()
    )

    high = (
        temp.loc[
            temp[score_col] >= 10,
            "time_at_home_pct",
        ]
        .to_numpy()
    )

    low_participants = (
        temp.loc[
            temp[score_col] < 10,
            "participant_id",
        ]
        .nunique()
    )

    high_participants = (
        temp.loc[
            temp[score_col] >= 10,
            "participant_id",
        ]
        .nunique()
    )

    fig, ax = plt.subplots(
        figsize=(7.4, 5.8)
    )

    ax.boxplot(
        [low, high],
        tick_labels=[
            f"<10\n"
            f"{len(low):,} windows\n"
            f"{low_participants} participants",

            f"≥10\n"
            f"{len(high):,} windows\n"
            f"{high_participants} participants",
        ],
        widths=0.52,
        showfliers=False,
    )

    # participant-window observations as lightly jittered points
    rng = np.random.default_rng(42)

    for x, vals in [
        (1, low),
        (2, high),
    ]:

        jitter = rng.normal(
            x,
            0.055,
            size=len(vals),
        )

        ax.scatter(
            jitter,
            vals,
            s=10,
            alpha=0.12,
        )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Weekly home time (%)"
    )

    ax.set_xlabel(
        f"{symptom_name} score"
    )

    ax.set_title(
        f"Home time by {symptom_name} symptom level",
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Each point is one clinical window; "
            "participants may contribute multiple windows."
        ),
        ha="center",
        fontsize=9.5,
    )

    fig.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    png = (
        OUT_DIR /
        f"{output_stem}.png"
    )

    pdf = (
        OUT_DIR /
        f"{output_stem}.pdf"
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

    return png, pdf


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    INPUT_PATH
)

print(
    "Loaded:",
    INPUT_PATH
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# IDENTIFY REQUIRED COLUMNS
# ============================================================

participant_col = find_column(
    df,
    exact_candidates=[
        "participant_id",
        "participant",
    ],
    contains_candidates=[
        "participantid",
    ],
)

home_col = find_column(
    df,
    exact_candidates=[
        "time_at_home_pct",
        "home_time_pct",
        "weekly_home_time_pct",
    ],
    contains_candidates=[
        "timeathomepct",
        "hometimepct",
    ],
)

phq_col = find_column(
    df,
    exact_candidates=[
        "depression_score",
        "phq9",
        "phq_9",
        "phq9_total",
        "phq_9_total",
        "phq_total",
    ],
    contains_candidates=[
        "phq9",
    ],
)

gad_col = find_column(
    df,
    exact_candidates=[
        "anxiety_score",
        "gad7",
        "gad_7",
        "gad7_total",
        "gad_7_total",
        "gad_total",
    ],
    contains_candidates=[
        "gad7",
    ],
)


required = {
    "participant": participant_col,
    "home time": home_col,
    "PHQ-9": phq_col,
    "GAD-7": gad_col,
}

missing = [
    key
    for key, value
    in required.items()
    if value is None
]

if missing:
    print()
    print(
        "Could not uniquely identify:",
        ", ".join(missing),
    )

    print()
    print("Available columns:")

    for c in df.columns:
        print(" ", c)

    raise SystemExit(1)


print()
print("Using columns:")
print(
    " participant:",
    participant_col,
)
print(
    " home time:  ",
    home_col,
)
print(
    " PHQ-9:      ",
    phq_col,
)
print(
    " GAD-7:      ",
    gad_col,
)


# Standardize names
df = df.rename(
    columns={
        participant_col:
            "participant_id",

        home_col:
            "time_at_home_pct",

        phq_col:
            "phq9",

        gad_col:
            "gad7",
    }
)


# Numeric conversion
for col in [
    "time_at_home_pct",
    "phq9",
    "gad7",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# Valid home time only
df = df[
    df["time_at_home_pct"]
    .between(
        0,
        100,
        inclusive="both",
    )
].copy()


# Valid questionnaire ranges
df.loc[
    ~df["phq9"].between(
        0,
        27,
        inclusive="both",
    ),
    "phq9",
] = np.nan

df.loc[
    ~df["gad7"].between(
        0,
        21,
        inclusive="both",
    ),
    "gad7",
] = np.nan


# ============================================================
# GROUP SUMMARIES
# ============================================================

phq_df = summarize_groups(
    df,
    "phq9",
    "PHQ-9",
)

gad_df = summarize_groups(
    df,
    "gad7",
    "GAD-7",
)


# ============================================================
# MIXED-EFFECTS MODELS
# ============================================================

phq_model_df = phq_df.copy()

phq_model_df[
    "phq9_ge10"
] = (
    phq_model_df["phq9"] >= 10
).astype(int)


gad_model_df = gad_df.copy()

gad_model_df[
    "gad7_ge10"
] = (
    gad_model_df["gad7"] >= 10
).astype(int)


print()
print("=" * 80)
print(
    "GEE — PHQ-9"
)
print("=" * 80)

phq_result = fit_repeated_model(
    phq_model_df,
    outcome="time_at_home_pct",
    group_col="phq9_ge10",
)

print(
    phq_result.summary()
)


print()
print("=" * 80)
print(
    "GEE — GAD-7"
)
print("=" * 80)

gad_result = fit_repeated_model(
    gad_model_df,
    outcome="time_at_home_pct",
    group_col="gad7_ge10",
)

print(
    gad_result.summary()
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results = []

for name, result, variable in [
    (
        "PHQ-9 ≥10",
        phq_result,
        "phq9_ge10",
    ),
    (
        "GAD-7 ≥10",
        gad_result,
        "gad7_ge10",
    ),
]:

    beta = (
        result.params[
            variable
        ]
    )

    se = (
        result.bse[
            variable
        ]
    )

    p = (
        result.pvalues[
            variable
        ]
    )

    ci = (
        result.conf_int()
        .loc[
            variable
        ]
    )

    results.append({
        "comparison":
            name,

        "beta_home_time_percentage_points":
            beta,

        "standard_error":
            se,

        "ci95_low":
            ci.iloc[0],

        "ci95_high":
            ci.iloc[1],

        "p_value":
            p,
    })


results_df = pd.DataFrame(
    results
)

results_path = (
    RESULTS_DIR /
    "home_time_phq9_gad7_gee.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)


print()
print("=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# FIGURES
# ============================================================

phq_png, phq_pdf = make_boxplot(
    df,
    score_col="phq9",
    symptom_name="PHQ-9",
    output_stem=(
        "weekly_home_time_by_phq9_ge10"
    ),
)


gad_png, gad_pdf = make_boxplot(
    df,
    score_col="gad7",
    symptom_name="GAD-7",
    output_stem=(
        "weekly_home_time_by_gad7_ge10"
    ),
)


# ============================================================
# COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5.6),
)


for ax, score_col, label in [
    (
        axes[0],
        "phq9",
        "PHQ-9",
    ),
    (
        axes[1],
        "gad7",
        "GAD-7",
    ),
]:

    temp = df[
        [
            "participant_id",
            "time_at_home_pct",
            score_col,
        ]
    ].dropna()

    low = temp.loc[
        temp[score_col] < 10,
        "time_at_home_pct",
    ]

    high = temp.loc[
        temp[score_col] >= 10,
        "time_at_home_pct",
    ]

    low_n = temp.loc[
        temp[score_col] < 10,
        "participant_id",
    ].nunique()

    high_n = temp.loc[
        temp[score_col] >= 10,
        "participant_id",
    ].nunique()

    ax.boxplot(
        [
            low.to_numpy(),
            high.to_numpy(),
        ],
        tick_labels=[
            (
                f"<10\n"
                f"{len(low):,} windows\n"
                f"{low_n} participants"
            ),
            (
                f"≥10\n"
                f"{len(high):,} windows\n"
                f"{high_n} participants"
            ),
        ],
        widths=0.48,
        showfliers=False,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_ylabel(
        "Weekly home time (%)"
    )

    ax.set_title(
        label,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )


fig.suptitle(
    "Home time by symptom burden",
    fontsize=18,
    fontweight="bold",
)

fig.text(
    0.5,
    0.015,
    (
        "Clinical windows are classified using the PHQ-9 or "
        "GAD-7 score measured in that same window."
    ),
    ha="center",
    fontsize=10,
)

fig.tight_layout(
    rect=[
        0,
        0.05,
        1,
        0.95,
    ]
)


combined_png = (
    OUT_DIR /
    "weekly_home_time_phq9_gad7_comparison.png"
)

combined_pdf = (
    OUT_DIR /
    "weekly_home_time_phq9_gad7_comparison.pdf"
)

fig.savefig(
    combined_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    combined_pdf,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved:")
print(phq_png)
print(phq_pdf)
print(gad_png)
print(gad_pdf)
print(combined_png)
print(combined_pdf)
print(results_path)
