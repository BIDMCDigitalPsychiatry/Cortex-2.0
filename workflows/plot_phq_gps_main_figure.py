from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/"
    "digital_clinic/feature_reliability/gps_clinical"
)

DATA = ROOT / "gps_clinical_weekly_features_with_home.parquet"

SUMMARY = (
    ROOT
    / "full_cohort_phq_analysis"
    / "phq_four_group_gps_summary.csv"
)

OUTDIR = ROOT / "full_cohort_phq_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / "figure_phq_gps_behavioral_constriction.png"
PDF = OUTDIR / "figure_phq_gps_behavioral_constriction.pdf"


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ============================================================
# DATA
# ============================================================

df = pd.read_parquet(DATA).copy()
summary = pd.read_csv(SUMMARY)

df["depression_score"] = pd.to_numeric(
    df["depression_score"],
    errors="coerce",
)

df["time_at_home_pct"] = pd.to_numeric(
    df["time_at_home_pct"],
    errors="coerce",
)

df["home_hours_per_day"] = (
    df["time_at_home_pct"] / 100 * 24
)

df["movement_km_per_day"] = (
    pd.to_numeric(
        df["displacement_distance_mean_m"],
        errors="coerce",
    )
    / 1000
)


bins = [-0.001, 4, 9, 14, 27]

labels = [
    "Minimal\n0–4",
    "Mild\n5–9",
    "Moderate\n10–14",
    "Mod. severe /\nsevere\n15–27",
]

summary_labels = [
    "Minimal (0–4)",
    "Mild (5–9)",
    "Moderate (10–14)",
    "Moderately severe / severe (15–27)",
]

df["phq_group"] = pd.cut(
    df["depression_score"],
    bins=bins,
    labels=summary_labels,
    ordered=True,
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5.4),
)

rng = np.random.default_rng(42)

panel_specs = [
    {
        "ax": axes[0],
        "feature_summary": "Movement distance (km/day)",
        "value_col": "movement_km_per_day",
        "title": "Daily movement falls with depressive symptom burden",
        "ylabel": "Daily movement distance (km/day)",
        "annotation": "−3.1 km/day per +5 PHQ-9 points",
        "ylim": (0, 55),
    },
    {
        "ax": axes[1],
        "feature_summary": "Time at home (h/day)",
        "value_col": "home_hours_per_day",
        "title": "Time at home rises at higher symptom severity",
        "ylabel": "Equivalent time at home (h/day)",
        "annotation": "+0.5 h/day per +5 PHQ-9 points",
        "ylim": (10, 22),
    },
]


for panel_index, spec in enumerate(panel_specs):

    ax = spec["ax"]

    sub_summary = summary[
        summary["feature"].eq(
            spec["feature_summary"]
        )
    ].copy()

    sub_summary["phq_group"] = pd.Categorical(
        sub_summary["phq_group"],
        categories=summary_labels,
        ordered=True,
    )

    sub_summary = sub_summary.sort_values(
        "phq_group"
    )

    x_positions = np.arange(4)

    # --------------------------------------------------------
    # faint raw weekly observations
    # --------------------------------------------------------

    for i, group in enumerate(summary_labels):

        vals = df.loc[
            df["phq_group"].eq(group),
            spec["value_col"],
        ].dropna()

        if len(vals) == 0:
            continue

        # limit displayed points for readability only
        if len(vals) > 220:
            vals = vals.sample(
                220,
                random_state=42,
            )

        jitter = rng.normal(
            loc=0,
            scale=0.07,
            size=len(vals),
        )

        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=10,
            alpha=0.12,
            edgecolors="none",
            zorder=1,
        )

    # --------------------------------------------------------
    # mean + cluster-bootstrap CI
    # --------------------------------------------------------

    means = sub_summary["mean"].to_numpy()

    lower = (
        means
        - sub_summary[
            "cluster_boot_ci_low"
        ].to_numpy()
    )

    upper = (
        sub_summary[
            "cluster_boot_ci_high"
        ].to_numpy()
        - means
    )

    bars = ax.bar(
        x_positions,
        means,
        width=0.62,
        alpha=0.72,
        edgecolor="black",
        linewidth=0.7,
        zorder=2,
    )

    ax.errorbar(
        x_positions,
        means,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )

    # --------------------------------------------------------
    # mean values above bars
    # --------------------------------------------------------

    for i, mean in enumerate(means):

        ax.text(
            i,
            mean + upper[i] + (
                0.8
                if panel_index == 0
                else 0.25
            ),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # n participants
    # --------------------------------------------------------

    ns = (
        sub_summary[
            "n_participants"
        ]
        .astype(int)
        .to_numpy()
    )

    tick_labels = [
        f"{label}\n(n={n})"
        for label, n in zip(
            labels,
            ns,
        )
    ]

    ax.set_xticks(
        x_positions,
        tick_labels,
    )

    ax.set_title(
        spec["title"],
        pad=14,
        fontweight="semibold",
    )

    ax.set_ylabel(
        spec["ylabel"]
    )

    ax.set_xlabel(
        "PHQ-9 severity"
    )

    ax.set_ylim(
        *spec["ylim"]
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    # regression annotation
    ax.text(
        0.03,
        0.96,
        spec["annotation"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="semibold",
    )

    # panel label
    ax.text(
        -0.12,
        1.05,
        chr(ord("A") + panel_index),
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


# ============================================================
# MAIN TITLE + FOOTNOTE
# ============================================================

fig.suptitle(
    "Higher depressive symptom burden is associated with behavioral constriction",
    fontsize=16,
    fontweight="semibold",
    y=0.98,
)

fig.text(
    0.5,
    0.015,
    (
        "Bars show weekly mean values; error bars are 95% participant-cluster bootstrap CIs. "
        "Points show a random subset of weekly observations for visual context. "
        "GPS features were derived from native irregular timestamps without fixed resampling."
    ),
    ha="center",
    va="bottom",
    fontsize=8.5,
)

plt.tight_layout(
    rect=[0.02, 0.07, 1, 0.93]
)

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

print("=" * 100)
print("FIGURE SAVED")
print("=" * 100)
print(PNG)
print(PDF)
