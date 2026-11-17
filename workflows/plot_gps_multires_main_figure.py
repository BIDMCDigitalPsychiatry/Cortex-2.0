from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "multires_clinical_benchmark/analysis"
)

TECH_CSV = BASE / "technical_fidelity_summary.csv"
WEEKLY_CSV = BASE / "weekly_clinical_multires_models.csv"

OUT_PNG = BASE / "figure_gps_multires_main.png"
OUT_PDF = BASE / "figure_gps_multires_main.pdf"


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# ORDER / LABELS
# ============================================================

RESOLUTION_ORDER = [
    "native",
    "1min",
    "5min",
    "10min",
    "15min",
    "30min",
    "60min",
    "120min",
    "240min",
]

X = np.arange(len(RESOLUTION_ORDER))

X_LABELS = [
    "Native",
    "1",
    "5",
    "10",
    "15",
    "30",
    "60",
    "120",
    "240",
]

FEATURE_ORDER = [
    "displacement",
    "radius_gyration",
    "entropy",
    "effective_places",
    "dominant_place_fraction",
    "observed_places",
]

FEATURE_LABELS = {
    "displacement": "Displacement",
    "radius_gyration": "Radius of gyration",
    "entropy": "Entropy",
    "effective_places": "Effective places",
    "dominant_place_fraction": "Dominant-place fraction",
    "observed_places": "Observed places",
}

# Matplotlib default qualitative palette
FEATURE_COLORS = {
    "displacement": "#1f77b4",
    "radius_gyration": "#ff7f0e",
    "entropy": "#2ca02c",
    "effective_places": "#d62728",
    "dominant_place_fraction": "#9467bd",
    "observed_places": "#8c564b",
}


# ============================================================
# LOAD DATA
# ============================================================

tech = pd.read_csv(TECH_CSV)
weekly = pd.read_csv(WEEKLY_CSV)

tech["resolution"] = pd.Categorical(
    tech["resolution"],
    categories=RESOLUTION_ORDER,
    ordered=True,
)

weekly["resolution"] = pd.Categorical(
    weekly["resolution"],
    categories=RESOLUTION_ORDER,
    ordered=True,
)


# ============================================================
# HELPERS
# ============================================================

def get_series(df, feature, value_col):
    d = (
        df.loc[df["feature"].eq(feature), ["resolution", value_col]]
        .dropna(subset=["resolution"])
        .drop_duplicates(subset=["resolution"])
        .set_index("resolution")
        .reindex(RESOLUTION_ORDER)
    )

    return d[value_col].to_numpy(dtype=float)


def get_beta_ci(df, outcome, feature):
    d = df.loc[
        df["outcome"].eq(outcome)
        & df["feature"].eq(feature),
        [
            "resolution",
            "within_beta",
            "within_ci_low",
            "within_ci_high",
        ],
    ].copy()

    d = (
        d.dropna(subset=["resolution"])
        .drop_duplicates(subset=["resolution"])
        .set_index("resolution")
        .reindex(RESOLUTION_ORDER)
    )

    beta = d["within_beta"].to_numpy(dtype=float)
    low = d["within_ci_low"].to_numpy(dtype=float)
    high = d["within_ci_high"].to_numpy(dtype=float)

    yerr = np.vstack([
        beta - low,
        high - beta,
    ])

    return beta, yerr


def panel_letter(ax, letter):
    ax.text(
        -0.10,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
        va="top",
    )


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12.5, 15.0),
    sharex=True,
)

ax1, ax2, ax3 = axes


# ============================================================
# PANEL A — TECHNICAL FIDELITY
# ============================================================

for feature in FEATURE_ORDER:

    y = get_series(
        tech,
        feature,
        "spearman",
    )

    ax1.plot(
        X,
        y,
        marker="o",
        linewidth=2.1,
        markersize=6.5,
        color=FEATURE_COLORS[feature],
        label=FEATURE_LABELS[feature],
    )

ax1.set_title(
    "Technical fidelity declines at different rates across GPS features",
    pad=14,
)
ax1.set_ylabel("Spearman correlation\nwith native")
ax1.set_ylim(0, 1.05)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.grid(
    axis="y",
    linewidth=0.8,
    alpha=0.22,
)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
panel_letter(ax1, "A")


# ============================================================
# PANEL B — PHQ-9
# ============================================================

for feature in FEATURE_ORDER:

    beta, yerr = get_beta_ci(
        weekly,
        "depression_score",
        feature,
    )

    ax2.errorbar(
        X,
        beta,
        yerr=yerr,
        fmt="o-",
        linewidth=1.9,
        markersize=5.8,
        capsize=3,
        color=FEATURE_COLORS[feature],
        label=FEATURE_LABELS[feature],
    )

ax2.axhline(
    0,
    color="black",
    linewidth=1.0,
    alpha=0.8,
)
ax2.set_title(
    "Within-person PHQ-9 associations remain stable for several features",
    pad=14,
)
ax2.set_ylabel("Within-person β\n(standardized feature)")
ax2.set_ylim(-0.40, 0.28)
ax2.set_yticks(np.arange(-0.4, 0.31, 0.1))
ax2.grid(
    axis="y",
    linewidth=0.8,
    alpha=0.22,
)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
panel_letter(ax2, "B")


# ============================================================
# PANEL C — GAD-7
# ============================================================

for feature in FEATURE_ORDER:

    beta, yerr = get_beta_ci(
        weekly,
        "anxiety_score",
        feature,
    )

    ax3.errorbar(
        X,
        beta,
        yerr=yerr,
        fmt="o-",
        linewidth=1.9,
        markersize=5.8,
        capsize=3,
        color=FEATURE_COLORS[feature],
        label=FEATURE_LABELS[feature],
    )

ax3.axhline(
    0,
    color="black",
    linewidth=1.0,
    alpha=0.8,
)
ax3.set_title(
    "Within-person GAD-7 associations are also relatively robust",
    pad=14,
)
ax3.set_ylabel("Within-person β\n(standardized feature)")
ax3.set_ylim(-0.40, 0.28)
ax3.set_yticks(np.arange(-0.4, 0.31, 0.1))
ax3.grid(
    axis="y",
    linewidth=0.8,
    alpha=0.22,
)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
panel_letter(ax3, "C")


# ============================================================
# X AXIS
# ============================================================

ax3.set_xticks(X)
ax3.set_xticklabels(X_LABELS)
ax3.set_xlabel("Sampling interval (minutes; Native = original irregular GPS)")


# ============================================================
# GLOBAL TITLE / LEGEND / CAPTION
# ============================================================

fig.suptitle(
    "Technical robustness and clinical validity of GPS features across temporal resolutions",
    fontsize=20,
    fontweight="bold",
    y=0.985,
)

handles, labels = ax1.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.055),
)

caption = (
    "Panel A shows Spearman correlation with native-resolution feature values. "
    "Panels B and C show within-person standardized mixed-model coefficients "
    "(β ± 95% CI) from weekly questionnaire models under the resolution-aware "
    "support condition. The central result is that some GPS features lose "
    "technical fidelity before they lose clinically meaningful longitudinal associations."
)

fig.text(
    0.5,
    0.014,
    caption,
    ha="center",
    va="bottom",
    fontsize=10.2,
)

fig.subplots_adjust(
    left=0.11,
    right=0.985,
    top=0.93,
    bottom=0.13,
    hspace=0.33,
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

print()
print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
print()

plt.show()
