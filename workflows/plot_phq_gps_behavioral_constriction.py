from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/full_cohort_phq_analysis"
)

SUMMARY_CSV = BASE / "phq_four_group_gps_summary.csv"

OUT_PNG = BASE / "figure_phq_gps_behavioral_constriction.png"
OUT_PDF = BASE / "figure_phq_gps_behavioral_constriction.pdf"


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BAR_COLOR = "#9B77C4"
BAR_EDGE = "#4D4355"

GROUP_ORDER = [
    "Minimal",
    "Mild",
    "Moderate",
    "Mod. severe / severe",
]

X_LABELS_BASE = [
    "Minimal\n0–4",
    "Mild\n5–9",
    "Moderate\n10–14",
    "Mod. severe /\nsevere\n15–27",
]


# ============================================================
# HELPERS
# ============================================================

def canonical_group(value):
    s = str(value).lower().replace("–", "-").replace("—", "-")

    if "0-4" in s or "minimal" in s:
        return "Minimal"

    if "5-9" in s or "mild" in s:
        return "Mild"

    if "10-14" in s or "moderate" in s:
        return "Moderate"

    if "15-27" in s or "severe" in s:
        return "Mod. severe / severe"

    return str(value)


def get_feature_rows(summary, feature_name):
    d = summary.loc[
        summary["feature"].eq(feature_name)
    ].copy()

    if d.empty:
        raise ValueError(
            f'Feature not found: "{feature_name}"\n'
            f'Available features:\n'
            f'{summary["feature"].drop_duplicates().tolist()}'
        )

    d["group_clean"] = d["phq_group"].map(canonical_group)

    d["group_clean"] = pd.Categorical(
        d["group_clean"],
        categories=GROUP_ORDER,
        ordered=True,
    )

    d = (
        d.sort_values("group_clean")
        .reset_index(drop=True)
    )

    if len(d) != 4:
        raise ValueError(
            f'Expected 4 PHQ-9 groups for "{feature_name}", '
            f"but found {len(d)}."
        )

    return d


def draw_panel(
    ax,
    data,
    title,
    ylabel,
    effect_text,
    panel_letter,
    ylim,
    yticks,
):
    x = np.arange(4)

    means = data["mean"].astype(float).to_numpy()

    ci_low = (
        data["cluster_boot_ci_low"]
        .astype(float)
        .to_numpy()
    )

    ci_high = (
        data["cluster_boot_ci_high"]
        .astype(float)
        .to_numpy()
    )

    yerr = np.vstack([
        means - ci_low,
        ci_high - means,
    ])

    # --------------------------------------------------------
    # Mean bars
    # --------------------------------------------------------

    ax.bar(
        x,
        means,
        width=0.62,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=1.0,
        alpha=0.86,
        zorder=2,
    )

    # --------------------------------------------------------
    # 95% participant-cluster bootstrap confidence intervals
    # --------------------------------------------------------

    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capsize=5,
        capthick=1.6,
        zorder=3,
    )

    # --------------------------------------------------------
    # Numerical mean labels
    # --------------------------------------------------------

    y_range = ylim[1] - ylim[0]
    offset = 0.018 * y_range

    for xi, mean, high in zip(
        x,
        means,
        ci_high,
    ):
        ax.text(
            xi,
            high + offset,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # --------------------------------------------------------
    # X labels + participant counts
    # --------------------------------------------------------

    ns = (
        data["n_participants"]
        .astype(int)
        .to_numpy()
    )

    labels = [
        f"{label}\n(n={n})"
        for label, n in zip(
            X_LABELS_BASE,
            ns,
        )
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_xlabel(
        "PHQ-9 severity",
        labelpad=8,
    )

    ax.set_ylabel(ylabel)

    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        title,
        pad=18,
    )

    # --------------------------------------------------------
    # Effect annotation
    # --------------------------------------------------------

    ax.text(
        0.03,
        0.92,
        effect_text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Panel letter
    # --------------------------------------------------------

    ax.text(
        -0.12,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
    )

    # --------------------------------------------------------
    # Grid / axes
    # --------------------------------------------------------

    ax.grid(
        axis="y",
        linewidth=0.8,
        alpha=0.22,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# LOAD DATA
# ============================================================

summary = pd.read_csv(SUMMARY_CSV)

required_columns = {
    "feature",
    "phq_group",
    "n_participants",
    "mean",
    "cluster_boot_ci_low",
    "cluster_boot_ci_high",
}

missing = required_columns - set(summary.columns)

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )


movement = get_feature_rows(
    summary,
    "Movement distance (km/day)",
)

home = get_feature_rows(
    summary,
    "Time at home (h/day)",
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7.2),
)


# ------------------------------------------------------------
# PANEL A
# ------------------------------------------------------------

draw_panel(
    axes[0],
    movement,
    title=(
        "Daily movement falls with "
        "depressive symptom burden"
    ),
    ylabel="Daily movement distance (km/day)",
    effect_text="−3.1 km/day per +5 PHQ-9 points",
    panel_letter="A",
    ylim=(0, 55),
    yticks=np.arange(0, 51, 10),
)


# ------------------------------------------------------------
# PANEL B
# ------------------------------------------------------------

draw_panel(
    axes[1],
    home,
    title=(
        "Time at home rises at "
        "higher symptom severity"
    ),
    ylabel="Equivalent time at home (h/day)",
    effect_text="+0.5 h/day per +5 PHQ-9 points",
    panel_letter="B",
    ylim=(10, 22),
    yticks=np.arange(10, 23, 2),
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    (
        "Higher depressive symptom burden is associated "
        "with behavioral constriction"
    ),
    fontsize=21,
    fontweight="bold",
    y=0.985,
)


# ============================================================
# CAPTION
# ============================================================

caption = (
    "Bars show weekly mean values; error bars are 95% "
    "participant-cluster bootstrap CIs. GPS features were "
    "derived from native irregular timestamps without fixed "
    "resampling."
)

fig.text(
    0.5,
    0.015,
    caption,
    ha="center",
    va="bottom",
    fontsize=10.5,
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.985,
    top=0.80,
    bottom=0.20,
    wspace=0.18,
)


# ============================================================
# SAVE
# ============================================================

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
print("Saved figure:")
print(OUT_PNG)
print(OUT_PDF)
print()

plt.show()
