from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/paper_replication/"
    "paper_replication_analysis_dataset.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/paper_replication"
)

PNG_OUT = OUT_DIR / "home_time_distribution_paper_style.png"
PDF_OUT = OUT_DIR / "home_time_distribution_paper_style.pdf"
STATS_OUT = OUT_DIR / "home_time_distribution_paper_style_stats.csv"

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

HOME_COLS = [
    (
        "Entire 14-day period",
        "home_14d_all_pct",
        "(A)",
    ),
    (
        "Weekdays only",
        "home_14d_weekday_matched_pct",
        "(B)",
    ),
    (
        "Weekends only",
        "home_14d_weekend_pct",
        "(C)",
    ),
]

for _, col, _ in HOME_COLS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

# ============================================================
# STYLE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

# Same binning across all three panels.
bins = np.arange(
    0,
    101,
    5,
)

# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(8.2, 10.2),
)

outer = fig.add_gridspec(
    3,
    1,
    hspace=0.34,
)

stats_rows = []

for i, (title, col, panel_label) in enumerate(HOME_COLS):

    x = (
        df[col]
        .dropna()
        .to_numpy(dtype=float)
    )

    # Separate compact boxplot area above histogram.
    inner = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer[i],
        height_ratios=[
            0.32,
            1.0,
        ],
        hspace=0.02,
    )

    ax_box = fig.add_subplot(
        inner[0]
    )

    ax_hist = fig.add_subplot(
        inner[1],
        sharex=ax_box,
    )

    # --------------------------------------------------------
    # BOX PLOT
    # --------------------------------------------------------

    bp = ax_box.boxplot(
        x,
        vert=False,
        widths=0.42,
        patch_artist=False,
        showmeans=True,
        meanprops={
            "marker": "+",
            "markersize": 8,
            "markeredgewidth": 1.2,
        },
        medianprops={
            "linewidth": 1.2,
        },
        whiskerprops={
            "linestyle": "--",
            "linewidth": 1.0,
        },
        capprops={
            "linewidth": 1.0,
        },
        boxprops={
            "linewidth": 1.0,
        },
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "alpha": 0.5,
        },
    )

    ax_box.set_xlim(
        0,
        100,
    )

    ax_box.set_yticks([])

    for spine in ax_box.spines.values():
        spine.set_visible(False)

    ax_box.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    ax_hist.hist(
        x,
        bins=bins,
        edgecolor="black",
        linewidth=0.6,
    )

    ax_hist.set_xlim(
        0,
        100,
    )

    ax_hist.set_ylabel(
        "Number of segments"
    )

    if i == 2:
        ax_hist.set_xlabel(
            "Home time (%)"
        )
    else:
        ax_hist.set_xlabel("")

    ax_hist.spines["top"].set_visible(
        False
    )

    ax_hist.spines["right"].set_visible(
        False
    )

    # --------------------------------------------------------
    # PANEL LABEL / TITLE / N
    # --------------------------------------------------------

    ax_hist.text(
        -0.10,
        1.34,
        panel_label,
        transform=ax_hist.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    ax_hist.text(
        -0.10,
        1.12,
        title,
        transform=ax_hist.transAxes,
        fontsize=10,
        fontweight="normal",
        rotation=90,
        va="top",
        ha="center",
    )

    ax_hist.text(
        0.02,
        0.93,
        f"N = {len(x)} segments",
        transform=ax_hist.transAxes,
        fontsize=9,
        va="top",
    )

    # --------------------------------------------------------
    # DESCRIPTIVE STATS
    # --------------------------------------------------------

    stats_rows.append(
        {
            "period":
                title,
            "n_segments":
                len(x),
            "mean":
                np.mean(x),
            "std":
                np.std(
                    x,
                    ddof=1,
                ),
            "min":
                np.min(x),
            "q1":
                np.percentile(
                    x,
                    25,
                ),
            "median":
                np.median(x),
            "q3":
                np.percentile(
                    x,
                    75,
                ),
            "max":
                np.max(x),
        }
    )

# ============================================================
# SAVE
# ============================================================

fig.subplots_adjust(
    left=0.19,
    right=0.97,
    top=0.98,
    bottom=0.07,
)

fig.savefig(
    PNG_OUT,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
)

plt.close(fig)

stats = pd.DataFrame(
    stats_rows
)

stats.to_csv(
    STATS_OUT,
    index=False,
)

print("=" * 100)
print("HOME-TIME DISTRIBUTION FIGURE")
print("=" * 100)

print(
    stats.round(2).to_string(
        index=False
    )
)

print()
print("Saved:")
print(" ", PNG_OUT)
print(" ", PDF_OUT)
print(" ", STATS_OUT)
