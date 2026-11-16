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
    "feature_reliability/gps_clinical/home_time"
)

DAILY_PATH = BASE / "daily_home_time.csv"
WEEKLY_PATH = BASE / "weekly_home_time.csv"

OUT_DIR = BASE / "figures"
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
# LOAD REAL DATA
# ============================================================

daily = pd.read_csv(
    DAILY_PATH
)

weekly = pd.read_csv(
    WEEKLY_PATH
)


# Keep only interpretable daily home-time values
daily_valid = daily[
    daily["time_at_home_pct"].notna()
].copy()

if "home_time_status" in daily_valid.columns:
    daily_valid = daily_valid[
        daily_valid["home_time_status"].eq(
            "observed"
        )
    ].copy()


# Keep only valid weekly values
weekly_valid = weekly[
    weekly["time_at_home_pct"].notna()
].copy()


print("=" * 80)
print("DAILY")
print("=" * 80)

print(
    "Valid participant-days:",
    len(daily_valid),
)

print(
    daily_valid[
        "time_at_home_pct"
    ]
    .describe(
        percentiles=[
            .10,
            .25,
            .50,
            .75,
            .90,
        ]
    )
    .to_string()
)


print()
print("=" * 80)
print("WEEKLY")
print("=" * 80)

print(
    "Valid clinical windows:",
    len(weekly_valid),
)

print(
    weekly_valid[
        "time_at_home_pct"
    ]
    .describe(
        percentiles=[
            .10,
            .25,
            .50,
            .75,
            .90,
        ]
    )
    .to_string()
)


# ============================================================
# FIGURE 1 — DAILY DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.2, 5.8)
)

values = (
    pd.to_numeric(
        daily_valid[
            "time_at_home_pct"
        ],
        errors="coerce",
    )
    .dropna()
    .to_numpy()
)


# 5-percentage-point bins
bins = np.arange(
    0,
    105,
    5,
)


ax.hist(
    values,
    bins=bins,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.9,
)


median = float(
    np.nanmedian(
        values
    )
)

mean = float(
    np.nanmean(
        values
    )
)


ax.axvline(
    median,
    linestyle="--",
    linewidth=1.8,
    label=(
        f"Median = "
        f"{median:.1f}%"
    ),
)

ax.axvline(
    mean,
    linestyle=":",
    linewidth=1.8,
    label=(
        f"Mean = "
        f"{mean:.1f}%"
    ),
)


ax.set_xlim(
    0,
    100,
)

ax.set_xlabel(
    "Daily home time (%)"
)

ax.set_ylabel(
    "Participant-days"
)

ax.set_title(
    "Distribution of daily home time",
    fontweight="bold",
)

ax.legend(
    frameon=False
)

ax.grid(
    axis="y",
    alpha=0.15,
)


ax.text(
    0.02,
    0.95,
    (
        f"N = {len(values):,} "
        f"participant-days"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
)


fig.tight_layout()


daily_png = (
    OUT_DIR
    / "daily_home_time_distribution.png"
)

daily_pdf = (
    OUT_DIR
    / "daily_home_time_distribution.pdf"
)


fig.savefig(
    daily_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    daily_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2 — WEEKLY BOXPLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.8, 5.8)
)

weekly_values = (
    pd.to_numeric(
        weekly_valid[
            "time_at_home_pct"
        ],
        errors="coerce",
    )
    .dropna()
    .to_numpy()
)


bp = ax.boxplot(
    weekly_values,
    vert=True,
    widths=0.45,
    patch_artist=False,
    showfliers=False,
)


weekly_median = float(
    np.nanmedian(
        weekly_values
    )
)

weekly_mean = float(
    np.nanmean(
        weekly_values
    )
)


ax.scatter(
    [1],
    [
        weekly_mean
    ],
    s=45,
    marker="o",
    label=(
        f"Mean = "
        f"{weekly_mean:.1f}%"
    ),
)


ax.set_ylim(
    0,
    100,
)

ax.set_xlim(
    0.55,
    1.45,
)

ax.set_xticks(
    [1]
)

ax.set_xticklabels(
    [
        "Weekly home time"
    ]
)

ax.set_ylabel(
    "Home time (%)"
)

ax.set_title(
    "Weekly home time across clinical windows",
    fontweight="bold",
)


ax.text(
    1.11,
    weekly_median,
    (
        f"Median "
        f"{weekly_median:.1f}%"
    ),
    va="center",
    fontsize=10,
)


ax.text(
    0.03,
    0.96,
    (
        f"N = {len(weekly_values):,} "
        f"clinical windows"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
)


ax.legend(
    frameon=False,
    loc="lower left",
)

ax.grid(
    axis="y",
    alpha=0.15,
)


fig.tight_layout()


weekly_png = (
    OUT_DIR
    / "weekly_home_time_boxplot.png"
)

weekly_pdf = (
    OUT_DIR
    / "weekly_home_time_boxplot.pdf"
)


fig.savefig(
    weekly_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    weekly_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.8)
)


# Left: daily histogram
ax = axes[0]

ax.hist(
    values,
    bins=bins,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.9,
)

ax.axvline(
    median,
    linestyle="--",
    linewidth=1.8,
)

ax.axvline(
    mean,
    linestyle=":",
    linewidth=1.8,
)

ax.set_xlim(
    0,
    100,
)

ax.set_xlabel(
    "Daily home time (%)"
)

ax.set_ylabel(
    "Participant-days"
)

ax.set_title(
    "Daily home time",
    fontweight="bold",
)

ax.grid(
    axis="y",
    alpha=0.15,
)

ax.text(
    0.02,
    0.95,
    (
        f"N = {len(values):,}\n"
        f"Median = {median:.1f}%"
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=10,
)


# Right: weekly boxplot
ax = axes[1]

ax.boxplot(
    weekly_values,
    vert=True,
    widths=0.42,
    patch_artist=False,
    showfliers=False,
)

ax.scatter(
    [1],
    [
        weekly_mean
    ],
    s=40,
    marker="o",
)

ax.set_ylim(
    0,
    100,
)

ax.set_xlim(
    0.55,
    1.45,
)

ax.set_xticks(
    [1]
)

ax.set_xticklabels(
    [
        "Weekly home time"
    ]
)

ax.set_ylabel(
    "Home time (%)"
)

ax.set_title(
    "Weekly home time",
    fontweight="bold",
)

ax.grid(
    axis="y",
    alpha=0.15,
)

ax.text(
    0.03,
    0.95,
    (
        f"N = {len(weekly_values):,}\n"
        f"Median = "
        f"{weekly_median:.1f}%"
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=10,
)


fig.suptitle(
    "General patterns of home time",
    fontsize=20,
    fontweight="bold",
    y=1.02,
)


fig.text(
    0.5,
    -0.01,
    (
        "Time at home varies widely across weeks, "
        "with many observations showing high home occupancy "
        "and substantial variability."
    ),
    ha="center",
    fontsize=11,
)


fig.tight_layout()


combined_png = (
    OUT_DIR
    / "home_time_general_patterns.png"
)

combined_pdf = (
    OUT_DIR
    / "home_time_general_patterns.pdf"
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
print(daily_png)
print(daily_pdf)
print(weekly_png)
print(weekly_pdf)
print(combined_png)
print(combined_pdf)
