from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/paper_replication"
)

INPUT = ROOT / "paper_replication_analysis_dataset.csv"

OUT_PNG = ROOT / "home_time_phq9_low_high_groups_nodots.png"
OUT_PDF = ROOT / "home_time_phq9_low_high_groups_nodots.pdf"
OUT_CSV = ROOT / "home_time_phq9_low_high_groups_nodots_summary.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT, low_memory=False)

df["phq9_total"] = pd.to_numeric(df["phq9_total"], errors="coerce")

# Keep only low (0–4) and high (>=10); exclude mild (5–9)
df = df[
    (df["phq9_total"].between(0, 4, inclusive="both")) |
    (df["phq9_total"] >= 10)
].copy()

df["phq_group"] = np.where(
    df["phq9_total"] >= 10,
    "High\nPHQ-9 ≥10",
    "Low\nPHQ-9 0–4",
)

group_order = [
    "Low\nPHQ-9 0–4",
    "High\nPHQ-9 ≥10",
]

panels = [
    ("home_14d_all_pct", "Entire 14-day period", "(A)"),
    ("home_14d_weekday_matched_pct", "Weekdays only", "(B)"),
    ("home_14d_weekend_pct", "Weekends only", "(C)"),
]


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize_panel(data, outcome):
    rows = []

    for g in group_order:
        x = pd.to_numeric(
            data.loc[data["phq_group"] == g, outcome],
            errors="coerce",
        ).dropna()

        n = len(x)
        mean = x.mean() if n > 0 else np.nan
        sd = x.std(ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        ci = 1.96 * se if n > 1 else np.nan

        rows.append(
            {
                "group": g,
                "n": n,
                "mean": mean,
                "sd": sd,
                "se": se,
                "ci95": ci,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BUILD SUMMARY TABLE
# ============================================================

all_summary = []

for outcome, title, panel in panels:
    s = summarize_panel(df, outcome)
    s["outcome"] = outcome
    s["panel_title"] = title
    s["panel"] = panel
    all_summary.append(s)

summary = pd.concat(all_summary, ignore_index=True)
summary.to_csv(OUT_CSV, index=False)


# ============================================================
# PLOT
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4), sharey=True)

colors = {
    "Low\nPHQ-9 0–4": "#1f77b4",
    "High\nPHQ-9 ≥10": "#ff7f0e",
}

for ax, (outcome, title, panel) in zip(axes, panels):
    s = summary[summary["outcome"] == outcome].copy()
    s["x"] = [0, 1]

    for _, row in s.iterrows():
        x = row["x"]
        mean = row["mean"]
        ci = row["ci95"]
        group = row["group"]
        n = int(row["n"])

        # Mean + 95% CI only (no raw dots)
        ax.errorbar(
            x,
            mean,
            yerr=ci,
            fmt="o",
            markersize=10,
            capsize=6,
            elinewidth=2,
            capthick=2,
            color=colors[group],
            markerfacecolor=colors[group],
            markeredgecolor=colors[group],
            zorder=3,
        )

        # n label near bottom
        ax.text(
            x,
            4,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        # mean label above point
        ax.text(
            x,
            mean + (ci if pd.notna(ci) else 0) + 3,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=colors[group],
        )

    ax.set_title(title, fontsize=15, pad=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_order, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("Home time (%)", fontsize=13)
    ax.text(
        -0.14,
        1.02,
        panel,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Home time in low vs high PHQ-9 groups",
    fontsize=18,
    y=1.02,
)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("LOW VS HIGH PHQ-9 HOME-TIME SUMMARY (NO DOTS)")
print("=" * 90)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\nSaved:")
print(f"  {OUT_PNG}")
print(f"  {OUT_PDF}")
print(f"  {OUT_CSV}")
