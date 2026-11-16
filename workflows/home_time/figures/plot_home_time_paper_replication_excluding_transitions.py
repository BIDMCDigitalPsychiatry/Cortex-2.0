from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/paper_replication/"
    "paper_replication_analysis_dataset.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/paper_replication"
)

PNG_OUT = OUT_DIR / "home_time_age_phq8_excluding_transitions.png"
PDF_OUT = OUT_DIR / "home_time_age_phq8_excluding_transitions.pdf"

df = pd.read_csv(DATA_PATH, low_memory=False)

for c in [
    "age",
    "phq8_total",
    "home_14d_all_pct",
    "home_14d_weekday_matched_pct",
    "home_14d_weekend_pct",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

zero_transition_ids = set(
    df.loc[
        np.isclose(
            df["home_14d_all_pct"],
            0.0,
            atol=1e-9,
        ),
        "participant_id",
    ]
)

filtered = df[
    ~df["participant_id"].isin(
        zero_transition_ids
    )
].copy()

print("=" * 100)
print("FIGURE SENSITIVITY DATASET")
print("=" * 100)
print("Full segments:", len(df))
print("Full participants:", df["participant_id"].nunique())
print("Excluded participants:", len(zero_transition_ids))
print("Remaining segments:", len(filtered))
print("Remaining participants:", filtered["participant_id"].nunique())


def panel(ax, data, xcol, ycol, label, title, xlabel, ylabel=None):
    d = data[[xcol, ycol]].dropna()

    ax.scatter(
        d[xcol],
        d[ycol],
        s=14,
        alpha=0.45,
        linewidths=0,
    )

    if len(d) >= 2 and d[xcol].nunique() >= 2:
        x = d[xcol].to_numpy(dtype=float)
        y = d[ycol].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)

        xs = np.linspace(
            x.min(),
            x.max(),
            200,
        )

        ys = slope * xs + intercept

        ax.plot(
            xs,
            ys,
            linewidth=1.8,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    ax.set_ylim(0, 100)

    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plt.rcParams["font.family"] = "Arial"

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8),
)

panel(
    axes[0, 0],
    filtered,
    "age",
    "home_14d_all_pct",
    "(A)",
    "Entire 14-day period",
    "Age at enrollment",
    "Home time (%)",
)

panel(
    axes[0, 1],
    filtered,
    "age",
    "home_14d_weekday_matched_pct",
    "(B)",
    "Weekdays only",
    "Age at enrollment",
)

panel(
    axes[0, 2],
    filtered,
    "age",
    "home_14d_weekend_pct",
    "(C)",
    "Weekends only",
    "Age at enrollment",
)

panel(
    axes[1, 0],
    filtered,
    "phq8_total",
    "home_14d_all_pct",
    "(D)",
    "Entire 14-day period",
    "PHQ-8 total score",
    "Home time (%)",
)

panel(
    axes[1, 1],
    filtered,
    "phq8_total",
    "home_14d_weekday_matched_pct",
    "(E)",
    "Weekdays only",
    "PHQ-8 total score",
)

panel(
    axes[1, 2],
    filtered,
    "phq8_total",
    "home_14d_weekend_pct",
    "(F)",
    "Weekends only",
    "PHQ-8 total score",
)

fig.tight_layout()

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

print()
print("Saved:")
print(PNG_OUT)
print(PDF_OUT)
