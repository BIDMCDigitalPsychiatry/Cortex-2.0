from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "multires_clinical_benchmark"
)

INPUT = BASE / "gps_multires_weekly_alignment_resolution_aware.parquet"

OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

OUT_CORR_PNG = OUT / "figure_saeb_style_phq_gps_correlations.png"
OUT_CORR_PDF = OUT / "figure_saeb_style_phq_gps_correlations.pdf"

OUT_GROUP_PNG = OUT / "figure_saeb_style_phq_gps_groups.png"
OUT_GROUP_PDF = OUT / "figure_saeb_style_phq_gps_groups.pdf"

OUT_STATS = OUT / "saeb_style_phq_gps_statistics.csv"


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(INPUT)

if "resolution" in df.columns:
    df = df.loc[df["resolution"].astype(str).eq("native")].copy()


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

def first_existing(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


participant_col = first_existing([
    "participant_id",
    "participant",
    "user_id",
    "id",
])

phq_col = first_existing([
    "depression_score",
    "phq9_total",
    "phq9_score",
    "PHQ9",
    "phq9",
])

date_col = first_existing([
    "assessment_datetime",
    "assessment_date",
    "survey_datetime",
    "survey_date",
    "datetime",
    "date",
    "timestamp",
])


if participant_col is None:
    raise RuntimeError(
        "Could not identify participant column.\n"
        f"Columns:\n{df.columns.tolist()}"
    )

if phq_col is None:
    raise RuntimeError(
        "Could not identify PHQ-9 column.\n"
        f"Columns:\n{df.columns.tolist()}"
    )


# ============================================================
# FEATURE DEFINITIONS
#
# Main figure deliberately focuses on interpretable mobility
# constructs with close precedent in the GPS-depression
# literature.
# ============================================================

FEATURE_CANDIDATES = {
    "Total distance": [
        "supported_displacement_distance_m_mean",
        "supported_displacement_distance_m_median",
    ],
    "Entropy": [
        "location_entropy_nats_mean",
        "location_entropy_nats_median",
    ],
    "Observed places": [
        "n_observed_places_mean",
        "n_observed_places_median",
    ],
    "Radius of gyration": [
        "supported_radius_of_gyration_m_mean",
        "supported_radius_of_gyration_m_median",
    ],
}



feature_cols = {}

for label, candidates in FEATURE_CANDIDATES.items():
    col = first_existing(candidates)

    if col is not None:
        feature_cols[label] = col


if len(feature_cols) == 0:
    raise RuntimeError(
        "None of the expected GPS feature columns were found.\n"
        f"Columns:\n{df.columns.tolist()}"
    )


print("\nUsing features:")
for label, col in feature_cols.items():
    print(f"  {label}: {col}")

print(f"\nPHQ-9 column: {phq_col}")
print(f"Participant column: {participant_col}")
print(f"Assessment date column: {date_col}")


# ============================================================
# ONE OBSERVATION PER PARTICIPANT
#
# Saeb et al. correlated participant-level GPS features with
# one PHQ-9 assessment. We therefore use the earliest aligned
# native-resolution assessment for each participant.
# ============================================================

keep = [
    participant_col,
    phq_col,
    *feature_cols.values(),
]

if date_col is not None:
    keep.append(date_col)

d = df[keep].copy()

d[phq_col] = pd.to_numeric(
    d[phq_col],
    errors="coerce",
)

if date_col is not None:
    d = d.sort_values(
        [participant_col, date_col],
        kind="stable",
    )
else:
    print(
        "\nWARNING: no assessment-date column found; "
        "using the first aligned native row per participant."
    )

d = (
    d.groupby(
        participant_col,
        as_index=False,
        sort=False,
    )
    .first()
)

print()
print("=" * 80)
print("SAEB-STYLE CROSS-SECTIONAL SAMPLE")
print("=" * 80)
print(f"Participants: {d[participant_col].nunique():,}")
print(
    f"PHQ-9 range: "
    f"{d[phq_col].min():.0f}–{d[phq_col].max():.0f}"
)


# ============================================================
# UNIT CLEANUP
# ============================================================

display_cols = {}

for label, col in feature_cols.items():

    x = pd.to_numeric(
        d[col],
        errors="coerce",
    )

    # Convert meter-valued distance features to km for display.
    if col.endswith("_m"):
        new_col = f"{col}_km_display"
        d[new_col] = x / 1000.0
        display_cols[label] = new_col
    else:
        d[col] = x
        display_cols[label] = col


# ============================================================
# CONTINUOUS CORRELATION STATISTICS
# ============================================================

records = []

for label, col in display_cols.items():

    z = d[
        [phq_col, col]
    ].dropna()

    if len(z) < 3:
        continue

    r, p = stats.pearsonr(
        z[col],
        z[phq_col],
    )

    records.append({
        "analysis": "pearson",
        "feature": label,
        "n": len(z),
        "r": r,
        "p": p,
    })


corr_stats = pd.DataFrame(records)

print("\nPearson correlations with PHQ-9:")
print(
    corr_stats[
        ["feature", "n", "r", "p"]
    ].round(4).to_string(index=False)
)


# ============================================================
# FIGURE 1 — SAEB-STYLE CORRELATIONS
# ============================================================

n_features = len(display_cols)

ncols = 2
nrows = int(np.ceil(n_features / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(9.2, 3.8 * nrows),
)

axes = np.atleast_1d(axes).ravel()


for ax, (label, col) in zip(
    axes,
    display_cols.items(),
):

    z = d[
        [phq_col, col]
    ].dropna()

    x = z[col].to_numpy(dtype=float)
    y = z[phq_col].to_numpy(dtype=float)

    r, p = stats.pearsonr(x, y)

    ax.scatter(
        x,
        y,
        s=28,
        edgecolor="none",
    )

    # Match Saeb logic:
    # fitted line shown only when P < .05.
    if p < 0.05 and len(np.unique(x)) > 1:

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        xx = np.linspace(
            np.nanmin(x),
            np.nanmax(x),
            200,
        )

        yy = intercept + slope * xx

        predicted = intercept + slope * x

        rmse = np.sqrt(
            np.mean(
                (y - predicted) ** 2
            )
        )

        ax.plot(
            xx,
            yy,
            linewidth=1.5,
        )

        # Saeb et al. displayed ± root mean square
        # deviation around the fitted model.
        ax.plot(
            xx,
            yy + rmse,
            linestyle=":",
            linewidth=1.0,
        )

        ax.plot(
            xx,
            yy - rmse,
            linestyle=":",
            linewidth=1.0,
        )

    ax.set_title(
        f"r={r:.2f}; P={p:.3g}",
        pad=6,
    )

    ax.set_ylabel("PHQ-9 score")

    if label in {
        "Total distance",
        "Radius of gyration",
    }:
        ax.set_xlabel(f"{label} (km)")
    else:
        ax.set_xlabel(label)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        linewidth=0.5,
        alpha=0.15,
    )


for ax in axes[n_features:]:
    ax.axis("off")


fig.suptitle(
    "GPS mobility features and depressive symptom severity",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)

fig.tight_layout()

fig.savefig(
    OUT_CORR_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_CORR_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# GROUP COMPARISON
#
# Saeb threshold:
# PHQ-9 <5 vs PHQ-9 >=5
# ============================================================

d["phq_group"] = np.where(
    d[phq_col] >= 5,
    "PHQ-9 ≥5",
    "PHQ-9 <5",
)

group_records = []


# ============================================================
# SCALE FEATURES TO 0–1 FOR DISPLAY, AS IN SAEB
# ============================================================

scaled = pd.DataFrame({
    participant_col: d[participant_col],
    "phq_group": d["phq_group"],
})

for label, col in display_cols.items():

    x = pd.to_numeric(
        d[col],
        errors="coerce",
    )

    xmin = x.min()
    xmax = x.max()

    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        scaled[label] = (
            (x - xmin)
            / (xmax - xmin)
        )
    else:
        scaled[label] = np.nan


# ============================================================
# T TESTS
# ============================================================

for label in display_cols:

    a = scaled.loc[
        scaled["phq_group"].eq("PHQ-9 <5"),
        label,
    ].dropna()

    b = scaled.loc[
        scaled["phq_group"].eq("PHQ-9 ≥5"),
        label,
    ].dropna()

    if len(a) < 2 or len(b) < 2:
        continue

    # Saeb reports t tests between the two groups.
    # Their paper does not specify an unequal-variance variant,
    # so this reproduces the conventional independent t test.
    t, p = stats.ttest_ind(
        a,
        b,
        equal_var=True,
        nan_policy="omit",
    )

    group_records.append({
        "analysis": "phq_group_ttest",
        "feature": label,
        "n_phq_lt5": len(a),
        "n_phq_ge5": len(b),
        "mean_phq_lt5": a.mean(),
        "mean_phq_ge5": b.mean(),
        "t": t,
        "p": p,
    })


group_stats = pd.DataFrame(group_records)

print("\nPHQ-9 <5 vs ≥5 t tests:")
print(
    group_stats[
        [
            "feature",
            "n_phq_lt5",
            "n_phq_ge5",
            "t",
            "p",
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# FIGURE 2 — GROUP COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.5, 5.8),
)

positions = np.arange(
    1,
    len(display_cols) + 1,
)

width = 0.28


for i, label in enumerate(
    display_cols.keys()
):

    a = scaled.loc[
        scaled["phq_group"].eq("PHQ-9 <5"),
        label,
    ].dropna()

    b = scaled.loc[
        scaled["phq_group"].eq("PHQ-9 ≥5"),
        label,
    ].dropna()

    bp1 = ax.boxplot(
        [a],
        positions=[positions[i] - width / 1.5],
        widths=width,
        patch_artist=False,
        showfliers=False,
    )

    bp2 = ax.boxplot(
        [b],
        positions=[positions[i] + width / 1.5],
        widths=width,
        patch_artist=False,
        showfliers=False,
    )

    row = group_stats.loc[
        group_stats["feature"].eq(label)
    ]

    if len(row):

        p = row.iloc[0]["p"]

        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = ""

        if stars:
            ymax = np.nanmax(
                np.concatenate([
                    a.to_numpy(),
                    b.to_numpy(),
                ])
            )

            y = min(
                1.07,
                ymax + 0.08,
            )

            ax.text(
                positions[i],
                y,
                stars,
                ha="center",
                va="bottom",
                fontsize=12,
            )


ax.set_xticks(positions)
ax.set_xticklabels(
    list(display_cols.keys()),
    rotation=0,
)

ax.set_ylabel("Feature value (scaled 0–1)")
ax.set_ylim(-0.05, 1.14)

ax.set_title(
    "GPS mobility features by depressive symptom group",
    fontsize=14,
    fontweight="bold",
)

# Simple manual legend to avoid additional visual clutter
ax.plot(
    [],
    [],
    label="PHQ-9 <5",
    linewidth=1.5,
)

ax.plot(
    [],
    [],
    label="PHQ-9 ≥5",
    linewidth=1.5,
)

ax.legend(
    frameon=False,
    loc="upper center",
    ncol=2,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.15,
)

fig.tight_layout()

fig.savefig(
    OUT_GROUP_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT_GROUP_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# SAVE STATISTICS
# ============================================================

all_stats = pd.concat(
    [
        corr_stats,
        group_stats,
    ],
    ignore_index=True,
    sort=False,
)

all_stats.to_csv(
    OUT_STATS,
    index=False,
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUT_CORR_PNG)
print(OUT_CORR_PDF)
print(OUT_GROUP_PNG)
print(OUT_GROUP_PDF)
print(OUT_STATS)
