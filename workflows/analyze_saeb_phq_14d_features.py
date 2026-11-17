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
    "multires_clinical_benchmark/saeb_replication"
)

INPUT = BASE / "saeb_phq_14d_5min_features.parquet"

OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

STATS_CSV = OUT / "saeb_phq_14d_5min_correlations.csv"
GROUP_CSV = OUT / "saeb_phq_14d_5min_group_tests.csv"

MAIN_PNG = OUT / "figure_saeb_phq_14d_5min_main.png"
MAIN_PDF = OUT / "figure_saeb_phq_14d_5min_main.pdf"

SECONDARY_PNG = OUT / "figure_saeb_phq_14d_5min_secondary.png"
SECONDARY_PDF = OUT / "figure_saeb_phq_14d_5min_secondary.pdf"

GROUP_PNG = OUT / "figure_saeb_phq_14d_5min_groups.png"
GROUP_PDF = OUT / "figure_saeb_phq_14d_5min_groups.pdf"


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = {
    "Total distance": "literature_total_distance_m",
    "Location variance": "literature_location_variance",
    "Number of clusters": "literature_n_clusters",
    "Entropy": "literature_entropy_nats",
    "Normalized entropy": "literature_normalized_entropy",
    "Homestay": "literature_homestay_fraction",

    "Transition time": "literature_transition_time_fraction",
    "Mean speed": "literature_speed_mean_kmh",
    "Speed variance": "literature_speed_variance_kmh2",
    "Raw entropy": "literature_raw_entropy",
}

MAIN_FEATURES = [
    "Total distance",
    "Location variance",
    "Number of clusters",
    "Entropy",
    "Normalized entropy",
    "Homestay",
]

SECONDARY_FEATURES = [
    "Transition time",
    "Mean speed",
    "Speed variance",
    "Raw entropy",
]


# ============================================================
# DISPLAY TRANSFORMATIONS
#
# Correlations use original values.
# These transformations are visualization only.
# ============================================================

def display_values(label, x):

    x = pd.to_numeric(
        x,
        errors="coerce",
    ).astype(float)

    if label == "Total distance":
        return x / 1000.0, "Total distance (km)"

    if label == "Homestay":
        return x * 100.0, "Homestay (%)"

    if label == "Transition time":
        return x * 100.0, "Transition time (%)"

    if label == "Mean speed":
        return x, "Mean speed (km/h)"

    if label == "Speed variance":
        return x, "Speed variance (km²/h²)"

    return x, label


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(INPUT)

print()
print("=" * 80)
print("SAEB PHQ-9 14-DAY / 5-MIN ANALYSIS")
print("=" * 80)

print(f"Rows: {len(df):,}")
print(
    f"Participants: "
    f"{df['participant_id'].nunique():,}"
)

print(
    f"Participants with GPS: "
    f"{(df['sampled_gps_points'] > 0).sum():,}"
)

print(
    f"Median observed days: "
    f"{df['observed_gps_days'].median():.1f}"
)

print(
    f"Median 5-min samples: "
    f"{df['sampled_gps_points'].median():,.0f}"
)


# ============================================================
# PEARSON + BOOTSTRAP
# ============================================================

N_BOOT = 1000
SEED = 2026

rng = np.random.default_rng(SEED)

records = []


for label, col in FEATURES.items():

    d = df[
        [
            "participant_id",
            "depression_score",
            col,
            "observed_gps_days",
            "sampled_gps_points",
        ]
    ].copy()

    d[col] = pd.to_numeric(
        d[col],
        errors="coerce",
    )

    d["depression_score"] = pd.to_numeric(
        d["depression_score"],
        errors="coerce",
    )

    d = d.dropna(
        subset=[
            "depression_score",
            col,
        ]
    )

    if len(d) < 3:
        continue

    x = d[col].to_numpy(dtype=float)
    y = d["depression_score"].to_numpy(dtype=float)

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        continue

    r, p = stats.pearsonr(
        x,
        y,
    )

    boot = []

    n = len(d)

    for _ in range(N_BOOT):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        xb = x[idx]
        yb = y[idx]

        if (
            np.std(xb) == 0
            or np.std(yb) == 0
        ):
            continue

        rb = stats.pearsonr(
            xb,
            yb,
        ).statistic

        if np.isfinite(rb):
            boot.append(rb)

    if len(boot):

        ci_low, ci_high = np.percentile(
            boot,
            [2.5, 97.5],
        )

    else:

        ci_low = np.nan
        ci_high = np.nan

    records.append({
        "feature": label,
        "column": col,
        "n": n,
        "pearson_r": r,
        "p_value": p,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_samples": len(boot),
        "median_observed_days": d[
            "observed_gps_days"
        ].median(),
        "median_sampled_points": d[
            "sampled_gps_points"
        ].median(),
    })


corr = pd.DataFrame(records)

corr.to_csv(
    STATS_CSV,
    index=False,
)

print()
print("=" * 80)
print("PEARSON CORRELATIONS")
print("=" * 80)

print(
    corr[
        [
            "feature",
            "n",
            "pearson_r",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "p_value",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# PHQ <5 vs >=5
# ============================================================

group_records = []

df["phq_group"] = np.where(
    df["depression_score"] >= 5,
    "PHQ-9 ≥5",
    "PHQ-9 <5",
)


for label, col in FEATURES.items():

    a = pd.to_numeric(
        df.loc[
            df["phq_group"].eq("PHQ-9 <5"),
            col,
        ],
        errors="coerce",
    ).dropna()

    b = pd.to_numeric(
        df.loc[
            df["phq_group"].eq("PHQ-9 ≥5"),
            col,
        ],
        errors="coerce",
    ).dropna()

    if len(a) < 2 or len(b) < 2:
        continue

    # Direct Saeb-style independent-samples t test.
    t, p = stats.ttest_ind(
        a,
        b,
        equal_var=True,
        nan_policy="omit",
    )

    group_records.append({
        "feature": label,
        "n_phq_lt5": len(a),
        "n_phq_ge5": len(b),
        "mean_phq_lt5": a.mean(),
        "mean_phq_ge5": b.mean(),
        "t": t,
        "p_value": p,
    })


groups = pd.DataFrame(group_records)

groups.to_csv(
    GROUP_CSV,
    index=False,
)

print()
print("=" * 80)
print("PHQ-9 <5 VS ≥5")
print("=" * 80)

print(
    groups[
        [
            "feature",
            "n_phq_lt5",
            "n_phq_ge5",
            "t",
            "p_value",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SAEB-STYLE SCATTER FUNCTION
# ============================================================

def plot_feature_panels(
    labels,
    *,
    output_png,
    output_pdf,
    title,
):

    n = len(labels)

    ncols = 2
    nrows = int(
        np.ceil(
            n / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            9.0,
            3.65 * nrows,
        ),
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    for ax, label in zip(
        axes,
        labels,
    ):

        col = FEATURES[label]

        d = df[
            [
                "depression_score",
                col,
            ]
        ].copy()

        d[col] = pd.to_numeric(
            d[col],
            errors="coerce",
        )

        d = d.dropna()

        raw_x = d[col]

        x_display, xlabel = (
            display_values(
                label,
                raw_x,
            )
        )

        x = x_display.to_numpy(
            dtype=float
        )

        y = d[
            "depression_score"
        ].to_numpy(
            dtype=float
        )

        row = corr.loc[
            corr["feature"].eq(label)
        ].iloc[0]

        r = row["pearson_r"]
        p = row["p_value"]
        lo = row["bootstrap_ci_low"]
        hi = row["bootstrap_ci_high"]

        ax.scatter(
            x,
            y,
            s=24,
            alpha=0.65,
            edgecolor="none",
        )

        # Saeb figure:
        # regression line only when P < .05.
        if (
            p < 0.05
            and len(np.unique(x)) > 1
        ):

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

            yy = (
                intercept
                + slope * xx
            )

            predicted = (
                intercept
                + slope * x
            )

            rmse = np.sqrt(
                np.mean(
                    (
                        y
                        - predicted
                    ) ** 2
                )
            )

            ax.plot(
                xx,
                yy,
                linewidth=1.5,
            )

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
            (
                f"r={r:.2f} "
                f"(95% CI {lo:.2f} to {hi:.2f}); "
                f"P={p:.3g}"
            ),
            pad=7,
        )

        ax.set_xlabel(
            xlabel
        )

        ax.set_ylabel(
            "PHQ-9 score"
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.grid(
            linewidth=0.5,
            alpha=0.12,
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout()

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# ============================================================
# MAIN LITERATURE FEATURES
# ============================================================

plot_feature_panels(
    MAIN_FEATURES,
    output_png=MAIN_PNG,
    output_pdf=MAIN_PDF,
    title=(
        "GPS mobility and depressive symptom severity\n"
        "14-day window, 5-minute sampling"
    ),
)


# ============================================================
# SECONDARY FEATURES
# ============================================================

plot_feature_panels(
    SECONDARY_FEATURES,
    output_png=SECONDARY_PNG,
    output_pdf=SECONDARY_PDF,
    title=(
        "Additional GPS mobility features and depressive symptoms\n"
        "14-day window, 5-minute sampling"
    ),
)


# ============================================================
# GROUP COMPARISON — CORE FEATURES ONLY
# ============================================================

scaled = {}

for label in MAIN_FEATURES:

    col = FEATURES[label]

    x = pd.to_numeric(
        df[col],
        errors="coerce",
    )

    xmin = x.min()
    xmax = x.max()

    if (
        np.isfinite(xmin)
        and np.isfinite(xmax)
        and xmax > xmin
    ):
        scaled[label] = (
            (x - xmin)
            / (xmax - xmin)
        )
    else:
        scaled[label] = (
            pd.Series(
                np.nan,
                index=df.index,
            )
        )


fig, ax = plt.subplots(
    figsize=(10.5, 5.8)
)

positions = np.arange(
    1,
    len(MAIN_FEATURES) + 1,
)

offset = 0.18
width = 0.28


for i, label in enumerate(
    MAIN_FEATURES
):

    x = scaled[label]

    low = x.loc[
        df["phq_group"].eq(
            "PHQ-9 <5"
        )
    ].dropna()

    high = x.loc[
        df["phq_group"].eq(
            "PHQ-9 ≥5"
        )
    ].dropna()

    ax.boxplot(
        [low],
        positions=[
            positions[i] - offset
        ],
        widths=width,
        showfliers=False,
    )

    ax.boxplot(
        [high],
        positions=[
            positions[i] + offset
        ],
        widths=width,
        showfliers=False,
    )

    result = groups.loc[
        groups[
            "feature"
        ].eq(label)
    ]

    if len(result):

        p = result.iloc[0][
            "p_value"
        ]

        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        else:
            marker = ""

        if marker:

            ymax = np.nanmax(
                np.concatenate(
                    [
                        low.to_numpy(),
                        high.to_numpy(),
                    ]
                )
            )

            ax.text(
                positions[i],
                min(
                    1.08,
                    ymax + 0.07,
                ),
                marker,
                ha="center",
                fontsize=12,
            )


ax.set_xticks(
    positions
)

ax.set_xticklabels(
    MAIN_FEATURES
)

ax.set_ylabel(
    "Feature value (scaled 0–1)"
)

ax.set_ylim(
    -0.05,
    1.13,
)

ax.set_title(
    "GPS mobility by depressive symptom group",
    fontsize=14,
    fontweight="bold",
)

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
    ncol=2,
    loc="upper center",
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.12,
)

fig.tight_layout()

fig.savefig(
    GROUP_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    GROUP_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

for p in [
    STATS_CSV,
    GROUP_CSV,
    MAIN_PNG,
    MAIN_PDF,
    SECONDARY_PNG,
    SECONDARY_PDF,
    GROUP_PNG,
    GROUP_PDF,
]:
    print(p)
