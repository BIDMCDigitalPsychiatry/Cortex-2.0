"""Publication-style figures for personalized GPS clinical degradation."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "personalized_clinical_degradation"
)

SUMMARY_PATH = ROOT / "combined_condition_summary.csv"
GEOMETRY_PATH = ROOT / "combined_geometry_contrasts.csv"

OUT_MAIN = ROOT / "personalized_gps_clinical_degradation_main_figure.png"
OUT_HEATMAP = ROOT / "personalized_gps_reliability_fingerprint_heatmap.png"


# ============================================================
# LOAD
# ============================================================

summary = pd.read_csv(SUMMARY_PATH)
geometry = pd.read_csv(GEOMETRY_PATH)

participants = [
    "LPE511",
    "VTE817",
    "BPH745",
]

participant_labels = {
    "LPE511": "LPE511\nDisplacement distance",
    "VTE817": "VTE817\nDisplacement distance",
    "BPH745": "BPH745\nDominant-place fraction",
}

participant_colors = {
    "LPE511": "#C44E52",
    "VTE817": "#4C72B0",
    "BPH745": "#55A868",
}

pattern_styles = {
    "distributed": "-",
    "contiguous": "--",
}

pattern_markers = {
    "distributed": "o",
    "contiguous": "s",
}


# ============================================================
# MAIN FIGURE
#
# Top row:
#   A1-A3 personalized degradation fingerprints
#
# Bottom left:
#   B geometry contrast
#
# Bottom right:
#   C feature degradation -> clinical consequence
# ============================================================

fig = plt.figure(
    figsize=(15, 9)
)

gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[1.0, 1.15],
    hspace=0.42,
    wspace=0.32,
)


# ------------------------------------------------------------
# PANEL A: PERSONALIZED CLINICAL DEGRADATION
# ------------------------------------------------------------

top_axes = []

global_ymax = max(
    0.5,
    summary["delta_mae_ci_high"].max() * 1.08,
)

for i, participant_id in enumerate(participants):

    ax = fig.add_subplot(
        gs[0, i]
    )

    top_axes.append(ax)

    participant_data = (
        summary[
            summary[
                "participant_id"
            ].eq(participant_id)
        ]
        .copy()
    )

    for pattern in [
        "distributed",
        "contiguous",
    ]:

        g = (
            participant_data[
                participant_data[
                    "pattern"
                ].eq(pattern)
            ]
            .sort_values(
                "missing_percent"
            )
        )

        x = g[
            "missing_percent"
        ].to_numpy()

        y = g[
            "mean_delta_mae"
        ].to_numpy()

        low = g[
            "delta_mae_ci_low"
        ].to_numpy()

        high = g[
            "delta_mae_ci_high"
        ].to_numpy()

        yerr = np.vstack(
            [
                y - low,
                high - y,
            ]
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=participant_colors[
                participant_id
            ],
            linestyle=pattern_styles[
                pattern
            ],
            marker=pattern_markers[
                pattern
            ],
            linewidth=2.2,
            markersize=6,
            capsize=3,
            capthick=1.2,
            label=pattern.capitalize(),
        )

    ax.axhline(
        0,
        color="0.35",
        linewidth=1,
        linestyle=":",
    )

    ax.set_title(
        participant_labels[
            participant_id
        ],
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xticks(
        [10, 25, 50, 75]
    )

    ax.set_xlabel(
        "GPS time masked (%)",
        fontsize=10,
    )

    ax.set_ylim(
        -0.35,
        global_ymax,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.8,
    )

    if i == 0:
        ax.set_ylabel(
            "Increase in depression\nprediction MAE (PHQ points)",
            fontsize=10,
        )
    else:
        ax.set_yticklabels([])

    if i == 2:
        ax.legend(
            frameon=False,
            fontsize=9,
            loc="upper left",
        )


fig.text(
    0.055,
    0.965,
    "A",
    fontsize=17,
    fontweight="bold",
)


# ------------------------------------------------------------
# PANEL B: GEOMETRY EFFECT
#
# Positive:
# contiguous missingness is more harmful than distributed.
# ------------------------------------------------------------

ax_b = fig.add_subplot(
    gs[1, 0]
)

geometry_plot = (
    geometry.copy()
)

geometry_plot[
    "label"
] = (
    geometry_plot[
        "participant_id"
    ]
    + " · "
    + geometry_plot[
        "missing_percent"
    ]
    .astype(int)
    .astype(str)
    + "%"
)

geometry_plot[
    "participant_order"
] = geometry_plot[
    "participant_id"
].map(
    {
        "LPE511": 0,
        "VTE817": 1,
        "BPH745": 2,
    }
)

geometry_plot = (
    geometry_plot.sort_values(
        [
            "participant_order",
            "missing_percent",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

y_positions = np.arange(
    len(geometry_plot)
)

for participant_id in participants:

    g = geometry_plot[
        geometry_plot[
            "participant_id"
        ].eq(participant_id)
    ]

    idx = g.index.to_numpy()

    x = g[
        "delta_mae_contiguous_minus_distributed"
    ].to_numpy()

    low = g[
        "delta_mae_geometry_ci_low"
    ].to_numpy()

    high = g[
        "delta_mae_geometry_ci_high"
    ].to_numpy()

    xerr = np.vstack(
        [
            x - low,
            high - x,
        ]
    )

    ax_b.errorbar(
        x,
        idx,
        xerr=xerr,
        fmt="o",
        color=participant_colors[
            participant_id
        ],
        markersize=6,
        capsize=3,
        linewidth=1.5,
    )

ax_b.axvline(
    0,
    color="0.3",
    linewidth=1,
    linestyle="--",
)

ax_b.set_yticks(
    y_positions
)

ax_b.set_yticklabels(
    geometry_plot[
        "label"
    ],
    fontsize=8.5,
)

ax_b.invert_yaxis()

ax_b.set_xlabel(
    "ΔMAE: contiguous − distributed\n(PHQ points)",
    fontsize=10,
)

ax_b.set_title(
    "Missingness geometry effect",
    fontsize=12,
    fontweight="bold",
)

ax_b.grid(
    axis="x",
    alpha=0.18,
)

ax_b.text(
    0.02,
    1.04,
    "B",
    transform=ax_b.transAxes,
    fontsize=17,
    fontweight="bold",
)


# ------------------------------------------------------------
# PANEL C:
# FEATURE DEGRADATION -> CLINICAL CONSEQUENCE
# ------------------------------------------------------------

ax_c = fig.add_subplot(
    gs[1, 1:]
)

for participant_id in participants:

    for pattern in [
        "distributed",
        "contiguous",
    ]:

        g = (
            summary[
                summary[
                    "participant_id"
                ].eq(participant_id)
                & summary[
                    "pattern"
                ].eq(pattern)
            ]
            .copy()
        )

        sizes = (
            35
            + g[
                "missing_percent"
            ].to_numpy()
            * 1.1
        )

        ax_c.scatter(
            100
            * g[
                "mean_feature_relative_error"
            ],
            g[
                "mean_delta_mae"
            ],
            s=sizes,
            color=participant_colors[
                participant_id
            ],
            marker=pattern_markers[
                pattern
            ],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
        )

        # connect degradation trajectory
        g = g.sort_values(
            "missing_percent"
        )

        ax_c.plot(
            100
            * g[
                "mean_feature_relative_error"
            ],
            g[
                "mean_delta_mae"
            ],
            color=participant_colors[
                participant_id
            ],
            linestyle=pattern_styles[
                pattern
            ],
            linewidth=1.4,
            alpha=0.65,
        )


ax_c.axhline(
    0,
    color="0.35",
    linewidth=1,
    linestyle=":",
)

ax_c.set_xlabel(
    "Personalized feature error (% of reference value)",
    fontsize=10,
)

ax_c.set_ylabel(
    "Increase in depression prediction MAE\n(PHQ points)",
    fontsize=10,
)

ax_c.set_title(
    "Feature degradation does not imply equal clinical impact",
    fontsize=12,
    fontweight="bold",
)

ax_c.grid(
    alpha=0.15,
)


# participant legend
participant_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=participant_colors[p],
        markeredgecolor="none",
        markersize=8,
        label=p,
    )
    for p in participants
]

legend1 = ax_c.legend(
    handles=participant_handles,
    title="Participant",
    frameon=False,
    loc="upper left",
)

ax_c.add_artist(
    legend1
)


# geometry legend
pattern_handles = [
    Line2D(
        [0],
        [0],
        marker=pattern_markers[p],
        linestyle=pattern_styles[p],
        color="0.25",
        markersize=6,
        label=p.capitalize(),
    )
    for p in [
        "distributed",
        "contiguous",
    ]
]

ax_c.legend(
    handles=pattern_handles,
    title="Missingness",
    frameon=False,
    loc="upper center",
)


ax_c.text(
    0.01,
    1.04,
    "C",
    transform=ax_c.transAxes,
    fontsize=17,
    fontweight="bold",
)


fig.suptitle(
    "Personalized propagation of GPS degradation into clinical prediction",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

fig.savefig(
    OUT_MAIN,
    dpi=350,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# RELIABILITY FINGERPRINT HEATMAP
# ============================================================

heat_rows = []

for participant_id in participants:

    for pattern in [
        "distributed",
        "contiguous",
    ]:

        g = (
            summary[
                summary[
                    "participant_id"
                ].eq(participant_id)
                & summary[
                    "pattern"
                ].eq(pattern)
            ]
            .copy()
        )

        g[
            "missing_percent_rounded"
        ] = (
            g[
                "missing_percent"
            ]
            .round()
            .astype(int)
        )

        g = g.set_index(
            "missing_percent_rounded"
        )

        row = {
            "label":
                f"{participant_id} · "
                f"{pattern.capitalize()}"
        }

        for pct in [
            10,
            25,
            50,
            75,
        ]:

            row[pct] = float(
                g.loc[
                    pct,
                    "mean_delta_mae",
                ]
            )

        heat_rows.append(
            row
        )


heat = pd.DataFrame(
    heat_rows
)

matrix = heat[
    [
        10,
        25,
        50,
        75,
    ]
].to_numpy()


# symmetric range around zero
max_abs = max(
    abs(
        np.nanmin(
            matrix
        )
    ),
    abs(
        np.nanmax(
            matrix
        )
    ),
)

if max_abs == 0:
    max_abs = 1.0


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

norm = TwoSlopeNorm(
    vmin=-max_abs,
    vcenter=0,
    vmax=max_abs,
)

im = ax.imshow(
    matrix,
    cmap="RdBu_r",
    norm=norm,
    aspect="auto",
)


ax.set_xticks(
    np.arange(4)
)

ax.set_xticklabels(
    [
        "10%",
        "25%",
        "50%",
        "75%",
    ]
)

ax.set_yticks(
    np.arange(
        len(heat)
    )
)

ax.set_yticklabels(
    heat[
        "label"
    ]
)

ax.set_xlabel(
    "GPS time masked"
)

ax.set_title(
    "Personalized GPS reliability fingerprints",
    fontsize=14,
    fontweight="bold",
)


# subtle participant separators
for y in [
    1.5,
    3.5,
]:

    ax.axhline(
        y,
        color="white",
        linewidth=3,
    )


# values inside cells
for i in range(
    matrix.shape[0]
):

    for j in range(
        matrix.shape[1]
    ):

        value = matrix[
            i,
            j,
        ]

        text_color = (
            "white"
            if abs(value)
            > 0.55 * max_abs
            else "black"
        )

        ax.text(
            j,
            i,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )


cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
)

cbar.set_label(
    "Increase in depression prediction MAE "
    "(PHQ points)"
)


fig.tight_layout()

fig.savefig(
    OUT_HEATMAP,
    dpi=350,
    bbox_inches="tight",
)

plt.close(fig)


print("=" * 90)
print("PUBLICATION FIGURES CREATED")
print("=" * 90)

print()
print("Main figure:")
print(OUT_MAIN)

print()
print("Reliability fingerprint heatmap:")
print(OUT_HEATMAP)
