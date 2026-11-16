from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/"
    "home_reference_stability/AMO327"
)

INPUT = (
    ROOT
    / "AMO327_home_threshold_sensitivity.csv"
)

OUT_PNG = (
    ROOT
    / "AMO327_home_threshold_heatmap.png"
)

OUT_PDF = (
    ROOT
    / "AMO327_home_threshold_heatmap.pdf"
)


df = pd.read_csv(
    INPUT
)

night_hours = [
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
]

valid_nights = [
    2,
    3,
    4,
    5,
    7,
]

calibration_days = [
    7,
    10,
    14,
    21,
    30,
    48,
]


plt.rcParams[
    "font.family"
] = "Arial"

plt.rcParams[
    "font.size"
] = 9.5


fig, axes = plt.subplots(
    1,
    5,
    figsize=(
        17,
        4.8,
    ),
    sharey=True,
)


# Use a common scale across all panels
finite = df[
    "distance_to_reference_home_m"
].dropna()

vmin = 0
vmax = finite.max()


for ax, h in zip(
    axes,
    night_hours,
):

    d = df[
        df[
            "min_night_observation_hours"
        ].eq(h)
    ].copy()

    matrix = np.full(
        (
            len(valid_nights),
            len(calibration_days),
        ),
        np.nan,
    )

    status = np.full(
        matrix.shape,
        "",
        dtype=object,
    )

    for i, n_nights in enumerate(
        valid_nights
    ):

        for j, n_days in enumerate(
            calibration_days
        ):

            row = d[
                d[
                    "min_valid_nights"
                ].eq(n_nights)
                &
                d[
                    "calibration_days"
                ].eq(n_days)
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            if (
                row[
                    "home_status"
                ]
                == "observed"
            ):

                matrix[
                    i,
                    j
                ] = row[
                    "distance_to_reference_home_m"
                ]

                status[
                    i,
                    j
                ] = (
                    f"{row['distance_to_reference_home_m']:.2f}"
                )

            else:

                status[
                    i,
                    j
                ] = "X"


    im = ax.imshow(
        matrix,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="Greys",
    )

    ax.set_xticks(
        np.arange(
            len(
                calibration_days
            )
        )
    )

    ax.set_xticklabels(
        calibration_days
    )

    ax.set_yticks(
        np.arange(
            len(
                valid_nights
            )
        )
    )

    ax.set_yticklabels(
        valid_nights
    )

    ax.set_xlabel(
        "Calibration days"
    )

    ax.set_title(
        f"≥ {h:g} h/night"
    )


    for i in range(
        len(valid_nights)
    ):

        for j in range(
            len(calibration_days)
        ):

            txt = status[
                i,
                j
            ]

            if not txt:
                continue

            if txt == "X":

                ax.text(
                    j,
                    i,
                    "No\ninference",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )

            else:

                value = matrix[
                    i,
                    j
                ]

                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                )


axes[0].set_ylabel(
    "Minimum valid nights"
)


cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="horizontal",
    fraction=0.055,
    pad=0.16,
)

cbar.set_label(
    "Distance to 48-day reference home (m)"
)


fig.suptitle(
    (
        "AMO327: sensitivity of inferred home location "
        "to nighttime evidence requirements"
    ),
    fontsize=14,
)

fig.text(
    0.5,
    0.02,
    (
        "Numbers show spatial deviation from the 48-day reference home; "
        "X indicates that available evidence was insufficient to infer home."
    ),
    ha="center",
    fontsize=9,
)

fig.subplots_adjust(
    top=0.82,
    bottom=0.28,
    wspace=0.10,
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

plt.close(fig)

print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
