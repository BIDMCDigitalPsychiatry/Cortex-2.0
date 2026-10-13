from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTDIR = Path("figures")
OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

PNG = (
    OUTDIR
    / "gps_nearby_summary_figure.png"
)

PDF = (
    OUTDIR
    / "gps_nearby_summary_figure.pdf"
)


# ============================================================
# PANEL A DATA
# ============================================================

groups = [
    "No onset",
    "GPS-specific onset",
    "Shared onset",
]

states = [
    "None",
    "Wi-Fi only",
    "Bluetooth only",
    "Wi-Fi + Bluetooth",
]

counts = {
    "No onset": {
        "None": 42755,
        "Wi-Fi only": 122448,
        "Bluetooth only": 669,
        "Wi-Fi + Bluetooth": 15409,
    },

    "GPS-specific onset": {
        "None": 684,
        "Wi-Fi only": 185,
        "Bluetooth only": 10,
        "Wi-Fi + Bluetooth": 7,
    },

    "Shared onset": {
        "None": 506,
        "Wi-Fi only": 1304,
        "Bluetooth only": 79,
        "Wi-Fi + Bluetooth": 502,
    },
}


# ============================================================
# PANEL B DATA
# ============================================================

forest_rows = [
    {
        "label":
            "GPS-specific onset — Wi-Fi only",
        "or":
            0.1774,
        "low":
            0.0768,
        "high":
            0.4098,
        "p":
            0.0001,
    },
    {
        "label":
            "GPS-specific onset — Bluetooth only",
        "or":
            0.5884,
        "low":
            0.2367,
        "high":
            1.4629,
        "p":
            0.2538,
    },
    {
        "label":
            "GPS-specific onset — Wi-Fi + Bluetooth",
        "or":
            0.1546,
        "low":
            0.0380,
        "high":
            0.6287,
        "p":
            0.0091,
    },
    {
        "label":
            "Shared onset — Wi-Fi only",
        "or":
            2.9090,
        "low":
            2.0098,
        "high":
            4.2104,
        "p":
            0.0000,
    },
    {
        "label":
            "Shared onset — Bluetooth only",
        "or":
            1.1402,
        "low":
            0.6344,
        "high":
            2.0493,
        "p":
            0.6609,
    },
    {
        "label":
            "Shared onset — Wi-Fi + Bluetooth",
        "or":
            4.0058,
        "low":
            2.2620,
        "high":
            7.0939,
        "p":
            0.0000,
    },
]


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(12, 13)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[
        1,
        1.25,
    ],
    hspace=0.28,
)


# ============================================================
# PANEL A
# ============================================================

ax1 = fig.add_subplot(
    gs[0, 0]
)

y = np.arange(
    len(groups)
)

left = np.zeros(
    len(groups),
    dtype=float,
)


for state in states:

    values = []

    for group in groups:

        total = sum(
            counts[group].values()
        )

        pct = (
            100
            * counts[group][state]
            / total
        )

        values.append(
            pct
        )

    bars = ax1.barh(
        y,
        values,
        left=left,
        label=state,
    )

    for rect, value in zip(
        bars,
        values,
    ):

        if value >= 8:

            ax1.text(
                (
                    rect.get_x()
                    + rect.get_width()
                    / 2
                ),
                (
                    rect.get_y()
                    + rect.get_height()
                    / 2
                ),
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
            )

    left += np.asarray(
        values
    )


ax1.set_yticks(
    y
)

ax1.set_yticklabels(
    groups
)

ax1.set_xlim(
    0,
    100,
)

ax1.set_xlabel(
    "Percent of risk hours / onsets"
)

ax1.set_title(
    "A. Nearby-device state distribution",
    loc="left",
    fontweight="bold",
)

ax1.legend(
    loc="lower right",
    frameon=True,
)


# ============================================================
# PANEL B
# ============================================================

ax2 = fig.add_subplot(
    gs[1, 0]
)


positions = [
    6,
    5,
    4,
    2.5,
    1.5,
    0.5,
]


for row, ypos in zip(
    forest_rows,
    positions,
):

    or_value = row["or"]
    low = row["low"]
    high = row["high"]

    ax2.errorbar(
        or_value,
        ypos,
        xerr=[
            [
                or_value - low
            ],
            [
                high - or_value
            ],
        ],
        fmt="o",
        capsize=4,
    )


ax2.axvline(
    1,
    linestyle="--",
    linewidth=1.2,
)

ax2.set_xscale(
    "log"
)

ax2.set_xlim(
    0.03,
    12,
)

ax2.set_yticks(
    positions
)

ax2.set_yticklabels(
    [
        row["label"]
        for row in forest_rows
    ]
)

ax2.set_xlabel(
    "Odds ratio "
    "(reference = no nearby evidence)"
)

ax2.set_title(
    "B. Nearby-device state model",
    loc="left",
    fontweight="bold",
)


for row, ypos in zip(
    forest_rows,
    positions,
):

    if row["p"] < 0.001:
        p_text = "p<0.001"
    else:
        p_text = (
            f"p={row['p']:.3f}"
        )

    text = (
        f"OR {row['or']:.2f} "
        f"({row['low']:.2f}, "
        f"{row['high']:.2f}), "
        f"{p_text}"
    )

    ax2.text(
        11.2,
        ypos,
        text,
        va="center",
        ha="right",
        fontsize=9,
    )


fig.suptitle(
    "Nearby-device context and GPS missingness onset",
    fontsize=16,
    fontweight="bold",
    y=0.985,
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.97,
    ]
)


fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)


print(
    "Saved:",
    PNG,
)

print(
    "Saved:",
    PDF,
)
