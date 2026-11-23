from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


UUID = "24E40C4C-A349-4F9F-93AB-01D00FB994AF"

ROOT = Path(
    "outputs/home_time_validation/extrasensory/"
    "participant_24E40C4C"
)

night = pd.read_csv(
    ROOT / "night_by_night_place_occupancy.csv"
)

places = pd.read_csv(
    ROOT / "persistent_place_nighttime_summary.csv"
)

# The night-level CSV was saved before place_label was attached,
# so recover the real Cortex place labels from the place summary.
night = night.merge(
    places[
        [
            "place_candidate_id",
            "place_label",
        ]
    ],
    on="place_candidate_id",
    how="left",
    validate="many_to_one",
)

night["date"] = pd.to_datetime(
    night["date"]
)

# Keep places ordered by total nighttime occupancy.
places = places.sort_values(
    "nighttime_fraction",
    ascending=False,
).reset_index(drop=True)

place_order = places["place_label"].tolist()

# ------------------------------------------------------------------
# Night × place matrix
# ------------------------------------------------------------------

matrix = (
    night.pivot_table(
        index="date",
        columns="place_label",
        values="nighttime_hours",
        aggfunc="sum",
        fill_value=0,
    )
    .reindex(
        columns=place_order,
        fill_value=0,
    )
)

night_totals = matrix.sum(axis=1)

# Percentage of each night's observed nighttime occupancy.
percent = (
    matrix.div(
        night_totals,
        axis=0,
    )
    * 100
)

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    gridspec_kw={
        "height_ratios": [1.4, 1.0]
    },
)

# ================================================================
# A. NIGHT-BY-NIGHT DISTRIBUTION
# ================================================================

ax = axes[0]

bottom = np.zeros(
    len(percent)
)

x = np.arange(
    len(percent)
)

for place in place_order:

    values = percent[
        place
    ].to_numpy()

    ax.bar(
        x,
        values,
        bottom=bottom,
        label=place,
        width=0.72,
    )

    bottom += values


ax.set_xticks(x)

ax.set_xticklabels(
    [
        d.strftime("%b %d")
        for d in percent.index
    ]
)

ax.set_ylim(
    0,
    100,
)

ax.set_ylabel(
    "Observed nighttime occupancy (%)"
)

ax.set_title(
    "A. Where was the participant during each observed night?"
)

ax.legend(
    title="Persistent place",
    frameon=False,
    ncol=min(
        6,
        len(place_order),
    ),
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
)

# Show actual observed nighttime hours above each night.
for i, hours in enumerate(
    night_totals
):
    ax.text(
        i,
        102,
        f"{hours:.1f} h",
        ha="center",
        va="bottom",
        fontsize=9,
    )


# ================================================================
# B. OVERALL NIGHTTIME OCCUPANCY
# ================================================================

ax = axes[1]

shares = (
    places[
        "nighttime_fraction"
    ]
    * 100
)

bars = ax.bar(
    places[
        "place_label"
    ],
    shares,
)

ax.axhline(
    50,
    linestyle="--",
    linewidth=1,
)

ax.set_ylabel(
    "Share of all observed nighttime occupancy (%)"
)

ax.set_xlabel(
    "Persistent Cortex place"
)

ax.set_title(
    "B. Overall nighttime occupancy and distance from reference home"
)

ymax = max(
    55,
    shares.max()
    + 12,
)

ax.set_ylim(
    0,
    ymax,
)

for bar, (_, row) in zip(
    bars,
    places.iterrows(),
):

    share = (
        row[
            "nighttime_fraction"
        ]
        * 100
    )

    distance_m = row[
        "distance_to_reference_home_m"
    ]

    if distance_m < 1000:
        distance_text = (
            f"{distance_m:.0f} m from reference home"
        )
    else:
        distance_text = (
            f"{distance_m / 1000:.1f} km from reference home"
        )

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height()
        + 1,
        f"{share:.1f}%\n{distance_text}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


fig.suptitle(
    "ExtraSensory participant 24E40C4C: nighttime location structure",
    fontsize=14,
)

fig.tight_layout()

out = (
    ROOT /
    "nighttime_place_distribution_clear.png"
)

fig.savefig(
    out,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("=" * 100)
print("PLACE SUMMARY")
print("=" * 100)

print(
    places[
        [
            "place_label",
            "nighttime_hours",
            "nighttime_fraction",
            "nights_present",
            "nights_won",
            "distance_to_reference_home_m",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

print()
print("=" * 100)
print("NIGHT-BY-NIGHT OCCUPANCY")
print("=" * 100)

print(
    matrix.to_string(
        float_format=lambda x: f"{x:.2f}",
    )
)

print()
print("Saved:")
print(out)
