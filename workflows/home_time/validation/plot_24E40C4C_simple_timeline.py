from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Recover P1/P2/... labels if needed
if "place_label" not in night.columns:
    night = night.merge(
        places[
            ["place_candidate_id", "place_label"]
        ],
        on="place_candidate_id",
        how="left",
        validate="many_to_one",
    )

night["date"] = pd.to_datetime(night["date"])

# Real spatial interpretation from this participant:
# P3 = ExtraSensory reference-home region in San Diego
# P1/P2/P4/P5/P6 = LA region
night["region"] = np.where(
    night["place_label"].eq("P3"),
    "San Diego",
    "LA",
)

daily = (
    night.groupby(
        ["date", "region"],
        as_index=False
    )["nighttime_hours"]
    .sum()
)

wide = (
    daily.pivot(
        index="date",
        columns="region",
        values="nighttime_hours"
    )
    .fillna(0)
)

for c in ["San Diego", "LA"]:
    if c not in wide.columns:
        wide[c] = 0.0

# Full calendar span so missing nights are visible
calendar = pd.date_range(
    wide.index.min(),
    wide.index.max(),
    freq="D"
)

timeline = pd.DataFrame(
    index=calendar
)

timeline = timeline.join(wide)

timeline[["San Diego", "LA"]] = (
    timeline[["San Diego", "LA"]]
    .fillna(0)
)

timeline["observed"] = (
    timeline["San Diego"]
    + timeline["LA"]
) > 0

timeline["region"] = "No nighttime data"

timeline.loc[
    timeline["San Diego"] > timeline["LA"],
    "region"
] = "San Diego"

timeline.loc[
    timeline["LA"] > timeline["San Diego"],
    "region"
] = "LA"

# ------------------------------------------------------------
# Real aggregate values
# ------------------------------------------------------------

sd_hours = wide["San Diego"].sum()
la_hours = wide["LA"].sum()
total_hours = sd_hours + la_hours

sd_pct = 100 * sd_hours / total_hours
la_pct = 100 * la_hours / total_hours

largest_single_place_pct = (
    places["nighttime_fraction"].max() * 100
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(9, 6)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1.25, 1],
    hspace=0.55,
)

# ============================================================
# A. TIMELINE
# ============================================================

ax = fig.add_subplot(gs[0])

x = np.arange(len(timeline))

y_map = {
    "San Diego": 1,
    "LA": 0,
    "No nighttime data": 0.5,
}

for i, (date, row) in enumerate(
    timeline.iterrows()
):
    region = row["region"]

    if region == "No nighttime data":
        ax.scatter(
            i,
            y_map[region],
            s=55,
            marker="x",
        )

        ax.text(
            i,
            y_map[region] + 0.12,
            "No data",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    else:
        hours = (
            row["San Diego"]
            if region == "San Diego"
            else row["LA"]
        )

        ax.scatter(
            i,
            y_map[region],
            s=180,
        )

        ax.text(
            i,
            y_map[region] + 0.13,
            f"{hours:.2f} h",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.plot(
    [0, len(timeline) - 1],
    [1, 1],
    linewidth=0.7,
    alpha=0.25,
)

ax.plot(
    [0, len(timeline) - 1],
    [0, 0],
    linewidth=0.7,
    alpha=0.25,
)

ax.set_yticks(
    [0, 1]
)

ax.set_yticklabels(
    [
        "LA region\n(~165 km away)",
        "Reference-home region\n(San Diego)",
    ]
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        d.strftime("%b %d")
        for d in timeline.index
    ]
)

ax.set_ylim(
    -0.35,
    1.35,
)

ax.set_title(
    "A. Observed nighttime location by calendar night",
    loc="left",
    fontweight="bold",
)

ax.spines[
    ["top", "right", "left"]
].set_visible(False)

ax.tick_params(
    axis="y",
    length=0,
)

# ============================================================
# B. REGIONAL OCCUPANCY
# ============================================================

ax = fig.add_subplot(gs[1])

ax.barh(
    ["Observed nighttime"],
    [sd_pct],
    left=0,
)

ax.barh(
    ["Observed nighttime"],
    [la_pct],
    left=sd_pct,
)

ax.text(
    sd_pct / 2,
    0,
    f"San Diego\n{sd_pct:.1f}%",
    ha="center",
    va="center",
    fontsize=11,
)

ax.text(
    sd_pct + la_pct / 2,
    0,
    f"LA region\n{la_pct:.1f}%",
    ha="center",
    va="center",
    fontsize=11,
)

ax.set_xlim(
    0,
    100,
)

ax.set_xlabel(
    "Share of all observed nighttime occupancy (%)"
)

ax.set_title(
    "B. Most nighttime observation occurred during the LA stay",
    loc="left",
    fontweight="bold",
)

ax.text(
    0,
    -0.75,
    (
        f"Cortex largest individual place = "
        f"{largest_single_place_pct:.1f}% "
        f"(<50% home-confidence threshold)"
    ),
    ha="left",
    va="center",
    fontsize=10,
)

ax.spines[
    ["top", "right", "left"]
].set_visible(False)

ax.tick_params(
    axis="y",
    length=0,
)

fig.suptitle(
    "ExtraSensory participant 24E40C4C",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

out = (
    ROOT /
    "nighttime_location_timeline_simple.png"
)

fig.savefig(
    out,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(out)
print()
print(
    f"San Diego: {sd_hours:.2f} h "
    f"({sd_pct:.1f}%)"
)
print(
    f"LA region: {la_hours:.2f} h "
    f"({la_pct:.1f}%)"
)
print(
    f"Largest individual Cortex place: "
    f"{largest_single_place_pct:.1f}%"
)
