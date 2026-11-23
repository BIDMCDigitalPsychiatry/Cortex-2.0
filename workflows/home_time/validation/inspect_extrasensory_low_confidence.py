from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from validate_extrasensory_home import (
    build_participant_places,
    infer_timezone,
)

from timezonefinder import TimezoneFinder


UUID = "24E40C4C-A349-4F9F-93AB-01D00FB994AF"

ROOT = Path("data/external/extrasensory")
LOCATION_DIR = ROOT / "absolute_location"
FEATURE_DIR = ROOT / "features_labels"

REFERENCE_FILE = (
    ROOT / "extrasensory_home_dominant_cluster.csv"
)

OUT = Path(
    "outputs/home_time_validation/extrasensory/"
    "participant_24E40C4C"
)
OUT.mkdir(parents=True, exist_ok=True)

MAX_GAP_MS = 60_000

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2
        * EARTH_RADIUS_M
        * np.arcsin(np.sqrt(a))
    )


# ---------------------------------------------------------
# Load ExtraSensory GPS + horizontal accuracy
# ---------------------------------------------------------

loc = pd.read_csv(
    LOCATION_DIR /
    f"{UUID}.absolute_locations.csv.gz"
)

features = pd.read_csv(
    FEATURE_DIR /
    f"{UUID}.features_labels.csv.gz",
    usecols=[
        "timestamp",
        "location:best_horizontal_accuracy",
    ],
)

data = loc.merge(
    features,
    on="timestamp",
    how="left",
    validate="one_to_one",
)

fixes = data.rename(
    columns={
        "location:best_horizontal_accuracy":
            "accuracy"
    }
)[
    [
        "timestamp",
        "latitude",
        "longitude",
        "accuracy",
    ]
].copy()

# ExtraSensory seconds -> Cortex milliseconds
fixes["timestamp"] = (
    pd.to_numeric(
        fixes["timestamp"],
        errors="coerce",
    )
    * 1000
)


# ---------------------------------------------------------
# Resolve timezone
# ---------------------------------------------------------

tf = TimezoneFinder()

timezone = infer_timezone(
    fixes,
    tf,
)

print("Participant:", UUID)
print("Timezone:", timezone)


# ---------------------------------------------------------
# Run the exact Cortex place pipeline
# ---------------------------------------------------------

(
    clean_fixes,
    assignments,
    places,
    supported_ms,
) = build_participant_places(
    fixes,
    timezone,
    MAX_GAP_MS,
)

print("Clean GPS fixes:", len(clean_fixes))
print("Persistent places:", len(places))
print(
    "Supported hours:",
    round(
        supported_ms / 3_600_000,
        2,
    ),
)


# ---------------------------------------------------------
# Independent ExtraSensory reference home
# ---------------------------------------------------------

ref = pd.read_csv(
    REFERENCE_FILE
)

ref = ref.loc[
    ref["uuid"] == UUID
].iloc[0]

reference_lat = float(
    ref["home_ref_lat"]
)

reference_lon = float(
    ref["home_ref_lon"]
)

print()
print(
    "Reference dominant fraction:",
    round(
        float(
            ref["dominant_fraction"]
        ),
        3,
    ),
)

print(
    "Reference p90:",
    round(
        float(
            ref["p90_distance_m"]
        ),
        1,
    ),
    "m",
)


# ---------------------------------------------------------
# Night-by-night occupancy: entire participant span
# ---------------------------------------------------------

starts = pd.to_numeric(
    assignments["episode_start_ms"],
    errors="coerce",
).to_numpy(float)

ends = pd.to_numeric(
    assignments["episode_end_ms"],
    errors="coerce",
).to_numpy(float)

place_ids = pd.to_numeric(
    assignments["place_candidate_id"],
    errors="coerce",
).to_numpy(int)

first_date = (
    pd.to_datetime(
        starts.min(),
        unit="ms",
        utc=True,
    )
    .tz_convert(timezone)
    .date()
)

last_date = (
    pd.to_datetime(
        ends.max(),
        unit="ms",
        utc=True,
    )
    .tz_convert(timezone)
    .date()
)

dates = pd.date_range(
    first_date,
    last_date,
    freq="D",
)

night_rows = []

for day in dates:

    date = day.date()

    night_start = pd.Timestamp(
        f"{date} 00:00:00"
    ).tz_localize(
        timezone
    )

    night_end = pd.Timestamp(
        f"{date} 06:00:00"
    ).tz_localize(
        timezone
    )

    lo = (
        night_start
        .tz_convert("UTC")
        .timestamp()
        * 1000
    )

    hi = (
        night_end
        .tz_convert("UTC")
        .timestamp()
        * 1000
    )

    overlap = np.maximum(
        0.0,
        np.minimum(
            ends,
            hi,
        )
        - np.maximum(
            starts,
            lo,
        ),
    )

    temp = pd.DataFrame(
        {
            "place_candidate_id":
                place_ids,
            "duration_ms":
                overlap,
        }
    )

    temp = temp.loc[
        temp["duration_ms"] > 0
    ]

    if temp.empty:
        continue

    sums = (
        temp
        .groupby(
            "place_candidate_id"
        )["duration_ms"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    total = float(
        sums.sum()
    )

    winner = int(
        sums.index[0]
    )

    for pid, duration in sums.items():
        night_rows.append(
            {
                "date": str(date),
                "place_candidate_id":
                    int(pid),
                "nighttime_ms":
                    float(duration),
                "nighttime_hours":
                    float(duration)
                    / 3_600_000,
                "night_fraction":
                    float(duration)
                    / total,
                "winner":
                    int(pid)
                    == winner,
            }
        )


night = pd.DataFrame(
    night_rows
)

night.to_csv(
    OUT / "night_by_night_place_occupancy.csv",
    index=False,
)


# ---------------------------------------------------------
# Aggregate nighttime occupancy by persistent place
# ---------------------------------------------------------

place_summary = (
    night
    .groupby(
        "place_candidate_id",
        as_index=False,
    )
    .agg(
        nighttime_hours=(
            "nighttime_hours",
            "sum",
        ),
        nights_present=(
            "date",
            "nunique",
        ),
        nights_won=(
            "winner",
            "sum",
        ),
    )
)

total_night_hours = (
    place_summary[
        "nighttime_hours"
    ].sum()
)

place_summary[
    "nighttime_fraction"
] = (
    place_summary[
        "nighttime_hours"
    ]
    / total_night_hours
)


# ---------------------------------------------------------
# Add Cortex place coordinates + distance to reference
# ---------------------------------------------------------

coord_cols = [
    "place_candidate_id",
    "representative_latitude",
    "representative_longitude",
]

place_summary = (
    place_summary
    .merge(
        places[
            coord_cols
        ],
        on="place_candidate_id",
        how="left",
    )
)

place_summary[
    "distance_to_reference_home_m"
] = haversine_m(
    place_summary[
        "representative_latitude"
    ].to_numpy(float),
    place_summary[
        "representative_longitude"
    ].to_numpy(float),
    reference_lat,
    reference_lon,
)

place_summary = (
    place_summary
    .sort_values(
        "nighttime_hours",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

place_summary[
    "rank"
] = (
    np.arange(
        len(place_summary)
    )
    + 1
)

place_summary[
    "place_label"
] = (
    "P"
    + place_summary[
        "rank"
    ].astype(str)
)

label_lookup = dict(
    zip(
        place_summary[
            "place_candidate_id"
        ],
        place_summary[
            "place_label"
        ],
    )
)

night[
    "place_label"
] = (
    night[
        "place_candidate_id"
    ].map(
        label_lookup
    )
)

place_summary.to_csv(
    OUT / "persistent_place_nighttime_summary.csv",
    index=False,
)


# ---------------------------------------------------------
# Console audit
# ---------------------------------------------------------

print()
print("=" * 100)
print("NIGHTTIME PLACE DISTRIBUTION")
print("=" * 100)

print(
    place_summary[
        [
            "place_label",
            "place_candidate_id",
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
print("NIGHT-BY-NIGHT WINNERS")
print("=" * 100)

winners = (
    night.loc[
        night["winner"]
    ][
        [
            "date",
            "place_label",
            "nighttime_hours",
            "night_fraction",
        ]
    ]
)

print(
    winners.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------

fig = plt.figure(
    figsize=(11, 11)
)

gs = fig.add_gridspec(
    3,
    1,
    height_ratios=[
        1.35,
        1.0,
        1.15,
    ],
    hspace=0.48,
)


# PANEL A — night-by-night winner
ax = fig.add_subplot(
    gs[0]
)

winner_table = (
    night.loc[
        night["winner"]
    ]
    .copy()
)

labels = (
    place_summary[
        "place_label"
    ].tolist()
)

y_lookup = {
    label: i
    for i, label in enumerate(
        labels
    )
}

for row in winner_table.itertuples():
    ax.scatter(
        pd.Timestamp(
            row.date
        ),
        y_lookup[
            row.place_label
        ],
        s=70,
    )

ax.set_yticks(
    list(
        y_lookup.values()
    )
)

ax.set_yticklabels(
    list(
        y_lookup.keys()
    )
)

ax.invert_yaxis()

ax.set_ylabel(
    "Dominant nighttime place"
)

ax.set_title(
    "A. Night-by-night dominant place (00:00–06:00)"
)

ax.grid(
    axis="x",
    alpha=0.2,
)


# PANEL B — cumulative nighttime occupancy
ax = fig.add_subplot(
    gs[1]
)

show = place_summary.copy()

ax.bar(
    show[
        "place_label"
    ],
    100
    * show[
        "nighttime_fraction"
    ],
)

ax.axhline(
    50,
    linestyle="--",
    linewidth=1,
)

ax.set_ylabel(
    "Share of observed nighttime occupancy (%)"
)

ax.set_xlabel(
    "Persistent Cortex place"
)

ax.set_title(
    "B. Total nighttime occupancy across full participant span"
)

for i, row in show.iterrows():
    ax.text(
        i,
        100
        * row[
            "nighttime_fraction"
        ]
        + 1,
        f"{100 * row['nighttime_fraction']:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )


# PANEL C — geography relative to reference home
ax = fig.add_subplot(
    gs[2]
)

lat0 = reference_lat
lon0 = reference_lon

meters_per_lat = 111_320.0
meters_per_lon = (
    111_320.0
    * np.cos(
        np.radians(
            lat0
        )
    )
)

x = (
    place_summary[
        "representative_longitude"
    ]
    - lon0
) * meters_per_lon

y = (
    place_summary[
        "representative_latitude"
    ]
    - lat0
) * meters_per_lat

sizes = (
    80
    + 500
    * place_summary[
        "nighttime_fraction"
    ]
)

ax.scatter(
    x,
    y,
    s=sizes,
)

ax.scatter(
    [0],
    [0],
    marker="*",
    s=220,
    label="ExtraSensory reference home",
)

for xx, yy, label in zip(
    x,
    y,
    place_summary[
        "place_label"
    ],
):
    ax.annotate(
        label,
        (xx, yy),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.axhline(
    0,
    linewidth=0.5,
)

ax.axvline(
    0,
    linewidth=0.5,
)

ax.set_xlabel(
    "East / west of reference home (m)"
)

ax.set_ylabel(
    "North / south of reference home (m)"
)

ax.set_title(
    "C. Persistent places relative to independent reference home"
)

ax.legend(
    frameon=False
)

ax.set_aspect(
    "equal",
    adjustable="datalim",
)


fig.suptitle(
    UUID,
    fontsize=13,
)

figure_path = (
    OUT /
    "nighttime_place_investigation.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("Saved:")
print(
    OUT /
    "night_by_night_place_occupancy.csv"
)
print(
    OUT /
    "persistent_place_nighttime_summary.csv"
)
print(
    figure_path
)
