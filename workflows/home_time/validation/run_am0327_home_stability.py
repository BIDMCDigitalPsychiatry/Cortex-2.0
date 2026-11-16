from pathlib import Path
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd


PID = "AMO327"

ROOT = Path("/shared/Digital Clinic/Cortex-2.0")

SOURCE = (
    ROOT
    / "workflows"
    / "build_weekly_home_time.py"
)

HOME_ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time"
)

DAILY_EXISTING = (
    HOME_ROOT
    / "daily_home_time.csv"
)

OUT_ROOT = (
    HOME_ROOT
    / "home_reference_stability"
    / PID
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DETERMINE AMO327'S TRUE AVAILABLE CALENDAR SPAN
# ============================================================

daily = pd.read_csv(
    DAILY_EXISTING,
    low_memory=False,
)

d = daily[
    daily["participant_id"]
    .astype(str)
    .eq(PID)
].copy()

if d.empty:
    raise RuntimeError(
        f"{PID} not found in {DAILY_EXISTING}"
    )

d["date"] = pd.to_datetime(
    d["date"]
)

first_date = d["date"].min()
last_date = d["date"].max()

full_span_days = int(
    (last_date - first_date).days
    + 1
)

print("=" * 90)
print("AMO327 HOME-INFERENCE STABILITY")
print("=" * 90)

print(
    "First available date:",
    first_date.date(),
)

print(
    "Last available date:",
    last_date.date(),
)

print(
    "Full calendar span:",
    full_span_days,
    "days",
)


WINDOWS = [
    3,
    5,
    7,
    10,
    14,
    21,
    30,
    full_span_days,
]

WINDOWS = list(
    dict.fromkeys(WINDOWS)
)


# ============================================================
# PREPARE TEMPORARY PRODUCTION RUNS
# ============================================================

original = SOURCE.read_text()

for n_days in WINDOWS:

    label = (
        "full"
        if n_days == full_span_days
        else f"days_{n_days}"
    )

    run_dir = (
        OUT_ROOT
        / label
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 90)
    print(
        f"RUNNING {PID}: {label}"
    )
    print("=" * 90)

    s = original


    # --------------------------------------------------------
    # Change only HOME_REFERENCE_DAYS
    # --------------------------------------------------------

    s, count = re.subn(
        r"HOME_REFERENCE_DAYS\s*=\s*\d+",
        f"HOME_REFERENCE_DAYS = {n_days}",
        s,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            "Could not patch HOME_REFERENCE_DAYS."
        )


    # --------------------------------------------------------
    # Redirect production home-time output root
    # --------------------------------------------------------

    replacement = (
        '/ "home_time"\n'
        '    / "home_reference_stability"\n'
        f'    / "{PID}"\n'
        f'    / "{label}"'
    )

    s, count = re.subn(
        r'/\s*"home_time"',
        replacement,
        s,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            "Could not redirect output directory."
        )


    # --------------------------------------------------------
    # Restrict participant loop to AMO327 only.
    #
    # Find the production loop and inject a guard immediately
    # after the loop header.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Restrict the exact production groupby loop to AMO327.
    #
    # Production:
    #   for participant_id, participant_raw in raw.groupby(...)
    # --------------------------------------------------------

    loop_text = "for participant_id, participant_raw in raw.groupby("

    idx = s.find(loop_text)

    if idx == -1:
        raise RuntimeError(
            "Could not locate the production raw.groupby participant loop."
        )

    # Find the colon ending the multiline for-loop header.
    line_start = s.rfind("\n", 0, idx) + 1

    pos = idx
    paren_balance = 0
    header_end = None

    while pos < len(s):

        ch = s[pos]

        if ch == "(":
            paren_balance += 1
        elif ch == ")":
            paren_balance -= 1
        elif ch == ":" and paren_balance == 0:
            header_end = pos
            break

        pos += 1

    if header_end is None:
        raise RuntimeError(
            "Could not determine end of participant groupby loop header."
        )

    indent = s[
        line_start:
        line_start + (
            len(s[line_start:idx])
            - len(s[line_start:idx].lstrip())
        )
    ]

    guard = (
        "\n"
        + indent
        + "    if str(participant_id) != "
        + repr(PID)
        + ":\n"
        + indent
        + "        continue"
    )

    s = (
        s[:header_end + 1]
        + guard
        + s[header_end + 1:]
    )


    # --------------------------------------------------------
    # Temporary executable
    # --------------------------------------------------------

    tmp = Path(
        f"/tmp/build_home_{PID}_{label}.py"
    )

    tmp.write_text(s)

    subprocess.run(
        [
            sys.executable,
            str(tmp),
        ],
        cwd=str(ROOT),
        env={
            **os.environ,
            "PYTHONPATH": "src",
        },
        check=True,
    )


# ============================================================
# COLLECT HOME ANCHORS
# ============================================================

rows = []

for n_days in WINDOWS:

    label = (
        "full"
        if n_days == full_span_days
        else f"days_{n_days}"
    )

    anchor_path = (
        OUT_ROOT
        / label
        / "participant_home_anchors.csv"
    )

    if not anchor_path.exists():

        print(
            "Missing:",
            anchor_path,
        )

        continue

    x = pd.read_csv(
        anchor_path,
        low_memory=False,
    )

    x = x[
        x["participant_id"]
        .astype(str)
        .eq(PID)
    ]

    if x.empty:
        continue

    r = x.iloc[0].to_dict()

    r["calibration_days"] = n_days
    r["calibration_label"] = label

    rows.append(r)


anchors = pd.DataFrame(
    rows
)

if anchors.empty:
    raise RuntimeError(
        "No AMO327 home anchors were produced."
    )


# ============================================================
# IDENTIFY HOME COORDINATE COLUMNS
# ============================================================

lat_candidates = [
    "representative_latitude",
    "home_latitude",
    "home_center_latitude",
]

lon_candidates = [
    "representative_longitude",
    "home_longitude",
    "home_center_longitude",
]

lat_col = next(
    (
        c
        for c in lat_candidates
        if c in anchors.columns
    ),
    None,
)

lon_col = next(
    (
        c
        for c in lon_candidates
        if c in anchors.columns
    ),
    None,
)

if lat_col is None or lon_col is None:

    print(
        "\nAvailable anchor columns:"
    )

    print(
        anchors.columns.tolist()
    )

    raise RuntimeError(
        "Could not identify home coordinate columns."
    )


# ============================================================
# FULL-STUDY REFERENCE
# ============================================================

ref = anchors[
    anchors[
        "calibration_label"
    ].eq("full")
]

if ref.empty:
    raise RuntimeError(
        "Full-study reference home was not produced."
    )

ref = ref.iloc[0]

ref_lat = float(
    ref[lat_col]
)

ref_lon = float(
    ref[lon_col]
)


# ============================================================
# HAVERSINE DISTANCE TO FULL-STUDY HOME
# ============================================================

def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):

    R = 6371000.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dp = np.radians(
        lat2 - lat1
    )

    dl = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dp / 2) ** 2
        +
        np.cos(p1)
        * np.cos(p2)
        * np.sin(dl / 2) ** 2
    )

    return (
        2
        * R
        * np.arcsin(
            np.sqrt(a)
        )
    )


distances = []

for _, row in anchors.iterrows():

    if (
        pd.isna(row[lat_col])
        or pd.isna(row[lon_col])
    ):
        distances.append(
            np.nan
        )
        continue

    distances.append(
        haversine_m(
            float(row[lat_col]),
            float(row[lon_col]),
            ref_lat,
            ref_lon,
        )
    )

anchors[
    "distance_to_full_home_m"
] = distances


# ============================================================
# SAME WINNING PLACE ID?
#
# Candidate IDs may not necessarily be comparable across
# separately fitted place models, so coordinate distance is
# the primary convergence metric.
# ============================================================

anchors = anchors.sort_values(
    "calibration_days"
)

summary_cols = [
    "calibration_days",
    "calibration_label",
]

for c in [
    "home_status",
    "home_reason",
    "home_confidence",
    "home_n_valid_nights",
    "home_n_nights_at_home",
    "home_nighttime_fraction",
    lat_col,
    lon_col,
    "distance_to_full_home_m",
]:

    if c in anchors.columns:
        summary_cols.append(c)


summary = anchors[
    summary_cols
].copy()

SUMMARY_OUT = (
    OUT_ROOT
    / "AMO327_home_reference_stability.csv"
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


print()
print("=" * 90)
print("AMO327 HOME STABILITY SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    SUMMARY_OUT,
)
