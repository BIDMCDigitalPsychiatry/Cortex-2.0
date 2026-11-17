#!/usr/bin/env python3

"""
Participant-checkpointed runner for
build_gps_clinical_weekly_features.py.

The original Cortex 2.0 feature builder is not modified.

Each participant is executed independently and its outputs are
persisted before moving to the next participant.

A restart skips participants whose checkpoint is complete.

Canonical combined outputs are replaced ONLY after all participants
complete successfully.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

REPO = Path(
    "/shared/Digital Clinic/Cortex-2.0"
)

ROOT = Path(
    "/shared/Digital Clinic/"
    "cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

ORIGINAL_SCRIPT = (
    REPO
    / "workflows/"
      "build_gps_clinical_weekly_features.py"
)

REQUESTED_PATH = (
    ROOT
    / "gps_clinical_weekly_requested_days.parquet"
)

RAW_PATH = (
    ROOT
    / "gps_clinical_weekly_raw_points.parquet"
)

DAILY_NAME = (
    "gps_clinical_weekly_daily_features.parquet"
)

WEEKLY_NAME = (
    "gps_clinical_weekly_features.parquet"
)

SHIM_DIR = (
    REPO
    / "workflows/"
      "_gps_clinical_checkpoint_shim"
)


# ============================================================
# VERIFY INPUTS
# ============================================================

for path in [
    ORIGINAL_SCRIPT,
    REQUESTED_PATH,
    RAW_PATH,
    SHIM_DIR / "sitecustomize.py",
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# SOURCE VERSION
# ============================================================

SOURCE_SHA256 = (
    hashlib.sha256(
        ORIGINAL_SCRIPT.read_bytes()
    )
    .hexdigest()
)

RUN_ID = (
    SOURCE_SHA256[:12]
)

CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints/"
      "gps_clinical_weekly_features"
    / RUN_ID
)

PARTICIPANT_ROOT = (
    CHECKPOINT_ROOT
    / "participants"
)

MANIFEST_PATH = (
    CHECKPOINT_ROOT
    / "checkpoint_manifest.csv"
)

PARTICIPANT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def utc_now():

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def safe_token(value):

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def atomic_json(
    payload,
    path,
):

    path = Path(
        path
    )

    tmp = (
        path.parent
        / (
            path.name
            + ".tmp"
        )
    )

    with open(
        tmp,
        "w",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
        )

    os.replace(
        tmp,
        path,
    )


def parquet_rows(path):

    path = Path(
        path
    )

    if not path.exists():
        return None

    return int(
        pq.ParquetFile(
            path
        )
        .metadata
        .num_rows
    )


def load_status(
    participant_dir,
):

    path = (
        participant_dir
        / "status.json"
    )

    if not path.exists():
        return None

    try:

        with open(path) as f:
            return json.load(f)

    except Exception:
        return None


def checkpoint_complete(
    participant_dir,
):

    status = load_status(
        participant_dir
    )

    if not status:
        return False

    if (
        status.get("status")
        != "complete"
    ):
        return False

    if (
        status.get(
            "source_sha256"
        )
        != SOURCE_SHA256
    ):
        return False

    daily = (
        participant_dir
        / DAILY_NAME
    )

    weekly = (
        participant_dir
        / WEEKLY_NAME
    )

    return (
        daily.exists()
        and weekly.exists()
    )


def refresh_manifest():

    records = []

    for status_path in sorted(
        PARTICIPANT_ROOT.glob(
            "*/status.json"
        )
    ):

        try:

            with open(
                status_path
            ) as f:

                records.append(
                    json.load(f)
                )

        except Exception:
            pass

    if not records:
        return

    df = pd.DataFrame(
        records
    )

    if "participant_index" in df.columns:

        df = df.sort_values(
            "participant_index"
        )

    tmp = (
        MANIFEST_PATH.parent
        / (
            MANIFEST_PATH.name
            + ".tmp"
        )
    )

    df.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        MANIFEST_PATH,
    )


def atomic_parquet(
    df,
    destination,
):

    destination = Path(
        destination
    )

    tmp = (
        destination.parent
        / (
            destination.stem
            + ".tmp"
            + destination.suffix
        )
    )

    df.to_parquet(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        destination,
    )


def combine_participant_output(
    participant_dirs,
    filename,
    destination,
):

    frames = []

    for participant_dir in participant_dirs:

        path = (
            participant_dir
            / filename
        )

        if not path.exists():

            raise RuntimeError(
                f"Missing completed shard: {path}"
            )

        frames.append(
            pd.read_parquet(
                path
            )
        )

    if not frames:

        raise RuntimeError(
            f"No shards found for {filename}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    sort_columns = [
        column
        for column in [
            "participant_id",
            "clinical_window_id",
            "date",
        ]
        if column in combined.columns
    ]

    if sort_columns:

        combined = (
            combined
            .sort_values(
                sort_columns
            )
            .reset_index(
                drop=True
            )
        )

    # Audit duplicates without silently dropping anything.
    if (
        filename == DAILY_NAME
        and {
            "clinical_window_id",
            "date",
        }.issubset(
            combined.columns
        )
    ):

        n_duplicates = int(
            combined.duplicated(
                [
                    "clinical_window_id",
                    "date",
                ]
            )
            .sum()
        )

        if n_duplicates:

            raise RuntimeError(
                "Duplicate clinical-window/day rows "
                f"in combined daily output: {n_duplicates}"
            )

    if (
        filename == WEEKLY_NAME
        and "clinical_window_id"
        in combined.columns
    ):

        n_duplicates = int(
            combined[
                "clinical_window_id"
            ]
            .duplicated()
            .sum()
        )

        if n_duplicates:

            raise RuntimeError(
                "Duplicate clinical_window_id rows "
                f"in combined weekly output: {n_duplicates}"
            )

    atomic_parquet(
        combined,
        destination,
    )

    return combined


# ============================================================
# PARTICIPANT MANIFEST
# ============================================================

requested = pd.read_parquet(
    REQUESTED_PATH
)

required_columns = {
    "participant_id",
    "mindlamp_uid",
}

missing = (
    required_columns
    - set(
        requested.columns
    )
)

if missing:

    raise RuntimeError(
        "Requested-days file is missing: "
        + ", ".join(
            sorted(
                missing
            )
        )
    )


participants = (
    requested[
        [
            "participant_id",
            "mindlamp_uid",
        ]
    ]
    .drop_duplicates()
    .copy()
)

participants[
    "participant_id"
] = (
    participants[
        "participant_id"
    ]
    .astype(str)
)

participants[
    "mindlamp_uid"
] = (
    participants[
        "mindlamp_uid"
    ]
    .astype(str)
)


uid_counts = (
    participants
    .groupby(
        "participant_id"
    )[
        "mindlamp_uid"
    ]
    .nunique()
)

bad = (
    uid_counts[
        uid_counts > 1
    ]
)

if not bad.empty:

    raise RuntimeError(
        "Participant IDs map to multiple mindLAMP UIDs:\n"
        + bad.to_string()
    )


participants = (
    participants
    .sort_values(
        "participant_id"
    )
    .reset_index(
        drop=True
    )
)


print(
    "=" * 80
)

print(
    "CORTEX 2.0 CLINICAL GPS FEATURE BUILD"
)

print(
    "PARTICIPANT-CHECKPOINTED EXECUTION"
)

print(
    "=" * 80
)

print(
    "Original feature script:",
    ORIGINAL_SCRIPT,
)

print(
    "Original script SHA256:",
    SOURCE_SHA256,
)

print(
    "Checkpoint root:",
    CHECKPOINT_ROOT,
)

print(
    "Participants requested:",
    len(
        participants
    ),
)

print(
    "Requested participant-days:",
    len(
        requested[
            [
                "participant_id",
                "date",
            ]
        ]
        .drop_duplicates()
    ),
)

print(
    flush=True,
)


# ============================================================
# RUN PARTICIPANTS
# ============================================================

failed = []

n_total = len(
    participants
)


for zero_index, row in participants.iterrows():

    participant_index = (
        zero_index + 1
    )

    participant_id = str(
        row[
            "participant_id"
        ]
    )

    mindlamp_uid = str(
        row[
            "mindlamp_uid"
        ]
    )

    token = safe_token(
        participant_id
    )

    participant_dir = (
        PARTICIPANT_ROOT
        / token
    )

    participant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        participant_dir
        / "status.json"
    )

    if checkpoint_complete(
        participant_dir
    ):

        status = load_status(
            participant_dir
        )

        print(
            f"[{participant_index}/{n_total}] "
            f"{participant_id}: "
            "already COMPLETE — skipping "
            f"(daily={status.get('daily_rows')}, "
            f"weekly={status.get('weekly_rows')})",
            flush=True,
        )

        continue


    participant_requested_days = int(
        requested[
            requested[
                "participant_id"
            ]
            .astype(str)
            .eq(
                participant_id
            )
        ][
            [
                "participant_id",
                "date",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


    running_status = {
        "participant_index":
            participant_index,

        "participant_id":
            participant_id,

        "mindlamp_uid":
            mindlamp_uid,

        "requested_days":
            participant_requested_days,

        "status":
            "running",

        "started_at":
            utc_now(),

        "completed_at":
            None,

        "return_code":
            None,

        "daily_rows":
            None,

        "weekly_rows":
            None,

        "source_sha256":
            SOURCE_SHA256,
    }

    atomic_json(
        running_status,
        status_path,
    )

    refresh_manifest()


    print()
    print(
        "-" * 80
    )

    print(
        f"[{participant_index}/{n_total}] "
        f"{participant_id} / {mindlamp_uid}"
    )

    print(
        "Requested days:",
        participant_requested_days,
    )

    print(
        "Checkpoint:",
        participant_dir,
    )

    print(
        "-" * 80,
        flush=True,
    )


    env = os.environ.copy()

    env[
        "GPS_CLINICAL_CHECKPOINT_PID"
    ] = participant_id

    env[
        "GPS_CLINICAL_CHECKPOINT_UID"
    ] = mindlamp_uid

    env[
        "GPS_CLINICAL_CHECKPOINT_OUTDIR"
    ] = str(
        participant_dir
    )

    old_pythonpath = (
        env.get(
            "PYTHONPATH",
            "",
        )
    )

    env[
        "PYTHONPATH"
    ] = (
        str(
            SHIM_DIR
        )
        + (
            os.pathsep
            + old_pythonpath
            if old_pythonpath
            else ""
        )
    )

    env[
        "PYTHONUNBUFFERED"
    ] = "1"


    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(
                ORIGINAL_SCRIPT
            ),
        ],
        cwd=str(
            REPO
        ),
        env=env,
        check=False,
    )


    daily_path = (
        participant_dir
        / DAILY_NAME
    )

    weekly_path = (
        participant_dir
        / WEEKLY_NAME
    )


    if (
        completed.returncode == 0
        and daily_path.exists()
        and weekly_path.exists()
    ):

        daily_rows = (
            parquet_rows(
                daily_path
            )
        )

        weekly_rows = (
            parquet_rows(
                weekly_path
            )
        )

        complete_status = {
            **running_status,

            "status":
                "complete",

            "completed_at":
                utc_now(),

            "return_code":
                0,

            "daily_rows":
                daily_rows,

            "weekly_rows":
                weekly_rows,
        }

        atomic_json(
            complete_status,
            status_path,
        )

        print(
            f"[{participant_index}/{n_total}] "
            f"{participant_id}: COMPLETE | "
            f"daily rows={daily_rows:,} | "
            f"weekly rows={weekly_rows:,}",
            flush=True,
        )

    else:

        failure_reason = []

        if completed.returncode != 0:

            failure_reason.append(
                "worker return code "
                f"{completed.returncode}"
            )

        if not daily_path.exists():

            failure_reason.append(
                f"missing {DAILY_NAME}"
            )

        if not weekly_path.exists():

            failure_reason.append(
                f"missing {WEEKLY_NAME}"
            )


        failed_status = {
            **running_status,

            "status":
                "failed",

            "completed_at":
                utc_now(),

            "return_code":
                int(
                    completed.returncode
                ),

            "failure_reason":
                "; ".join(
                    failure_reason
                ),
        }

        atomic_json(
            failed_status,
            status_path,
        )

        failed.append(
            participant_id
        )

        print(
            f"[{participant_index}/{n_total}] "
            f"{participant_id}: FAILED — "
            + "; ".join(
                failure_reason
            ),
            flush=True,
        )


    refresh_manifest()


# ============================================================
# FINAL STATUS
# ============================================================

complete_dirs = []

failed_final = []

for _, row in participants.iterrows():

    participant_id = str(
        row[
            "participant_id"
        ]
    )

    participant_dir = (
        PARTICIPANT_ROOT
        / safe_token(
            participant_id
        )
    )

    if checkpoint_complete(
        participant_dir
    ):

        complete_dirs.append(
            participant_dir
        )

    else:

        failed_final.append(
            participant_id
        )


print()
print(
    "=" * 80
)

print(
    "CHECKPOINT SUMMARY"
)

print(
    "=" * 80
)

print(
    "Complete participants:",
    len(
        complete_dirs
    ),
    "/",
    n_total,
)

print(
    "Incomplete/failed:",
    len(
        failed_final
    ),
)

print(
    "Manifest:",
    MANIFEST_PATH,
)


# ============================================================
# ONLY PROMOTE FINAL OUTPUT WHEN EVERY PARTICIPANT IS COMPLETE
# ============================================================

if failed_final:

    print()
    print(
        "Canonical output files were NOT replaced."
    )

    print(
        "Progress is safely checkpointed."
    )

    print(
        "Rerun the same job to resume."
    )

    print(
        "Incomplete participant IDs:"
    )

    for participant_id in failed_final:

        print(
            "  ",
            participant_id,
        )

    sys.exit(2)


print()
print(
    "All participants complete."
)

print(
    "Combining participant shards..."
)


daily = combine_participant_output(
    complete_dirs,
    DAILY_NAME,
    ROOT / DAILY_NAME,
)

weekly = combine_participant_output(
    complete_dirs,
    WEEKLY_NAME,
    ROOT / WEEKLY_NAME,
)


print()
print(
    "=" * 80
)

print(
    "FINAL CLINICAL GPS FEATURE BUILD COMPLETE"
)

print(
    "=" * 80
)

print(
    "Daily rows:",
    f"{len(daily):,}",
)

print(
    "Weekly rows:",
    f"{len(weekly):,}",
)

if (
    "participant_id"
    in daily.columns
):

    print(
        "Daily-feature participants:",
        daily[
            "participant_id"
        ].nunique(),
    )

if (
    "participant_id"
    in weekly.columns
):

    print(
        "Weekly-feature participants:",
        weekly[
            "participant_id"
        ].nunique(),
    )

print(
    "Saved daily:",
    ROOT / DAILY_NAME,
)

print(
    "Saved weekly:",
    ROOT / WEEKLY_NAME,
)

print(
    "Checkpoint manifest:",
    MANIFEST_PATH,
)

print(
    "=" * 80
)
