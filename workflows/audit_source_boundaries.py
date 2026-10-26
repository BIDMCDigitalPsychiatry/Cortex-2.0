from pathlib import Path

import pandas as pd


WINDOWS = Path(
    "../cortex_longitudinal/data/study_windows.csv"
)

PASSIVE = Path("../passive")

STREAMS = {
    "gps": ("gps", "gps"),
    "accelerometer": ("acc", "acc"),
    "screen": ("screen", "screen"),
}

OUT = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/source_boundary_audit.csv"
)


def scan_files(paths):
    min_ms = None
    max_ms = None
    n_rows = 0

    for path in paths:

        for chunk in pd.read_csv(
            path,
            usecols=["timestamp"],
            chunksize=1_000_000,
        ):
            ts = pd.to_numeric(
                chunk["timestamp"],
                errors="coerce",
            ).dropna()

            if len(ts) == 0:
                continue

            n_rows += len(ts)

            cmin = int(ts.min())
            cmax = int(ts.max())

            min_ms = (
                cmin
                if min_ms is None
                else min(min_ms, cmin)
            )

            max_ms = (
                cmax
                if max_ms is None
                else max(max_ms, cmax)
            )

    return min_ms, max_ms, n_rows


windows = pd.read_csv(WINDOWS)

results = []

for i, row in windows.iterrows():

    pid = str(row["participant_id"])
    uid = str(row["mindlamp_uid"])

    clinical_start = pd.Timestamp(
        row["clinical_start"],
        tz="UTC",
    )

    clinical_end = pd.Timestamp(
        row["clinical_end"],
        tz="UTC",
    )

    final_day_end = (
        clinical_end
        + pd.Timedelta(days=1)
    )

    for stream, (folder, prefix) in STREAMS.items():

        paths = sorted(
            (PASSIVE / folder).glob(
                f"{prefix}_{uid}_*.csv"
            )
        )

        if not paths:
            results.append({
                "participant_id": pid,
                "mindlamp_uid": uid,
                "stream": stream,
                "n_files": 0,
                "n_rows": 0,
                "raw_start_utc": None,
                "raw_end_utc": None,
                "has_clinical_start_day": False,
                "has_clinical_end_day": False,
                "ends_before_clinical_end_day": True,
            })
            continue

        min_ms, max_ms, n_rows = scan_files(
            paths
        )

        if max_ms is None:
            continue

        raw_start = pd.to_datetime(
            min_ms,
            unit="ms",
            utc=True,
        )

        raw_end = pd.to_datetime(
            max_ms,
            unit="ms",
            utc=True,
        )

        has_start_day = (
            raw_end >= clinical_start
            and raw_start < (
                clinical_start
                + pd.Timedelta(days=1)
            )
        )

        has_end_day = (
            raw_end >= clinical_end
            and raw_start < final_day_end
        )

        results.append({
            "participant_id": pid,
            "mindlamp_uid": uid,
            "stream": stream,
            "n_files": len(paths),
            "n_rows": n_rows,
            "raw_start_utc": raw_start,
            "raw_end_utc": raw_end,
            "has_clinical_start_day": has_start_day,
            "has_clinical_end_day": has_end_day,
            "ends_before_clinical_end_day": (
                raw_end < clinical_end
            ),
            "hours_from_raw_end_to_clinical_end_start": (
                clinical_end - raw_end
            ).total_seconds() / 3600,
        })

    if (i + 1) % 25 == 0:
        print(
            f"Processed {i + 1}/{len(windows)}",
            flush=True,
        )


out = pd.DataFrame(results)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)

print("\n=== SUMMARY ===")

summary = (
    out.groupby("stream")
    .agg(
        participants=("participant_id", "size"),
        with_files=("n_files", lambda x: int((x > 0).sum())),
        with_final_day=(
            "has_clinical_end_day",
            "sum",
        ),
        ending_before_final_day=(
            "ends_before_clinical_end_day",
            "sum",
        ),
    )
)

print(summary)

print("\nSaved:")
print(OUT)
