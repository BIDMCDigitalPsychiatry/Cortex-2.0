import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter


DAY_MS = 86_400_000

BENCHMARK = Path(
    "../cortex_longitudinal/data/"
    "benchmark_5_full/"
    "benchmark_5_daily_master.csv"
)

PASSIVE_ROOT = Path("../passive")

PARTICIPANT_MAP = Path(
    "../cortex_longitudinal/data/"
    "study_windows.csv"
)

OUTPUT_ROOT = Path(
    "../cortex2_outputs/"
    "digital_clinic/"
    "validation"
)


def stream_daily_dq(
    adapter,
    *,
    stream,
    uid,
    day_starts,
    bin_seconds,
    chunksize,
):
    """
    Incrementally reproduce Cortex 2.0 temporal-bin observability
    without holding raw sensor observations in memory.
    """

    day_starts = np.asarray(
        day_starts,
        dtype=np.int64,
    )

    bin_ms = int(
        bin_seconds * 1000
    )

    if DAY_MS % bin_ms != 0:
        raise ValueError(
            "This validation expects bins that divide "
            "a UTC day exactly."
        )

    bins_per_day = (
        DAY_MS // bin_ms
    )

    occupied = np.zeros(
        (
            len(day_starts),
            bins_per_day,
        ),
        dtype=bool,
    )

    overall_start = int(
        day_starts.min()
    )

    overall_end = int(
        day_starts.max()
        + DAY_MS
    )

    chunks_seen = 0
    rows_seen = 0
    rows_in_window = 0

    for chunk in adapter.iter_canonical(
        stream,
        chunksize=chunksize,
        source_participant_id=uid,
        start_ms=overall_start,
        end_ms=overall_end,
    ):
        chunks_seen += 1
        rows_seen += len(chunk)

        ts = pd.to_numeric(
            chunk["timestamp_ms"],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=np.int64
        )

        ts = ts[
            (ts >= overall_start)
            &
            (ts < overall_end)
        ]

        rows_in_window += len(ts)

        if len(ts) == 0:
            continue

        # Locate each timestamp within one benchmark UTC day.
        day_index = (
            np.searchsorted(
                day_starts,
                ts,
                side="right",
            )
            - 1
        )

        valid_day = (
            (day_index >= 0)
            &
            (day_index < len(day_starts))
        )

        ts = ts[valid_day]
        day_index = day_index[
            valid_day
        ]

        offsets = (
            ts
            - day_starts[
                day_index
            ]
        )

        valid_offset = (
            (offsets >= 0)
            &
            (offsets < DAY_MS)
        )

        ts = ts[valid_offset]
        day_index = day_index[
            valid_offset
        ]
        offsets = offsets[
            valid_offset
        ]

        bin_index = (
            offsets
            // bin_ms
        ).astype(
            np.int64
        )

        occupied[
            day_index,
            bin_index,
        ] = True

    dq = (
        occupied.sum(axis=1)
        / bins_per_day
    )

    return dq, {
        "chunks_seen":
            chunks_seen,
        "rows_seen":
            rows_seen,
        "rows_in_window":
            rows_in_window,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stream",
        choices=[
            "gps",
            "accelerometer",
        ],
        required=True,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
    )

    args = parser.parse_args()

    benchmark = pd.read_csv(
        BENCHMARK
    )

    benchmark["timestamp"] = (
        pd.to_numeric(
            benchmark["timestamp"],
            errors="raise",
        ).astype(
            np.int64
        )
    )

    benchmark["date"] = (
        pd.to_datetime(
            benchmark["date"]
        ).dt.date.astype(str)
    )

    if args.stream == "gps":
        cortex_column = (
            "gps_data_quality"
        )

        bin_seconds = 600

    else:
        cortex_column = (
            "acc_data_quality"
        )

        bin_seconds = 1

    adapter = MindLAMPAdapter(
        study_id="digital_clinic",
        passive_root=PASSIVE_ROOT,
        participant_map=PARTICIPANT_MAP,
    )

    results = []

    groups = benchmark.groupby(
        [
            "participant_id",
            "mindlamp_uid",
        ],
        sort=False,
    )

    for (
        participant_id,
        uid,
    ), b in groups:

        b = (
            b.sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        print(
            f"\n{participant_id} / {uid}"
            f" / {args.stream}",
            flush=True,
        )

        cortex2_dq, meta = (
            stream_daily_dq(
                adapter,
                stream=args.stream,
                uid=uid,
                day_starts=(
                    b[
                        "timestamp"
                    ].to_numpy()
                ),
                bin_seconds=(
                    bin_seconds
                ),
                chunksize=(
                    args.chunksize
                ),
            )
        )

        cortex_dq = pd.to_numeric(
            b[cortex_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        diff = np.abs(
            cortex2_dq
            - cortex_dq
        )

        for i in range(
            len(b)
        ):
            results.append({
                "participant_id":
                    participant_id,

                "mindlamp_uid":
                    uid,

                "date":
                    b.loc[
                        i,
                        "date",
                    ],

                "study_day":
                    int(
                        b.loc[
                            i,
                            "study_day",
                        ]
                    ),

                "stream":
                    args.stream,

                "cortex_dq":
                    cortex_dq[i],

                "cortex2_dq":
                    cortex2_dq[i],

                "abs_diff":
                    diff[i],

                "match_1e12":
                    bool(
                        diff[i]
                        <= 1e-12
                    ),
            })

        print(
            "  days:",
            len(b),
        )

        print(
            "  exact within 1e-12:",
            int(
                np.sum(
                    diff <= 1e-12
                )
            ),
            "/",
            len(b),
        )

        print(
            "  max abs diff:",
            float(
                np.nanmax(diff)
            ),
        )

        print(
            "  raw rows read:",
            f"{meta['rows_seen']:,}",
        )

        print(
            "  rows in clinical window:",
            f"{meta['rows_in_window']:,}",
        )

    out = pd.DataFrame(
        results
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        OUTPUT_ROOT
        / (
            "legacy_cortex_vs_cortex2_dq_5_"
            f"{args.stream}.csv"
        )
    )

    out.to_csv(
        comparison_path,
        index=False,
    )

    summary = (
        out.groupby(
            [
                "participant_id",
                "stream",
            ],
            as_index=False,
        )
        .agg(
            days=(
                "date",
                "size",
            ),

            matching_days=(
                "match_1e12",
                "sum",
            ),

            max_abs_diff=(
                "abs_diff",
                "max",
            ),

            mean_abs_diff=(
                "abs_diff",
                "mean",
            ),
        )
    )

    summary[
        "mismatching_days"
    ] = (
        summary["days"]
        - summary[
            "matching_days"
        ]
    )

    summary_path = (
        OUTPUT_ROOT
        / (
            "legacy_cortex_vs_cortex2_dq_5_"
            f"{args.stream}_summary.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n=== SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    mismatches = out[
        ~out["match_1e12"]
    ]

    if len(mismatches):
        print(
            "\n=== NON-MATCHING DAYS ==="
        )

        print(
            mismatches[
                [
                    "participant_id",
                    "date",
                    "study_day",
                    "cortex_dq",
                    "cortex2_dq",
                    "abs_diff",
                ]
            ].to_string(
                index=False
            )
        )

    else:
        print(
            "\nAll days match within 1e-12."
        )

    print(
        "\nSaved:"
    )
    print(
        comparison_path
    )
    print(
        summary_path
    )


if __name__ == "__main__":
    main()
