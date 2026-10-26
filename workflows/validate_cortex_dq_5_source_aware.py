import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter
from cortex2_engine.quality.source_availability import (
    SourceBoundaryStatus,
    classify_source_interval,
    resolve_source_boundary,
    source_interval_overlap,
)

from validate_cortex_dq_5 import (
    BENCHMARK,
    PASSIVE_ROOT,
    PARTICIPANT_MAP,
    OUTPUT_ROOT,
    DAY_MS,
    stream_daily_dq,
)


SOURCE_INTERVALS = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/source_intervals.parquet"
)

SOURCE_MANIFEST_DIR = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/source_files"
)


OUTSIDE_STATUSES = {
    SourceBoundaryStatus.OUTSIDE_DECLARED_END,
    SourceBoundaryStatus.OUTSIDE_DECLARED_START,
    SourceBoundaryStatus.NO_INTERVAL_OVERLAP,
}


def resolve_episode_boundary(
    source_intervals,
    *,
    participant_id,
    uid,
    stream,
):
    rows = source_intervals[
        source_intervals["participant_id"]
        .astype(str)
        .eq(str(participant_id))
        &
        source_intervals["source_participant_id"]
        .astype(str)
        .eq(str(uid))
    ]

    episode_ids = (
        rows["episode_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(episode_ids) > 1:
        raise RuntimeError(
            f"Multiple episodes found for "
            f"{participant_id} / {uid}: "
            f"{episode_ids.tolist()}"
        )

    if len(episode_ids) == 0:
        return None, {
            "boundary_known": False,
            "boundary_scope": None,
            "boundary_source": None,
            "source_start_ms": None,
            "source_end_ms": None,
        }

    episode_id = episode_ids[0]

    boundary = resolve_source_boundary(
        source_intervals,
        episode_id=episode_id,
        stream=stream,
    )

    return episode_id, boundary


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

    benchmark = pd.read_csv(BENCHMARK)

    benchmark["timestamp"] = pd.to_numeric(
        benchmark["timestamp"],
        errors="raise",
    ).astype(np.int64)

    benchmark["date"] = (
        pd.to_datetime(
            benchmark["date"]
        )
        .dt.date
        .astype(str)
    )

    source_intervals = pd.read_parquet(
        SOURCE_INTERVALS
    )

    if args.stream == "gps":
        cortex_column = "gps_data_quality"
        bin_seconds = 600
    else:
        cortex_column = "acc_data_quality"
        bin_seconds = 1

    adapter = MindLAMPAdapter(
        study_id="digital_clinic",
        passive_root=PASSIVE_ROOT,
        participant_map=PARTICIPANT_MAP,
        source_manifest_dir=SOURCE_MANIFEST_DIR,
    )

    results = []

    groups = benchmark.groupby(
        [
            "participant_id",
            "mindlamp_uid",
        ],
        sort=False,
    )

    for (participant_id, uid), b in groups:

        b = (
            b.sort_values("timestamp")
            .reset_index(drop=True)
        )

        print(
            f"\n{participant_id} / {uid}"
            f" / {args.stream}",
            flush=True,
        )

        day_starts = (
            b["timestamp"]
            .to_numpy(
                dtype=np.int64
            )
        )

        raw_cortex2_dq, meta = stream_daily_dq(
            adapter,
            stream=args.stream,
            uid=uid,
            day_starts=day_starts,
            bin_seconds=bin_seconds,
            chunksize=args.chunksize,
        )

        episode_id, boundary = (
            resolve_episode_boundary(
                source_intervals,
                participant_id=participant_id,
                uid=uid,
                stream=args.stream,
            )
        )

        statuses = []
        fractions = []

        for day_start in day_starts:
            day_end = int(
                day_start + DAY_MS
            )

            status = classify_source_interval(
                requested_start_ms=int(day_start),
                requested_end_ms=day_end,
                source_start_ms=boundary[
                    "source_start_ms"
                ],
                source_end_ms=boundary[
                    "source_end_ms"
                ],
            )

            overlap = source_interval_overlap(
                requested_start_ms=int(day_start),
                requested_end_ms=day_end,
                source_start_ms=boundary[
                    "source_start_ms"
                ],
                source_end_ms=boundary[
                    "source_end_ms"
                ],
            )

            statuses.append(status)
            fractions.append(
                overlap[
                    "source_available_fraction"
                ]
            )

        source_unavailable = np.array(
            [
                s in OUTSIDE_STATUSES
                for s in statuses
            ],
            dtype=bool,
        )

        fractions = np.asarray(
            fractions,
            dtype=float,
        )

        partial_source = (
            (~source_unavailable)
            & (fractions < 1.0)
        )

        boundary_unknown = np.array(
            [
                s
                == SourceBoundaryStatus.BOUNDARY_UNKNOWN
                for s in statuses
            ],
            dtype=bool,
        )

        # Preserve the raw observability result for audit.
        # For the Cortex full-day benchmark, days that were
        # definitely outside or only partly inside the declared
        # source interval are not comparable.
        cortex2_dq = raw_cortex2_dq.copy()

        cortex2_dq[
            source_unavailable
            | partial_source
        ] = np.nan

        cortex_dq = pd.to_numeric(
            b[cortex_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        comparable = (
            np.isfinite(cortex2_dq)
            & np.isfinite(cortex_dq)
        )

        diff = np.full(
            len(b),
            np.nan,
            dtype=float,
        )

        diff[comparable] = np.abs(
            cortex2_dq[comparable]
            - cortex_dq[comparable]
        )

        match = np.zeros(
            len(b),
            dtype=bool,
        )

        match[comparable] = (
            diff[comparable]
            <= 1e-12
        )

        for i in range(len(b)):
            results.append({
                "participant_id":
                    participant_id,

                "mindlamp_uid":
                    uid,

                "episode_id":
                    episode_id,

                "date":
                    b.loc[i, "date"],

                "study_day":
                    int(
                        b.loc[
                            i,
                            "study_day",
                        ]
                    ),

                "stream":
                    args.stream,

                "source_status":
                    statuses[i].value,

                "source_available_fraction":
                    fractions[i],

                "boundary_scope":
                    boundary[
                        "boundary_scope"
                    ],

                "boundary_source":
                    boundary[
                        "boundary_source"
                    ],

                "source_unavailable":
                    bool(
                        source_unavailable[i]
                    ),

                "partial_source":
                    bool(
                        partial_source[i]
                    ),

                "boundary_unknown":
                    bool(
                        boundary_unknown[i]
                    ),

                "cortex_dq":
                    cortex_dq[i],

                "cortex2_dq_raw":
                    raw_cortex2_dq[i],

                "cortex2_dq":
                    cortex2_dq[i],

                "comparable":
                    bool(
                        comparable[i]
                    ),

                "abs_diff":
                    diff[i],

                "match_1e12":
                    bool(
                        match[i]
                    ),
            })

        n_comparable = int(
            comparable.sum()
        )

        n_matching = int(
            match.sum()
        )

        print(
            "  days:",
            len(b),
        )

        print(
            "  source unavailable:",
            int(
                source_unavailable.sum()
            ),
        )

        print(
            "  partial source:",
            int(
                partial_source.sum()
            ),
        )

        print(
            "  boundary unknown:",
            int(
                boundary_unknown.sum()
            ),
        )

        print(
            "  comparable days:",
            n_comparable,
        )

        print(
            "  exact within 1e-12:",
            n_matching,
            "/",
            n_comparable,
        )

        if n_comparable:
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

    out = pd.DataFrame(results)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        OUTPUT_ROOT
        / (
            "legacy_cortex_vs_cortex2_dq_5_"
            f"{args.stream}_source_aware.csv"
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
            source_unavailable_days=(
                "source_unavailable",
                "sum",
            ),
            partial_source_days=(
                "partial_source",
                "sum",
            ),
            boundary_unknown_days=(
                "boundary_unknown",
                "sum",
            ),
            comparable_days=(
                "comparable",
                "sum",
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

    summary["mismatching_days"] = (
        summary["comparable_days"]
        - summary["matching_days"]
    )

    summary_path = (
        OUTPUT_ROOT
        / (
            "legacy_cortex_vs_cortex2_dq_5_"
            f"{args.stream}_source_aware_summary.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n=== SUMMARY ===")

    print(
        summary.to_string(
            index=False
        )
    )

    mismatches = out[
        out["comparable"]
        & ~out["match_1e12"]
    ]

    if len(mismatches):
        print(
            "\n=== TRUE COMPARABLE MISMATCHES ==="
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
                    "source_status",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "\nAll comparable days match within 1e-12."
        )

    excluded = out[
        out["source_unavailable"]
        | out["partial_source"]
    ]

    if len(excluded):
        print(
            "\n=== DAYS EXCLUDED BY SOURCE AVAILABILITY ==="
        )

        print(
            excluded[
                [
                    "participant_id",
                    "date",
                    "study_day",
                    "source_status",
                    "source_available_fraction",
                    "cortex_dq",
                    "cortex2_dq_raw",
                    "cortex2_dq",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(comparison_path)
    print(summary_path)


if __name__ == "__main__":
    main()
