from pathlib import Path
import argparse

import pandas as pd

from cortex2_engine.adapters.mindlamp import (
    MindLAMPAdapter,
)
from cortex2_engine.canonical.reader import (
    EpisodeReader,
)
from cortex2_engine.missingness.hourly import (
    classify_hourly_gap_pattern,
)
from cortex2_engine.quality.hourly import (
    episode_hourly_quality,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--limit",
    type=int,
    default=None,
)

parser.add_argument(
    "--overwrite",
    action="store_true",
)

args = parser.parse_args()


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

QUALITY = ROOT / "quality"

CHECKPOINT = (
    QUALITY
    / "hourly_missingness"
    / "episodes"
)

CHECKPOINT.mkdir(
    parents=True,
    exist_ok=True,
)


episodes = pd.read_parquet(
    ROOT / "inventory/episodes.parquet"
)

source_intervals = pd.read_parquet(
    ROOT / "inventory/source_intervals.parquet"
)

context = pd.read_parquet(
    QUALITY
    / "episode_stream_context.parquet"
)


# ------------------------------------------------------------
# Episodes with BOTH GPS and ACC evidence
# ------------------------------------------------------------

ev = (
    context[
        context["stream"].isin(
            [
                "gps",
                "accelerometer",
            ]
        )
    ]
    .pivot(
        index="episode_id",
        columns="stream",
        values="has_stream_evidence",
    )
    .fillna(False)
)

eligible_ids = ev[
    ev["gps"]
    &
    ev["accelerometer"]
].index.astype(str)


selected = episodes[
    episodes["episode_id"]
    .astype(str)
    .isin(eligible_ids)
].copy()


selected = selected.sort_values(
    [
        "participant_id",
        "episode_id",
    ]
)


if args.limit is not None:
    selected = selected.head(
        args.limit
    )


print(
    "Eligible GPS+ACC episodes:",
    len(eligible_ids),
)

print(
    "Processing this run:",
    len(selected),
)


# ------------------------------------------------------------
# Adapter
# ------------------------------------------------------------

participant_map = episodes[
    [
        "participant_id",
        "source_participant_id",
    ]
].drop_duplicates()

participant_map_path = (
    ROOT
    / "inventory"
    / "cortex2_participant_map.csv"
)

participant_map.to_csv(
    participant_map_path,
    index=False,
)


adapter = MindLAMPAdapter(
    study_id="digital_clinic",
    passive_root=Path(
        "../passive"
    ),
    participant_map=participant_map_path,
    source_id_column=(
        "source_participant_id"
    ),
    participant_id_column=(
        "participant_id"
    ),
    source_manifest_dir=(
        ROOT
        / "inventory/source_files"
    ),
)

reader = EpisodeReader(
    adapter,
    episodes,
)


# ------------------------------------------------------------
# Stream evidence lookup
# ------------------------------------------------------------

screen_evidence = (
    context[
        context["stream"]
        .eq("screen")
    ]
    .set_index("episode_id")[
        "has_stream_evidence"
    ]
    .to_dict()
)


summaries = []


for n, ep in enumerate(
    selected.itertuples(
        index=False
    ),
    start=1,
):

    episode_id = str(
        ep.episode_id
    )

    participant_id = str(
        ep.participant_id
    )

    output = (
        CHECKPOINT
        / f"{episode_id}.parquet"
    )

    print(
        f"\n[{n}/{len(selected)}] "
        f"{participant_id} | "
        f"{episode_id}"
    )

    if (
        output.exists()
        and not args.overwrite
    ):
        print(
            "checkpoint exists"
        )

        x = pd.read_parquet(
            output
        )

    else:

        gps = episode_hourly_quality(
            reader,
            source_intervals,
            episode_id=episode_id,
            stream="gps",
        )

        acc = episode_hourly_quality(
            reader,
            source_intervals,
            episode_id=episode_id,
            stream="accelerometer",
        )

        has_screen = bool(
            screen_evidence.get(
                episode_id,
                False,
            )
        )

        if has_screen:

            screen = episode_hourly_quality(
                reader,
                source_intervals,
                episode_id=episode_id,
                stream="screen",
            )

        key = [
            "episode_id",
            "hour_index",
            "start_ms",
            "end_ms",
            "hour_start_utc",
            "hour_end_utc",
        ]

        g = gps[
            key
            + [
                "participant_id",
                "source_participant_id",
                "hour_observed",
                "data_quality",
                "source_unavailable",
                "partial_source",
                "boundary_unknown",
            ]
        ].rename(
            columns={
                "hour_observed":
                    "gps_observed",

                "data_quality":
                    "gps_dq",

                "source_unavailable":
                    "gps_source_unavailable",

                "partial_source":
                    "gps_partial_source",

                "boundary_unknown":
                    "gps_boundary_unknown",
            }
        )

        a = acc[
            key
            + [
                "hour_observed",
                "data_quality",
                "source_unavailable",
                "partial_source",
                "boundary_unknown",
            ]
        ].rename(
            columns={
                "hour_observed":
                    "acc_observed",

                "data_quality":
                    "acc_dq",

                "source_unavailable":
                    "acc_source_unavailable",

                "partial_source":
                    "acc_partial_source",

                "boundary_unknown":
                    "acc_boundary_unknown",
            }
        )

        x = g.merge(
            a,
            on=key,
            how="outer",
            validate="one_to_one",
        )

        if has_screen:

            s = screen[
                key
                + [
                    "hour_observed",
                    "source_unavailable",
                    "partial_source",
                    "boundary_unknown",
                ]
            ].rename(
                columns={
                    "hour_observed":
                        "screen_event_observed",

                    "source_unavailable":
                        "screen_source_unavailable",

                    "partial_source":
                        "screen_partial_source",

                    "boundary_unknown":
                        "screen_boundary_unknown",
                }
            )

            x = x.merge(
                s,
                on=key,
                how="left",
                validate="one_to_one",
            )

        else:

            x[
                "screen_event_observed"
            ] = False

            x[
                "screen_source_unavailable"
            ] = False

            x[
                "screen_partial_source"
            ] = False

            x[
                "screen_boundary_unknown"
            ] = True

        x[
            "screen_episode_evidence"
        ] = has_screen


        x["gap_pattern"] = [
            classify_hourly_gap_pattern(
                gps_observed=bool(go),
                acc_observed=bool(ao),
                screen_event_observed=bool(so),
                gps_source_unavailable=bool(gsu),
                acc_source_unavailable=bool(asu),
                gps_partial_source=bool(gp),
                acc_partial_source=bool(ap),
            ).value
            for (
                go,
                ao,
                so,
                gsu,
                asu,
                gp,
                ap,
            ) in zip(
                x["gps_observed"],
                x["acc_observed"],
                x[
                    "screen_event_observed"
                ],
                x[
                    "gps_source_unavailable"
                ],
                x[
                    "acc_source_unavailable"
                ],
                x[
                    "gps_partial_source"
                ],
                x[
                    "acc_partial_source"
                ],
            )
        ]


        x[
            "any_boundary_unknown"
        ] = (
            x["gps_boundary_unknown"]
            |
            x["acc_boundary_unknown"]
        )


        x.to_parquet(
            output,
            index=False,
        )


    counts = (
        x["gap_pattern"]
        .value_counts()
        .to_dict()
    )

    eligible_hours = int(
        (
            x["gap_pattern"]
            !=
            "source_unavailable_or_partial"
        ).sum()
    )

    summaries.append({
        "participant_id":
            participant_id,

        "episode_id":
            episode_id,

        "hours_total":
            len(x),

        "eligible_hours":
            eligible_hours,

        "both_observed_hours":
            counts.get(
                "both_observed",
                0,
            ),

        "gps_specific_gap_hours":
            counts.get(
                "gps_specific_gap",
                0,
            ),

        "acc_specific_gap_hours":
            counts.get(
                "acc_specific_gap",
                0,
            ),

        "shared_gap_with_screen_activity_hours":
            counts.get(
                "shared_gap_with_screen_activity",
                0,
            ),

        "shared_gap_without_screen_activity_evidence_hours":
            counts.get(
                "shared_gap_without_screen_activity_evidence",
                0,
            ),

        "source_unavailable_or_partial_hours":
            counts.get(
                "source_unavailable_or_partial",
                0,
            ),
    })


summary = pd.DataFrame(
    summaries
)


for col in [
    "both_observed_hours",
    "gps_specific_gap_hours",
    "acc_specific_gap_hours",
    "shared_gap_with_screen_activity_hours",
    "shared_gap_without_screen_activity_evidence_hours",
]:

    summary[
        col.replace(
            "_hours",
            "_fraction",
        )
    ] = (
        summary[col]
        /
        summary[
            "eligible_hours"
        ].replace(
            0,
            pd.NA,
        )
    )


summary_path = (
    QUALITY
    / "hourly_missingness"
    / "episode_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print(
    "\n" + "=" * 70
)

print(
    "SUMMARY"
)

print(
    "=" * 70
)


totals = summary[
    [
        "both_observed_hours",
        "gps_specific_gap_hours",
        "acc_specific_gap_hours",
        "shared_gap_with_screen_activity_hours",
        "shared_gap_without_screen_activity_evidence_hours",
        "source_unavailable_or_partial_hours",
    ]
].sum()


print(
    totals.to_string()
)


print(
    "\nSaved:"
)

print(summary_path)
