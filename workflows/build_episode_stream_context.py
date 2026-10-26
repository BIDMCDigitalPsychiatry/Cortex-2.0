from pathlib import Path

import pandas as pd

from cortex2_engine.quality.source_availability import (
    resolve_source_boundary,
)
from cortex2_engine.quality.stream_evidence import (
    classify_stream_evidence_context,
)


ROOT = Path(
    "../cortex2_outputs/digital_clinic"
)

EPISODES = ROOT / "inventory/episodes.parquet"
SOURCE_INTERVALS = ROOT / "inventory/source_intervals.parquet"

EVIDENCE = (
    ROOT
    / "quality/episode_stream_evidence.csv"
)

OUTPUT = (
    ROOT
    / "quality/episode_stream_context.parquet"
)


episodes = pd.read_parquet(
    EPISODES
)

source_intervals = pd.read_parquet(
    SOURCE_INTERVALS
)

evidence = pd.read_csv(
    EVIDENCE
)


if evidence[
    "has_stream_evidence"
].dtype == object:
    evidence[
        "has_stream_evidence"
    ] = (
        evidence[
            "has_stream_evidence"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )


episode_meta = episodes[
    [
        "episode_id",
        "start_ms",
        "end_ms",
    ]
].copy()

episode_meta["start_year"] = (
    pd.to_datetime(
        episode_meta["start_ms"],
        unit="ms",
        utc=True,
    ).dt.year
)


rows = []

for row in evidence.itertuples(
    index=False
):

    boundary = resolve_source_boundary(
        source_intervals,
        episode_id=str(
            row.episode_id
        ),
        stream=str(
            row.stream
        ),
    )

    context = (
        classify_stream_evidence_context(
            boundary_known=boundary[
                "boundary_known"
            ],
            boundary_scope=boundary[
                "boundary_scope"
            ],
            has_stream_evidence=bool(
                row.has_stream_evidence
            ),
        )
    )

    rows.append({
        "participant_id":
            str(row.participant_id),

        "episode_id":
            str(row.episode_id),

        "source_participant_id":
            str(
                row.source_participant_id
            ),

        "stream":
            str(row.stream),

        "has_stream_evidence":
            bool(
                row.has_stream_evidence
            ),

        "overlapping_files":
            int(
                row.overlapping_files
            ),

        "boundary_known":
            bool(
                boundary[
                    "boundary_known"
                ]
            ),

        "boundary_scope":
            boundary[
                "boundary_scope"
            ],

        "boundary_source":
            boundary[
                "boundary_source"
            ],

        "stream_evidence_context":
            context.value,
    })


out = pd.DataFrame(
    rows
)

out = out.merge(
    episode_meta,
    on="episode_id",
    how="left",
    validate="many_to_one",
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_parquet(
    OUTPUT,
    index=False,
)


print(
    "Rows:",
    len(out)
)

print(
    "Episodes:",
    out["episode_id"].nunique()
)


print(
    "\nContext by stream:"
)

print(
    pd.crosstab(
        out["stream"],
        out[
            "stream_evidence_context"
        ],
        margins=True,
    ).to_string()
)


print(
    "\nContext by start year:"
)

print(
    pd.crosstab(
        [
            out["start_year"],
            out["stream"],
        ],
        out[
            "stream_evidence_context"
        ],
    ).to_string()
)


print("\nSaved:")
print(OUTPUT)
