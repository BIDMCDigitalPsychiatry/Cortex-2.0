import argparse
from pathlib import Path

import pandas as pd

from cortex2_engine.adapters.mindlamp import MindLAMPAdapter
from cortex2_engine.canonical.reader import EpisodeReader
from cortex2_engine.quality.episode import episode_daily_quality


EPISODES_PATH = Path(
    "../cortex2_outputs/digital_clinic/inventory/episodes.parquet"
)

SOURCE_INTERVALS_PATH = Path(
    "../cortex2_outputs/digital_clinic/inventory/source_intervals.parquet"
)

SOURCE_MANIFEST_DIR = Path(
    "../cortex2_outputs/digital_clinic/inventory/source_files"
)

PASSIVE_ROOT = Path("../passive")

OUTPUT_ROOT = Path(
    "../cortex2_outputs/digital_clinic/quality"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stream",
        choices=[
            "gps",
            "accelerometer",
            "screen",
        ],
        required=True,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    episodes = pd.read_parquet(
        EPISODES_PATH
    )

    source_intervals = pd.read_parquet(
        SOURCE_INTERVALS_PATH
    )

    adapter = MindLAMPAdapter(
        study_id="digital_clinic",
        passive_root=PASSIVE_ROOT,
        source_manifest_dir=SOURCE_MANIFEST_DIR,
    )

    reader = EpisodeReader(
        adapter,
        episodes,
    )

    checkpoint_dir = (
        OUTPUT_ROOT
        / "episodes"
        / args.stream
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors = []

    scanned = 0
    reused = 0

    total = len(episodes)

    for i, ep in enumerate(
        episodes.itertuples(index=False),
        start=1,
    ):
        episode_id = str(
            ep.episode_id
        )

        output_path = (
            checkpoint_dir
            / f"{episode_id}.parquet"
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            reused += 1

        else:
            try:
                daily = episode_daily_quality(
                    reader,
                    source_intervals,
                    episode_id=episode_id,
                    stream=args.stream,
                    chunksize=args.chunksize,
                )

                daily.to_parquet(
                    output_path,
                    index=False,
                )

                scanned += 1

            except Exception as exc:
                errors.append({
                    "episode_id":
                        episode_id,
                    "participant_id":
                        str(ep.participant_id),
                    "source_participant_id":
                        str(ep.source_participant_id),
                    "stream":
                        args.stream,
                    "error":
                        repr(exc),
                })

        if (
            i % 25 == 0
            or i == total
        ):
            print(
                f"{i:,}/{total:,} | "
                f"computed={scanned:,} | "
                f"reused={reused:,} | "
                f"errors={len(errors):,}",
                flush=True,
            )

    error_path = (
        OUTPUT_ROOT
        / f"{args.stream}_errors.csv"
    )

    pd.DataFrame(
        errors
    ).to_csv(
        error_path,
        index=False,
    )

    if errors:
        print("\nERRORS:")
        print(
            pd.DataFrame(errors)
            .to_string(index=False)
        )

        raise RuntimeError(
            f"{len(errors)} episode(s) failed. "
            f"See {error_path}"
        )

    frames = []

    for ep in episodes.itertuples(
        index=False
    ):
        path = (
            checkpoint_dir
            / f"{ep.episode_id}.parquet"
        )

        frames.append(
            pd.read_parquet(path)
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    out = out.sort_values(
        [
            "participant_id",
            "episode_id",
            "study_day",
        ]
    ).reset_index(
        drop=True
    )

    final_path = (
        OUTPUT_ROOT
        / f"{args.stream}_daily.parquet"
    )

    out.to_parquet(
        final_path,
        index=False,
    )

    print("\n=== COHORT SUMMARY ===")

    print(
        "Episodes:",
        out["episode_id"].nunique(),
    )

    print(
        "Participants:",
        out["participant_id"].nunique(),
    )

    print(
        "Daily rows:",
        len(out),
    )

    print("\nSource status:")
    print(
        out["source_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nSource unavailable days:",
        int(
            out[
                "source_unavailable"
            ].sum()
        ),
    )

    print(
        "Partial-source days:",
        int(
            out[
                "partial_source"
            ].sum()
        ),
    )

    print(
        "Boundary-unknown days:",
        int(
            out[
                "boundary_unknown"
            ].sum()
        ),
    )

    if "data_quality" in out.columns:
        print(
            "\nDQ available days:",
            int(
                out[
                    "data_quality"
                ].notna()
                .sum()
            ),
        )

        print(
            "DQ zero days:",
            int(
                out[
                    "data_quality"
                ].eq(0)
                .sum()
            ),
        )

    if "observed" in out.columns:
        print(
            "\nScreen-observed days:",
            int(
                out["observed"]
                .eq(1)
                .sum()
            ),
        )

    print("\nSaved:")
    print(final_path)
    print(error_path)


if __name__ == "__main__":
    main()
