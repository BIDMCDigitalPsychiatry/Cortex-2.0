from pathlib import Path

import pandas as pd


class SourceFileIndex:
    """
    Persistent index of raw source files and their temporal bounds.

    The index is source-platform agnostic. Downstream code asks for:

        stream
        source identity
        [start_ms, end_ms)

    and receives only source files that can overlap that request.
    """

    REQUIRED_COLUMNS = {
        "stream",
        "source_participant_id",
        "source_file",
        "size_bytes",
        "modified_ns",
        "min_timestamp_ms",
        "max_timestamp_ms",
        "scan_status",
    }

    def __init__(
        self,
        manifest_dir: str | Path,
    ):
        self.manifest_dir = Path(
            manifest_dir
        ).resolve()

        self._cache = {}

    def _load(
        self,
        stream: str,
    ) -> pd.DataFrame:

        if stream in self._cache:
            return self._cache[stream]

        path = (
            self.manifest_dir
            / f"{stream}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"No source manifest for stream "
                f"'{stream}': {path}"
            )

        df = pd.read_parquet(path)

        missing = (
            self.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{path} missing required manifest "
                f"columns: {sorted(missing)}"
            )

        self._cache[stream] = df

        return df

    def candidate_records(
        self,
        stream: str,
        *,
        source_participant_id: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        verify_files: bool = True,
    ) -> pd.DataFrame:
        """
        Select raw files that can overlap a requested interval.

        Overlap rule for [start_ms, end_ms):

            file_max >= start_ms
            file_min < end_ms
        """

        df = self._load(
            stream
        ).copy()

        if source_participant_id is not None:
            df = df[
                df[
                    "source_participant_id"
                ].astype(str).eq(
                    str(source_participant_id)
                )
            ]

        bad = df[
            ~df["scan_status"].eq("ok")
        ]

        if len(bad):
            examples = (
                bad["source_file"]
                .head(5)
                .tolist()
            )

            raise RuntimeError(
                "Source manifest contains scan errors "
                f"for requested data. Examples: {examples}"
            )

        # Empty source files cannot contribute observations.
        df = df[
            df["min_timestamp_ms"].notna()
            &
            df["max_timestamp_ms"].notna()
        ]

        if start_ms is not None:
            df = df[
                pd.to_numeric(
                    df["max_timestamp_ms"]
                )
                >= int(start_ms)
            ]

        if end_ms is not None:
            df = df[
                pd.to_numeric(
                    df["min_timestamp_ms"]
                )
                < int(end_ms)
            ]

        df = df.sort_values(
            [
                "min_timestamp_ms",
                "source_file",
            ]
        ).reset_index(drop=True)

        if verify_files:

            for row in df.itertuples(
                index=False
            ):
                path = Path(
                    row.source_file
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Indexed source file no longer exists: "
                        f"{path}"
                    )

                stat = path.stat()

                if (
                    int(stat.st_size)
                    != int(row.size_bytes)
                    or int(stat.st_mtime_ns)
                    != int(row.modified_ns)
                ):
                    raise RuntimeError(
                        "Source file changed after manifest "
                        f"creation: {path}. "
                        "Rebuild its source manifest."
                    )

        return df

    def candidate_files(
        self,
        stream: str,
        **kwargs,
    ) -> list[Path]:

        df = self.candidate_records(
            stream,
            **kwargs,
        )

        return [
            Path(p)
            for p in df["source_file"]
        ]

    def selection_summary(
        self,
        stream: str,
        **kwargs,
    ) -> dict:

        all_rows = self._load(
            stream
        )

        selected = self.candidate_records(
            stream,
            **kwargs,
        )

        return {
            "stream":
                stream,

            "manifest_files":
                int(len(all_rows)),

            "selected_files":
                int(len(selected)),

            "selected_bytes":
                int(
                    selected[
                        "size_bytes"
                    ].sum()
                )
                if len(selected)
                else 0,
        }
