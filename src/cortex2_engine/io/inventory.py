from pathlib import Path

import pandas as pd

from cortex2_engine.config import StudyConfig


def build_raw_file_manifest(
    config: StudyConfig,
) -> pd.DataFrame:

    rows = []

    for stream_name, stream in config.streams.items():

        root = Path(stream.path)

        if not root.exists():
            raise FileNotFoundError(
                f"{stream_name}: {root}"
            )

        for path in root.glob(
            stream.filename_pattern
        ):

            if not path.is_file():
                continue

            stat = path.stat()

            rows.append({
                "stream":
                    stream_name,

                "path":
                    str(path.resolve()),

                "filename":
                    path.name,

                "size_bytes":
                    stat.st_size,

                "modified_ns":
                    stat.st_mtime_ns,
            })

    return pd.DataFrame(rows)
