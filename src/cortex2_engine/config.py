from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, Field


class StreamConfig(BaseModel):
    path: Path
    filename_pattern: str


class WindowsConfig(BaseModel):
    path: Path
    participant_id_column: str = "participant_id"
    sensor_id_column: str = "mindlamp_uid"
    start_column: str = "clinical_start"
    end_column: str = "clinical_end"


class StudyConfig(BaseModel):
    name: str
    timezone: str = "UTC"
    output_dir: Path
    windows: WindowsConfig
    streams: Dict[str, StreamConfig] = Field(default_factory=dict)


def load_study_config(path: str | Path) -> StudyConfig:
    path = Path(path).resolve()

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    config = StudyConfig.model_validate(raw)

    # Resolve relative paths relative to the config file.
    base = path.parent

    if not config.windows.path.is_absolute():
        config.windows.path = (
            base / config.windows.path
        ).resolve()

    if not config.output_dir.is_absolute():
        config.output_dir = (
            base / config.output_dir
        ).resolve()

    for stream in config.streams.values():
        if not stream.path.is_absolute():
            stream.path = (
                base / stream.path
            ).resolve()

    return config
