from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, field_validator

from cortex2_engine.time import parse_clock


class StreamConfig(BaseModel):
    path: Path
    filename_pattern: str


class WindowsConfig(BaseModel):
    path: Path
    participant_id_column: str = "participant_id"
    sensor_id_column: str = "mindlamp_uid"
    start_column: str = "clinical_start"
    end_column: str = "clinical_end"


class AdapterConfig(BaseModel):
    """How ``cortex2 features`` reads raw data.

    ``kind="mindlamp"`` uses ``passive_root`` (``<root>/<prefix>/<prefix>_<uid>_*.csv``)
    and optional ``participant_map`` / ``source_manifest_dir``.
    ``kind="generic_csv"`` uses ``config_path`` (a generic CSV adapter YAML).
    """

    kind: Literal["mindlamp", "generic_csv"] = "mindlamp"
    passive_root: Path | None = None
    participant_map: Path | None = None
    source_manifest_dir: Path | None = None
    config_path: Path | None = None


class StudyConfig(BaseModel):
    name: str
    timezone: str = "UTC"
    # Local clock time at which a feature day begins (configs/default.yaml
    # time.day_start). Days are [day_start, next day_start) in `timezone`.
    day_start: str = "00:00"
    output_dir: Path
    windows: WindowsConfig
    streams: dict[str, StreamConfig] = Field(default_factory=dict)
    # Optional; needed only by `cortex2 features`.
    adapter: AdapterConfig | None = None

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown IANA timezone: {value!r}") from exc
        return value

    @field_validator("day_start")
    @classmethod
    def _valid_day_start(cls, value: str) -> str:
        parse_clock(value)
        return value


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

    if config.adapter is not None:
        for field in (
            "passive_root",
            "participant_map",
            "source_manifest_dir",
            "config_path",
        ):
            value = getattr(config.adapter, field)
            if value is not None and not value.is_absolute():
                setattr(config.adapter, field, (base / value).resolve())

    for stream in config.streams.values():
        if not stream.path.is_absolute():
            stream.path = (
                base / stream.path
            ).resolve()

    return config
