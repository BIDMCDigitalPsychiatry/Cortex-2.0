"""Build a study adapter and its episodes from a ``StudyConfig``."""

from __future__ import annotations

import pandas as pd

from cortex2_engine.adapters.base import BaseAdapter
from cortex2_engine.canonical.episodes import build_episodes_from_windows
from cortex2_engine.config import StudyConfig
from cortex2_engine.windows.clinical_windows import load_clinical_windows


def build_adapter(config: StudyConfig) -> BaseAdapter:
    """Instantiate the adapter named in ``config.adapter``.

    Raises
    ------
    ValueError
        When the config has no ``adapter`` section or a required path is
        missing.
    """

    a = config.adapter

    if a is None:
        raise ValueError("Study config has no 'adapter' section; `cortex2 features` needs one.")

    if a.kind == "mindlamp":
        if a.passive_root is None:
            raise ValueError("adapter.passive_root is required for kind 'mindlamp'")

        from cortex2_engine.adapters.mindlamp import MindLAMPAdapter

        return MindLAMPAdapter(
            study_id=config.name,
            passive_root=a.passive_root,
            participant_map=a.participant_map,
            source_id_column=config.windows.sensor_id_column,
            participant_id_column=config.windows.participant_id_column,
            source_manifest_dir=a.source_manifest_dir,
        )

    if a.config_path is None:
        raise ValueError("adapter.config_path is required for kind 'generic_csv'")

    from cortex2_engine.adapters.generic.csv import GenericCSVAdapter

    return GenericCSVAdapter(a.config_path)


def build_study_episodes(config: StudyConfig) -> pd.DataFrame:
    """Canonical episodes from the study's clinical windows (inclusive end dates)."""

    windows = load_clinical_windows(config.windows)

    return build_episodes_from_windows(
        windows,
        study_id=config.name,
        participant_id_column="participant_id",
        source_participant_id_column="sensor_id",
        start_column="clinical_start",
        end_column="clinical_end",
        inclusive_end_date=True,
        source_platform=config.adapter.kind if config.adapter else None,
    )
