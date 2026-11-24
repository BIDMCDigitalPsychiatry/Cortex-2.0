"""Persistent retrospective GPS place/home reference shared by Cortex workflows.

References are per participant and built once from the union of configured raw
sources. Never silently fall back to re-clustering a different source window.
"""
from pathlib import Path
import json
import os
import re

import numpy as np
import pandas as pd

REFERENCE_ROOT = Path(os.environ.get(
    "CORTEX2_PLACE_REFERENCE_DIR",
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/shared_place_reference_v1",
))
SCHEMA_VERSION = 1


def _folder(participant_id, root=None):
    name = str(participant_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Unsafe participant identifier for reference directory")
    return Path(root or REFERENCE_ROOT) / name


def save_reference(participant_id, assignments, places, home, dates,
                   *, root=None, compatibility_multiplier=1.0, source_names=()):
    folder = _folder(participant_id, root)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Reference already exists; refusing overwrite: {folder}")
    required = {"date", "place_candidate_id", "duration_ms", "episode_start_ms", "episode_end_ms"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"Assignments lack {sorted(required - set(assignments.columns))}")
    if assignments["date"].isna().any():
        raise ValueError("Null dates in persistent reference")
    unique_dates = sorted({str(d) for d in dates})
    if not set(assignments["date"].astype(str)).issubset(unique_dates):
        raise ValueError("Assignment dates outside reference coverage")
    folder.mkdir(parents=True, exist_ok=False)
    assignments.to_parquet(folder / "assignments.parquet", index=False)
    places.to_parquet(folder / "places.parquet", index=False)
    def native(x):
        if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "participant_id": str(participant_id),
        "scope": "retrospective_union_of_configured_raw_sources",
        "dates": unique_dates,
        "compatibility_multiplier": compatibility_multiplier,
        "source_names": list(source_names),
        "home": {k: native(v) for k, v in home.items()},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return folder


def load_reference(participant_id, *, required_dates=(), root=None):
    folder = _folder(participant_id, root)
    metadata_path = folder / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing shared place reference: {folder}. Build it with "
            "workflows/build_gps_shared_place_reference.py before running the workflow."
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unrecognized shared place reference schema")
    if metadata.get("participant_id") != str(participant_id):
        raise ValueError("Reference participant mismatch")
    uncovered = {str(d) for d in required_dates} - set(metadata["dates"])
    if uncovered:
        raise ValueError(
            f"Shared reference for {participant_id} does not cover "
            f"requested dates: {sorted(uncovered)[:8]}. Rebuild the reference "
            "from the combined source-day manifest, not per workflow."
        )
    assignments = pd.read_parquet(folder / "assignments.parquet")
    places = pd.read_parquet(folder / "places.parquet")
    if not assignments.empty:
        if not {"date", "place_candidate_id", "duration_ms"}.issubset(assignments.columns):
            raise ValueError("Malformed saved assignments")
    return assignments, places, metadata["home"], metadata
