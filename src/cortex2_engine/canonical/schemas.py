from importlib.resources import files

import yaml


SCHEMA_NAMES = {
    "study",
    "episodes",
    "participants",
    "gps",
    "accelerometer",
    "screen",
    "source_intervals",
    "clinical",
    "clinical_windows",

    "device_usage",
    "nearby_device",}


def load_schema(name: str) -> dict:
    """
    Load one Cortex 2.0 canonical schema specification.
    """

    if name not in SCHEMA_NAMES:
        raise ValueError(
            f"Unknown Cortex 2.0 schema: {name}. "
            f"Available: {sorted(SCHEMA_NAMES)}"
        )

    path = (
        files("cortex2_engine")
        / "schema_specs"
        / f"{name}.yaml"
    )

    with path.open("r") as f:
        return yaml.safe_load(f)


def list_schemas() -> list[str]:
    return sorted(SCHEMA_NAMES)
