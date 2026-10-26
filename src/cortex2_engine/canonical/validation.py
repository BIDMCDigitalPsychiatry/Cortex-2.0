from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cortex2_engine.canonical.schemas import load_schema


@dataclass
class ValidationReport:
    schema: str
    n_rows: int
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str):
        self.valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)


def _check_required_columns(
    df: pd.DataFrame,
    schema: dict,
    report: ValidationReport,
):
    required = list(
        schema.get("required", {}).keys()
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        report.add_error(
            f"Missing required columns: {missing}"
        )


def _check_required_nulls(
    df: pd.DataFrame,
    schema: dict,
    report: ValidationReport,
):
    for col in schema.get("required", {}):

        if col not in df.columns:
            continue

        n = int(
            df[col].isna().sum()
        )

        if n:
            report.add_error(
                f"{col}: {n} missing required values"
            )


def _check_numeric_constraints(
    df: pd.DataFrame,
    schema: dict,
    report: ValidationReport,
):
    constraints = schema.get(
        "constraints",
        {}
    )

    for col, rules in constraints.items():

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if "min" in rules:

            n = int(
                (values < rules["min"])
                .fillna(False)
                .sum()
            )

            if n:
                report.add_error(
                    f"{col}: {n} values below "
                    f"{rules['min']}"
                )

        if "max" in rules:

            n = int(
                (values > rules["max"])
                .fillna(False)
                .sum()
            )

            if n:
                report.add_error(
                    f"{col}: {n} values above "
                    f"{rules['max']}"
                )


def _check_timestamps(
    df: pd.DataFrame,
    report: ValidationReport,
):
    if (
        "timestamp_ms" not in df.columns
        or "timestamp_utc" not in df.columns
    ):
        return

    ms = pd.to_numeric(
        df["timestamp_ms"],
        errors="coerce",
    )

    dt = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    invalid_dt = int(
        dt.isna().sum()
    )

    if invalid_dt:
        report.add_error(
            f"timestamp_utc: {invalid_dt} invalid values"
        )
        return

    reconstructed = (
        dt.astype("int64")
        // 1_000_000
    )

    diff = (
        reconstructed
        - ms
    ).abs()

    mismatch = int(
        (diff > 1).fillna(False).sum()
    )

    if mismatch:
        report.add_error(
            f"{mismatch} rows have timestamp_ms and "
            "timestamp_utc differing by >1 ms"
        )


def _check_allowed_events(
    df: pd.DataFrame,
    schema: dict,
    report: ValidationReport,
):
    allowed = schema.get(
        "allowed_events"
    )

    if (
        not allowed
        or "event" not in df.columns
    ):
        return

    invalid = (
        ~df["event"].isin(allowed)
        & df["event"].notna()
    )

    n = int(invalid.sum())

    if n:
        values = (
            df.loc[invalid, "event"]
            .astype(str)
            .unique()[:10]
            .tolist()
        )

        report.add_error(
            f"{n} invalid event values. "
            f"Examples: {values}"
        )


def validate_canonical(
    df: pd.DataFrame,
    schema_name: str,
) -> ValidationReport:
    """
    Validate a DataFrame against a Cortex 2.0 canonical schema.

    Validation never silently modifies or deletes data.
    """

    schema = load_schema(
        schema_name
    )

    report = ValidationReport(
        schema=schema_name,
        n_rows=len(df),
    )

    _check_required_columns(
        df,
        schema,
        report,
    )

    if not report.valid:
        return report

    _check_required_nulls(
        df,
        schema,
        report,
    )

    _check_numeric_constraints(
        df,
        schema,
        report,
    )

    _check_timestamps(
        df,
        report,
    )

    _check_allowed_events(
        df,
        schema,
        report,
    )

    return report
