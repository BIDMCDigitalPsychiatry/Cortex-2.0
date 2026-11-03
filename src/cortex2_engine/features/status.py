"""Unified interpretation status for Cortex 2.0 feature values.

Every scalar feature ``<name>`` is emitted with a ``<name>_status`` string drawn
from :class:`FeatureStatus`. The status says how the value may be interpreted;
it is not a quality score. See ``docs/status_vocabulary.md``.

What this is NOT: a spatial or trajectory evidence label. ``spatial_status``
and ``trajectory_status`` on GPS edges describe one edge's geometry and keep
their own vocabularies (``gps_spatial_semantics``,
``gps_trajectory_plausibility``).
"""

from __future__ import annotations

import math
from enum import Enum


class FeatureStatus(str, Enum):
    """Interpretation status for a participant-feature-time observation."""

    OBSERVED = "observed"
    """Value computed with adequate observability."""

    OBSERVED_ZERO = "observed_zero"
    """Adequate observability and the construct is genuinely absent."""

    INSUFFICIENT_OBSERVABILITY = "insufficient_observability"
    """Some data, below the feature's documented minimum; value is NaN."""

    NO_RAW_DATA = "no_raw_data"
    """Stream present for the episode, zero records in the window."""

    SOURCE_ABSENT = "source_absent"
    """Stream not available for this episode/platform (e.g. telephony on iOS)."""

    EXTRACTION_FAILED = "extraction_failed"
    """Processing error; value NaN, ``error_type`` populated."""

    ANALYSIS_MASKED = "analysis_masked"
    """Excluded by a downstream rule; value retained."""

    CENSORED = "censored"
    """Value is a lower bound (bout hit a cap, window truncated)."""

    NOT_APPLICABLE = "not_applicable"
    """The construct is undefined for this unit (e.g. median of zero bouts)."""

    FEATURE_EMPTY = "feature_empty"
    """Deprecated. Never emitted; kept so old outputs still parse."""


#: Statuses under which a numeric value may be interpreted behaviourally.
VALUE_BEARING_STATUSES = frozenset({
    FeatureStatus.OBSERVED,
    FeatureStatus.OBSERVED_ZERO,
    FeatureStatus.CENSORED,
    FeatureStatus.ANALYSIS_MASKED,
})

#: Members that feature code may emit (everything except deprecated ones).
EMITTABLE_STATUSES = frozenset(
    member for member in FeatureStatus
    if member is not FeatureStatus.FEATURE_EMPTY
)


def _is_missing(
    value: object,
) -> bool:

    if value is None:
        return True

    try:
        return math.isnan(float(value))

    except (TypeError, ValueError):
        return False


def status_from_counts(
    n_observed_ms: float | None,
    minimum_observed_ms: float,
    value: float | None,
    *,
    n_records: int | None = None,
) -> FeatureStatus:
    """Derive a feature status from observed time and the computed value.

    Parameters
    ----------
    n_observed_ms
        Observed (supported) time behind the value, in milliseconds.
    minimum_observed_ms
        The feature's documented minimum observed time.
    value
        The computed value (NaN/None when it could not be computed).
    n_records
        Raw record count in the window, when known. ``0`` yields
        ``no_raw_data`` regardless of ``n_observed_ms``.

    Returns
    -------
    FeatureStatus
        ``no_raw_data`` → ``insufficient_observability`` → ``not_applicable``
        (missing value at adequate observability) → ``observed_zero`` →
        ``observed``, in that order of precedence.

    Notes
    -----
    Observed time, not record count, decides adequacy (rule 1). A value of
    exactly zero is only ``observed_zero`` when observability is adequate
    (rule 2).
    """

    if minimum_observed_ms < 0:
        raise ValueError(
            "minimum_observed_ms must be >= 0"
        )

    if n_records is not None and int(n_records) == 0:
        return FeatureStatus.NO_RAW_DATA

    if (
        _is_missing(n_observed_ms)
        or float(n_observed_ms) <= 0
        or float(n_observed_ms) < minimum_observed_ms
    ):
        return FeatureStatus.INSUFFICIENT_OBSERVABILITY

    if _is_missing(value):
        return FeatureStatus.NOT_APPLICABLE

    if float(value) == 0.0:
        return FeatureStatus.OBSERVED_ZERO

    return FeatureStatus.OBSERVED


def gated_value(
    value: float | None,
    status: FeatureStatus | str,
) -> float:
    """Return ``value`` only when ``status`` licenses interpreting it.

    Parameters
    ----------
    value
        Computed value.
    status
        Its status.

    Returns
    -------
    float
        ``float(value)`` for value-bearing statuses, otherwise NaN.
    """

    status = FeatureStatus(status)

    if status in VALUE_BEARING_STATUSES and not _is_missing(value):
        return float(value)

    return float("nan")
