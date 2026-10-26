from enum import Enum


class FeatureStatus(str, Enum):
    """Interpretation status for a participant-feature-time observation."""

    OBSERVED = "observed"
    OBSERVED_ZERO = "observed_zero"
    NO_RAW_DATA = "no_raw_data"
    FEATURE_EMPTY = "feature_empty"
    EXTRACTION_FAILED = "extraction_failed"
    ANALYSIS_MASKED = "analysis_masked"
    NOT_APPLICABLE = "not_applicable"
