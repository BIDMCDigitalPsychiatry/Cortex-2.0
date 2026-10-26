from enum import Enum


class StreamEvidenceContext(str, Enum):
    """
    Evidence available for interpreting sensor observability.

    These states describe evidence, not assumptions about whether
    a device was actually prescribed, worn, enabled, or functioning.
    """

    DECLARED_STREAM_WITH_EVIDENCE = (
        "declared_stream_boundary__evidence_present"
    )

    DECLARED_STREAM_NO_EVIDENCE = (
        "declared_stream_boundary__evidence_absent"
    )

    PARTICIPANT_BOUNDARY_WITH_EVIDENCE = (
        "participant_boundary_only__evidence_present"
    )

    PARTICIPANT_BOUNDARY_NO_EVIDENCE = (
        "participant_boundary_only__evidence_absent"
    )

    BOUNDARY_UNKNOWN_WITH_EVIDENCE = (
        "boundary_unknown__evidence_present"
    )

    BOUNDARY_UNKNOWN_NO_EVIDENCE = (
        "boundary_unknown__evidence_absent"
    )


def classify_stream_evidence_context(
    *,
    boundary_known: bool,
    boundary_scope: str | None,
    has_stream_evidence: bool,
):
    """
    Combine declared source-boundary provenance with observed
    episode-level stream evidence.

    This function does not infer instrumentation or sensor failure.
    """

    has_stream_evidence = bool(
        has_stream_evidence
    )

    if not boundary_known:
        return (
            StreamEvidenceContext.BOUNDARY_UNKNOWN_WITH_EVIDENCE
            if has_stream_evidence
            else StreamEvidenceContext.BOUNDARY_UNKNOWN_NO_EVIDENCE
        )

    if boundary_scope == "stream":
        return (
            StreamEvidenceContext.DECLARED_STREAM_WITH_EVIDENCE
            if has_stream_evidence
            else StreamEvidenceContext.DECLARED_STREAM_NO_EVIDENCE
        )

    if boundary_scope == "participant":
        return (
            StreamEvidenceContext.PARTICIPANT_BOUNDARY_WITH_EVIDENCE
            if has_stream_evidence
            else StreamEvidenceContext.PARTICIPANT_BOUNDARY_NO_EVIDENCE
        )

    raise ValueError(
        f"Unsupported boundary_scope: {boundary_scope}"
    )
