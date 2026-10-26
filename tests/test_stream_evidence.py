from cortex2_engine.quality.stream_evidence import (
    StreamEvidenceContext,
    classify_stream_evidence_context,
)


def test_declared_stream_with_evidence():
    result = classify_stream_evidence_context(
        boundary_known=True,
        boundary_scope="stream",
        has_stream_evidence=True,
    )

    assert result == (
        StreamEvidenceContext.DECLARED_STREAM_WITH_EVIDENCE
    )


def test_participant_boundary_without_evidence():
    result = classify_stream_evidence_context(
        boundary_known=True,
        boundary_scope="participant",
        has_stream_evidence=False,
    )

    assert result == (
        StreamEvidenceContext.PARTICIPANT_BOUNDARY_NO_EVIDENCE
    )


def test_unknown_boundary_with_evidence():
    result = classify_stream_evidence_context(
        boundary_known=False,
        boundary_scope=None,
        has_stream_evidence=True,
    )

    assert result == (
        StreamEvidenceContext.BOUNDARY_UNKNOWN_WITH_EVIDENCE
    )


def test_unknown_boundary_without_evidence():
    result = classify_stream_evidence_context(
        boundary_known=False,
        boundary_scope=None,
        has_stream_evidence=False,
    )

    assert result == (
        StreamEvidenceContext.BOUNDARY_UNKNOWN_NO_EVIDENCE
    )
