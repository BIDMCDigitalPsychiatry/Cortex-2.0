import pandas as pd

from cortex2_engine.quality.source_availability import (
    SourceBoundaryStatus,
    classify_source_interval,
    resolve_source_boundary,
    source_interval_overlap,
)


def test_stream_specific_boundary_has_priority():
    df = pd.DataFrame([
        {
            "episode_id": "E1",
            "stream": None,
            "source_start_ms": None,
            "source_end_ms": 1000,
            "boundary_source": "participant",
        },
        {
            "episode_id": "E1",
            "stream": "gps",
            "source_start_ms": None,
            "source_end_ms": 900,
            "boundary_source": "sensor",
        },
    ])

    r = resolve_source_boundary(
        df,
        episode_id="E1",
        stream="gps",
    )

    assert r["boundary_scope"] == "stream"
    assert r["source_end_ms"] == 900


def test_participant_boundary_is_fallback():
    df = pd.DataFrame([
        {
            "episode_id": "E1",
            "stream": None,
            "source_start_ms": None,
            "source_end_ms": 1000,
            "boundary_source": "participant",
        },
    ])

    r = resolve_source_boundary(
        df,
        episode_id="E1",
        stream="accelerometer",
    )

    assert r["boundary_scope"] == "participant"
    assert r["source_end_ms"] == 1000


def test_unknown_boundary_stays_unknown():
    df = pd.DataFrame(columns=[
        "episode_id",
        "stream",
        "source_start_ms",
        "source_end_ms",
        "boundary_source",
    ])

    r = resolve_source_boundary(
        df,
        episode_id="E1",
        stream="gps",
    )

    assert r["boundary_known"] is False


def test_interval_after_declared_end_is_outside():
    status = classify_source_interval(
        requested_start_ms=1000,
        requested_end_ms=2000,
        source_end_ms=1000,
    )

    assert (
        status
        == SourceBoundaryStatus.OUTSIDE_DECLARED_END
    )


def test_partial_source_overlap_is_quantified():
    r = source_interval_overlap(
        requested_start_ms=0,
        requested_end_ms=1000,
        source_end_ms=500,
    )

    assert r["overlap_ms"] == 500
    assert r["source_available_fraction"] == 0.5
    assert r["overlap_start_ms"] == 0
    assert r["overlap_end_ms"] == 500
