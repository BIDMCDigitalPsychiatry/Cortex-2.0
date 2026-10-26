import pandas as pd
import pytest

from cortex2_engine.features.gps_location_episodes import (
    build_location_compatible_episodes,
)

from cortex2_engine.features.gps_spatial_semantics import (
    SPATIAL_STATUS_ACCURACY_OVERLAP,
    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
    SPATIAL_STATUS_TEMPORAL_GAP,
)


SECOND = 1000


def _edge(
    start,
    end,
    *,
    status=SPATIAL_STATUS_ACCURACY_OVERLAP,
    supported=True,
    lat1=42.0,
    lon1=-71.0,
    lat2=42.0,
    lon2=-71.0,
    distance_km=0.0,
    accuracy=10.0,
):
    return {
        "start_ms":
            start,

        "end_ms":
            end,

        "duration_ms":
            end - start,

        "supported":
            supported,

        "spatial_status":
            status,

        "start_latitude":
            lat1,

        "start_longitude":
            lon1,

        "end_latitude":
            lat2,

        "end_longitude":
            lon2,

        "distance_km":
            distance_km,

        "start_accuracy":
            accuracy,

        "end_accuracy":
            accuracy,
    }


def test_contiguous_overlap_edges_form_one_episode():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
        ),
        _edge(
            SECOND,
            2 * SECOND,
        ),
        _edge(
            2 * SECOND,
            3 * SECOND,
        ),
    ])

    result = (
        build_location_compatible_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 1

    row = result.iloc[0]

    assert (
        row["episode_start_ms"]
        == 0
    )

    assert (
        row["episode_end_ms"]
        == 3 * SECOND
    )

    assert (
        row["duration_ms"]
        == 3 * SECOND
    )

    assert (
        row["n_edges"]
        == 3
    )


def test_displacement_edge_breaks_episode():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
        ),

        _edge(
            SECOND,
            2 * SECOND,
            status=(
                SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
            ),
        ),

        _edge(
            2 * SECOND,
            3 * SECOND,
        ),
    ])

    result = (
        build_location_compatible_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 2

    assert (
        result[
            "duration_ms"
        ].tolist()
        == [
            SECOND,
            SECOND,
        ]
    )


def test_temporal_gap_breaks_episode():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
        ),

        _edge(
            SECOND,
            10 * SECOND,
            supported=False,
            status=(
                SPATIAL_STATUS_TEMPORAL_GAP
            ),
        ),

        _edge(
            10 * SECOND,
            11 * SECOND,
        ),
    ])

    result = (
        build_location_compatible_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 2


def test_nonadjacent_overlap_edges_do_not_merge():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
        ),

        _edge(
            5 * SECOND,
            6 * SECOND,
        ),
    ])

    result = (
        build_location_compatible_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 2


def test_no_overlap_edges_returns_empty():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
            status=(
                SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
            ),
        )
    ])

    result = (
        build_location_compatible_episodes(
            edges
        )
    )

    assert result.empty


def test_representative_location_is_duration_weighted():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
            lat1=0.0,
            lon1=0.0,
            lat2=0.0,
            lon2=0.0,
        ),

        _edge(
            SECOND,
            3 * SECOND,
            lat1=2.0,
            lon1=0.0,
            lat2=2.0,
            lon2=0.0,
        ),
    ])

    result = (
        build_location_compatible_episodes(
            edges
        )
    )

    row = result.iloc[0]

    assert (
        row[
            "representative_latitude"
        ]
        == pytest.approx(
            4 / 3
        )
    )

    assert (
        row[
            "representative_longitude"
        ]
        == pytest.approx(
            0.0
        )
    )


def test_path_distance_is_preserved():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
            distance_km=0.002,
        ),

        _edge(
            SECOND,
            2 * SECOND,
            distance_km=0.003,
        ),
    ])

    result = (
        build_location_compatible_episodes(
            edges
        )
    )

    assert (
        result.iloc[0][
            "path_distance_m"
        ]
        == pytest.approx(
            5.0
        )
    )


def test_missing_required_column_raises():

    edges = pd.DataFrame({
        "start_ms": [
            0
        ],
        "end_ms": [
            SECOND
        ],
    })

    with pytest.raises(
        ValueError
    ):
        build_location_compatible_episodes(
            edges
        )
