import pandas as pd
import pytest

from raphe.features.gps_anchor_episodes import (
    build_anchor_compatible_location_episodes,
)

from raphe.features.gps_spatial_semantics import (
    SPATIAL_STATUS_ACCURACY_OVERLAP,
    SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY,
    SPATIAL_STATUS_TEMPORAL_GAP,
)


SECOND = 1000


def _edge(
    start,
    end,
    *,
    lat1=0.0,
    lat2=0.0,
    lon1=0.0,
    lon2=0.0,
    accuracy=10.0,
    status=SPATIAL_STATUS_ACCURACY_OVERLAP,
    supported=True,
    distance_km=0.0,
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


def test_stable_edges_form_one_anchor_episode():

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
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 1

    assert (
        result.iloc[0][
            "duration_ms"
        ]
        == 3 * SECOND
    )

    assert (
        result.iloc[0][
            "n_edges"
        ]
        == 3
    )


def test_anchor_rule_breaks_transitive_chain():

    # About 7.8 m per step in latitude.
    #
    # Consecutive points remain within 10 m combined accuracy,
    # but the third point is ~15.6 m from the original anchor.
    step = 0.00007

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
            lat1=0.0,
            lat2=step,
            accuracy=5.0,
            distance_km=0.0078,
        ),
        _edge(
            SECOND,
            2 * SECOND,
            lat1=step,
            lat2=2 * step,
            accuracy=5.0,
            distance_km=0.0078,
        ),
    ])

    result = (
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 2

    assert result[
        "n_edges"
    ].tolist() == [
        1,
        1,
    ]


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
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 2


def test_temporal_gap_breaks_episode():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
        ),
        _edge(
            SECOND,
            10 * SECOND,
            status=(
                SPATIAL_STATUS_TEMPORAL_GAP
            ),
            supported=False,
        ),
        _edge(
            10 * SECOND,
            11 * SECOND,
        ),
    ])

    result = (
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 2


def test_nonadjacent_edges_do_not_merge():

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
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    assert len(
        result
    ) == 2


def test_no_eligible_edges_returns_empty():

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
            status=(
                SPATIAL_STATUS_DISPLACEMENT_EXCEEDS_ACCURACY
            ),
        ),
    ])

    result = (
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    assert result.empty


def test_anchor_ratio_never_exceeds_one_inside_episode():

    step = 0.00002

    edges = pd.DataFrame([
        _edge(
            0,
            SECOND,
            lat1=0.0,
            lat2=step,
            accuracy=10,
            distance_km=0.0022,
        ),
        _edge(
            SECOND,
            2 * SECOND,
            lat1=step,
            lat2=2 * step,
            accuracy=10,
            distance_km=0.0022,
        ),
    ])

    result = (
        build_anchor_compatible_location_episodes(
            edges
        )
    )

    assert (
        result[
            "max_anchor_ratio"
        ]
        <= 1.0
    ).all()


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
        build_anchor_compatible_location_episodes(
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
        "start_ms": [0],
        "end_ms": [SECOND],
    })

    with pytest.raises(
        ValueError
    ):
        build_anchor_compatible_location_episodes(
            edges
        )
