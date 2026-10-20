import pandas as pd
import pytest

from raphe.features.gps_place_candidates import (
    build_place_candidates,
)


def _segment(
    start,
    *,
    lat,
    duration=1000,
    anchor_accuracy=5.0,
    spread=5.0,
):
    return {
        "episode_start_ms":
            start,

        "episode_end_ms":
            start + duration,

        "duration_ms":
            duration,

        "anchor_latitude":
            lat,

        "anchor_longitude":
            0.0,

        "anchor_accuracy_m":
            anchor_accuracy,

        "representative_latitude":
            lat,

        "representative_longitude":
            0.0,

        "max_anchor_distance_m":
            spread,
    }


def test_nearby_segments_share_candidate():

    # ~11 m apart; each support radius = 10 m.
    episodes = pd.DataFrame([
        _segment(
            0,
            lat=0.0,
        ),
        _segment(
            1000,
            lat=0.00010,
        ),
    ])

    assignments, places = (
        build_place_candidates(
            episodes,
            compatibility_multiplier=1.0,
        )
    )

    assert len(
        places
    ) == 1

    assert assignments[
        "place_candidate_id"
    ].nunique() == 1


def test_far_segments_are_separate():

    episodes = pd.DataFrame([
        _segment(
            0,
            lat=0.0,
        ),
        _segment(
            1000,
            lat=0.001,
        ),
    ])

    _, places = (
        build_place_candidates(
            episodes,
            compatibility_multiplier=1.0,
        )
    )

    assert len(
        places
    ) == 2


def test_fixed_seed_prevents_transitive_chaining():

    # Approx:
    # A-B = 15 m
    # B-C = 15 m
    # A-C = 30 m
    #
    # Radius sum = 20 m.
    #
    # B can join A.
    # C must not join A merely because it overlaps B.

    episodes = pd.DataFrame([
        _segment(
            0,
            lat=0.0,
            duration=3000,
        ),
        _segment(
            1000,
            lat=0.000135,
            duration=2000,
        ),
        _segment(
            2000,
            lat=0.000270,
            duration=1000,
        ),
    ])

    assignments, places = (
        build_place_candidates(
            episodes,
            compatibility_multiplier=1.0,
        )
    )

    assert len(
        places
    ) == 2

    assert assignments[
        "place_candidate_id"
    ].tolist() == [
        0,
        0,
        1,
    ]


def test_longest_segment_becomes_first_seed():

    episodes = pd.DataFrame([
        _segment(
            0,
            lat=0.0,
            duration=1000,
        ),
        _segment(
            1000,
            lat=0.01,
            duration=5000,
        ),
    ])

    _, places = (
        build_place_candidates(
            episodes
        )
    )

    assert places.iloc[0][
        "seed_episode_start_ms"
    ] == 1000


def test_multiplier_controls_compatibility():

    episodes = pd.DataFrame([
        _segment(
            0,
            lat=0.0,
        ),
        _segment(
            1000,
            lat=0.00027,
        ),
    ])

    _, strict_places = (
        build_place_candidates(
            episodes,
            compatibility_multiplier=1.0,
        )
    )

    _, relaxed_places = (
        build_place_candidates(
            episodes,
            compatibility_multiplier=2.0,
        )
    )

    assert len(
        strict_places
    ) == 2

    assert len(
        relaxed_places
    ) == 1


def test_place_duration_is_preserved():

    episodes = pd.DataFrame([
        _segment(
            0,
            lat=0.0,
            duration=2000,
        ),
        _segment(
            2000,
            lat=0.00005,
            duration=3000,
        ),
    ])

    _, places = (
        build_place_candidates(
            episodes
        )
    )

    assert places.iloc[0][
        "total_duration_ms"
    ] == pytest.approx(
        5000
    )


def test_invalid_multiplier_raises():

    episodes = pd.DataFrame([
        _segment(
            0,
            lat=0.0,
        )
    ])

    with pytest.raises(
        ValueError
    ):
        build_place_candidates(
            episodes,
            compatibility_multiplier=0,
        )


def test_missing_column_raises():

    episodes = pd.DataFrame({
        "duration_ms": [
            1000
        ]
    })

    with pytest.raises(
        ValueError
    ):
        build_place_candidates(
            episodes
        )
