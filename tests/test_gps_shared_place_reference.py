"""Tests for canonical per-participant place/home reference persistence."""
import numpy as np
import pandas as pd
import pytest

from cortex2_engine.features.gps_shared_place_reference import save_reference, load_reference
from cortex2_engine.features.gps_location_features import compute_home_away_location_entropy


def test_reference_roundtrip_and_same_home(tmp_path):
    assignments = pd.DataFrame({
        'date': ['2023-12-11', '2023-12-12', '2023-12-12'],
        'place_candidate_id': [1, 1, 2],
        'duration_ms': [5_000, 2_000, 3_000],
        'episode_start_ms': [1, 2, 3],
        'episode_end_ms': [2, 3, 4],
    })
    places = pd.DataFrame({'place_candidate_id': [1, 2]})
    home = {'home_place_candidate_id': 1, 'home_status': 'observed_or_reconstructed',
            'home_confidence': 0.8}
    save_reference('AMO327', assignments, places, home,
                   ['2023-12-11', '2023-12-12', '2023-12-13'], root=tmp_path)
    a, p, h, metadata = load_reference(
        'AMO327', required_dates=['2023-12-12', '2023-12-13'], root=tmp_path)
    pd.testing.assert_frame_equal(a, assignments)
    pd.testing.assert_frame_equal(p, places)
    assert h == home
    assert metadata['scope'].startswith('retrospective')
    d = a.loc[a.date.eq('2023-12-12')]
    result = compute_home_away_location_entropy(
        d, home_place_candidate_id=h['home_place_candidate_id'],
        home_status=h['home_status'])
    assert result['away_place_count'] == 1
    assert np.isclose(result['home_stationary_fraction'], 0.4)


def test_rejects_missing_dates_and_refuses_overwrite(tmp_path):
    a = pd.DataFrame({
        'date': ['2023-12-11'], 'place_candidate_id': [1],
        'duration_ms': [5.], 'episode_start_ms': [1], 'episode_end_ms': [2],
    })
    p = pd.DataFrame({'place_candidate_id': [1]})
    save_reference('AMO327', a, p, {'home_place_candidate_id': 1},
                   ['2023-12-11'], root=tmp_path)
    with pytest.raises(ValueError, match='does not cover'):
        load_reference('AMO327', required_dates=['2023-12-12'], root=tmp_path)
    with pytest.raises(FileExistsError):
        save_reference('AMO327', a, p, {'home_place_candidate_id': 1},
                       ['2023-12-11'], root=tmp_path)
