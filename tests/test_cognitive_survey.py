import numpy as np
import pandas as pd
import pytest
from gps_synthetic import local_ms
from pydantic import ValidationError

from cortex2_engine.features.cognitive_variability import (
    compute_session_variability,
    ex_gaussian_tau_mom,
    session_rt_variability,
    weekly_cognitive_summary,
)
from cortex2_engine.features.survey_scoring import (
    SurveyScoringSpec,
    load_builtin_spec,
    normalise_text,
    score_surveys,
)

TZ = "America/New_York"


def test_ex_gaussian_tau_recovers_simulated_value():
    rng = np.random.default_rng(3)
    rt = rng.normal(400, 40, 200_000) + rng.exponential(150, 200_000)
    assert ex_gaussian_tau_mom(rt) == pytest.approx(150, rel=0.05)

    assert np.isnan(ex_gaussian_tau_mom(np.array([1.0, 2.0])))
    assert np.isnan(ex_gaussian_tau_mom(np.array([5.0, 5.0, 5.0])))
    assert np.isnan(ex_gaussian_tau_mom(-rng.exponential(1, 1000)))  # negative skew
    # Extreme skew is capped at the ex-Gaussian maximum (tau = SD).
    spike = np.array([1.0] * 50 + [1000.0])
    assert ex_gaussian_tau_mom(spike) == pytest.approx(np.std(spike, ddof=1))


def _trials(rts, start=0, task="jewels_a", session="s1"):
    return pd.DataFrame({
        "task": task,
        "session_id": session,
        "timestamp_ms": start + np.arange(len(rts)) * 2000,
        "response_time_ms": rts,
    })


def test_session_trimming_and_metrics():
    rts = [100, 500, 520, 540, 560, 580, 600, 620, 640, 660, 5000]
    out = session_rt_variability(_trials(rts))

    assert out["n_trials"] == 11
    assert out["n_trials_kept"] == 9  # 100 ms anticipation and 5 s lapse removed
    assert out["rt_slope_ms_per_trial"] == pytest.approx(20.0)
    kept = np.array(rts[1:-1], float)
    assert out["rt_cov"] == pytest.approx(kept.std(ddof=1) / kept.mean())
    assert out["rt_tau_ms_status"] == "not_applicable"  # symmetric: no skew
    assert out["rt_cov_status"] == "observed"


def test_session_gates():
    few = session_rt_variability(_trials([500, 510, 520]))
    assert few["rt_cov_status"] == "insufficient_observability"

    none = session_rt_variability(_trials([np.nan, np.nan]))
    assert none["rt_cov_status"] == "no_raw_data"

    flat = session_rt_variability(_trials([500] * 10))
    assert flat["rt_cov"] == 0.0 and flat["rt_cov_status"] == "observed_zero"
    assert flat["rt_slope_ms_per_trial_status"] in ("observed_zero", "observed")

    with pytest.raises(ValueError):
        session_rt_variability(_trials([500] * 10), min_trials=2)


def test_sessions_and_weekly_summary():
    rng = np.random.default_rng(4)
    frames = []
    for i, day in enumerate(["2026-06-01", "2026-06-03", "2026-06-05", "2026-06-09"]):
        frames.append(_trials(rng.normal(500, 50, 20).clip(200), start=local_ms(day, 10, TZ), session=f"s{i}"))
    frames.append(_trials([500, 510], start=local_ms("2026-06-02", 10, TZ), session="short"))
    cognitive = pd.concat(frames, ignore_index=True)

    sessions = compute_session_variability(cognitive, tz=TZ)
    assert len(sessions) == 5
    assert sessions.loc[sessions["session_id"] == "s3", "date_local"].item() == "2026-06-09"

    weekly = weekly_cognitive_summary(sessions)
    first = weekly[weekly["week_start"] == "2026-06-01"].iloc[0]
    assert first["n_sessions"] == 4
    assert first["rt_cov_n_sessions"] == 3
    assert first["rt_cov_weekly_median"] == pytest.approx(
        sessions.loc[sessions["session_id"].isin(["s0", "s1", "s2"]), "rt_cov"].median()
    )
    assert weekly["week_start"].tolist() == ["2026-06-01", "2026-06-08"]

    only_short = weekly_cognitive_summary(sessions[sessions["session_id"] == "short"])
    assert only_short["rt_cov_weekly_median_status"].item() == "insufficient_observability"

    assert compute_session_variability(cognitive.iloc[0:0], tz=TZ).empty
    assert weekly_cognitive_summary(sessions.iloc[0:0]).empty
    with pytest.raises(ValueError, match="Missing required cognitive columns"):
        compute_session_variability(cognitive.drop(columns="task"), tz=TZ)


PHQ = load_builtin_spec("phq9")
ITEMS = list(PHQ.questions)


def _survey(answers, *, activity="act1", participant="U1", start=0, instrument="PHQ-9"):
    return pd.DataFrame({
        "timestamp_ms": start + np.arange(len(answers)) * 1000,
        "instrument": instrument,
        "item": [q for q, _ in answers],
        "value_text": [str(v) for _, v in answers],
        "value_numeric": pd.to_numeric(pd.Series([v for _, v in answers]), errors="coerce"),
        "activity_id": activity,
        "source_participant_id": participant,
    })


def test_phq9_scoring_from_text_and_numbers():
    answers = [(q, "Several days") for q in ITEMS[:5]] + [(q, 2) for q in ITEMS[5:]]
    answers.append(("Free text comment", "felt ok"))
    out = score_surveys(_survey(answers), PHQ)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["score"] == 5 + 8
    assert row["score_status"] == "observed"
    assert row["n_items_valid"] == 9 and row["n_items_expected"] == 9
    assert row["participant_id"] == "U1"


def test_minimum_items_no_prorating_invalid_and_zero():
    seven = [(q, 1) for q in ITEMS[:7]]
    assert score_surveys(_survey(seven), PHQ)["score"].item() == 7

    six = [(q, 1) for q in ITEMS[:6]] + [(ITEMS[6], 9)]  # 9 is out of range
    out = score_surveys(_survey(six), PHQ).iloc[0]
    assert np.isnan(out["score"])
    assert out["score_status"] == "insufficient_observability"
    assert out["n_items_invalid"] == 1

    zero = score_surveys(_survey([(q, "Not at all") for q in ITEMS]), PHQ).iloc[0]
    assert zero["score"] == 0.0 and zero["score_status"] == "observed_zero"


def test_aliases_sessions_duplicates_and_other_instruments():
    aliased = [(f"phq9_{i}", 3) for i in range(1, 10)]
    a = _survey(aliased, activity=None, instrument="phq9")
    b = _survey(aliased, activity=None, instrument="phq9", start=10 * 3_600_000)
    out = score_surveys(pd.concat([a, b], ignore_index=True), PHQ)
    assert out["score"].tolist() == [27.0, 27.0]
    assert out["session_id"].nunique() == 2

    dup = _survey([(q, 0) for q in ITEMS] + [(ITEMS[0], 3)])
    assert score_surveys(dup, PHQ)["score"].item() == 3

    gad = load_builtin_spec("gad7")
    assert score_surveys(_survey([(q, 1) for q in ITEMS]), gad).empty
    assert score_surveys(_survey([("unrelated", 1)]), PHQ).empty


def test_reverse_items_and_spec_validation():
    spec = SurveyScoringSpec(
        instrument="Mini",
        category_list=["a", "b"],
        questions={"q1": {"category": "a"}, "q2": {"category": "a"}, "q3": {"category": "b"}},
        reverse_items=["q2"],
        item_min=1,
        item_max=5,
        min_items={"a": 1},
    )
    df = _survey([("q1", 4), ("q2", 4), ("q3", 5)], instrument="mini").drop(columns=["value_text"])
    out = score_surveys(df, spec).set_index("category")
    assert out.loc["a", "score"] == 4 + 2
    assert out.loc["b", "score"] == 5

    base = {"instrument": "x", "category_list": ["t"], "item_min": 0, "item_max": 3,
            "questions": {"q": {"category": "t"}}}
    for bad in (
        {"item_max": 0},
        {"questions": {"q": {"category": "nope"}}},
        {"questions": {"q": {"category": "t", "map": "nope"}}},
        {"reverse_items": ["zz"]},
        {"min_items": {"t": 2}},
    ):
        with pytest.raises(ValidationError):
            SurveyScoringSpec(**{**base, **bad})

    with pytest.raises(ValueError, match="No built-in survey spec"):
        load_builtin_spec("nope")
    with pytest.raises(ValueError, match="Missing required survey columns"):
        score_surveys(pd.DataFrame({"item": []}), PHQ)
    assert normalise_text("  Feeling  down, depressed! ") == "feeling down depressed"
