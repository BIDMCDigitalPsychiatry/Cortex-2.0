"""Survey scoring from the canonical survey stream.

Construct
---------
Instrument scores (for example PHQ-9 total) from one-row-per-item survey
responses. This ports the legacy Cortex ``scoring_dict`` idea (a category
list, per-question scoring rules and named answer maps) into a validated
pydantic model, :class:`SurveyScoringSpec`.

Rules
-----
- Items are matched on normalised text (lower case, punctuation removed,
  whitespace collapsed) against each question's text and ``aliases``.
- An item's value is the answer's ``value_text`` looked up in the
  question's ``map`` when one is set. If there is no map, or the text is not
  in it, ``value_numeric`` is used.
- Values outside ``[item_min, item_max]`` are invalid and are not scored.
- Reverse items score ``item_min + item_max - value``.
- A category score is the **sum** of its valid items when at least
  ``min_items[category]`` are valid (default: all). There is no pro-rating;
  otherwise the score is NaN with ``insufficient_observability``.
- A session is one ``activity_id``. Rows without one are grouped by
  ``session_gap_ms`` of silence.

Built-in specs: ``schema_specs/survey_specs/phq9.yaml`` (Kroenke et al. 2001)
and ``gad7.yaml`` (Spitzer et al. 2006), loaded with
:func:`load_builtin_spec`.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from cortex2_engine.features.status import FeatureStatus

SCORE_COLUMNS = [
    "participant_id",
    "instrument",
    "session_id",
    "timestamp_ms",
    "category",
    "score",
    "score_status",
    "n_items_valid",
    "n_items_expected",
    "n_items_invalid",
]


def normalise_text(value: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""

    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


class QuestionScoring(BaseModel):
    """Scoring rule for one question."""

    category: str
    map: str | None = None
    aliases: list[str] = Field(default_factory=list)


class SurveyScoringSpec(BaseModel):
    """Scoring specification for one instrument.

    Attributes
    ----------
    instrument
        Canonical instrument name; ``instrument_aliases`` are also accepted.
    category_list
        Categories scored (for example ``["total"]``).
    questions
        Question text → :class:`QuestionScoring`.
    maps
        Named answer maps: answer text → score.
    reverse_items
        Question texts scored ``item_min + item_max - value``.
    item_min, item_max
        Valid item range.
    min_items
        Valid items needed per category (default: every item).
    """

    instrument: str
    instrument_aliases: list[str] = Field(default_factory=list)
    category_list: list[str]
    questions: dict[str, QuestionScoring]
    maps: dict[str, dict[str, float]] = Field(default_factory=dict)
    reverse_items: list[str] = Field(default_factory=list)
    item_min: float
    item_max: float
    min_items: dict[str, int] = Field(default_factory=dict)
    reference: str | None = None
    aggregation: Literal["sum"] = "sum"

    @model_validator(mode="after")
    def _check(self) -> SurveyScoringSpec:

        if self.item_max <= self.item_min:
            raise ValueError("item_max must exceed item_min")

        for text, q in self.questions.items():
            if q.category not in self.category_list:
                raise ValueError(f"question {text!r}: unknown category {q.category!r}")
            if q.map is not None and q.map not in self.maps:
                raise ValueError(f"question {text!r}: unknown map {q.map!r}")

        unknown_reverse = set(self.reverse_items) - set(self.questions)
        if unknown_reverse:
            raise ValueError(f"reverse_items not in questions: {sorted(unknown_reverse)}")

        for category, n in self.min_items.items():
            n_items = sum(q.category == category for q in self.questions.values())
            if category not in self.category_list or not 1 <= n <= n_items:
                raise ValueError(f"min_items[{category!r}] must be in [1, {n_items}]")

        return self

    def matches_instrument(self, name: str) -> bool:
        """Whether ``name`` refers to this instrument."""

        names = {normalise_text(self.instrument)}
        names |= {normalise_text(a) for a in self.instrument_aliases}
        return normalise_text(name) in names

    def question_lookup(self) -> dict[str, str]:
        """Normalised text (and aliases) → canonical question text."""

        lookup = {}
        for text, q in self.questions.items():
            for variant in [text, *q.aliases]:
                lookup[normalise_text(variant)] = text
        return lookup


def load_builtin_spec(name: str) -> SurveyScoringSpec:
    """Load a packaged spec (``"phq9"`` or ``"gad7"``)."""

    path = resources.files("cortex2_engine.schema_specs").joinpath(
        f"survey_specs/{name}.yaml"
    )

    if not path.is_file():
        raise ValueError(f"No built-in survey spec named {name!r}")

    return SurveyScoringSpec.model_validate(yaml.safe_load(path.read_text()))


def _sessions(x: pd.DataFrame, session_gap_ms: float) -> pd.Series:

    t = pd.to_numeric(x["timestamp_ms"], errors="coerce")
    by_gap = (t.diff() > session_gap_ms).cumsum().astype(str)
    fallback = "t" + t.groupby(by_gap).transform("min").astype("Int64").astype(str)

    if "activity_id" in x.columns:
        activity = x["activity_id"].astype("string")
        return activity.where(activity.notna(), fallback).astype(str)

    return fallback


def score_surveys(
    survey: pd.DataFrame,
    spec: SurveyScoringSpec,
    *,
    session_gap_ms: float = 3_600_000,
) -> pd.DataFrame:
    """Score every session of ``spec.instrument`` in a survey table.

    Parameters
    ----------
    survey
        Canonical survey rows (``timestamp_ms``, ``instrument``, ``item``,
        ``value_numeric`` and/or ``value_text``; optional ``activity_id`` and
        ``participant_id``/``source_participant_id``).
    spec
        Scoring specification.
    session_gap_ms
        Silence that splits sessions without an ``activity_id`` (1 h).

    Returns
    -------
    pandas.DataFrame
        One row per participant × session × category (``SCORE_COLUMNS``).
        Unmatched items are ignored; duplicate answers keep the last.
    """

    missing = {"timestamp_ms", "instrument", "item"} - set(survey.columns)

    if missing:
        raise ValueError("Missing required survey columns: " + ", ".join(sorted(missing)))

    x = survey[survey["instrument"].astype(str).map(spec.matches_instrument)].copy()

    if x.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    lookup = spec.question_lookup()
    x["question"] = x["item"].astype(str).map(normalise_text).map(lookup)
    x = x[x["question"].notna()].sort_values("timestamp_ms", kind="mergesort")

    if x.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    participant_column = (
        "participant_id"
        if "participant_id" in x.columns and x["participant_id"].notna().any()
        else ("source_participant_id" if "source_participant_id" in x.columns else None)
    )
    x["_participant"] = x[participant_column].astype(str) if participant_column else ""

    numeric = (
        pd.to_numeric(x["value_numeric"], errors="coerce")
        if "value_numeric" in x.columns
        else pd.Series(np.nan, index=x.index)
    )
    text = (
        x["value_text"].astype("string").fillna("").map(normalise_text)
        if "value_text" in x.columns
        else pd.Series("", index=x.index)
    )

    maps = {
        name: {normalise_text(k): float(v) for k, v in m.items()}
        for name, m in spec.maps.items()
    }

    values = []
    for question, answer, number in zip(x["question"], text, numeric, strict=True):
        map_name = spec.questions[question].map
        value = maps[map_name].get(answer, np.nan) if map_name else np.nan
        values.append(number if np.isnan(value) else value)

    x["_value"] = np.asarray(values, dtype=float)
    x["_invalid"] = x["_value"].notna() & ~x["_value"].between(spec.item_min, spec.item_max)
    x.loc[x["_invalid"], "_value"] = np.nan

    reverse = x["question"].isin(spec.reverse_items)
    x.loc[reverse, "_value"] = spec.item_min + spec.item_max - x.loc[reverse, "_value"]

    rows = []

    for participant, by_participant in x.groupby("_participant", sort=True):
        sessions = _sessions(by_participant, session_gap_ms)

        for session, items in by_participant.groupby(sessions, sort=False):
            last = items.drop_duplicates("question", keep="last")

            for category in spec.category_list:
                expected = [
                    t for t, q in spec.questions.items() if q.category == category
                ]
                c = last[last["question"].isin(expected)]
                valid = c["_value"].dropna()
                need = spec.min_items.get(category, len(expected))

                if len(valid) >= need:
                    score = float(valid.sum())
                    status = (
                        FeatureStatus.OBSERVED_ZERO if score == 0 else FeatureStatus.OBSERVED
                    )
                else:
                    score = np.nan
                    status = FeatureStatus.INSUFFICIENT_OBSERVABILITY

                rows.append({
                    "participant_id": participant or None,
                    "instrument": spec.instrument,
                    "session_id": session,
                    "timestamp_ms": int(pd.to_numeric(items["timestamp_ms"]).min()),
                    "category": category,
                    "score": score,
                    "score_status": status.value,
                    "n_items_valid": len(valid),
                    "n_items_expected": len(expected),
                    "n_items_invalid": int(c["_invalid"].sum()),
                })

    return pd.DataFrame(rows, columns=SCORE_COLUMNS)
