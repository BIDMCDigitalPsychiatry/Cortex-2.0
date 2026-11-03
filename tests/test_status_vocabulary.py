"""Every status string emitted under features/ must be a FeatureStatus member."""

import ast
import math
from pathlib import Path

import pytest

import cortex2_engine.features as features_pkg
from cortex2_engine.features.status import (
    EMITTABLE_STATUSES,
    FeatureStatus,
    gated_value,
    status_from_counts,
)

FEATURES_DIR = Path(features_pkg.__file__).parent

# Edge-level evidence labels with their own documented vocabularies.
NON_FEATURE_STATUS_KEYS = {
    "spatial_status",
    "trajectory_status",
}

EMITTABLE = {member.value for member in EMITTABLE_STATUSES}
MEMBER_NAMES = {member.name for member in EMITTABLE_STATUSES}


def _is_status_key(name: str) -> bool:
    return (
        (name == "status" or name.endswith("_status"))
        and name not in NON_FEATURE_STATUS_KEYS
    )


def _string_constants(node: ast.AST) -> list[str]:
    """String literals that can become the status value itself.

    Direct constants and the branches of conditional / boolean expressions
    count; strings used as arguments (column names, messages) do not.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]

    if isinstance(node, ast.IfExp):
        return _string_constants(node.body) + _string_constants(node.orelse)

    if isinstance(node, ast.BoolOp):
        return [s for v in node.values for s in _string_constants(v)]

    return []


def _emitted_status_literals(tree: ast.AST) -> list[tuple[int, str]]:

    found = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_status_key(key.value)
                ):
                    found += [
                        (value.lineno, s) for s in _string_constants(value)
                    ]

        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )

            for target in targets:
                name = None

                if isinstance(target, ast.Name):
                    name = target.id

                elif isinstance(target, ast.Subscript) and isinstance(
                    target.slice, ast.Constant
                ):
                    name = target.slice.value

                if (
                    isinstance(name, str)
                    and _is_status_key(name)
                    and node.value is not None
                ):
                    found += [
                        (node.lineno, s)
                        for s in _string_constants(node.value)
                    ]

    return found


def _feature_modules():
    return sorted(
        p for p in FEATURES_DIR.glob("*.py") if p.name != "status.py"
    )


@pytest.mark.parametrize(
    "path",
    _feature_modules(),
    ids=lambda p: p.name,
)
def test_emitted_status_literals_are_members(path):

    tree = ast.parse(path.read_text())

    bad = [
        (line, value)
        for line, value in _emitted_status_literals(tree)
        if value not in EMITTABLE
    ]

    assert not bad, f"{path.name}: non-vocabulary statuses {bad}"


@pytest.mark.parametrize(
    "path",
    _feature_modules(),
    ids=lambda p: p.name,
)
def test_feature_status_attributes_exist_and_are_emittable(path):

    tree = ast.parse(path.read_text())

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "FeatureStatus"
        ):
            assert node.attr in MEMBER_NAMES, (
                f"{path.name}:{node.lineno} FeatureStatus.{node.attr}"
            )


def test_scanner_detects_a_bad_literal():

    tree = ast.parse(
        'out = {"x_status": "observed_value"}\n'
        'y_status = "observed" if a else "no_raw_data"\n'
        'z_status = f(df["n_samples"])\n'
        'result["spatial_status"] = "temporal_gap"\n'
    )

    values = [v for _, v in _emitted_status_literals(tree)]

    assert sorted(values) == ["no_raw_data", "observed", "observed_value"]


def test_vocabulary_is_exactly_the_documented_set():

    assert EMITTABLE == {
        "observed",
        "observed_zero",
        "insufficient_observability",
        "no_raw_data",
        "source_absent",
        "extraction_failed",
        "analysis_masked",
        "censored",
        "not_applicable",
    }

    assert FeatureStatus.FEATURE_EMPTY not in EMITTABLE_STATUSES
    assert FeatureStatus("observed") is FeatureStatus.OBSERVED


@pytest.mark.parametrize(
    ("observed_ms", "minimum_ms", "value", "n_records", "expected"),
    [
        (10.0, 5.0, 3.0, None, FeatureStatus.OBSERVED),
        (10.0, 5.0, 0.0, None, FeatureStatus.OBSERVED_ZERO),
        (4.0, 5.0, 0.0, None, FeatureStatus.INSUFFICIENT_OBSERVABILITY),
        (0.0, 0.0, 0.0, None, FeatureStatus.INSUFFICIENT_OBSERVABILITY),
        (None, 0.0, 1.0, None, FeatureStatus.INSUFFICIENT_OBSERVABILITY),
        (math.nan, 0.0, 1.0, None, FeatureStatus.INSUFFICIENT_OBSERVABILITY),
        (10.0, 5.0, math.nan, None, FeatureStatus.NOT_APPLICABLE),
        (10.0, 5.0, None, None, FeatureStatus.NOT_APPLICABLE),
        (10.0, 5.0, 3.0, 0, FeatureStatus.NO_RAW_DATA),
        (10.0, 5.0, 3.0, 4, FeatureStatus.OBSERVED),
        (5.0, 5.0, 1.0, None, FeatureStatus.OBSERVED),
    ],
)
def test_status_from_counts(observed_ms, minimum_ms, value, n_records, expected):

    assert (
        status_from_counts(
            observed_ms,
            minimum_ms,
            value,
            n_records=n_records,
        )
        is expected
    )


def test_status_from_counts_rejects_negative_minimum():

    with pytest.raises(ValueError):
        status_from_counts(1.0, -1.0, 1.0)


def test_gated_value():

    assert gated_value(0.0, FeatureStatus.OBSERVED_ZERO) == 0.0
    assert gated_value(2.0, "censored") == 2.0
    assert math.isnan(gated_value(2.0, FeatureStatus.INSUFFICIENT_OBSERVABILITY))
    assert math.isnan(gated_value(None, FeatureStatus.OBSERVED))
    assert math.isnan(gated_value(2.0, FeatureStatus.NOT_APPLICABLE))
