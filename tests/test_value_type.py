"""Tests for AssignmentValueType."""

from __future__ import annotations

import numpy as np

from linear_assignment import AssignmentValueType


class TestAssignmentValueType:
    """Threshold predicate behavior for COST and SCORE semantics."""

    def test_cost_threshold_predicate(self) -> None:
        predicate = AssignmentValueType.COST.create_threshold_predicate(threshold=3.0)
        values = np.array([1.0, 3.0, 4.0])
        result = predicate(values)
        np.testing.assert_array_equal(result, np.array([True, True, False]))

    def test_score_threshold_predicate(self) -> None:
        predicate = AssignmentValueType.SCORE.create_threshold_predicate(threshold=3.0)
        values = np.array([1.0, 3.0, 4.0])
        result = predicate(values)
        np.testing.assert_array_equal(result, np.array([False, True, True]))

    def test_enum_values(self) -> None:
        assert AssignmentValueType.COST.value == "Cost"
        assert AssignmentValueType.SCORE.value == "Score"
