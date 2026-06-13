"""Tests for AssignmentMatrix functional mixin."""

from __future__ import annotations

import numpy as np
import pytest

from linear_assignment import AssignmentMatrix, AssignmentValueType


class TestConvertCostFormat:
    """Cost-format conversion."""

    def test_cost_matrix_is_identity(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            type=AssignmentValueType.COST,
        )
        converted = matrix.convert_cost_format()
        assert converted is matrix
        assert converted.type is AssignmentValueType.COST

    def test_score_matrix_is_negated(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 5.0], [4.0, 2.0]]),
            type=AssignmentValueType.SCORE,
        )
        converted = matrix.convert_cost_format()
        assert converted.type is AssignmentValueType.COST
        np.testing.assert_array_equal(converted.value, -matrix.value)


class TestMutualOptimalMask:
    """Mutual-optimal mask creation."""

    def test_cost_mutual_optimal_mask(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 9.0], [8.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        mask = matrix.create_mutual_optimal_mask()
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_score_mutual_optimal_mask(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[9.0, 1.0], [2.0, 8.0]]),
            type=AssignmentValueType.SCORE,
        )
        mask = matrix.create_mutual_optimal_mask()
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_filter_suboptimal_pair(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 9.0], [8.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        matrix.filter_suboptimal_pair(replace_value=99.0)
        expected = np.array([[1.0, 99.0], [99.0, 2.0]])
        np.testing.assert_array_equal(matrix.value, expected)


class TestRatioTest:
    """Ratio test along rows and columns."""

    @pytest.fixture
    def cost_matrix(self) -> AssignmentMatrix:
        return AssignmentMatrix(
            value=np.array(
                [
                    [1.0, 5.0, 3.0],
                    [4.0, 2.0, 6.0],
                ]
            ),
            type=AssignmentValueType.COST,
        )

    def test_axis_0_cost(self, cost_matrix: AssignmentMatrix) -> None:
        closest, ratios = cost_matrix.ratio_test(axis=0)
        assert closest.shape == (3,)
        assert ratios.shape == (3,)
        np.testing.assert_array_equal(closest, np.array([1.0, 2.0, 3.0]))

    def test_axis_1_cost(self, cost_matrix: AssignmentMatrix) -> None:
        closest, ratios = cost_matrix.ratio_test(axis=1)
        assert closest.shape == (2,)
        assert ratios.shape == (2,)
        np.testing.assert_array_equal(closest, np.array([1.0, 2.0]))

    def test_axis_0_score(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 5.0], [4.0, 2.0]]),
            type=AssignmentValueType.SCORE,
        )
        closest, ratios = matrix.ratio_test(axis=0)
        np.testing.assert_array_equal(closest, np.array([4.0, 5.0]))
        assert ratios.shape == (2,)

    def test_single_row_axis_0_uses_inf_second_values(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 2.0, 3.0]]),
            type=AssignmentValueType.COST,
        )
        _, ratios = matrix.ratio_test(axis=0)
        assert ratios.shape == (3,)
        assert np.all(np.isfinite(ratios))

    def test_single_column_axis_1_uses_inf_second_values(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0], [2.0], [3.0]]),
            type=AssignmentValueType.COST,
        )
        _, ratios = matrix.ratio_test(axis=1)
        assert ratios.shape == (3,)

    def test_invalid_axis_raises(self, cost_matrix: AssignmentMatrix) -> None:
        with pytest.raises(ValueError, match="axis must be 0 or 1"):
            cost_matrix.ratio_test(axis=2)


class TestThresholdFiltering:
    """Threshold predicates, masks, and assignment filtering."""

    def test_create_mask_cost(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 5.0], [4.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        mask = matrix.create_mask(threshold=3.0)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_create_mask_score(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 5.0], [4.0, 2.0]]),
            type=AssignmentValueType.SCORE,
        )
        mask = matrix.create_mask(threshold=3.0)
        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_create_threshold_predicate_delegates_to_type(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        predicate = matrix.create_threshold_predicate(threshold=1.5)
        np.testing.assert_array_equal(predicate(np.array([1.0, 2.0])), np.array([True, False]))

    def test_filter_assignment_by_threshold(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 5.0], [4.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        rows = np.array([0, 1, 0], dtype=np.int64)
        cols = np.array([0, 1, 1], dtype=np.int64)
        filtered_rows, filtered_cols = matrix.filter_assignment_by_threshold(
            rows, cols, threshold=3.0
        )
        np.testing.assert_array_equal(filtered_rows, np.array([0, 1]))
        np.testing.assert_array_equal(filtered_cols, np.array([0, 1]))

    def test_filter_assignment_empty_input(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)
        out_rows, out_cols = matrix.filter_assignment_by_threshold(rows, cols, threshold=1.0)
        assert out_rows.size == 0
        assert out_cols.size == 0

    def test_filter_assignment_mismatched_lengths_raises(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        with pytest.raises(ValueError, match="same shape"):
            matrix.filter_assignment_by_threshold(
                np.array([0], dtype=np.int64),
                np.array([0, 1], dtype=np.int64),
                threshold=1.0,
            )
