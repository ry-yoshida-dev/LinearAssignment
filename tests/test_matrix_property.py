"""Tests for AssignmentMatrix property mixin."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from linear_assignment import AssignmentMatrix, AssignmentValueType
from linear_assignment.array_types import NumericArray


class TestAssignmentMatrixProperty:
    """Scalar and shape properties."""

    @pytest.fixture
    def sample(self) -> AssignmentMatrix:
        return AssignmentMatrix(
            value=np.array(
                [
                    [1.0, 5.0, 3.0],
                    [4.0, 2.0, 6.0],
                ]
            ),
            type=AssignmentValueType.COST,
        )

    def test_shape(self, sample: AssignmentMatrix) -> None:
        assert sample.shape == (2, 3)

    def test_min_max_mean(self, sample: AssignmentMatrix) -> None:
        assert sample.min == 1.0
        assert sample.max == 6.0
        assert sample.mean == 3.5

    def test_flatten(self, sample: AssignmentMatrix) -> None:
        flat = sample.flatten
        np.testing.assert_array_equal(flat, sample.value.ravel())

    def test_best_worst_value_cost(self, sample: AssignmentMatrix) -> None:
        assert sample.best_value == 1.0
        assert sample.worst_value == 6.0

    def test_best_worst_value_score(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 5.0], [4.0, 2.0]]),
            type=AssignmentValueType.SCORE,
        )
        assert matrix.best_value == 5.0
        assert matrix.worst_value == 1.0

    def test_shape_raises_for_invalid_ndim(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        matrix.value = cast(NumericArray, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="not 2D"):
            _ = matrix.shape
