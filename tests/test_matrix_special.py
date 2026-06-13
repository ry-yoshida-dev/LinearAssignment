"""Tests for AssignmentMatrix dunder-method mixin."""

from __future__ import annotations

import numpy as np
import pytest

from linear_assignment import AssignmentMatrix, AssignmentValueType


class TestAssignmentMatrixSpecial:
    """Indexing, arithmetic, and string representation."""

    @pytest.fixture
    def matrix_a(self) -> AssignmentMatrix:
        return AssignmentMatrix(
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            type=AssignmentValueType.COST,
        )

    @pytest.fixture
    def matrix_b(self) -> AssignmentMatrix:
        return AssignmentMatrix(
            value=np.array([[10.0, 20.0], [30.0, 40.0]]),
            type=AssignmentValueType.COST,
        )

    def test_getitem_scalar(self, matrix_a: AssignmentMatrix) -> None:
        assert matrix_a[0, 1] == 2.0

    def test_getitem_row_slice(self, matrix_a: AssignmentMatrix) -> None:
        row = matrix_a[0]
        np.testing.assert_array_equal(row, np.array([1.0, 2.0]))

    def test_setitem(self, matrix_a: AssignmentMatrix) -> None:
        matrix_a[1, 1] = 99.0
        assert matrix_a.value[1, 1] == 99.0

    def test_add(self, matrix_a: AssignmentMatrix, matrix_b: AssignmentMatrix) -> None:
        result = matrix_a + matrix_b
        np.testing.assert_array_equal(result.value, matrix_a.value + matrix_b.value)
        assert result.type is AssignmentValueType.COST

    def test_mul(self, matrix_a: AssignmentMatrix, matrix_b: AssignmentMatrix) -> None:
        result = matrix_a * matrix_b
        np.testing.assert_array_equal(result.value, matrix_a.value * matrix_b.value)
        assert result.type is AssignmentValueType.COST

    def test_str_contains_shape(self, matrix_a: AssignmentMatrix) -> None:
        rendered = str(matrix_a)
        assert "AssignmentMatrix" in rendered
        assert "(2, 2)" in rendered
