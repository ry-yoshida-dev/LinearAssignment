"""Tests for AssignmentMatrix core validation."""

from __future__ import annotations

import numpy as np
import pytest

from linear_assignment import AssignmentMatrix, AssignmentValueType


class TestAssignmentMatrixCore:
    """Construction and validation of AssignmentMatrix."""

    def test_valid_2d_matrix(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 2.0], [3.0, 4.0]]),
            type=AssignmentValueType.COST,
        )
        assert matrix.shape == (2, 2)

    def test_rejects_1d_array(self) -> None:
        with pytest.raises(ValueError, match="not 2D"):
            AssignmentMatrix(
                value=np.array([1.0, 2.0, 3.0]),
                type=AssignmentValueType.COST,
            )

    def test_rejects_3d_array(self) -> None:
        with pytest.raises(ValueError, match="not 2D"):
            AssignmentMatrix(
                value=np.ones((2, 2, 2)),
                type=AssignmentValueType.COST,
            )

    def test_accepts_integer_dtype(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1, 2], [3, 4]], dtype=np.int32),
            type=AssignmentValueType.SCORE,
        )
        assert matrix.type is AssignmentValueType.SCORE
