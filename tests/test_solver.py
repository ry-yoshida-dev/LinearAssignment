"""Tests for AssignmentSolver base behavior."""

from __future__ import annotations

import numpy as np

from linear_assignment import AssignmentMatrix, AssignmentSolverMethod, AssignmentValueType
from linear_assignment.solvers import HungarianAssignmentSolver


class TestAssignmentSolverBase:
    """Shared solver interface."""

    def test_1x1_matrix_shortcut(self) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[42.0]]),
            type=AssignmentValueType.COST,
        )
        solver = HungarianAssignmentSolver()
        rows, cols = solver.run_assignment(matrix)
        np.testing.assert_array_equal(rows, np.array([0]))
        np.testing.assert_array_equal(cols, np.array([0]))

    def test_solver_str(self) -> None:
        solver = AssignmentSolverMethod.HUNGARIAN.build()
        assert str(solver) == "HungarianAssignmentSolver()"
