"""Tests for HungarianAssignmentSolver."""

from __future__ import annotations

import numpy as np

from linear_assignment import AssignmentMatrix, AssignmentSolverMethod, AssignmentValueType
from linear_assignment.solvers import HungarianAssignmentSolver

from tests.helpers import (
    assert_valid_assignment,
    assignment_index_pairs,
    assignment_total,
    brute_force_optimal_cost,
    brute_force_optimal_score,
)


class TestHungarianAssignmentSolver:
    """Optimal assignment via scipy linear_sum_assignment."""

    def test_square_cost_matrix_is_optimal(
        self,
        hungarian_solver: HungarianAssignmentSolver,
        cost_matrix_3x3: AssignmentMatrix,
    ) -> None:
        rows, cols = hungarian_solver.run_assignment(cost_matrix_3x3)
        num_rows, num_cols = cost_matrix_3x3.shape
        assert_valid_assignment(rows, cols, num_rows, num_cols)
        total = assignment_total(cost_matrix_3x3, rows, cols)
        expected = brute_force_optimal_cost(cost_matrix_3x3.value)
        assert abs(total - expected) < 1e-9

    def test_rectangular_score_matrix_is_optimal(
        self,
        hungarian_solver: HungarianAssignmentSolver,
        score_matrix_2x3: AssignmentMatrix,
    ) -> None:
        rows, cols = hungarian_solver.run_assignment(score_matrix_2x3)
        num_rows, num_cols = score_matrix_2x3.shape
        assert_valid_assignment(rows, cols, num_rows, num_cols)
        total = assignment_total(score_matrix_2x3, rows, cols)
        expected = brute_force_optimal_score(score_matrix_2x3.value)
        assert abs(total - expected) < 1e-9

    def test_more_rows_than_columns(
        self,
        hungarian_solver: HungarianAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array(
                [
                    [1.0, 4.0],
                    [2.0, 5.0],
                    [3.0, 6.0],
                ]
            ),
            type=AssignmentValueType.COST,
        )
        rows, cols = hungarian_solver.run_assignment(matrix)
        assert_valid_assignment(rows, cols, *matrix.shape)
        total = assignment_total(matrix, rows, cols)
        expected = brute_force_optimal_cost(matrix.value)
        assert abs(total - expected) < 1e-9

    def test_method_property(
        self,
        hungarian_solver: HungarianAssignmentSolver,
    ) -> None:
        assert hungarian_solver.method is AssignmentSolverMethod.HUNGARIAN

    def test_known_2x2_assignment(
        self,
        hungarian_solver: HungarianAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 100.0], [100.0, 1.0]]),
            type=AssignmentValueType.COST,
        )
        rows, cols = hungarian_solver.run_assignment(matrix)
        assert assignment_index_pairs(rows, cols) == {(0, 0), (1, 1)}
