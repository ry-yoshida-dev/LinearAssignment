"""Tests for MutualOptimalAssignmentSolver."""

from __future__ import annotations

import numpy as np

from linear_assignment import AssignmentMatrix, AssignmentSolverMethod, AssignmentValueType
from linear_assignment.solvers import MutualOptimalAssignmentSolver

from tests.helpers import assert_valid_assignment, assignment_index_pairs


class TestMutualOptimalAssignmentSolver:
    """Mutual-optimal bipartite matching."""

    def test_perfect_mutual_optimal_matching(
        self,
        mutual_optimal_solver: MutualOptimalAssignmentSolver,
        mutual_optimal_cost_matrix: AssignmentMatrix,
    ) -> None:
        rows, cols = mutual_optimal_solver.run_assignment(mutual_optimal_cost_matrix)
        assert_valid_assignment(rows, cols, *mutual_optimal_cost_matrix.shape)
        assert assignment_index_pairs(rows, cols) == {(0, 0), (1, 1)}

    def test_global_minimum_is_always_mutual_optimal(
        self,
        mutual_optimal_solver: MutualOptimalAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array(
                [
                    [5.0, 1.0, 9.0],
                    [2.0, 8.0, 3.0],
                    [7.0, 4.0, 6.0],
                ]
            ),
            type=AssignmentValueType.COST,
        )
        mask = matrix.create_mutual_optimal_mask()
        assert mask.any()
        global_min_pos = np.unravel_index(np.argmin(matrix.value), matrix.value.shape)
        assert mask[global_min_pos]

        rows, cols = mutual_optimal_solver.run_assignment(matrix)
        assert_valid_assignment(rows, cols, *matrix.shape)
        for row, col in zip(rows, cols, strict=True):
            assert mask[row, col]

    def test_partial_mutual_optimal_matching(
        self,
        mutual_optimal_solver: MutualOptimalAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array(
                [
                    [1.0, 9.0, 8.0],
                    [7.0, 2.0, 6.0],
                    [5.0, 4.0, 3.0],
                ]
            ),
            type=AssignmentValueType.COST,
        )
        rows, cols = mutual_optimal_solver.run_assignment(matrix)
        assert_valid_assignment(rows, cols, *matrix.shape)
        mask = matrix.create_mutual_optimal_mask()
        for row, col in zip(rows, cols, strict=True):
            assert mask[row, col]

    def test_score_matrix(
        self,
        mutual_optimal_solver: MutualOptimalAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[9.0, 1.0], [2.0, 8.0]]),
            type=AssignmentValueType.SCORE,
        )
        rows, cols = mutual_optimal_solver.run_assignment(matrix)
        assert assignment_index_pairs(rows, cols) == {(0, 0), (1, 1)}

    def test_method_property(
        self,
        mutual_optimal_solver: MutualOptimalAssignmentSolver,
    ) -> None:
        assert mutual_optimal_solver.method is AssignmentSolverMethod.MUTUAL_OPTIMAL
