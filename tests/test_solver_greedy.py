"""Tests for GreedyAssignmentSolver."""

from __future__ import annotations

import numpy as np

from linear_assignment import AssignmentMatrix, AssignmentSolverMethod, AssignmentValueType
from linear_assignment.array_types import IndexArray
from linear_assignment.solvers.greedy import GreedyAssignmentSolver

from tests.helpers import assert_valid_assignment, assignment_index_pairs


class TestGreedyAssignmentSolver:
    """Greedy heuristic assignment."""

    def test_returns_valid_assignment(
        self,
        greedy_solver: GreedyAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array(
                [
                    [9.0, 1.0, 5.0],
                    [2.0, 8.0, 3.0],
                    [4.0, 6.0, 7.0],
                ]
            ),
            type=AssignmentValueType.COST,
        )
        rows, cols = greedy_solver.run_assignment(matrix)
        assert_valid_assignment(rows, cols, *matrix.shape)

    def test_cost_picks_lowest_values_first(
        self,
        greedy_solver: GreedyAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 100.0], [100.0, 2.0]]),
            type=AssignmentValueType.COST,
        )
        rows, cols = greedy_solver.run_assignment(matrix)
        assert assignment_index_pairs(rows, cols) == {(0, 0), (1, 1)}

    def test_score_picks_highest_values_first(
        self,
        greedy_solver: GreedyAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 100.0], [100.0, 2.0]]),
            type=AssignmentValueType.SCORE,
        )
        rows, cols = greedy_solver.run_assignment(matrix)
        assert assignment_index_pairs(rows, cols) == {(0, 1), (1, 0)}

    def test_select_greedy_matches_static(self) -> None:
        sorted_indices: IndexArray = np.array([1, 0, 3, 2], dtype=np.int64)
        rows, cols = GreedyAssignmentSolver._select_greedy_matches(  # pyright: ignore[reportPrivateUsage]
            sorted_indices, num_rows=2, num_cols=2
        )
        np.testing.assert_array_equal(rows, np.array([0, 1]))
        np.testing.assert_array_equal(cols, np.array([1, 0]))

    def test_method_property(
        self,
        greedy_solver: GreedyAssignmentSolver,
    ) -> None:
        assert greedy_solver.method is AssignmentSolverMethod.GREEDY

    def test_rectangular_matrix(
        self,
        greedy_solver: GreedyAssignmentSolver,
    ) -> None:
        matrix = AssignmentMatrix(
            value=np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]),
            type=AssignmentValueType.COST,
        )
        rows, cols = greedy_solver.run_assignment(matrix)
        assert_valid_assignment(rows, cols, *matrix.shape)
        assert len(rows) == min(*matrix.shape)
