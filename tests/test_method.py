"""Tests for AssignmentSolverMethod factory."""

from __future__ import annotations

from linear_assignment import AssignmentSolverMethod
from linear_assignment.solvers import (
    GreedyAssignmentSolver,
    HungarianAssignmentSolver,
    MutualOptimalAssignmentSolver,
)


class TestAssignmentSolverMethod:
    """Solver factory and enum metadata."""

    def test_hungarian_build(self) -> None:
        solver = AssignmentSolverMethod.HUNGARIAN.build()
        assert isinstance(solver, HungarianAssignmentSolver)
        assert solver.method is AssignmentSolverMethod.HUNGARIAN

    def test_mutual_optimal_build(self) -> None:
        solver = AssignmentSolverMethod.MUTUAL_OPTIMAL.build()
        assert isinstance(solver, MutualOptimalAssignmentSolver)
        assert solver.method is AssignmentSolverMethod.MUTUAL_OPTIMAL

    def test_greedy_build(self) -> None:
        solver = AssignmentSolverMethod.GREEDY.build()
        assert isinstance(solver, GreedyAssignmentSolver)
        assert solver.method is AssignmentSolverMethod.GREEDY

    def test_enum_values(self) -> None:
        assert AssignmentSolverMethod.HUNGARIAN.value == "Hungarian"
        assert AssignmentSolverMethod.MUTUAL_OPTIMAL.value == "MutualOptimal"
        assert AssignmentSolverMethod.GREEDY.value == "Greedy"
