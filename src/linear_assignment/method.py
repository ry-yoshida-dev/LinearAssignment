from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solver import AssignmentSolver

class AssignmentSolverMethod(Enum):
    """
    AssignmentSolverMethod is the method of the assignment solver.

    Attributes
    ----------
    HUNGARIAN: Uses the Hungarian algorithm for optimal assignment.
    MUTUAL_OPTIMAL: Uses the mutual optimal matching.
    GREEDY: Uses a greedy approach for assignment.
    """
    HUNGARIAN = "Hungarian"
    MUTUAL_OPTIMAL = "MutualOptimal"
    GREEDY = "Greedy"

    def build(self) -> "AssignmentSolver":
        from .solvers import (
            GreedyAssignmentSolver,
            HungarianAssignmentSolver,
            MutualOptimalAssignmentSolver,
        )

        match self:
            case AssignmentSolverMethod.HUNGARIAN:
                return HungarianAssignmentSolver()
            case AssignmentSolverMethod.MUTUAL_OPTIMAL:
                return MutualOptimalAssignmentSolver()
            case AssignmentSolverMethod.GREEDY:
                return GreedyAssignmentSolver()
