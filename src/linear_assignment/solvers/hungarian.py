from typing import cast
from scipy.optimize import linear_sum_assignment  # type: ignore

from ..array_types import AssignmentPairIndices
from ..matrix import AssignmentMatrix
from ..method import AssignmentSolverMethod
from ..solver import AssignmentSolver


class HungarianAssignmentSolver(AssignmentSolver):
    """
    HungarianAssignmentSolver is a solver for assignment problems using the Hungarian algorithm.
    """
    def _run(
        self,
        matrix: AssignmentMatrix,
    ) -> AssignmentPairIndices:
        """
        Solve assignment problem using the Hungarian algorithm.
        
        This method uses scipy's linear_sum_assignment which implements
        the Hungarian algorithm for optimal assignment.
        
        Parameters:
        ----------
        matrix: AssignmentMatrix
            Input assignment matrix.
                   
        Returns:
            tuple: (row_indices, col_indices) representing the optimal assignment
        """
        assignment = cast(
            AssignmentPairIndices,
            linear_sum_assignment(matrix.convert_cost_format().value),
        )
        return assignment

    @property
    def method(self) -> AssignmentSolverMethod:
        """
        Return solver method type.
        
        Returns:
        -------
        AssignmentSolverMethod: The method type.
        """
        return AssignmentSolverMethod.HUNGARIAN
