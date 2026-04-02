import numpy as np
from typing import cast
from scipy.sparse import csr_matrix # type: ignore
from scipy.sparse.csgraph import maximum_bipartite_matching  # type: ignore

from ..matrix import AssignmentMatrix
from ..method import AssignmentSolverMethod
from ..solver import AssignmentSolver


class MutualOptimalAssignmentSolver(AssignmentSolver):
    """
    MutualOptimalAssignmentSolver is a solver for assignment problems using the mutual optimal matching.
    
    This solver filters out suboptimal pairs by keeping only mutual optimal matches.
    """

    def _run(
        self,
        matrix: AssignmentMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve assignment problem using the mutual optimal matching.
        
        This method filters out suboptimal pairs by keeping only mutual optimal matches.
        
        Parameters:
        ----------
        matrix: AssignmentMatrix
            Input assignment matrix.
                   
        Returns:
        -------
            tuple: (row_indices, col_indices) representing the mutual optimal assignment
        """
        mask = matrix.create_mutual_optimal_mask()

        matching = cast(
            np.ndarray,
            maximum_bipartite_matching(
                csr_matrix(mask.astype(np.int8)),
                perm_type="row",
            ),
        )
        row_indices = np.where(matching != -1)[0].astype(np.int64)
        col_indices = matching[row_indices].astype(np.int64)
        return row_indices, col_indices

    @property
    def method(self) -> AssignmentSolverMethod:
        """
        Return solver method type.
        
        Returns:
        -------
        AssignmentSolverMethod: The method type.
        """
        return AssignmentSolverMethod.MUTUAL_OPTIMAL