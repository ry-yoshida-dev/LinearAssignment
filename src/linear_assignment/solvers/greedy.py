import numpy as np
from importlib import import_module

from ..array_types import AssignmentPairIndices, IndexArray
from ..matrix import AssignmentMatrix
from ..method import AssignmentSolverMethod
from ..solver import AssignmentSolver
from ..value_type import AssignmentValueType

njit = import_module("numba").njit

class GreedyAssignmentSolver(AssignmentSolver):
    """
    GreedyAssignmentSolver is a solver for assignment problems using a greedy approach.
    
    This solver sorts the matrix entries and greedily assigns the best available pairs while avoiding conflicts.
    """
    def _run(
        self,
        matrix: AssignmentMatrix,
    ) -> AssignmentPairIndices:
        """
        Solve assignment problem using a greedy approach.
        
        This method sorts the matrix entries and greedily assigns the best available pairs while avoiding conflicts.
        
        Parameters:
        ----------
        matrix: AssignmentMatrix
            Input assignment matrix.
                   
        Returns:
        -------
            tuple: (row_indices, col_indices) representing the greedy assignment
        """
        num_rows, num_cols = matrix.shape
        matrix_value = matrix.value
        flat = matrix_value.ravel()

        # Sort all matrix positions once in C, then greedily pick non-conflicting pairs.
        sorted_indices = np.argsort(flat)
        if matrix.type == AssignmentValueType.SCORE:
            sorted_indices = sorted_indices[::-1]
        return self._select_greedy_matches(sorted_indices, num_rows, num_cols)

    @staticmethod
    @njit(cache=True)  # pyright: ignore[reportUntypedFunctionDecorator]
    def _select_greedy_matches(
        sorted_indices: IndexArray,
        num_rows: int,
        num_cols: int,
    ) -> AssignmentPairIndices:
        """
        Select greedy matches from the sorted indices.
        
        Parameters:
        ----------
        sorted_indices: IndexArray
            The sorted indices of the matrix.
        num_rows: int
            The number of rows in the matrix.
        num_cols: int
            The number of columns in the matrix.

        Returns:
        -------
        tuple: (row_indices, col_indices) representing the greedy assignment
        """
        used_rows = np.zeros(num_rows, dtype=np.bool_)
        used_cols = np.zeros(num_cols, dtype=np.bool_)
        max_matches = min(num_rows, num_cols)

        row_indices = np.empty(max_matches, dtype=np.int64)
        col_indices = np.empty(max_matches, dtype=np.int64)
        match_count = 0

        for flat_idx in sorted_indices:
            i = flat_idx // num_cols
            j = flat_idx % num_cols
            if used_rows[i] or used_cols[j]:
                continue
            row_indices[match_count] = i
            col_indices[match_count] = j
            used_rows[i] = True
            used_cols[j] = True
            match_count += 1
            if match_count == max_matches:
                break

        return row_indices[:match_count], col_indices[:match_count]

    @property
    def method(self) -> AssignmentSolverMethod:
        """
        Return solver method type.
        
        Returns:
        -------
        AssignmentSolverMethod: The method type.
        """
        return AssignmentSolverMethod.GREEDY