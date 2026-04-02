from abc import ABC, abstractmethod
import numpy as np

from .matrix import AssignmentMatrix
from .method import AssignmentSolverMethod

class AssignmentSolver(ABC):
    """
    Abstract base class for assignment problem solvers.

    Concrete solvers must implement `_run()` with their own strategy.
    """
    def run_assignment(
        self, 
        matrix: AssignmentMatrix
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the assignment algorithm on the provided matrix.
        
        Parameters:
        ----------
        matrix: AssignmentMatrix
            Input assignment matrix.
                    
        Returns:
        -------
            tuple: (row_indices, col_indices) representing the optimal assignment
        """
        if matrix.shape == (1, 1):
            return np.array([0]), np.array([0])
        return self._run(matrix)

    @abstractmethod
    def _run(
        self,
        matrix: AssignmentMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run assignment logic for each concrete solver.
        
        Parameters:
        ----------
        matrix: AssignmentMatrix
            Input assignment matrix.
                   
        Returns:
        -------
            tuple: (row_indices, col_indices) representing the optimal assignment
        """

    @property
    @abstractmethod
    def method(self) -> AssignmentSolverMethod:
        """
        Return solver method type.
        
        Returns:
        -------
        AssignmentSolverMethod: The method type.
        """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
