import numpy as np
from enum import Enum
from typing import Callable

class AssignmentValueType(Enum):
    """
    AssignmentValueType is the type of the matrix value.
    COST: Low-value is better
    SCORE: High-value is better
    """
    COST = "Cost"
    SCORE = "Score"

    def create_threshold_predicate(
        self, 
        threshold: float
        ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a threshold predicate function based on the assignment value type.

        Parameters
        ----------
        threshold: float
            The threshold value.

        Returns
        -------
        Callable: The condition function.
        """
        match self:
            case AssignmentValueType.COST:
                return lambda value: value <= threshold
            case AssignmentValueType.SCORE:
                return lambda value: value >= threshold