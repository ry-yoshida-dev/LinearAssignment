from collections.abc import Callable
from enum import Enum

from .array_types import BoolArray, NumericArray


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
        ) -> Callable[[NumericArray], BoolArray]:
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