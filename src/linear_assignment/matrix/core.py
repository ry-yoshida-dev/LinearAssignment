"""Assignment matrix dataclass composed from mixins."""

from __future__ import annotations

from dataclasses import dataclass

from ..array_types import NumericArray
from ..value_type import AssignmentValueType

from .mixin import (
    AssignmentMatrixFunctionMixin,
    AssignmentMatrixPropertyMixin,
    AssignmentMatrixSpecialMixin,
)


@dataclass
class AssignmentMatrix(
    AssignmentMatrixPropertyMixin,
    AssignmentMatrixFunctionMixin,
    AssignmentMatrixSpecialMixin,
):
    """
    Container class for assignment matrices.

    Attributes
    ----------
    value : NumericArray
        Matrix with shape ``(N, M)``.
    type : AssignmentValueType
        Whether lower (COST) or higher (SCORE) values are preferred.
    """

    value: NumericArray
    type: AssignmentValueType

    def __post_init__(self) -> None:
        if self.value.ndim != 2:
            raise ValueError(
                f"Shape of the matrix is not 2D. Shape: {self.value.shape}"
            )
