"""Backing-store protocol for assignment matrices."""

from __future__ import annotations

from typing import Protocol

from ...array_types import NumericArray
from ...value_type import AssignmentValueType


class AssignmentMatrixBacking(Protocol):
    """
    Minimum backing store required by matrix mixins and ``AssignmentMatrix``.

    Implemented by :class:`~linear_assignment.matrix.core.AssignmentMatrix` (storage)
    and mixins that inherit this protocol.

    Attributes
    ----------
    value : NumericArray
        Two-dimensional assignment matrix.
    type : AssignmentValueType
        Whether lower (COST) or higher (SCORE) values are preferred.
    """

    value: NumericArray
    type: AssignmentValueType
