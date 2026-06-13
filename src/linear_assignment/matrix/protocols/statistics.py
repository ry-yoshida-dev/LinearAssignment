"""Statistics protocol for assignment matrices."""

from __future__ import annotations

from typing import Protocol

from ...array_types import NumericArray

from .backing import AssignmentMatrixBacking


class AssignmentMatrixStatistics(AssignmentMatrixBacking, Protocol):
    """
    Read-only statistics surface used by dependent operations.

    Attributes
    ----------
    flatten : NumericArray
        One-dimensional view of ``value``.
    min : float
        Minimum entry in ``value``.
    max : float
        Maximum entry in ``value``.
    """

    @property
    def flatten(self) -> NumericArray: ...

    @property
    def min(self) -> float: ...

    @property
    def max(self) -> float: ...

    @property
    def mean(self) -> float: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def best_value(self) -> float: ...

    @property
    def worst_value(self) -> float: ...
