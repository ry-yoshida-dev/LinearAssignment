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
    def flatten(self) -> NumericArray:
        """One-dimensional view of ``value``."""
        ...

    @property
    def min(self) -> float:
        """Minimum entry in ``value``."""
        ...

    @property
    def max(self) -> float:
        """Maximum entry in ``value``."""
        ...

    @property
    def mean(self) -> float:
        """Mean of the entries in ``value``."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Number of rows and columns of ``value``."""
        ...

    @property
    def best_value(self) -> float:
        """Most preferred entry in ``value`` for the matrix type."""
        ...

    @property
    def worst_value(self) -> float:
        """Least preferred entry in ``value`` for the matrix type."""
        ...
