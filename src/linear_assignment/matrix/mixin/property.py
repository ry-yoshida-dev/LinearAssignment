"""Property mixin for assignment matrices."""

from __future__ import annotations

import numpy as np

from ...array_types import NumericArray
from ...value_type import AssignmentValueType
from ..protocols import AssignmentMatrixBacking


class AssignmentMatrixPropertyMixin(AssignmentMatrixBacking):
    """Provides property access to assignment matrix attributes and derived values."""
    """
    Derived scalar and shape properties for an assignment matrix.
    """

    @property
    def best_value(self) -> float:
        """
        Optimal scalar under the current value type semantics.

        Returns
        -------
        float
            Minimum for COST, maximum for SCORE.
        """
        return self.min if self.type == AssignmentValueType.COST else self.max

    @property
    def worst_value(self) -> float:
        """
        Worst scalar under the current value type semantics.

        Returns
        -------
        float
            Maximum for COST, minimum for SCORE.
        """
        return self.max if self.type == AssignmentValueType.COST else self.min

    @property
    def mean(self) -> float:
        """
        Mean of all matrix entries.

        Returns
        -------
        float
            Mean of all values in ``value``.
        """
        return float(np.mean(self.value))

    @property
    def shape(self) -> tuple[int, int]:
        """
        Validated two-dimensional shape.

        Returns
        -------
        tuple[int, int]
            ``(rows, cols)``.

        Raises
        ------
        ValueError
            If ``value`` is not two-dimensional.
        """
        shape = self.value.shape
        if len(shape) != 2:
            raise ValueError(f"Shape of the matrix is not 2D. Shape: {shape}")
        return int(shape[0]), int(shape[1])

    @property
    def flatten(self) -> NumericArray:
        """
        Flatten the matrix.

        Returns
        -------
        NumericArray
            One-dimensional view of ``value``.
        """
        return self.value.flatten()

    @property
    def max(self) -> float:
        """
        Maximum entry in the matrix.

        Returns
        -------
        float
            Global maximum.
        """
        return float(np.max(self.value))

    @property
    def min(self) -> float:
        """
        Minimum entry in the matrix.

        Returns
        -------
        float
            Global minimum.
        """
        return float(np.min(self.value))
