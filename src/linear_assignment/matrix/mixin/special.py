"""Dunder-method mixin for assignment matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...array_types import NumericArray

from ..protocols.backing import AssignmentMatrixBacking

if TYPE_CHECKING:
    from ..core import AssignmentMatrix


class AssignmentMatrixSpecialMixin(AssignmentMatrixBacking):
    """
    Indexing, arithmetic, and string representation for assignment matrices.
    """

    def __getitem__(
        self,
        key: tuple[int, int] | int,
    ) -> float | NumericArray:
        """
        Index into the underlying ndarray.

        Parameters
        ----------
        key : tuple[int, int] | int
            Index or slice key.

        Returns
        -------
        float | NumericArray
            Scalar or sub-array from ``value``.
        """
        return self.value[key]

    def __setitem__(
        self,
        key: Any,
        value: float | NumericArray,
    ) -> None:
        """
        Assign into the underlying ndarray.

        Parameters
        ----------
        key : Any
            Index or slice key accepted by ``value``.
        value : float | NumericArray
            Value(s) to assign.
        """
        self.value[key] = value

    def __add__(
        self,
        other: AssignmentMatrix,
    ) -> AssignmentMatrix:
        """
        Element-wise addition with another matrix.

        Parameters
        ----------
        other : AssignmentMatrix
            Matrix with the same shape.

        Returns
        -------
        AssignmentMatrix
            Sum matrix preserving ``type``.
        """
        from ..core import AssignmentMatrix

        return AssignmentMatrix(
            value=self.value + other.value,
            type=self.type,
        )

    def __mul__(
        self,
        other: AssignmentMatrix,
    ) -> AssignmentMatrix:
        """
        Element-wise multiplication with another matrix.

        Parameters
        ----------
        other : AssignmentMatrix
            Matrix with the same shape.

        Returns
        -------
        AssignmentMatrix
            Product matrix preserving ``type``.
        """
        from ..core import AssignmentMatrix

        return AssignmentMatrix(
            value=self.value * other.value,
            type=self.type,
        )

    def __str__(self) -> str:
        """
        Human-readable summary of the matrix.

        Returns
        -------
        str
            Formatted string with shape information.
        """
        return (
            f"AssignmentMatrix(\n"
            f"  matrix.shape={self.value.shape},\n"
            f")"
        )
