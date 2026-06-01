"""Functional operations mixin for assignment matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np

from ...value_type import AssignmentValueType

from ..protocols.backing import AssignmentMatrixBacking

if TYPE_CHECKING:
    from ..core import AssignmentMatrix


class AssignmentMatrixFunctionMixin(AssignmentMatrixBacking):
    """
    Filtering, masking, and analysis operations on assignment matrices.
    """

    def convert_cost_format(self) -> AssignmentMatrix:
        """
        Convert the matrix to cost format.

        Returns
        -------
        AssignmentMatrix
            Matrix in cost format.
        """
        from ..core import AssignmentMatrix

        match self.type:
            case AssignmentValueType.COST:
                return cast(AssignmentMatrix, self)
            case AssignmentValueType.SCORE:
                return AssignmentMatrix(
                    value=-self.value,
                    type=AssignmentValueType.COST,
                )

    def filter_suboptimal_pair(self, replace_value: float) -> None:
        """
        Filter out suboptimal pairs by keeping only mutual optimal matches.

        Parameters
        ----------
        replace_value : float
            Value to replace filtered elements.
        """
        keep_mask = self.create_mutual_optimal_mask()
        self.value[~keep_mask] = replace_value

    def create_mutual_optimal_mask(self) -> np.ndarray:
        """
        Create a mask that keeps only mutually optimal row/column pairs.

        Returns
        -------
        np.ndarray
            Boolean mask where True means keep this pair.
        """
        match self.type:
            case AssignmentValueType.COST:
                col_optimal_mask = self.value == self.value.min(axis=0, keepdims=True)
                row_optimal_mask = self.value == self.value.min(axis=1, keepdims=True)
            case AssignmentValueType.SCORE:
                col_optimal_mask = self.value == self.value.max(axis=0, keepdims=True)
                row_optimal_mask = self.value == self.value.max(axis=1, keepdims=True)
        return row_optimal_mask & col_optimal_mask

    def ratio_test(self, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a ratio test along the specified axis.

        Compares the best value to the second-best along each row or column.
        Used to identify reliable matches before assignment.

        Parameters
        ----------
        axis : int
            Axis along which the reduction is performed.
            ``axis=0`` computes per-column results; ``axis=1`` per-row.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(closest_values, ratio_test_results)``.

        Raises
        ------
        ValueError
            If ``axis`` is not 0 or 1.
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")

        len_row, len_col = self.value.shape
        eps = np.finfo(float).eps

        match self.type:
            case AssignmentValueType.COST:
                closest_values = np.min(self.value, axis=axis)
                if axis == 0:
                    second_values = (
                        np.partition(self.value, 1, axis=axis)[1, :]
                        if len_row >= 2
                        else np.full(len_col, np.inf)
                    )
                else:
                    second_values = (
                        np.partition(self.value, 1, axis=axis)[:, 1]
                        if len_col >= 2
                        else np.full(len_row, np.inf)
                    )
                ratio_test_results = np.divide(closest_values + eps, second_values + eps)
            case AssignmentValueType.SCORE:
                closest_values = np.max(self.value, axis=axis)
                if axis == 0:
                    second_values = (
                        np.partition(self.value, -2, axis=axis)[-2, :]
                        if len_row >= 2
                        else np.zeros(len_col)
                    )
                else:
                    second_values = (
                        np.partition(self.value, -2, axis=axis)[:, -2]
                        if len_col >= 2
                        else np.zeros(len_row)
                    )
                ratio_test_results = np.divide(second_values + eps, closest_values + eps)
        return closest_values, ratio_test_results

    def create_threshold_predicate(
        self,
        threshold: float,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a threshold predicate based on the assignment value type.

        Parameters
        ----------
        threshold : float
            Threshold value.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Predicate applied element-wise to arrays.
        """
        return self.type.create_threshold_predicate(threshold=threshold)

    def create_mask(self, threshold: float) -> np.ndarray:
        """
        Create a boolean mask matrix from a threshold.

        Parameters
        ----------
        threshold : float
            Threshold value.

        Returns
        -------
        np.ndarray
            Mask matrix.
        """
        threshold_predicate = self.create_threshold_predicate(threshold=threshold)
        return threshold_predicate(self.value)

    def filter_assignment_by_threshold(
        self,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter assignment pairs by matrix values against a threshold.

        For each ``(row, col)`` pair, the value at ``value[row, col]`` is checked
        using the same rule as :meth:`create_mask` (COST: ``<= threshold``,
        SCORE: ``>= threshold``). Pairs that fail the check are dropped.

        Parameters
        ----------
        row_indices : np.ndarray
            Row indices from an assignment solver.
        col_indices : np.ndarray
            Column indices aligned with ``row_indices``.
        threshold : float
            Threshold applied to assigned cell values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Filtered ``(row_indices, col_indices)`` with only passing pairs.

        Raises
        ------
        ValueError
            If ``row_indices`` and ``col_indices`` differ in length.
        """
        rows = np.asarray(row_indices, dtype=np.int64)
        cols = np.asarray(col_indices, dtype=np.int64)
        if rows.shape != cols.shape:
            raise ValueError(
                "row_indices and col_indices must have the same shape; "
                f"got {rows.shape} and {cols.shape}."
            )
        if rows.size == 0:
            return rows, cols

        threshold_predicate = self.create_threshold_predicate(threshold=threshold)
        paired_values = self.value[rows, cols]
        keep_mask = threshold_predicate(paired_values)
        return rows[keep_mask], cols[keep_mask]
