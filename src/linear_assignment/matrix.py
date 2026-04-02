from __future__ import annotations
import numpy as np
from numpy import floating
from dataclasses import dataclass
from typing import Any, Callable

from .value_type import AssignmentValueType

@dataclass
class AssignmentMatrix:
    """
    Container class for assignment matrices.

    Attributes:
    ----------
    value: np.ndarray
        The matrix value with shape (N, M).
    type: AssignmentValueType
        The type of the matrix value.
    """
    value: np.ndarray
    type: AssignmentValueType

    def __post_init__(self):
        if self.value.ndim != 2:
            raise ValueError(f"Shape of the matrix is not 2D. Shape: {self.value.shape}")

    @property
    def best_value(self) -> float:
        return self.min if self.type == AssignmentValueType.COST else self.max
    
    @property
    def worst_value(self) -> float:
        return self.max if self.type == AssignmentValueType.COST else self.min

    @property
    def mean(self) -> floating:
        return np.mean(self.flatten)

    @property
    def shape(self) -> tuple[int, int]:
        shape = self.value.shape
        if len(shape) != 2:
            raise ValueError(f"Shape of the matrix is not 2D. Shape: {shape}")
        return shape

    def __getitem__(
        self, 
        key: tuple[int, int] | int
        ) -> float | np.ndarray:
        return self.value[key]

    def __setitem__(
        self, 
        key: Any, 
        value: float | np.ndarray
        ) -> None:
        self.value[key] = value

    def convert_cost_format(self) -> AssignmentMatrix:
        """
        Convert the matrix to cost format.
        
        Returns
        -------
        AssignmentMatrix:
            Matrix in cost format.
        """
        match self.type:
            case AssignmentValueType.COST:
                return self
            case AssignmentValueType.SCORE:
                return AssignmentMatrix(
                    value=-self.value, 
                    type=AssignmentValueType.COST
                    )

    def filter_suboptimal_pair(
        self, 
        replace_value: float
        ) -> None:
        """
        Filter out suboptimal pairs by keeping only mutual optimal matches.
        
        Parameters
        ----------
        replace_value: float
            Value to replace filtered elements
        """
        keep_mask = self.create_mutual_optimal_mask()
        self.value[~keep_mask] = replace_value

    def create_mutual_optimal_mask(self) -> np.ndarray:
        """
        Create a mask that keeps only mutually optimal row/column pairs.

        Returns
        -------
        np.ndarray:
            Boolean mask where True means "keep this pair".
        """
        match self.type:
            case AssignmentValueType.COST:
                col_optimal_mask = self.value == self.value.min(axis=0, keepdims=True)
                row_optimal_mask = self.value == self.value.min(axis=1, keepdims=True)
            case AssignmentValueType.SCORE:
                col_optimal_mask = self.value == self.value.max(axis=0, keepdims=True)
                row_optimal_mask = self.value == self.value.max(axis=1, keepdims=True)
        return row_optimal_mask & col_optimal_mask

    def ratio_test(
        self, 
        axis: int = 0
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a ratio test on the matrix, comparing the second largest value to the largest value along the specified axis.
        This method is used to identify reliable matches before assignment.

        Parameters
        ----------
        axis: int
            Axis along which the reduction is performed.
            axis=0 computes per-column results, axis=1 computes per-row results.

        Returns
        -------
        tuple: (closest_values, ratio_test_results)
        """
        if axis not in [0, 1]:
            raise ValueError("axis must be 0 or 1")

        len_row, len_col = self.value.shape
        eps = np.finfo(float).eps

        match self.type:
            case AssignmentValueType.COST:
                closest_values = np.min(self.value, axis=axis)
                if axis == 0:
                    second_values = np.partition(self.value, 1, axis=axis)[1, :] if len_row >= 2 else np.full(len_col, np.inf)
                else:
                    second_values = np.partition(self.value, 1, axis=axis)[:, 1] if len_col >= 2 else np.full(len_row, np.inf)
                ratio_test_results = np.divide(closest_values + eps, second_values + eps)
            case AssignmentValueType.SCORE:
                closest_values = np.max(self.value, axis=axis)
                if axis == 0:
                    second_values = np.partition(self.value, -2, axis=axis)[-2, :] if len_row >= 2 else np.zeros(len_col)
                else:
                    second_values = np.partition(self.value, -2, axis=axis)[:, -2] if len_col >= 2 else np.zeros(len_row)
                ratio_test_results = np.divide(second_values + eps, closest_values + eps)
        return closest_values, ratio_test_results

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
        return self.type.create_threshold_predicate(threshold=threshold)

    def create_mask(
        self, 
        threshold: float
        ) -> np.ndarray:
        """
        Create a mask matrix based on the threshold.

        Parameters
        ----------
        threshold: float
            The threshold value.

        Returns
        -------
        np.ndarray: The mask matrix.
        """
        threshold_predicate = self.create_threshold_predicate(threshold=threshold)
        mask = threshold_predicate(self.value)
        return mask

    def __add__(
        self, 
        other: AssignmentMatrix
        ) -> AssignmentMatrix:
        return AssignmentMatrix(
            value=self.value + other.value, 
            type=self.type
            )

    def __mul__( 
        self, 
        other: AssignmentMatrix
        ) -> AssignmentMatrix:
        return AssignmentMatrix(
            value=self.value * other.value, 
            type=self.type
            )

    @property
    def flatten(self) -> np.ndarray:    
        """
        Flatten the matrix.
        """
        return self.value.flatten()

    @property
    def max(self) -> float:
        return np.max(self.flatten)

    @property
    def min(self) -> float:
        return np.min(self.flatten)

    def __str__(self):
        """
        String representation of AssignmentMatrix.
        
        Returns:
            str: Formatted string showing matrix shape and contents
        """
        return (
        f"AssignmentMatrix(\n"
        f"  matrix.shape={self.value.shape},\n"
        f")"
        )

