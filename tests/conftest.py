"""Shared fixtures for linear_assignment tests."""

from __future__ import annotations

import numpy as np
import pytest

from linear_assignment import AssignmentMatrix, AssignmentValueType
from linear_assignment.solvers import (
    GreedyAssignmentSolver,
    HungarianAssignmentSolver,
    MutualOptimalAssignmentSolver,
)


@pytest.fixture
def cost_matrix_3x3() -> AssignmentMatrix:
    """Small square cost matrix with a unique Hungarian optimum."""
    return AssignmentMatrix(
        value=np.array(
            [
                [1.0, 5.0, 3.0],
                [4.0, 2.0, 6.0],
                [7.0, 8.0, 1.0],
            ]
        ),
        type=AssignmentValueType.COST,
    )


@pytest.fixture
def score_matrix_2x3() -> AssignmentMatrix:
    """Rectangular score matrix (more columns than rows)."""
    return AssignmentMatrix(
        value=np.array(
            [
                [0.1, 0.7, 0.4],
                [0.3, 0.2, 0.8],
            ]
        ),
        type=AssignmentValueType.SCORE,
    )


@pytest.fixture
def mutual_optimal_cost_matrix() -> AssignmentMatrix:
    """Cost matrix where mutual-optimal pairs form a perfect matching."""
    return AssignmentMatrix(
        value=np.array(
            [
                [1.0, 9.0],
                [8.0, 2.0],
            ]
        ),
        type=AssignmentValueType.COST,
    )


@pytest.fixture
def hungarian_solver() -> HungarianAssignmentSolver:
    """Hungarian solver instance."""
    return HungarianAssignmentSolver()


@pytest.fixture
def greedy_solver() -> GreedyAssignmentSolver:
    """Greedy solver instance."""
    return GreedyAssignmentSolver()


@pytest.fixture
def mutual_optimal_solver() -> MutualOptimalAssignmentSolver:
    """Mutual-optimal solver instance."""
    return MutualOptimalAssignmentSolver()
