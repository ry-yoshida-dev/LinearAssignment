"""Test helpers for assignment verification."""

from __future__ import annotations

import numpy as np

from linear_assignment import AssignmentMatrix
from linear_assignment.array_types import IndexArray, NumericArray


def _index_permutations(pool_size: int, pick: int) -> list[tuple[int, ...]]:
    """Return all ``pick``-length permutations of ``range(pool_size)``."""
    results: list[tuple[int, ...]] = []
    _collect_index_permutations(
        available=tuple(range(pool_size)),
        pick=pick,
        prefix=(),
        results=results,
    )
    return results


def _collect_index_permutations(
    available: tuple[int, ...],
    pick: int,
    prefix: tuple[int, ...],
    results: list[tuple[int, ...]],
) -> None:
    """Recursively collect index permutations into ``results``."""
    if pick == 0:
        results.append(prefix)
        return

    for index, value in enumerate(available):
        remainder = available[:index] + available[index + 1 :]
        _collect_index_permutations(
            available=remainder,
            pick=pick - 1,
            prefix=prefix + (value,),
            results=results,
        )


def assignment_index_pairs(
    row_indices: IndexArray,
    col_indices: IndexArray,
) -> set[tuple[int, int]]:
    """Convert solver output arrays into a set of integer index pairs."""
    rows = [int(value) for value in row_indices.tolist()]
    cols = [int(value) for value in col_indices.tolist()]
    return set(zip(rows, cols, strict=True))


def assert_valid_assignment(
    row_indices: IndexArray,
    col_indices: IndexArray,
    num_rows: int,
    num_cols: int,
) -> None:
    """Verify assignment indices are in bounds and conflict-free."""
    assert row_indices.shape == col_indices.shape
    assert len(np.unique(row_indices)) == len(row_indices)
    assert len(np.unique(col_indices)) == len(col_indices)
    assert np.all((row_indices >= 0) & (row_indices < num_rows))
    assert np.all((col_indices >= 0) & (col_indices < num_cols))
    assert len(row_indices) <= min(num_rows, num_cols)


def assignment_total(
    matrix: AssignmentMatrix,
    row_indices: IndexArray,
    col_indices: IndexArray,
) -> float:
    """Sum assigned cell values."""
    return float(np.sum(matrix.value[row_indices, col_indices]))


def brute_force_optimal_cost(cost_matrix: NumericArray) -> float:
    """Exhaustive minimum total cost for small matrices."""
    num_rows, num_cols = cost_matrix.shape
    match_count = min(num_rows, num_cols)
    best_cost = float("inf")

    if num_rows <= num_cols:
        for col_perm in _index_permutations(num_cols, match_count):
            cost = sum(cost_matrix[i, col_perm[i]] for i in range(num_rows))
            best_cost = min(best_cost, cost)
    else:
        for row_perm in _index_permutations(num_rows, match_count):
            cost = sum(cost_matrix[row_perm[j], j] for j in range(num_cols))
            best_cost = min(best_cost, cost)

    return best_cost


def brute_force_optimal_score(score_matrix: NumericArray) -> float:
    """Exhaustive maximum total score for small matrices."""
    num_rows, num_cols = score_matrix.shape
    match_count = min(num_rows, num_cols)
    best_score = float("-inf")

    if num_rows <= num_cols:
        for col_perm in _index_permutations(num_cols, match_count):
            score = sum(score_matrix[i, col_perm[i]] for i in range(num_rows))
            best_score = max(best_score, score)
    else:
        for row_perm in _index_permutations(num_rows, match_count):
            score = sum(score_matrix[row_perm[j], j] for j in range(num_cols))
            best_score = max(best_score, score)

    return best_score
