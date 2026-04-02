# LinearAssignment

## Overview

LinearAssignment (`linear_assignment`) is a Python package for solving bipartite assignment problems from cost/score matrices.
It provides a unified interface over multiple solver strategies:

- Hungarian (global optimum)
- Greedy (fast heuristic)
- Mutual-optimal (mutual best pair filtering)

For package-level details, see [src/linear_assignment/README.md](src/linear_assignment/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

If you only need dependencies:

```bash
pip install -r requirements.txt
```

## Quick example

```python
import numpy as np

from linear_assignment import (
    AssignmentMatrix,
    AssignmentSolverMethod,
    AssignmentValueType,
)

matrix = AssignmentMatrix(
    value=np.array([
        [0.1, 0.7, 0.4],
        [0.3, 0.2, 0.8],
    ]),
    type=AssignmentValueType.COST,
)

solver = AssignmentSolverMethod.HUNGARIAN.build()
row_idx, col_idx = solver.run_assignment(matrix)
print(row_idx, col_idx)
```

## Notes

- `AssignmentMatrix.type` controls whether lower (`COST`) or higher (`SCORE`) values are preferred.
- All solvers return `(row_indices, col_indices)` as `numpy.ndarray`.
