# LinearAssignment

## Overview

`linear_assignment` solves bipartite assignment problems on cost or score matrices through one interface over several solvers. `AssignmentMatrix.type` selects whether lower (`COST`) or higher (`SCORE`) values are preferred, and every solver returns `(row_indices, col_indices)`.

| Method | Class | Characteristics |
| ------ | ----- | --------------- |
| `HUNGARIAN` | `HungarianAssignmentSolver` | Global optimum via `scipy.optimize.linear_sum_assignment` |
| `GREEDY` | `GreedyAssignmentSolver` | Fast heuristic; sorted pick without row/column conflicts |
| `MUTUAL_OPTIMAL` | `MutualOptimalAssignmentSolver` | Keeps only mutually optimal row/column pairs |

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np

from linear_assignment import (
    AssignmentMatrix,
    AssignmentSolverMethod,
    AssignmentValueType,
)

matrix = AssignmentMatrix(
    value=np.array(
        [
            [0.1, 0.7, 0.4],
            [0.3, 0.2, 0.8],
        ]
    ),
    type=AssignmentValueType.COST,
)

solver = AssignmentSolverMethod.HUNGARIAN.build()
row_indices, col_indices = solver.run_assignment(matrix)
```
