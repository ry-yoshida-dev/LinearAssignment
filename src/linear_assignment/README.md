# linear_assignment

## Overview

`linear_assignment` provides a compact API for assignment matching on 2D matrices.
The package centers on:

- `AssignmentMatrix`: matrix object with value semantics (`COST`/`SCORE`)
- `AssignmentSolver`: abstract solver interface
- `AssignmentSolverMethod`: enum factory for concrete solvers

## Components

| Component | Description |
| --------- | ----------- |
| [`matrix.py`](./matrix.py) | `AssignmentMatrix` and matrix utility operations |
| [`value_type.py`](./value_type.py) | `AssignmentValueType` enum (`COST` / `SCORE`) |
| [`solver.py`](./solver.py) | `AssignmentSolver` abstract base |
| [`method.py`](./method.py) | `AssignmentSolverMethod` enum and solver factory (`build()`) |
| [`solvers/`](./solvers/) | Concrete implementations (Hungarian, Greedy, MutualOptimal) |

## Supported solvers

| Method | Class | Characteristics |
| ------ | ----- | --------------- |
| `HUNGARIAN` | `HungarianAssignmentSolver` | Global optimum via `scipy.optimize.linear_sum_assignment` |
| `GREEDY` | `GreedyAssignmentSolver` | Fast heuristic; sorted pick without row/col conflicts |
| `MUTUAL_OPTIMAL` | `MutualOptimalAssignmentSolver` | Keeps only mutually optimal row/col pairs |
