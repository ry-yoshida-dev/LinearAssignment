# linear_assignment

## Overview

Assignment matrices with cost/score semantics, an abstract solver interface, and an enum factory for the concrete solvers.

## Components

| Component | Description |
| --------- | ----------- |
| [`matrix/`](./matrix/README.md) | `AssignmentMatrix` and its mixins and protocols |
| [`array_types.py`](./array_types.py) | NumPy dtype aliases (`NumericArray`, `IndexArray`, `BoolArray`, …) |
| [`value_type.py`](./value_type.py) | `AssignmentValueType` enum (`COST` / `SCORE`) |
| [`solver.py`](./solver.py) | `AssignmentSolver` abstract base |
| [`method.py`](./method.py) | `AssignmentSolverMethod` enum and solver factory (`build()`) |
| [`solvers/`](./solvers/README.md) | Concrete solver implementations |
