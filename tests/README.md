# Tests

## Overview

Pytest suite for the `linear_assignment` package under `src/`. Covers matrix mixins, value types, solver factory methods, and all three assignment solvers.

## Components

| File | Description |
| --- | --- |
| [helpers.py](helpers.py) | Assignment verification and brute-force reference helpers |
| [conftest.py](conftest.py) | Shared pytest fixtures |
| [test_value_type.py](test_value_type.py) | `AssignmentValueType` threshold predicates |
| [test_method.py](test_method.py) | `AssignmentSolverMethod` factory |
| [test_matrix_core.py](test_matrix_core.py) | `AssignmentMatrix` construction validation |
| [test_matrix_property.py](test_matrix_property.py) | Property mixin (shape, min/max, best/worst) |
| [test_matrix_function.py](test_matrix_function.py) | Function mixin (masks, ratio test, filtering) |
| [test_matrix_special.py](test_matrix_special.py) | Dunder methods (indexing, arithmetic, str) |
| [test_solver.py](test_solver.py) | Base solver (1×1 shortcut, str) |
| [test_solver_hungarian.py](test_solver_hungarian.py) | Hungarian optimal assignment |
| [test_solver_greedy.py](test_solver_greedy.py) | Greedy heuristic assignment |
| [test_solver_mutual_optimal.py](test_solver_mutual_optimal.py) | Mutual-optimal matching |
| [test_init.py](test_init.py) | Public API exports |

## Examples

Run all tests from the repository root:

```bash
pytest
```

Install dev dependencies first:

```bash
pip install -e ".[dev]"
```
