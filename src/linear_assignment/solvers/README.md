# solvers

## Overview

Concrete `AssignmentSolver` implementations.

## Components

| Component | Description |
| --------- | ----------- |
| [`hungarian.py`](./hungarian.py) | `HungarianAssignmentSolver`: global optimum |
| [`greedy.py`](./greedy.py) | `GreedyAssignmentSolver`: greedy pick of the best available pairs |
| [`mutual_optimal.py`](./mutual_optimal.py) | `MutualOptimalAssignmentSolver`: mutually optimal pairs only |
