# mixin

## Overview

Mixins composed into `AssignmentMatrix`, each adding a focused slice of behavior.

## Components

| Component | Description |
| --------- | ----------- |
| [`property.py`](./property.py) | `AssignmentMatrixPropertyMixin` (shape, min/max, mean, best/worst) |
| [`function.py`](./function.py) | `AssignmentMatrixFunctionMixin` (masking, ratio test, cost conversion, assignment filtering) |
| [`special.py`](./special.py) | `AssignmentMatrixSpecialMixin` (indexing, arithmetic, `__str__`) |
