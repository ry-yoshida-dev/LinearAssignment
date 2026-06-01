# mixin

## Overview

Mixin classes composed into `AssignmentMatrix` in `core.py`. Each mixin adds a focused slice of behavior on top of the shared `value` / `type` fields.

## Components

| Component | Description |
| --------- | ----------- |
| [`property.py`](./property.py) | `AssignmentMatrixPropertyMixin` (shape, min/max, mean, best/worst) |
| [`function.py`](./function.py) | `AssignmentMatrixFunctionMixin` (masking, ratio test, cost conversion, assignment filtering) |
| [`special.py`](./special.py) | `AssignmentMatrixSpecialMixin` (indexing, arithmetic, `__str__`) |
