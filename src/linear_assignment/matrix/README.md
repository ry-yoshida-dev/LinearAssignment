# matrix

## Overview

`AssignmentMatrix`, a 2D cost or score matrix with statistics, masking, and assignment filtering.

## Components

| Component | Description |
| --------- | ----------- |
| [`core.py`](./core.py) | `AssignmentMatrix` dataclass wiring mixins together |
| [`protocols/`](./protocols/README.md) | `AssignmentMatrixBacking` and `AssignmentMatrixStatistics` protocols |
| [`mixin/`](./mixin/README.md) | Property, function, and special mixins composed into `AssignmentMatrix` |

## Examples

```python
import numpy as np
from linear_assignment import AssignmentMatrix, AssignmentValueType

matrix = AssignmentMatrix(
    value=np.array([[1.0, 4.0], [3.0, 2.0]]),
    type=AssignmentValueType.COST,
)
mask = matrix.create_mask(threshold=2.5)
row_indices, col_indices = matrix.filter_assignment_by_threshold(
    np.array([0, 1]),
    np.array([0, 1]),
    threshold=2.5,
)
cost_matrix = matrix.convert_cost_format()
```
