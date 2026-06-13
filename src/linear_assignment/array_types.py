"""NumPy array type aliases for linear assignment."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

NumericDType: TypeAlias = np.integer[Any] | np.floating[Any]
NumericArray: TypeAlias = NDArray[NumericDType]
BoolArray: TypeAlias = NDArray[np.bool_]
IndexArray: TypeAlias = NDArray[np.integer[Any]]
AssignmentPairIndices: TypeAlias = tuple[IndexArray, IndexArray]
