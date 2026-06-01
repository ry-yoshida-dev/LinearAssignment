"""Assignment matrix package."""

from .core import AssignmentMatrix
from .protocols import AssignmentMatrixBacking, AssignmentMatrixStatistics

__all__ = [
    "AssignmentMatrix",
    "AssignmentMatrixBacking",
    "AssignmentMatrixStatistics",
]
