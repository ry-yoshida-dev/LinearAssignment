"""Shared attribute declarations for assignment matrix mixins."""

from __future__ import annotations

from ...array_types import NumericArray
from ...value_type import AssignmentValueType


class AssignmentMatrixMixinBase:
    """
    Declares backing attributes consumed by matrix mixins.

    Implemented structurally by
    :class:`~linear_assignment.matrix.core.AssignmentMatrix`.
    """

    value: NumericArray
    type: AssignmentValueType
