"""Tests for public package exports."""

from __future__ import annotations

import linear_assignment


class TestPublicExports:
    """Verify __all__ symbols are importable from the package root."""

    def test_all_exports_exist(self) -> None:
        for name in linear_assignment.__all__:
            assert hasattr(linear_assignment, name)

    def test_all_length(self) -> None:
        assert len(linear_assignment.__all__) == 9
