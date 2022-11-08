from __future__ import annotations

from enum import Enum
from threading import Lock
from typing import Any, Type, TypeVar

from solus import Singleton

__all__ = (
    # ordering
    "Ordering",
    # markers
    "Marker",
    "marker",
    # no default
    "NoDefault",
    "no_default",
)


class Ordering(Enum):
    """Represents ordering."""

    LESS = -1
    """The left item is *less* than the right item."""
    EQUAL = 0
    """The left item is *equal* to the right item."""
    GREATER = 1
    """The left item is *greater* than the right item."""

    def is_less(self) -> bool:
        """Checks if the ordering is [`LESS`][iters.types.Ordering.LESS].

        Returns:
            Whether the ordering is [`LESS`][iters.types.Ordering.LESS].
        """
        return self is type(self).LESS

    def is_equal(self) -> bool:
        """Checks if the ordering is [`EQUAL`][iters.types.Ordering.EQUAL].

        Returns:
            Whether the ordering is [`EQUAL`][iters.types.Ordering.EQUAL].
        """
        return self is type(self).EQUAL

    def is_greater(self) -> bool:
        """Checks if the ordering is [`GREATER`][iters.types.Ordering.GREATER].

        Returns:
            Whether the ordering is [`GREATER`][iters.types.Ordering.GREATER].
        """
        return self is type(self).GREATER

    def is_less_or_equal(self) -> bool:
        """Checks if the ordering is [`LESS`][iters.types.Ordering.LESS] or
        [`EQUAL`][iters.types.Ordering.EQUAL].

        This is equivalent to:

        ```python
        ordering.is_less() or ordering.is_equal()
        ```

        Returns:
            Whether the ordering is [`LESS`][iters.types.Ordering.LESS]
                or [`EQUAL`][iters.types.Ordering.EQUAL].
        """
        return self.is_less() or self.is_equal()

    def is_not_equal(self) -> bool:
        """Checks if the ordering is not [`EQUAL`][iters.types.Ordering.EQUAL].

        This is equivalent to:

        ```python
        not ordering.is_equal()
        ```

        Returns:
            Whether the ordering is not [`EQUAL`][iters.types.Ordering.EQUAL].
        """
        return not self.is_equal()

    def is_greater_or_equal(self) -> bool:
        """Checks if the ordering is [`GREATER`][iters.types.Ordering.GREATER] or
        [`EQUAL`][iters.types.Ordering.EQUAL].

        This is equivalent to:

        ```python
        ordering.is_greater() or ordering.is_equal()
        ```

        Returns:
            Whether the ordering is [`GREATER`][iters.types.Ordering.GREATER]
                or [`EQUAL`][iters.types.Ordering.EQUAL].
        """
        return self.is_greater() or self.is_equal()


class NoDefault(Singleton):
    """Represents the absence of default values."""


no_default = NoDefault()
"""The instance of [`NoDefault`][iters.types.NoDefault]."""


class Marker(Singleton):
    """Represents markers used for various checks."""


marker = Marker()
"""The instance of [`Marker`][iters.types.Marker]."""
