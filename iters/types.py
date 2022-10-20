from __future__ import annotations

from enum import Enum
from threading import Lock
from typing import Any, Type, TypeVar

from iters.typing import get_name

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


S = TypeVar("S")


class SingletonType(type):
    _INSTANCES = {}  # type: ignore
    _LOCK = Lock()

    def __call__(cls: Type[S], *args: Any, **kwargs: Any) -> S:
        instances = cls._INSTANCES  # type: ignore
        lock = cls._LOCK  # type: ignore

        # use double-checked locking

        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = super().__call__(*args, **kwargs)  # type: ignore

        return instances[cls]  # type: ignore


class Singleton(metaclass=SingletonType):
    def __repr__(self) -> str:
        return get_name(type(self))


singleton = Singleton()


class NoDefault(Singleton):
    pass


no_default = NoDefault()


class Marker(Singleton):
    pass


marker = Marker()
