from __future__ import annotations

from typing import Any, TypeVar, Union

from solus import Singleton
from typing_extensions import TypeGuard
from wraps.option import Null, Option, Some

__all__ = (
    # markers
    "Marker",
    "marker",
    # no default
    "NoDefault",
    "no_default",
    # wrap marked
    "wrap_marked",
    # type guards
    "is_marker",
    "is_not_marker",
    "is_no_default",
    "is_not_no_default",
)

T = TypeVar("T")


class NoDefault(Singleton):
    """Represents the absence of default values."""


no_default = NoDefault()
"""The instance of [`NoDefault`][iters.types.NoDefault]."""


NoDefaultOr = Union[NoDefault, T]


def is_no_default(item: Any) -> TypeGuard[NoDefault]:
    """Checks if the `item` is [`NoDefault`][iters.types.NoDefault].

    Returns:
        Whether the `item` is [`NoDefault`][iters.types.NoDefault].
    """
    return item is no_default


def is_not_no_default(item: NoDefaultOr[T]) -> TypeGuard[T]:
    """Checks if the `item` is not [`NoDefault`][iters.types.NoDefault].

    Returns:
        Whether the `item` is not [`NoDefault`][iters.types.NoDefault].
    """
    return item is not no_default


class Marker(Singleton):
    """Represents markers used for various checks."""


marker = Marker()
"""The instance of [`Marker`][iters.types.Marker]."""


MarkerOr = Union[Marker, T]


def is_marker(item: Any) -> TypeGuard[Marker]:
    """Checks if the `item` is [`Marker`][iters.types.Marker].

    Returns:
        Whether the `item` is [`Marker`][iters.types.Marker].
    """
    return item is marker


def is_not_marker(item: MarkerOr[T]) -> TypeGuard[T]:
    """Checks if the `item` is not [`Marker`][iters.types.Marker].

    Returns:
        Whether the `item` is not [`Marker`][iters.types.Marker].
    """
    return item is not marker


def wrap_marked(item: MarkerOr[T]) -> Option[T]:
    return Some(item) if is_not_marker(item) else Null()
