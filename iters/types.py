from __future__ import annotations

from typing import TypeVar, Union

from solus import Singleton
from typing_extensions import TypeIs
from wraps.option import NULL, Option, Some

__all__ = (
    # markers
    "Marker",
    "marker",
    # wrap marked
    "wrap_marked",
    # type guards
    "is_marker",
)

T = TypeVar("T")


class NoDefault(Singleton):
    pass


no_default = NoDefault()


NoDefaultOr = Union[NoDefault, T]


def is_no_default(item: NoDefaultOr[T]) -> TypeIs[NoDefault]:
    return item is no_default


class Marker(Singleton):
    """Represents markers used for various checks."""


marker = Marker()
"""The instance of [`Marker`][iters.types.Marker]."""


MarkerOr = Union[Marker, T]


def is_marker(item: MarkerOr[T]) -> TypeIs[Marker]:
    """Checks if the `item` is [`Marker`][iters.types.Marker].

    Returns:
        Whether the `item` is [`Marker`][iters.types.Marker].
    """
    return item is marker


def wrap_marked(item: MarkerOr[T]) -> Option[T]:
    """Convertes [`MarkerOr[T]`][iters.types.MarkerOr] to [`Option[T]`][wraps.option.Option].

    Arguments:
        item: The item to convert.

    Returns:
        The converted item.
    """
    return NULL if is_marker(item) else Some(item)
