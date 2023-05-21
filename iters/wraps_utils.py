from typing import Iterable, Iterator, TypeVar

from typing_aliases import Binary, Pair
from wraps.result import Result

__all__ = ("coalesce",)

T = TypeVar("T")


def coalesce(function: Binary[T, T, Result[T, Pair[T]]], iterable: Iterable[T]) -> Iterator[T]:
    ...
