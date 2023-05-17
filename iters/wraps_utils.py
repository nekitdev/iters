from typing import Iterable, Iterator, TypeVar

from funcs.typing import Binary
from wraps.result import Result

from iters.typing import Pair

__all__ = ("coalesce",)

T = TypeVar("T")


def coalesce(function: Binary[T, T, Result[T, Pair[T]]], iterable: Iterable[T]) -> Iterator[T]:
    ...
