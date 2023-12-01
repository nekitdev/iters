from typing import Iterable, Iterator, TypeVar

from typing_aliases import Binary, Unary
from wraps.option import Option, is_some

__all__ = ("filter_map_option", "scan")

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")


def scan(state: S, function: Binary[S, T, Option[U]], iterable: Iterable[T]) -> Iterator[U]:
    for item in iterable:
        option = function(state, item)

        if is_some(option):
            yield option.unwrap()

        else:
            break


def filter_map_option(function: Unary[T, Option[U]], iterable: Iterable[T]) -> Iterator[U]:
    for option in map(function, iterable):
        if is_some(option):
            yield option.unwrap()
