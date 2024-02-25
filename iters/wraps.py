from typing import Iterable, Iterator, TypeVar

from typing_aliases import Binary, Unary
from wraps.option import Option

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")


def scan(state: S, function: Binary[S, T, Option[U]], iterable: Iterable[T]) -> Iterator[U]:
    for item in iterable:
        option = function(state, item)

        if option.is_some():
            yield option.unwrap()

        else:
            break


def filter_map_option(function: Unary[T, Option[U]], iterable: Iterable[T]) -> Iterator[U]:
    for item in iterable:
        option = function(item)

        if option.is_some():
            yield option.unwrap()
