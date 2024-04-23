from typing import Iterable, Iterator, TypeVar

from typing_aliases import Binary, Unary
from wraps.primitives.option import NULL, Option, Some
from wraps.primitives.result import Error, Ok, Result

from iters.types import is_marker, marker
from iters.utils import chain

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


def exactly_one(iterable: Iterable[T]) -> Result[T, Option[Iterator[T]]]:
    iterator = iter(iterable)

    first = next(iterator, marker)

    if is_marker(first):
        return Error(NULL)

    second = next(iterator, marker)

    if not is_marker(second):
        return Error(Some(chain((first, second), iterator)))

    return Ok(first)


def at_most_one(iterable: Iterable[T]) -> Result[Option[T], Iterator[T]]:
    iterator = iter(iterable)

    first = next(iterator, marker)

    if is_marker(first):
        return Ok(NULL)

    second = next(iterator, marker)

    if not is_marker(second):
        return Error(chain((first, second), iterator))

    return Ok(Some(first))
