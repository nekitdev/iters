from typing import Iterable, Iterator, TypeVar

from funcs.typing import Predicate, Unary
from wraps.result import Result

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")


def map_ok(function: Unary[T, U], iterable: Iterable[Result[T, E]]) -> Iterator[Result[U, E]]:
    for result in iterable:
        yield result.map(function)


def filter_ok_predicate(predicate: Predicate[T]) -> Predicate[Result[T, E]]:
    def result_predicate(result: Result[T, E]) -> bool:
        return result.is_ok() and predicate(result.unwrap())

    return result_predicate


def filter_ok(predicate: Predicate[T], iterable: Iterable[Result[T, E]]) -> Iterator[Result[T, E]]:
    return filter(filter_ok_predicate(predicate), iterable)
