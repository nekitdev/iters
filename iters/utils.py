from __future__ import annotations

from builtins import zip as standard_zip
from collections import Counter as counter_dict
from collections import deque
from functools import reduce as standard_reduce
from itertools import accumulate as standard_accumulate
from itertools import chain, combinations, combinations_with_replacement, compress
from itertools import count as standard_count
from itertools import cycle
from itertools import dropwhile as standard_drop_while
from itertools import filterfalse as filter_false
from itertools import groupby as standard_group
from itertools import islice as iter_slice
from itertools import permutations
from itertools import product as standard_product
from itertools import takewhile as standard_take_while
from itertools import tee as standard_copy
from itertools import zip_longest as standard_zip_longest
from math import copysign as copy_sign
from operator import add
from operator import ge as greater_or_equal
from operator import gt as greater
from operator import le as less_or_equal
from operator import lt as less
from operator import mul
from sys import version_info
from typing import (
    Any,
    ContextManager,
    Counter,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from funcs.typing import (
    AnyErrorType,
    Binary,
    Compare,
    DynamicTuple,
    EmptyTuple,
    Inspect,
    Nullary,
    Predicate,
    Tuple1,
    Tuple2,
    Tuple3,
    Tuple4,
    Tuple5,
    Tuple6,
    Tuple7,
    Tuple8,
    Unary,
)
from funcs.unpacking import unpack_binary
from orderings import LenientOrdered, Ordering, StrictOrdered
from typing_extensions import Literal, Never

from iters.types import is_marker, is_no_default, is_not_marker, marker, no_default
from iters.typing import (
    ForEach,
    Pair,
    Product,
    RecursiveIterable,
    Sum,
    Validate,
    is_bytes,
    is_string,
)

__all__ = (
    "accumulate_fold",
    "accumulate_reduce",
    "accumulate_product",
    "accumulate_sum",
    "all_equal",
    "all_unique",
    "all_unique_fast",
    "append",
    "at",
    "at_or_last",
    "cartesian_power",
    "cartesian_product",
    "chain",
    "chain_from_iterable",
    "chunks",
    "collapse",
    "combinations",
    "combinations_with_replacement",
    "combine",
    "compare",
    "compress",
    "consume",
    "contains",
    "contains_identity",
    "copy",
    "copy_infinite",
    "copy_unsafe",
    "count",
    "count_dict",
    "cycle",
    "distribute",
    "distribute_infinite",
    "distribute_unsafe",
    "divide",
    "drop",
    "drop_while",
    "duplicates",
    "duplicates_fast",
    "empty",
    "filter_except",
    "filter_false",
    "filter_false_map",
    "filter_map",
    "find_all",
    "find",
    "find_or_first",
    "find_or_last",
    "first",
    "flat_map",
    "flatten",
    "fold",
    "for_each",
    "group",
    "group_dict",
    "group_list",
    "groups",
    "groups_longest",
    "has_next",
    "inspect",
    "interleave",
    "interleave_longest",
    "intersperse",
    "intersperse_with",
    "is_empty",
    "is_sorted",
    "iter_chunks",
    "iter_chunks_infinite",
    "iter_chunks_unsafe",
    "iter_except",
    "iter_function",
    "iter_length",
    "iter_slice",
    "iter_windows",
    "iter_with",
    "iterate",
    "last",
    "last_with_tail",
    "list_windows",
    "map_except",
    "min_max",
    "next_of",
    "once",
    "once_with",
    "pad",
    "pad_with",
    "pairs",
    "pairs_longest",
    "pairs_windows",
    "partition",
    "partition_infinite",
    "partition_unsafe",
    "peek",
    "permutations",
    "position_all",
    "position",
    "power_set",
    "prepend",
    "product",
    "reduce",
    "remove",
    "remove_duplicates",
    "repeat",
    "repeat_each",
    "repeat_last",
    "repeat_with",
    "rest",
    "reverse",
    "skip",
    "skip_while",
    "sort",
    "spy",
    "step_by",
    "sum",
    "tail",
    "take",
    "take_while",
    "transpose",
    "tuple_windows",
    "unique",
    "unique_fast",
    "zip",
    "zip_equal",
    "zip_longest",
)

PYTHON_3_8 = version_info >= (3, 8, 0)
PYTHON_3_10 = version_info >= (3, 10, 0)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

R = TypeVar("R")

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")

# I = TypeVar("I", bound=Iterable)  # I[T]

Q = TypeVar("Q", bound=Hashable)

S = TypeVar("S", bound=Sum)
P = TypeVar("P", bound=Product)

LT = TypeVar("LT", bound=LenientOrdered)
ST = TypeVar("ST", bound=StrictOrdered)


def take_while(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> Iterator[T]:
    return standard_take_while(predicate or bool, iterable)


def drop_while(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> Iterator[T]:
    return standard_drop_while(predicate or bool, iterable)


chain_from_iterable = chain.from_iterable
skip_while = drop_while


@overload
def compare(left_iterable: Iterable[ST], right_iterable: Iterable[ST], key: None = ...) -> Ordering:
    ...


@overload
def compare(left_iterable: Iterable[T], right_iterable: Iterable[T], key: Unary[T, ST]) -> Ordering:
    ...


def compare(
    left_iterable: Iterable[Any],
    right_iterable: Iterable[Any],
    key: Optional[Unary[Any, Any]] = None,
) -> Ordering:
    if key is None:
        return compare_simple(left_iterable, right_iterable)

    return compare_by(left_iterable, right_iterable, key)


def compare_simple(left_iterable: Iterable[ST], right_iterable: Iterable[ST]) -> Ordering:
    for left, right in zip_longest(left_iterable, right_iterable, fill=marker):
        if is_marker(left):
            return Ordering.LESS

        if is_marker(right):
            return Ordering.GREATER

        if left < right:  # type: ignore
            return Ordering.LESS

        if left > right:  # type: ignore
            return Ordering.GREATER

    return Ordering.EQUAL


def compare_by(
    left_iterable: Iterable[T], right_iterable: Iterable[T], key: Unary[T, ST]
) -> Ordering:
    return compare_simple(map(key, left_iterable), map(key, right_iterable))


def iter_function(function: Nullary[T], sentinel: V) -> Iterator[T]:
    return iter(function, sentinel)


def empty() -> Iterator[Never]:
    return
    yield  # type: ignore


def once(value: T) -> Iterator[T]:
    yield value


def once_with(function: Nullary[T]) -> Iterator[T]:
    yield function()


def repeat(value: T, count: Optional[int] = None) -> Iterator[T]:
    if count is None:
        while True:
            yield value

    else:
        for _ in range(count):
            yield value


def repeat_with(function: Nullary[T], count: Optional[int] = None) -> Iterator[T]:
    if count is None:
        while True:
            yield function()

    else:
        for _ in range(count):
            yield function()


def repeat_factory(count: int) -> Unary[T, Iterator[T]]:
    def actual_repeat(value: T) -> Iterator[T]:
        return repeat(value, count)

    return actual_repeat


def repeat_each(count: int, iterable: Iterable[T]) -> Iterator[T]:
    return flat_map(repeat_factory(count), iterable)


@overload
def repeat_last(iterable: Iterable[T]) -> Iterator[T]:
    ...


@overload
def repeat_last(iterable: Iterable[T], default: U) -> Iterator[Union[T, U]]:
    ...


def repeat_last(iterable: Iterable[Any], default: Any = no_default) -> Iterator[Any]:
    item = marker

    for item in iterable:
        yield item

    if is_marker(item):
        if is_no_default(default):
            return

        item = default

    yield from repeat(item)


def count(start: int = 0, step: int = 1) -> Iterator[int]:
    return standard_count(start, step)


def tabulate(function: Unary[int, T], start: int = 0, step: int = 1) -> Iterator[T]:
    return map(function, count(start, step))


def consume(iterable: Iterable[T]) -> None:
    deque(iterable, 0)


def for_each(function: ForEach[T], iterable: Iterable[T]) -> None:
    for item in iterable:
        function(item)


FIRST_ON_EMPTY = "first() called on an empty iterable"


@overload
def first(iterable: Iterable[T]) -> T:
    ...


@overload
def first(iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def first(iterable: Iterable[Any], default: Any = no_default) -> Any:
    iterator = iter(iterable)

    result = next(iterator, marker)

    if is_marker(result):
        if is_no_default(default):
            raise ValueError(FIRST_ON_EMPTY)

        return default

    return result


LAST_ON_EMPTY = "last() called on an empty iterable"


@overload
def last(iterable: Iterable[T]) -> T:
    ...


@overload
def last(iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def last(iterable: Iterable[Any], default: Any = no_default) -> Any:
    result = marker

    try:
        iterator = reversed(iterable)  # type: ignore

        result = next(iterator, marker)

    except TypeError:
        for result in iterable:  # noqa
            pass

    if is_marker(result):
        if is_no_default(default):
            raise ValueError(LAST_ON_EMPTY)

        return default

    return result


REDUCE_ON_EMPTY = "reduce() called on an empty iterable"


def reduce(function: Binary[T, T, T], iterable: Iterable[T]) -> T:
    empty, iterator = is_empty(iterable)

    if empty:
        raise ValueError(REDUCE_ON_EMPTY)

    return standard_reduce(function, iterator)


def fold(initial: U, function: Binary[U, T, U], iterable: Iterable[T]) -> U:
    return standard_reduce(function, iterable, initial)


def accumulate_reduce(function: Binary[T, T, T], iterable: Iterable[T]) -> Iterator[T]:
    return standard_accumulate(iterable, function)


def accumulate_fold(initial: U, function: Binary[U, T, U], iterable: Iterable[T]) -> Iterator[U]:
    if PYTHON_3_8:
        return standard_accumulate(iterable, function, initial=initial)  # type: ignore

    return standard_accumulate(prepend(initial, iterable), function)  # type: ignore


@overload
def accumulate_sum(iterable: Iterable[S]) -> Iterator[S]:
    ...


@overload
def accumulate_sum(iterable: Iterable[S], initial: S) -> Iterator[S]:
    ...


def accumulate_sum(iterable: Iterable[Any], initial: Any = no_default) -> Iterator[Any]:
    if is_no_default(initial):
        return accumulate_reduce(add, iterable)

    return accumulate_fold(initial, add, iterable)


@overload
def accumulate_product(iterable: Iterable[P]) -> Iterator[P]:
    ...


@overload
def accumulate_product(iterable: Iterable[P], initial: P) -> Iterator[P]:
    ...


def accumulate_product(iterable: Iterable[Any], initial: Any = no_default) -> Iterator[Any]:
    if is_no_default(initial):
        return accumulate_reduce(mul, iterable)

    return accumulate_fold(initial, mul, iterable)


AT_ON_EMPTY = "at() called on an empty iterable"


@overload
def at(index: int, iterable: Iterable[T]) -> T:
    ...


@overload
def at(index: int, iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def at(index: int, iterable: Iterable[Any], default: Any = no_default) -> Any:
    iterator = drop(index, iterable)

    result = next(iterator, marker)

    if is_marker(result):
        if is_no_default(default):
            raise ValueError(AT_ON_EMPTY)

        return default

    return result


AT_OR_LAST_ON_EMPTY = "at_or_last() called on an empty iterable"


@overload
def at_or_last(index: int, iterable: Iterable[T]) -> T:
    ...


@overload
def at_or_last(index: int, iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def at_or_last(index: int, iterable: Iterable[Any], default: Any = no_default) -> Any:
    length = index + 1

    iterator = take(length, iterable)

    result = last(iterator, marker)

    if is_marker(result):
        if is_no_default(default):
            raise ValueError(AT_OR_LAST_ON_EMPTY)

        return default

    return result


@overload
def copy_unsafe(iterable: Iterable[T]) -> Pair[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[0]) -> EmptyTuple:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[1]) -> Tuple1[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[2]) -> Tuple2[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[3]) -> Tuple3[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[4]) -> Tuple4[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[5]) -> Tuple5[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[6]) -> Tuple6[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[7]) -> Tuple7[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: Literal[8]) -> Tuple8[Iterator[T]]:
    ...


@overload
def copy_unsafe(iterable: Iterable[T], copies: int) -> DynamicTuple[Iterator[T]]:
    ...


def copy_unsafe(iterable: Iterable[T], copies: int = 2) -> DynamicTuple[Iterator[T]]:
    return standard_copy(iterable, copies)


copy_infinite = copy_unsafe


@overload
def copy(iterable: Iterable[T]) -> Pair[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[0]) -> EmptyTuple:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[1]) -> Tuple1[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[2]) -> Tuple2[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[3]) -> Tuple3[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[4]) -> Tuple4[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[5]) -> Tuple5[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[6]) -> Tuple6[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[7]) -> Tuple7[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: Literal[8]) -> Tuple8[Iterator[T]]:
    ...


@overload
def copy(iterable: Iterable[T], copies: int) -> DynamicTuple[Iterator[T]]:
    ...


def copy(iterable: Iterable[T], copies: int = 2) -> DynamicTuple[Iterator[T]]:
    collected = tuple(iterable)

    return tuple(iter(collected) for _ in range(copies))


def drop(size: int, iterable: Iterable[T]) -> Iterator[T]:
    return iter_slice(iterable, size, None)


skip = drop


def rest(iterable: Iterable[T]) -> Iterator[T]:
    return drop(1, iterable)


def take(size: int, iterable: Iterable[T]) -> Iterator[T]:
    return iter_slice(iterable, size)


def step_by(step: int, iterable: Iterable[T]) -> Iterator[T]:
    return iter_slice(iterable, None, None, step)


@overload
def groups(size: Literal[0], iterable: Iterable[T]) -> Iterator[Never]:
    ...


@overload
def groups(size: Literal[1], iterable: Iterable[T]) -> Iterator[Tuple1[T]]:
    ...


@overload
def groups(size: Literal[2], iterable: Iterable[T]) -> Iterator[Tuple2[T]]:
    ...


@overload
def groups(size: Literal[3], iterable: Iterable[T]) -> Iterator[Tuple3[T]]:
    ...


@overload
def groups(size: Literal[4], iterable: Iterable[T]) -> Iterator[Tuple4[T]]:
    ...


@overload
def groups(size: Literal[5], iterable: Iterable[T]) -> Iterator[Tuple5[T]]:
    ...


@overload
def groups(size: Literal[6], iterable: Iterable[T]) -> Iterator[Tuple6[T]]:
    ...


@overload
def groups(size: Literal[7], iterable: Iterable[T]) -> Iterator[Tuple7[T]]:
    ...


@overload
def groups(size: Literal[8], iterable: Iterable[T]) -> Iterator[Tuple8[T]]:
    ...


@overload
def groups(size: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
    ...


def groups(size: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
    return zip(*repeat(iter(iterable), size))


@overload
def groups_longest(size: Literal[0], iterable: Iterable[T]) -> Iterator[Never]:
    ...


@overload
def groups_longest(size: Literal[1], iterable: Iterable[T]) -> Iterator[Tuple1[T]]:
    ...


@overload
def groups_longest(size: Literal[2], iterable: Iterable[T]) -> Iterator[Tuple2[Optional[T]]]:
    ...


@overload
def groups_longest(size: Literal[3], iterable: Iterable[T]) -> Iterator[Tuple3[Optional[T]]]:
    ...


@overload
def groups_longest(size: Literal[4], iterable: Iterable[T]) -> Iterator[Tuple4[Optional[T]]]:
    ...


@overload
def groups_longest(size: Literal[5], iterable: Iterable[T]) -> Iterator[Tuple5[Optional[T]]]:
    ...


@overload
def groups_longest(size: Literal[6], iterable: Iterable[T]) -> Iterator[Tuple6[Optional[T]]]:
    ...


@overload
def groups_longest(size: Literal[7], iterable: Iterable[T]) -> Iterator[Tuple7[Optional[T]]]:
    ...


@overload
def groups_longest(size: Literal[8], iterable: Iterable[T]) -> Iterator[Tuple8[Optional[T]]]:
    ...


@overload
def groups_longest(size: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[Optional[T]]]:
    ...


@overload
def groups_longest(size: Literal[0], iterable: Iterable[T], fill: U) -> Iterator[Never]:
    ...


@overload
def groups_longest(size: Literal[1], iterable: Iterable[T], fill: U) -> Iterator[Tuple1[T]]:
    ...


@overload
def groups_longest(
    size: Literal[2], iterable: Iterable[T], fill: U
) -> Iterator[Tuple2[Union[T, U]]]:
    ...


@overload
def groups_longest(
    size: Literal[3], iterable: Iterable[T], fill: U
) -> Iterator[Tuple3[Union[T, U]]]:
    ...


@overload
def groups_longest(
    size: Literal[4], iterable: Iterable[T], fill: U
) -> Iterator[Tuple4[Union[T, U]]]:
    ...


@overload
def groups_longest(
    size: Literal[5], iterable: Iterable[T], fill: U
) -> Iterator[Tuple5[Union[T, U]]]:
    ...


@overload
def groups_longest(
    size: Literal[6], iterable: Iterable[T], fill: U
) -> Iterator[Tuple6[Union[T, U]]]:
    ...


@overload
def groups_longest(
    size: Literal[7], iterable: Iterable[T], fill: U
) -> Iterator[Tuple7[Union[T, U]]]:
    ...


@overload
def groups_longest(
    size: Literal[8], iterable: Iterable[T], fill: U
) -> Iterator[Tuple8[Union[T, U]]]:
    ...


@overload
def groups_longest(
    size: int, iterable: Iterable[T], fill: U
) -> Iterator[DynamicTuple[Union[T, U]]]:
    ...


def groups_longest(
    size: int, iterable: Iterable[Any], fill: Optional[Any] = None
) -> Iterator[DynamicTuple[Any]]:
    return zip_longest(*repeat(iter(iterable), size), fill=fill)


@overload
def pairs_longest(iterable: Iterable[T]) -> Iterator[Pair[Optional[T]]]:
    ...


@overload
def pairs_longest(iterable: Iterable[T], fill: U) -> Iterator[Pair[Union[T, U]]]:
    ...


def pairs_longest(iterable: Iterable[Any], fill: Optional[Any] = None) -> Iterator[Pair[Any]]:
    return groups_longest(2, iterable, fill)


def pairs(iterable: Iterable[T]) -> Iterator[Pair[T]]:
    return groups(2, iterable)


def flatten(nested: Iterable[Iterable[T]]) -> Iterator[T]:
    return chain_from_iterable(nested)


def flat_map(function: Unary[T, Iterable[U]], iterable: Iterable[T]) -> Iterator[U]:
    return flatten(map(function, iterable))


def filter_map(
    predicate: Optional[Predicate[T]], function: Unary[T, U], iterable: Iterable[T]
) -> Iterator[U]:
    if predicate is None:
        for item in iterable:
            if item:
                yield function(item)

    else:
        for item in iterable:
            if predicate(item):
                yield function(item)


def filter_false_map(
    predicate: Optional[Predicate[T]], function: Unary[T, U], iterable: Iterable[T]
) -> Iterator[U]:
    if predicate is None:
        for item in iterable:
            if not item:
                yield function(item)

    else:
        for item in iterable:
            if not predicate(item):
                yield function(item)


def partition_unsafe(
    predicate: Optional[Predicate[T]], iterable: Iterable[T]
) -> Pair[Iterator[T]]:
    for_true, for_false = copy_unsafe(iterable)

    return filter(predicate, for_true), filter_false(predicate, for_false)


partition_infinite = partition_unsafe


def partition(
    predicate: Optional[Predicate[T]], iterable: Iterable[T]
) -> Pair[Iterator[T]]:
    for_true, for_false = copy(iterable)

    return filter(predicate, for_true), filter_false(predicate, for_false)


def prepend(value: T, iterable: Iterable[T]) -> Iterator[T]:
    return chain(once(value), iterable)


def append(value: T, iterable: Iterable[T]) -> Iterator[T]:
    return chain(iterable, once(value))


@overload
def group(iterable: Iterable[T], key: None = ...) -> Iterator[Tuple[T, Iterator[T]]]:
    ...


@overload
def group(iterable: Iterable[T], key: Unary[T, U]) -> Iterator[Tuple[U, Iterator[T]]]:
    ...


def group(
    iterable: Iterable[Any], key: Optional[Unary[Any, Any]] = None
) -> Iterator[Tuple[Any, Iterator[Any]]]:
    return standard_group(iterable, key)


@overload
def group_list(iterable: Iterable[T], key: None = ...) -> Iterator[Tuple[T, List[T]]]:
    ...


@overload
def group_list(iterable: Iterable[T], key: Unary[T, U]) -> Iterator[Tuple[U, List[T]]]:
    ...


def group_list(
    iterable: Iterable[Any], key: Optional[Unary[Any, Any]] = None
) -> Iterator[Tuple[Any, List[Any]]]:
    for group_key, group_iterator in group(iterable, key):
        yield (group_key, list(group_iterator))


@overload
def group_dict(iterable: Iterable[Q], key: None = ...) -> Dict[Q, List[Q]]:
    ...


@overload
def group_dict(iterable: Iterable[T], key: Unary[T, Q]) -> Dict[Q, List[T]]:
    ...


def group_dict(
    iterable: Iterable[Any], key: Optional[Unary[Any, Any]] = None
) -> Dict[Any, List[Any]]:
    result: Dict[Any, List[Any]] = {}

    for group_key, group_iterator in group(iterable, key):
        result.setdefault(group_key, []).extend(group_iterator)

    return result


@overload
def count_dict(iterable: Iterable[Q], key: None = ...) -> Counter[Q]:
    ...


@overload
def count_dict(iterable: Iterable[T], key: Unary[T, Q]) -> Counter[Q]:
    ...


def count_dict(iterable: Iterable[Any], key: Optional[Unary[Any, Any]] = None) -> Counter[Any]:
    return counter_dict(iterable if key is None else map(key, iterable))


def chunks(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    iterator = iter(iterable)

    while True:
        chunk = list(take(size, iterator))

        if not chunk:
            break

        yield chunk


def iter_chunks_unsafe(size: int, iterable: Iterable[T]) -> Iterator[Iterator[T]]:
    source = iter(iterable)

    while True:
        empty, source = is_empty(source)

        if empty:
            return

        source, iterator = copy_unsafe(source)

        yield take(size, iterator)

        consume(take(size, source))


iter_chunks_infinite = iter_chunks_unsafe


def iter_chunks(size: int, iterable: Iterable[T]) -> Iterator[Iterator[T]]:
    source = iter(iterable)

    while True:
        empty, source = is_empty(source)

        if empty:
            return

        source, iterator = copy(source)

        yield take(size, iterator)

        consume(take(size, source))


def iter_length(iterable: Iterable[T]) -> int:
    counting = count()

    consume(zip(iterable, counting))

    return next(counting)


@overload
def sum(iterable: Iterable[S]) -> S:
    ...


@overload
def sum(iterable: Iterable[S], initial: S) -> S:
    ...


def sum(iterable: Iterable[Any], initial: Any = no_default) -> Any:
    if is_no_default(initial):
        return reduce(add, iterable)

    return fold(initial, add, iterable)


@overload
def product(iterable: Iterable[P]) -> P:
    ...


@overload
def product(iterable: Iterable[P], initial: P) -> P:
    ...


def product(iterable: Iterable[Any], initial: Any = no_default) -> Any:
    if is_no_default(initial):
        return reduce(mul, iterable)

    return fold(initial, mul, iterable)


def iterate(function: Unary[T, T], value: T, count: Optional[int] = None) -> Iterator[T]:
    if count is None:
        while True:
            yield value
            value = function(value)

    else:
        for _ in range(count):
            yield value
            value = function(value)


def iter_with(context_manager: ContextManager[Iterable[T]]) -> Iterator[T]:
    with context_manager as iterable:
        yield from iterable


def walk(node: RecursiveIterable[T]) -> Iterator[T]:
    if is_string(node) or is_bytes(node):
        yield node  # type: ignore
        return

    try:
        tree = iter(node)  # type: ignore

    except TypeError:
        yield node  # type: ignore
        return

    else:
        for child in tree:
            yield from walk(child)


def collapse(iterable: Iterable[RecursiveIterable[T]]) -> Iterator[T]:
    return walk(iterable)


def pad(
    value: T,
    iterable: Iterable[T],
    size: Optional[int] = None,
    *,
    multiple: bool = False,
) -> Iterator[T]:
    if size is None:
        yield from chain(iterable, repeat(value))

    else:
        count = 0

        for item in iterable:
            count += 1
            yield item

        length = (size - count) % size if multiple else (size - count)

        if length > 0:
            yield from repeat(value, length)


def pad_with(
    function: Unary[int, T],
    iterable: Iterable[T],
    size: Optional[int] = None,
    *,
    multiple: bool = False,
) -> Iterator[T]:
    index = 0

    for item in iterable:
        yield item
        index += 1

    if size is None:
        while True:
            yield function(index)
            index += 1

    else:
        length = (size - index) % size if multiple else (size - index)

        if length > 0:
            for index in range(index, index + length):
                yield function(index)


def contains(value: U, iterable: Iterable[T]) -> bool:
    return any(item == value for item in iterable)


def contains_identity(value: T, iterable: Iterable[T]) -> bool:
    return any(item is value for item in iterable)


@overload
def all_unique_fast(iterable: Iterable[Q], key: None = ...) -> bool:
    ...


@overload
def all_unique_fast(iterable: Iterable[T], key: Unary[T, Q]) -> bool:
    ...


def all_unique_fast(iterable: Iterable[Any], key: Optional[Unary[Any, Any]] = None) -> bool:
    is_unique, _ = is_empty(duplicates_fast(iterable, key))

    return is_unique


def all_unique(iterable: Iterable[T], key: Optional[Unary[T, U]] = None) -> bool:
    is_unique, _ = is_empty(duplicates(iterable, key))

    return is_unique


def all_equal(iterable: Iterable[T], key: Optional[Unary[T, U]] = None) -> bool:
    groups = group(iterable, key)

    return is_marker(next(groups, marker)) or is_marker(next(groups, marker))


def remove(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> Iterator[T]:
    return filter_false(predicate, iterable)


# XXX: more concise name?


def remove_duplicates(iterable: Iterable[T], key: Optional[Unary[T, U]] = None) -> Iterator[T]:
    for item in group(iterable, key):
        _, iterator = item

        yield next(iterator)


def spy(size: int, iterable: Iterable[T]) -> Tuple[List[T], Iterator[T]]:
    iterator = iter(iterable)

    head = list(take(size, iterator))

    return head.copy(), chain(head, iterator)


PEEK_ON_EMPTY = "peek() called on an empty iterable"


@overload
def peek(iterable: Iterable[T]) -> Tuple[T, Iterator[T]]:
    ...


@overload
def peek(iterable: Iterable[T], default: U) -> Tuple[Union[T, U], Iterator[T]]:
    ...


def peek(iterable: Iterable[Any], default: Any = no_default) -> Tuple[Any, Iterator[Any]]:
    iterator = iter(iterable)

    result = next(iterator, marker)

    if is_marker(result):
        if is_no_default(default):
            raise ValueError(PEEK_ON_EMPTY)

        return (default, iterator)

    return (result, prepend(result, iterator))


def has_next(iterable: Iterable[T]) -> Tuple[bool, Iterator[T]]:
    result, iterator = peek(iterable, marker)

    return (is_not_marker(result), iterator)


def is_empty(iterable: Iterable[T]) -> Tuple[bool, Iterator[T]]:
    result, iterator = peek(iterable, marker)

    return (is_marker(result), iterator)


def next_of(iterator: Iterator[T]) -> Nullary[T]:
    def call() -> T:
        return next(iterator)

    return call


def next_of_iterable(iterable: Iterable[T]) -> Nullary[T]:
    return next_of(iter(iterable))


def combine(*iterables: Iterable[T]) -> Iterator[T]:
    pending = len(iterables)
    nexts = cycle(map(next_of_iterable, iterables))

    while pending:
        try:
            for next in nexts:
                yield next()

        except StopIteration:
            pending -= 1
            nexts = cycle(take(pending, nexts))


@overload
def distribute_unsafe(count: Literal[0], iterable: Iterable[T]) -> EmptyTuple:
    ...


@overload
def distribute_unsafe(count: Literal[1], iterable: Iterable[T]) -> Tuple1[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: Literal[2], iterable: Iterable[T]) -> Tuple2[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: Literal[3], iterable: Iterable[T]) -> Tuple3[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: Literal[4], iterable: Iterable[T]) -> Tuple4[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: Literal[5], iterable: Iterable[T]) -> Tuple5[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: Literal[6], iterable: Iterable[T]) -> Tuple6[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: Literal[7], iterable: Iterable[T]) -> Tuple7[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: Literal[8], iterable: Iterable[T]) -> Tuple8[Iterator[T]]:
    ...


@overload
def distribute_unsafe(count: int, iterable: Iterable[T]) -> DynamicTuple[Iterator[T]]:
    ...


def distribute_unsafe(count: int, iterable: Iterable[T]) -> DynamicTuple[Iterator[T]]:
    iterators = copy_unsafe(iterable, count)

    return tuple(step_by(count, drop(index, iterator)) for index, iterator in enumerate(iterators))


distribute_infinite = distribute_unsafe


@overload
def distribute(count: Literal[0], iterable: Iterable[T]) -> EmptyTuple:
    ...


@overload
def distribute(count: Literal[1], iterable: Iterable[T]) -> Tuple1[Iterator[T]]:
    ...


@overload
def distribute(count: Literal[2], iterable: Iterable[T]) -> Tuple2[Iterator[T]]:
    ...


@overload
def distribute(count: Literal[3], iterable: Iterable[T]) -> Tuple3[Iterator[T]]:
    ...


@overload
def distribute(count: Literal[4], iterable: Iterable[T]) -> Tuple4[Iterator[T]]:
    ...


@overload
def distribute(count: Literal[5], iterable: Iterable[T]) -> Tuple5[Iterator[T]]:
    ...


@overload
def distribute(count: Literal[6], iterable: Iterable[T]) -> Tuple6[Iterator[T]]:
    ...


@overload
def distribute(count: Literal[7], iterable: Iterable[T]) -> Tuple7[Iterator[T]]:
    ...


@overload
def distribute(count: Literal[8], iterable: Iterable[T]) -> Tuple8[Iterator[T]]:
    ...


@overload
def distribute(count: int, iterable: Iterable[T]) -> DynamicTuple[Iterator[T]]:
    ...


def distribute(count: int, iterable: Iterable[T]) -> DynamicTuple[Iterator[T]]:
    iterators = copy(iterable, count)

    return tuple(step_by(count, drop(index, iterator)) for index, iterator in enumerate(iterators))


def divide(count: int, iterable: Iterable[T]) -> Iterator[Iterator[T]]:
    array = list(iterable)

    size, last = divmod(len(array), count)

    stop = 0

    for index in range(count):
        start = stop
        stop += size

        if index < last:
            stop += 1

        yield iter(array[start:stop])


def intersperse(value: T, iterable: Iterable[T]) -> Iterator[T]:
    return rest(interleave(repeat(value), iterable))


def intersperse_with(function: Nullary[T], iterable: Iterable[T]) -> Iterator[T]:
    return rest(interleave(repeat_with(function), iterable))


def interleave(*iterables: Iterable[T]) -> Iterator[T]:
    return flatten(zip(*iterables))


def interleave_longest(*iterables: Iterable[T]) -> Iterator[T]:
    iterator = flatten(zip_longest(*iterables, fill=marker))
    return (item for item in iterator if is_not_marker(item))


def position_all(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> Iterator[int]:
    if predicate is None:
        for index, item in enumerate(iterable):
            if item:
                yield index

    else:
        for index, item in enumerate(iterable):
            if predicate(item):
                yield index


POSITION_NO_MATCH = "position() has not found any matches"


@overload
def position(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> int:
    ...


@overload
def position(predicate: Optional[Predicate[T]], iterable: Iterable[T], default: U) -> Union[int, U]:
    ...


def position(
    predicate: Optional[Predicate[T]], iterable: Iterable[T], default: Any = no_default
) -> Any:
    index = next(position_all(predicate, iterable), None)

    if index is None:
        if is_no_default(default):
            raise ValueError(POSITION_NO_MATCH)

        return default

    return index


def find_all(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> Iterator[T]:
    return filter(predicate, iterable)


FIND_NO_MATCH = "find() has not found any matches"
FIND_ON_EMPTY = "find() called on an empty iterable"


@overload
def find(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> T:
    ...


@overload
def find(predicate: Optional[Predicate[T]], iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def find(
    predicate: Optional[Predicate[Any]], iterable: Iterable[Any], default: Any = no_default
) -> Any:
    item = marker

    if predicate is None:
        for item in iterable:
            if item:
                return item

    else:
        for item in iterable:
            if predicate(item):
                return item

    if is_no_default(default):
        raise ValueError(FIND_ON_EMPTY if is_marker(item) else FIND_NO_MATCH)

    return default


FIND_OR_FIRST_ON_EMPTY = "find_or_first() called on an empty iterable"


@overload
def find_or_first(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> T:
    ...


@overload
def find_or_first(
    predicate: Optional[Predicate[T]], iterable: Iterable[T], default: U
) -> Union[T, U]:
    ...


def find_or_first(
    predicate: Optional[Predicate[Any]], iterable: Iterable[Any], default: Any = no_default
) -> Any:
    iterator = iter(iterable)

    first = next(iterator, marker)

    if is_marker(first):
        if is_no_default(default):
            raise ValueError(FIND_OR_FIRST_ON_EMPTY)

        first = default

    iterator = prepend(first, iterator)

    if predicate is None:
        for item in iterator:
            if item:
                return item

    else:
        for item in iterator:
            if predicate(item):
                return item

    return first


FIND_OR_LAST_ON_EMPTY = "find_or_last() called on an empty iterable"


@overload
def find_or_last(predicate: Optional[Predicate[T]], iterable: Iterable[T]) -> T:
    ...


@overload
def find_or_last(
    predicate: Optional[Predicate[T]], iterable: Iterable[T], default: U
) -> Union[T, U]:
    ...


def find_or_last(
    predicate: Optional[Predicate[Any]], iterable: Iterable[Any], default: Any = no_default
) -> Any:
    item = marker

    if predicate is None:
        for item in iterable:
            if item:
                return item

    else:
        for item in iterable:
            if predicate(item):
                return item

    if is_marker(item):
        if is_no_default(default):
            raise ValueError(FIND_OR_LAST_ON_EMPTY)

        return default

    return item


MIN_MAX_ON_EMPTY = "min_max() called on an empty iterable"


@overload
def min_max(iterable: Iterable[ST], *, key: None = ...) -> Pair[ST]:
    ...


@overload
def min_max(iterable: Iterable[T], *, key: Unary[T, ST]) -> Pair[T]:
    ...


@overload
def min_max(iterable: Iterable[ST], *, key: None = ..., default: U) -> Union[Pair[ST], U]:
    ...


@overload
def min_max(iterable: Iterable[T], *, key: Unary[T, ST], default: U) -> Union[Pair[T], U]:
    ...


def min_max(
    iterable: Iterable[Any],
    *,
    key: Optional[Unary[Any, Any]] = None,
    default: Any = no_default,
) -> Any:
    iterator = iter(iterable)

    result = next(iterator, marker)

    if is_marker(result):
        if is_no_default(default):
            raise ValueError(MIN_MAX_ON_EMPTY)

        return default

    return min_max_simple(iterator, result) if key is None else min_max_by(iterator, result, key)


def min_max_simple(iterable: Iterable[ST], value: ST) -> Pair[ST]:
    low = high = value

    for item in iterable:
        if item < low:
            low = item

        if high < item:
            high = item

    return (low, high)


def min_max_by(iterable: Iterable[T], value: T, key: Unary[T, ST]) -> Pair[T]:
    low = high = value
    low_key = high_key = key(value)

    for item in iterable:
        item_key = key(item)

        if item_key < low_key:
            low_key = item_key
            low = item

        if high_key < item_key:
            high_key = item_key
            high = item

    return (low, high)


def filter_except(
    validate: Validate[T], iterable: Iterable[T], *errors: AnyErrorType
) -> Iterator[T]:
    for item in iterable:
        try:
            validate(item)

        except errors:
            pass

        else:
            yield item


def map_except(function: Unary[T, U], iterable: Iterable[T], *errors: AnyErrorType) -> Iterator[U]:
    for item in iterable:
        try:
            yield function(item)

        except errors:
            pass


def iter_except(function: Nullary[T], *errors: AnyErrorType) -> Iterator[T]:
    try:
        while True:
            yield function()

    except errors:
        pass


LAST_WITH_TAIL_ON_EMPTY = "last_with_tail() called on an empty iterable"


@overload
def last_with_tail(iterable: Iterable[T]) -> T:
    ...


@overload
def last_with_tail(iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def last_with_tail(iterable: Iterable[Any], default: Any = no_default) -> Any:
    try:
        iterator = reversed(iterable)  # type: ignore

    except TypeError:
        iterator = tail(1, iterable)

    result = next(iterator, marker)

    if is_marker(result):
        if is_no_default(default):
            raise ValueError(LAST_WITH_TAIL_ON_EMPTY)

        return default

    return result


def tail(size: int, iterable: Iterable[T]) -> Iterator[T]:
    return iter(deque(iterable, size))


STRICT_TRUE = True
STRICT_FALSE = False
REVERSE_TRUE = True
REVERSE_FALSE = False


COMPARE: Dict[
    Pair[bool],
    Union[Compare[LenientOrdered, LenientOrdered], Compare[StrictOrdered, StrictOrdered]],
] = {
    (STRICT_FALSE, REVERSE_FALSE): less_or_equal,
    (STRICT_FALSE, REVERSE_TRUE): greater_or_equal,
    (STRICT_TRUE, REVERSE_FALSE): less,
    (STRICT_TRUE, REVERSE_TRUE): greater,
}


@overload
def is_sorted(
    iterable: Iterable[LT],
    key: None = ...,
    *,
    strict: Literal[False] = ...,
    reverse: bool = ...,
) -> bool:
    ...


@overload
def is_sorted(
    iterable: Iterable[ST],
    key: None = ...,
    *,
    strict: Literal[True],
    reverse: bool = ...,
) -> bool:
    ...


@overload
def is_sorted(
    iterable: Iterable[T],
    key: Unary[T, LT],
    *,
    strict: Literal[False] = ...,
    reverse: bool = ...,
) -> bool:
    ...


@overload
def is_sorted(
    iterable: Iterable[T],
    key: Unary[T, ST],
    *,
    strict: Literal[True],
    reverse: bool = ...,
) -> bool:
    ...


def is_sorted(
    iterable: Iterable[Any],
    key: Optional[Unary[Any, Any]] = None,
    *,
    strict: bool = False,
    reverse: bool = False,
) -> bool:
    return (
        is_sorted_simple(iterable, strict=strict, reverse=reverse)
        if key is None
        else is_sorted_by(iterable, key, strict=strict, reverse=reverse)
    )


def is_sorted_simple(
    iterable: Iterable[Any], *, strict: bool = False, reverse: bool = False
) -> bool:
    compare = COMPARE[strict, reverse]
    return all(map(unpack_binary(compare), pairs_windows(iterable)))


def is_sorted_by(
    iterable: Iterable[Any], key: Unary[Any, Any], *, strict: bool = False, reverse: bool = False
) -> bool:
    return is_sorted_simple(map(key, iterable), strict=strict, reverse=reverse)


@overload
def sort(iterable: Iterable[ST], *, key: None = ..., reverse: bool = ...) -> Iterator[ST]:
    ...


@overload
def sort(iterable: Iterable[T], *, key: Unary[T, ST], reverse: bool = ...) -> Iterator[T]:
    ...


def sort(
    iterable: Iterable[Any],
    *,
    key: Optional[Unary[Any, Any]] = None,
    reverse: bool = False,
) -> Iterator[Any]:
    return iter(sorted(iterable, key=key, reverse=reverse))


def reverse(iterable: Iterable[T]) -> Iterator[T]:
    try:
        return reversed(iterable)  # type: ignore

    except TypeError:
        return reversed(list(iterable))


# def circular_list_windows(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
#     ...


# def circular_tuple_windows(size: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
#     ...


# def circular_set_windows(size: int, iterable: Iterable[T]) -> Iterator[Set[T]]:
#     ...


def list_windows(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    iterator = iter(iterable)

    window = deque(take(size, iterator), size)

    # list(window) to copy, since windows are mutable

    if len(window) == size:
        yield list(window)

    window_append = window.append

    for item in iterator:
        window_append(item)
        yield list(window)


@overload
def tuple_windows(size: Literal[0], iterable: Iterable[T]) -> Iterator[EmptyTuple]:
    ...


@overload
def tuple_windows(size: Literal[1], iterable: Iterable[T]) -> Iterator[Tuple1[T]]:
    ...


@overload
def tuple_windows(size: Literal[2], iterable: Iterable[T]) -> Iterator[Tuple2[T]]:
    ...


@overload
def tuple_windows(size: Literal[3], iterable: Iterable[T]) -> Iterator[Tuple3[T]]:
    ...


@overload
def tuple_windows(size: Literal[4], iterable: Iterable[T]) -> Iterator[Tuple4[T]]:
    ...


@overload
def tuple_windows(size: Literal[5], iterable: Iterable[T]) -> Iterator[Tuple5[T]]:
    ...


@overload
def tuple_windows(size: Literal[6], iterable: Iterable[T]) -> Iterator[Tuple6[T]]:
    ...


@overload
def tuple_windows(size: Literal[7], iterable: Iterable[T]) -> Iterator[Tuple7[T]]:
    ...


@overload
def tuple_windows(size: Literal[8], iterable: Iterable[T]) -> Iterator[Tuple8[T]]:
    ...


@overload
def tuple_windows(size: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
    ...


def tuple_windows(size: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
    iterator = iter(iterable)

    window = deque(take(size, iterator), size)

    if len(window) == size:
        yield tuple(window)

    window_append = window.append

    for item in iterator:
        window_append(item)
        yield tuple(window)


def pairs_windows(iterable: Iterable[T]) -> Iterator[Pair[T]]:
    return tuple_windows(2, iterable)


def iter_windows(size: int, iterable: Iterable[T]) -> Iterator[Iterator[T]]:
    for window in list_windows(size, iterable):
        yield iter(window)


def set_windows(size: int, iterable: Iterable[Q]) -> Iterator[Set[Q]]:
    iterator = iter(iterable)

    window = deque(take(size, iterator), size)

    if len(window) == size:
        yield set(window)

    window_append = window.append

    for item in iterator:
        window_append(item)
        yield set(window)


def inspect(function: Inspect[T], iterable: Iterable[T]) -> Iterator[T]:
    for item in iterable:
        function(item)
        yield item


def duplicates_fast_simple(iterable: Iterable[Q]) -> Iterator[Q]:
    seen: Set[Q] = set()
    add_to_seen = seen.add

    for item in iterable:
        if item in seen:
            yield item

        else:
            add_to_seen(item)


def duplicates_fast_by(iterable: Iterable[T], key: Unary[T, Q]) -> Iterator[T]:
    seen_values: Set[Q] = set()
    add_to_seen_values = seen_values.add

    for item in iterable:
        value = key(item)

        if value in seen_values:
            yield item

        else:
            add_to_seen_values(value)


@overload
def duplicates_fast(iterable: Iterable[Q], key: None = ...) -> Iterator[Q]:
    ...


@overload
def duplicates_fast(iterable: Iterable[T], key: Unary[T, Q]) -> Iterator[T]:
    ...


def duplicates_fast(
    iterable: Iterable[Any], key: Optional[Unary[Any, Any]] = None
) -> Iterator[Any]:
    return duplicates_fast_simple(iterable) if key is None else duplicates_fast_by(iterable, key)


def duplicates_simple(iterable: Iterable[T]) -> Iterator[T]:
    seen_set: Set[T] = set()
    add_to_seen_set = seen_set.add
    seen_list: List[T] = []
    add_to_seen_list = seen_list.append

    for item in iterable:
        try:
            if item in seen_set:
                yield item

            else:
                add_to_seen_set(item)

        except TypeError:
            if item in seen_list:
                yield item

            else:
                add_to_seen_list(item)


def duplicates_by(iterable: Iterable[T], key: Unary[T, U]) -> Iterator[T]:
    seen_values_set: Set[U] = set()
    add_to_seen_values_set = seen_values_set.add
    seen_values_list: List[U] = []
    add_to_seen_values_list = seen_values_list.append

    for item in iterable:
        value = key(item)

        try:
            if value in seen_values_set:
                yield item

            else:
                add_to_seen_values_set(value)

        except TypeError:
            if value in seen_values_list:
                yield item

            else:
                add_to_seen_values_list(value)


def duplicates(iterable: Iterable[T], key: Optional[Unary[T, U]] = None) -> Iterator[T]:
    return duplicates_simple(iterable) if key is None else duplicates_by(iterable, key)


def unique_fast_simple(iterable: Iterable[Q]) -> Iterator[Q]:
    seen: Set[Q] = set()
    add_to_seen = seen.add

    for item in iterable:
        if item not in seen:
            add_to_seen(item)

            yield item


def unique_fast_by(iterable: Iterable[T], key: Unary[T, Q]) -> Iterator[T]:
    seen_values: Set[Q] = set()
    add_to_seen_values = seen_values.add

    for item in iterable:
        value = key(item)

        if value not in seen_values:
            add_to_seen_values(value)

            yield item


@overload
def unique_fast(iterable: Iterable[Q], key: None = ...) -> Iterator[Q]:
    ...


@overload
def unique_fast(iterable: Iterable[T], key: Unary[T, Q]) -> Iterator[T]:
    ...


def unique_fast(iterable: Iterable[Any], key: Optional[Unary[Any, Any]] = None) -> Iterator[Any]:
    return unique_fast_simple(iterable) if key is None else unique_fast_by(iterable, key)


def unique_simple(iterable: Iterable[T]) -> Iterator[T]:
    seen_set: Set[T] = set()
    add_to_seen_set = seen_set.add
    seen_list: List[T] = []
    add_to_seen_list = seen_list.append

    for item in iterable:
        try:
            if item not in seen_set:
                add_to_seen_set(item)

                yield item

        except TypeError:
            if item not in seen_list:
                add_to_seen_list(item)

                yield item


def unique_by(iterable: Iterable[T], key: Unary[T, U]) -> Iterator[T]:
    seen_values_set: Set[U] = set()
    add_to_seen_values_set = seen_values_set.add
    seen_values_list: List[U] = []
    add_to_seen_values_list = seen_values_list.append

    for item in iterable:
        value = key(item)

        try:
            if value not in seen_values_set:
                add_to_seen_values_set(value)

                yield item

        except TypeError:
            if value not in seen_values_list:
                add_to_seen_values_list(value)

                yield item


def unique(iterable: Iterable[T], key: Optional[Unary[T, U]] = None) -> Iterator[T]:
    return unique_simple(iterable) if key is None else unique_by(iterable, key)


@overload
def zip() -> Iterator[Never]:
    ...


@overload
def zip(__iterable_a: Iterable[A]) -> Iterator[Tuple[A]]:
    ...


@overload
def zip(__iterable_a: Iterable[A], __iterable_b: Iterable[B]) -> Iterator[Tuple[A, B]]:
    ...


@overload
def zip(
    __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
) -> Iterator[Tuple[A, B, C]]:
    ...


@overload
def zip(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
) -> Iterator[Tuple[A, B, C, D]]:
    ...


@overload
def zip(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
) -> Iterator[Tuple[A, B, C, D, E]]:
    ...


@overload
def zip(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
) -> Iterator[Tuple[A, B, C, D, E, F]]:
    ...


@overload
def zip(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
) -> Iterator[Tuple[A, B, C, D, E, F, G]]:
    ...


@overload
def zip(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
    __iterable_h: Iterable[H],
) -> Iterator[Tuple[A, B, C, D, E, F, G, H]]:
    ...


@overload
def zip(
    __iterable_a: Iterable[Any],
    __iterable_b: Iterable[Any],
    __iterable_c: Iterable[Any],
    __iterable_d: Iterable[Any],
    __iterable_e: Iterable[Any],
    __iterable_f: Iterable[Any],
    __iterable_g: Iterable[Any],
    __iterable_h: Iterable[Any],
    __iterable_n: Iterable[Any],
    *iterables: Iterable[Any],
) -> Iterator[DynamicTuple[Any]]:
    ...


def zip(*iterables: Iterable[Any]) -> Iterator[DynamicTuple[Any]]:
    return standard_zip(*iterables)


@overload
def zip_equal() -> Iterator[Never]:
    ...


@overload
def zip_equal(__iterable_a: Iterable[A]) -> Iterator[Tuple[A]]:
    ...


@overload
def zip_equal(__iterable_a: Iterable[A], __iterable_b: Iterable[B]) -> Iterator[Tuple[A, B]]:
    ...


@overload
def zip_equal(
    __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
) -> Iterator[Tuple[A, B, C]]:
    ...


@overload
def zip_equal(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
) -> Iterator[Tuple[A, B, C, D]]:
    ...


@overload
def zip_equal(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
) -> Iterator[Tuple[A, B, C, D, E]]:
    ...


@overload
def zip_equal(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
) -> Iterator[Tuple[A, B, C, D, E, F]]:
    ...


@overload
def zip_equal(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
) -> Iterator[Tuple[A, B, C, D, E, F, G]]:
    ...


@overload
def zip_equal(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
    __iterable_h: Iterable[H],
) -> Iterator[Tuple[A, B, C, D, E, F, G, H]]:
    ...


@overload
def zip_equal(
    __iterable_a: Iterable[Any],
    __iterable_b: Iterable[Any],
    __iterable_c: Iterable[Any],
    __iterable_d: Iterable[Any],
    __iterable_e: Iterable[Any],
    __iterable_f: Iterable[Any],
    __iterable_g: Iterable[Any],
    __iterable_h: Iterable[Any],
    __iterable_n: Iterable[Any],
    *iterables: Iterable[Any],
) -> Iterator[DynamicTuple[Any]]:
    ...


def zip_equal(*iterables: Iterable[Any]) -> Iterator[DynamicTuple[Any]]:
    if PYTHON_3_10:
        return standard_zip(*iterables, strict=True)  # type: ignore

    return zip_equal_simple(*iterables)


SINGULAR = " "
PLURAL = "s 1-"

SHORTER = "zip_equal() argument {short} is shorter than argument{plural}{index}"
LONGER = "zip_equal() argument {long} is longer than argument{plural}{index}"


def format_shorter(index: int) -> str:
    return SHORTER.format(
        short=index + 1,
        index=index,
        plural=(PLURAL if index - 1 else SINGULAR),
    )


def format_longer(index: int) -> str:
    return LONGER.format(
        long=index + 1,
        index=index,
        plural=(PLURAL if index - 1 else SINGULAR),
    )


def zip_equal_simple(*iterables: Iterable[Any]) -> Iterator[DynamicTuple[Any]]:
    if not iterables:
        return  # early return

    for item in zip_longest(*iterables, fill=marker):  # check for length
        head, *tail = item

        if is_marker(head):  # argument longer than previous arguments
            for index, value in enumerate(tail, 1):
                if value is not marker:
                    raise ValueError(format_longer(index))

        else:  # argument shorter than previous ones
            for index, value in enumerate(tail, 1):
                if is_marker(value):
                    raise ValueError(format_shorter(index))

        yield item  # simply yield if everything is alright


@overload
def zip_longest() -> Iterator[Never]:
    ...


@overload
def zip_longest(__iterable_a: Iterable[A]) -> Iterator[Tuple[A]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A], __iterable_b: Iterable[B]
) -> Iterator[Tuple[Optional[A], Optional[B]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
) -> Iterator[Tuple[Optional[A], Optional[B], Optional[C]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
) -> Iterator[Tuple[Optional[A], Optional[B], Optional[C], Optional[D]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
) -> Iterator[Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
) -> Iterator[Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E], Optional[F]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
) -> Iterator[
    Tuple[
        Optional[A],
        Optional[B],
        Optional[C],
        Optional[D],
        Optional[E],
        Optional[F],
        Optional[G],
    ]
]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
    __iterable_h: Iterable[H],
) -> Iterator[
    Tuple[
        Optional[A],
        Optional[B],
        Optional[C],
        Optional[D],
        Optional[E],
        Optional[F],
        Optional[G],
        Optional[H],
    ]
]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[Any],
    __iterable_b: Iterable[Any],
    __iterable_c: Iterable[Any],
    __iterable_d: Iterable[Any],
    __iterable_e: Iterable[Any],
    __iterable_f: Iterable[Any],
    __iterable_g: Iterable[Any],
    __iterable_h: Iterable[Any],
    __iterable_n: Iterable[Any],
    *iterables: Iterable[Any],
) -> Iterator[DynamicTuple[Optional[Any]]]:
    ...


@overload
def zip_longest(*, fill: T) -> Iterator[Never]:
    ...


@overload
def zip_longest(__iterable_a: Iterable[A], *, fill: T) -> Iterator[Tuple[A]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A], __iterable_b: Iterable[B], *, fill: T
) -> Iterator[Tuple[Union[A, T], Union[B, T]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    *,
    fill: T,
) -> Iterator[Tuple[Union[A, T], Union[B, T], Union[C, T]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    *,
    fill: T,
) -> Iterator[Tuple[Union[A, T], Union[B, T], Union[C, T], Union[D, T]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    *,
    fill: T,
) -> Iterator[Tuple[Union[A, T], Union[B, T], Union[C, T], Union[D, T], Union[E, T]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    *,
    fill: T,
) -> Iterator[Tuple[Union[A, T], Union[B, T], Union[C, T], Union[D, T], Union[E, T], Union[F, T]]]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
    *,
    fill: T,
) -> Iterator[
    Tuple[
        Union[A, T],
        Union[B, T],
        Union[C, T],
        Union[D, T],
        Union[E, T],
        Union[F, T],
        Union[G, T],
    ]
]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
    __iterable_h: Iterable[H],
    *,
    fill: T,
) -> Iterator[
    Tuple[
        Union[A, T],
        Union[B, T],
        Union[C, T],
        Union[D, T],
        Union[E, T],
        Union[F, T],
        Union[G, T],
        Union[H, T],
    ]
]:
    ...


@overload
def zip_longest(
    __iterable_a: Iterable[Any],
    __iterable_b: Iterable[Any],
    __iterable_c: Iterable[Any],
    __iterable_d: Iterable[Any],
    __iterable_e: Iterable[Any],
    __iterable_f: Iterable[Any],
    __iterable_g: Iterable[Any],
    __iterable_h: Iterable[Any],
    __iterable_n: Iterable[Any],
    *iterables: Iterable[Any],
    fill: T,
) -> Iterator[DynamicTuple[Union[Any, T]]]:
    ...


def zip_longest(
    *iterables: Iterable[Any], fill: Optional[Any] = None
) -> Iterator[DynamicTuple[Any]]:
    return standard_zip_longest(*iterables, fillvalue=fill)


def transpose(iterable: Iterable[Iterable[T]]) -> Iterator[DynamicTuple[T]]:
    return zip_equal(*iterable)


def power_set(iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
    array = list(iterable)

    return flatten(combinations(array, count) for count in inclusive(range(len(array))))


def inclusive(non_inclusive: range) -> range:
    step = non_inclusive.step

    return range(non_inclusive.start, non_inclusive.stop + sign(step), step)


def sign(value: int) -> int:
    return int(copy_sign(1, value))


@overload
def cartesian_product() -> Iterator[EmptyTuple]:
    ...


@overload
def cartesian_product(__iterable_a: Iterable[A]) -> Iterator[Tuple[A]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[A], __iterable_b: Iterable[B]
) -> Iterator[Tuple[A, B]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
) -> Iterator[Tuple[A, B, C]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
) -> Iterator[Tuple[A, B, C, D]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
) -> Iterator[Tuple[A, B, C, D, E]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
) -> Iterator[Tuple[A, B, C, D, E, F]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
) -> Iterator[Tuple[A, B, C, D, E, F, G]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[A],
    __iterable_b: Iterable[B],
    __iterable_c: Iterable[C],
    __iterable_d: Iterable[D],
    __iterable_e: Iterable[E],
    __iterable_f: Iterable[F],
    __iterable_g: Iterable[G],
    __iterable_h: Iterable[H],
) -> Iterator[Tuple[A, B, C, D, E, F, G, H]]:
    ...


@overload
def cartesian_product(
    __iterable_a: Iterable[Any],
    __iterable_b: Iterable[Any],
    __iterable_c: Iterable[Any],
    __iterable_d: Iterable[Any],
    __iterable_e: Iterable[Any],
    __iterable_f: Iterable[Any],
    __iterable_g: Iterable[Any],
    __iterable_h: Iterable[Any],
    __iterable_n: Iterable[Any],
    *iterables: Iterable[Any],
) -> Iterator[DynamicTuple[Any]]:
    ...


def cartesian_product(*iterables: Iterable[Any]) -> Iterator[DynamicTuple[Any]]:
    return standard_product(*iterables)


@overload
def cartesian_power(power: Literal[0], iterable: Iterable[T]) -> Iterator[EmptyTuple]:
    ...


@overload
def cartesian_power(power: Literal[1], iterable: Iterable[T]) -> Iterator[Tuple1[T]]:
    ...


@overload
def cartesian_power(power: Literal[2], iterable: Iterable[T]) -> Iterator[Tuple2[T]]:
    ...


@overload
def cartesian_power(power: Literal[3], iterable: Iterable[T]) -> Iterator[Tuple3[T]]:
    ...


@overload
def cartesian_power(power: Literal[4], iterable: Iterable[T]) -> Iterator[Tuple4[T]]:
    ...


@overload
def cartesian_power(power: Literal[5], iterable: Iterable[T]) -> Iterator[Tuple5[T]]:
    ...


@overload
def cartesian_power(power: Literal[6], iterable: Iterable[T]) -> Iterator[Tuple6[T]]:
    ...


@overload
def cartesian_power(power: Literal[7], iterable: Iterable[T]) -> Iterator[Tuple7[T]]:
    ...


@overload
def cartesian_power(power: Literal[8], iterable: Iterable[T]) -> Iterator[Tuple8[T]]:
    ...


@overload
def cartesian_power(power: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
    ...


def cartesian_power(power: int, iterable: Iterable[T]) -> Iterator[DynamicTuple[T]]:
    return standard_product(iterable, repeat=power)
