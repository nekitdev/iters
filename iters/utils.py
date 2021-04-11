from builtins import zip as std_zip
from collections import deque
from functools import reduce
from itertools import (
    chain,
    compress,
    count,
    cycle,
    dropwhile as drop_while,
    filterfalse as filter_false,
    islice as iter_slice,
    repeat,
    starmap as star_map,
    takewhile as take_while,
    tee as copy,
    zip_longest as std_zip_longest,
)
from operator import add, attrgetter as get_attr_factory, mul
from sys import hexversion as hex_version
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    List,
    Optional,
    Reversible,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
    overload,
)

from iters.types import Marker, MarkerOr, marker

__all__ = (
    "append",
    "at",
    "at_or_last",
    "chain",
    "chain_from_iterable",
    "collapse",
    "compress",
    "copy",
    "copy_infinite",
    "copy_safe",
    "count",
    "cycle",
    "distinct",
    "drop",
    "drop_while",
    "exhaust",
    "filter_false",
    "first",
    "flatten",
    "fold",
    "get",
    "group",
    "group_longest",
    "iter_chunk",
    "iter_len",
    "iter_slice",
    "iterate",
    "last",
    "list_chunk",
    "partition",
    "partition_infinite",
    "partition_safe",
    "prepend",
    "product",
    "reduce",
    "repeat",
    "side_effect",
    "star_map",
    "step_by",
    "sum",
    "take",
    "take_while",
    "tuple_chunk",
    "with_iter",
    "zip",
    "zip_longest",
)

PY_380 = hex_version == 0x30800F0  # 3.8.0

T = TypeVar("T")
U = TypeVar("U")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


@overload
def zip(__iterable_1: Iterable[T1]) -> Iterator[Tuple[T1]]:
    ...


@overload
def zip(__iterable_1: Iterable[T1], __iterable_2: Iterable[T2]) -> Iterator[Tuple[T1, T2]]:
    ...


@overload
def zip(
    __iterable_1: Iterable[T1], __iterable_2: Iterable[T2], __iterable_3: Iterable[T3]
) -> Iterator[Tuple[T1, T2, T3]]:
    ...


@overload
def zip(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
    __iterable_4: Iterable[T4],
) -> Iterator[Tuple[T1, T2, T3, T4]]:
    ...


@overload
def zip(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
    __iterable_4: Iterable[T4],
    __iterable_5: Iterable[T5],
) -> Iterator[Tuple[T1, T2, T3, T4, T5]]:
    ...


@overload
def zip(
    __iterable_1: Iterable[Any],
    __iterable_2: Iterable[Any],
    __iterable_3: Iterable[Any],
    __iterable_4: Iterable[Any],
    __iterable_5: Iterable[Any],
    __iterable_6: Iterable[Any],
    *iterables: Iterable[Any],
) -> Iterator[Tuple[Any, ...]]:
    ...


def zip(*iterables: Iterable[Any]) -> Iterator[Tuple[Any, ...]]:
    return std_zip(*iterables)


@overload
def zip_longest(__iterable_1: Iterable[T1]) -> Iterator[Tuple[Optional[T1]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
) -> Iterator[Tuple[Optional[T1], Optional[T2]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
) -> Iterator[Tuple[Optional[T1], Optional[T2], Optional[T3]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
    __iterable_4: Iterable[T4],
) -> Iterator[Tuple[Optional[T1], Optional[T2], Optional[T3], Optional[T4]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
    __iterable_4: Iterable[T4],
    __iterable_5: Iterable[T5],
) -> Iterator[Tuple[Optional[T1], Optional[T2], Optional[T3], Optional[T4], Optional[T5]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[Any],
    __iterable_2: Iterable[Any],
    __iterable_3: Iterable[Any],
    __iterable_4: Iterable[Any],
    __iterable_5: Iterable[Any],
    __iterable_6: Iterable[Any],
    *iterables: Iterable[Any],
) -> Iterator[Tuple[Optional[Any], ...]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1], *, fill: T
) -> Iterator[Tuple[Union[T1, T]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1], __iterable_2: Iterable[T2], *, fill: T
) -> Iterator[Tuple[Union[T1, T], Union[T2, T]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
    *,
    fill: T,
) -> Iterator[Tuple[Union[T1, T], Union[T2, T], Union[T3, T]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
    __iterable_4: Iterable[T4],
    *,
    fill: T,
) -> Iterator[Tuple[Union[T1, T], Union[T2, T], Union[T3, T], Union[T4, T]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[T1],
    __iterable_2: Iterable[T2],
    __iterable_3: Iterable[T3],
    __iterable_4: Iterable[T4],
    __iterable_5: Iterable[T5],
    *,
    fill: T,
) -> Iterator[Tuple[Union[T1, T], Union[T2, T], Union[T3, T], Union[T4, T], Union[T5, T]]]:
    ...


@overload
def zip_longest(
    __iterable_1: Iterable[Any],
    __iterable_2: Iterable[Any],
    __iterable_3: Iterable[Any],
    __iterable_4: Iterable[Any],
    __iterable_5: Iterable[Any],
    __iterable_6: Iterable[Any],
    *iterables: Iterable[Any],
    fill: T,
) -> Iterator[Tuple[Union[Any, T], ...]]:
    ...


@no_type_check
def zip_longest(
    *iterables: Iterable[Any], fill: Optional[T] = None
) -> Iterator[Tuple[Union[Any, Optional[T]], ...]]:
    return std_zip_longest(*iterables, fillvalue=fill)


chain_from_iterable = chain.from_iterable


def exhaust(iterator: Iterator[T], amount: Optional[int] = None) -> None:
    if amount is None:
        deque(iterator, maxlen=0)

    else:
        exhaust(take(iterator, amount))


@overload
def first(iterable: Iterable[T], default: Marker = marker) -> T:
    ...


@overload
def first(iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def first(iterable: Iterable[T], default: MarkerOr[U] = marker) -> Union[T, U]:
    try:
        return next(iter(iterable))

    except StopIteration as error:
        if default is marker:
            raise ValueError("first() called on an empty iterable.") from error

        return cast(T, default)


@overload
def last(iterable: Iterable[T], default: Marker = marker) -> T:
    ...


@overload
def last(iterable: Iterable[T], default: U) -> Union[T, U]:
    ...


def last(iterable: Iterable[T], default: MarkerOr[T] = marker) -> T:
    try:
        if isinstance(iterable, Sequence):
            return cast(Sequence[T], iterable)[-1]

        elif isinstance(iterable, Reversible) and not PY_380:
            return next(reversed(cast(Reversible[T], iterable)))

        else:
            return deque(iterable, maxlen=1)[-1]

    except (IndexError, ValueError, StopIteration) as error:
        if default is marker:
            raise ValueError("last() called on an empty iterable.") from error

        return cast(T, default)


def fold(iterable: Iterable[T], function: Callable[[U, T], U], initial: U) -> U:
    return reduce(function, iterable, initial)


DOT = "."
DUNDER = "__"


def get(iterable: Iterable[T], **attrs: U) -> Iterator[T]:
    names = tuple(attr.replace(DUNDER, DOT) for attr in attrs.keys())
    expected = tuple(attrs.values())

    # special case for one attribute -> we recieve pure values instead of tuples

    if len(expected) == 1:
        expected = first(expected)  # type: ignore

    get_attrs = get_attr_factory(*names)

    def predicate(item: T) -> bool:
        return get_attrs(item) == expected

    return filter(predicate, iterable)


@overload
def at(iterable: Iterable[T], n: int, default: Marker = marker) -> T:
    ...


@overload
def at(iterable: Iterable[T], n: int, default: U) -> Union[T, U]:
    ...


def at(iterable: Iterable[T], n: int, default: MarkerOr[U] = marker) -> Union[T, U]:
    try:
        return next(iter_slice(iterable, n, None))

    except StopIteration as error:
        if default is marker:
            raise ValueError("at() called with n larger than iterable length.") from error

        return cast(U, default)


@overload
def at_or_last(iterable: Iterable[T], n: int, default: Marker = marker) -> T:
    ...


@overload
def at_or_last(iterable: Iterable[T], n: int, default: U) -> Union[T, U]:
    ...


def at_or_last(
    iterable: Iterable[T], n: int, default: MarkerOr[U] = marker
) -> Union[T, U]:
    return last(iter_slice(iterable, n + 1), default=default)


def copy_safe(iterable: Iterable[T], n: int = 2) -> Tuple[Iterator[T], ...]:
    collected = tuple(iterable)

    return tuple(iter(collected) for _ in range(n))


copy_infinite = copy


def drop(iterable: Iterable[T], n: int) -> Iterator[T]:
    return iter_slice(iterable, n, None)


def take(iterable: Iterable[T], n: int) -> Iterator[T]:
    return iter_slice(iterable, n)


def step_by(iterable: Iterable[T], step: int) -> Iterator[T]:
    return iter_slice(iterable, None, None, step)


def group(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    iterators = (iter(iterable),) * n

    return zip(*iterators)


@overload
def group_longest(iterable: Iterable[T], n: int) -> Iterator[Tuple[Optional[T], ...]]:
    ...


@overload
def group_longest(iterable: Iterable[T], n: int, fill: T) -> Iterator[Tuple[T, ...]]:
    ...


def group_longest(
    iterable: Iterable[T], n: int, fill: Optional[T] = None
) -> Iterator[Tuple[Optional[T], ...]]:
    iterators = (iter(iterable),) * n

    return zip_longest(*iterators, fill=fill)


def flatten(iterable: Iterable[Iterable[T]]) -> Iterator[T]:
    return chain_from_iterable(iterable)


def partition(
    iterable: Iterable[T], predicate: Callable[[T], Any] = bool
) -> Tuple[Iterator[T], Iterator[T]]:
    for_true, for_false = copy(iterable)

    return filter(predicate, for_true), filter_false(predicate, for_false)


partition_infinite = partition


def partition_safe(
    iterable: Iterable[T], predicate: Callable[[T], Any] = bool
) -> Tuple[Iterator[T], Iterator[T]]:
    for_true, for_false = copy_safe(iterable)

    return filter(predicate, for_true), filter_false(predicate, for_false)


def prepend(iterable: Iterable[T], item: T) -> Iterator[T]:
    return chain((item,), iterable)


def append(iterable: Iterable[T], item: T) -> Iterator[T]:
    return chain(iterable, (item,))


def list_chunk(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    iterator = iter(iterable)

    while True:
        part = list(take(iterator, n))

        if not part:
            break

        yield part


def tuple_chunk(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    iterator = iter(iterable)

    while True:
        part = tuple(take(iterator, n))

        if not part:
            break

        yield part


def iter_chunk(iterable: Iterable[T], n: int) -> Iterator[Iterator[T]]:
    source = iter(iterable)

    while True:
        item = next(source, marker)

        if item is marker:
            return

        source, iterator = copy(prepend(source, cast(T, item)))

        yield take(iterator, n)

        exhaust(source, n)


def iter_len(iterable: Iterable[T]) -> int:
    counter = count()

    deque(zip(counter, iterable), maxlen=0)

    return next(counter)


def sum(iterable: Iterable[T], start: MarkerOr[T] = marker) -> T:
    if start is marker:
        return reduce(add, iterable)

    return reduce(add, iterable, cast(T, start))


def product(iterable: Iterable[T], start: MarkerOr[T] = marker) -> T:
    if start is marker:
        return reduce(mul, iterable)

    return reduce(mul, iterable, cast(T, start))


def iterate(function: Callable[[T], T], value: T) -> Iterator[T]:
    while True:
        yield value
        value = function(value)


def with_iter(context_manager: ContextManager[Iterable[T]]) -> Iterator[T]:
    with context_manager as iterable:
        yield from iterable


def collapse(
    iterable: Iterable[T],
    base_type: Optional[Type[Any]] = None,
    levels: Optional[int] = None,
) -> Iterator[T]:
    def walk(node: Union[T, Iterable[T]], level: int) -> Iterator[T]:
        if (
            ((levels is not None) and (level > levels))
            or isinstance(node, (str, bytes))
            or ((base_type is not None) and isinstance(node, base_type))
        ):
            yield cast(T, node)
            return

        try:
            tree = iter(node)  # type: ignore

        except TypeError:
            yield cast(T, node)

            return

        else:
            for child in tree:
                for item in walk(child, level + 1):
                    yield item

    yield from walk(iterable, 0)


def side_effect(
    iterable: Iterable[T],
    function: Callable[[T], None],
    before: Optional[Callable[[], None]] = None,
    after: Optional[Callable[[], None]] = None,
) -> Iterator[T]:
    try:
        if before is not None:
            before()

        for item in iterable:
            function(item)

            yield item

    finally:
        if after is not None:
            after()


def distinct(iterable: Iterable[T], key: Optional[Callable[[T], U]] = None) -> Iterator[T]:
    if key is None:
        seen_set: Set[T] = set()
        add_to_seen_set = seen_set.add
        seen_list: List[T] = []
        add_to_seen_list = seen_list.append

        for element in iterable:
            try:
                if element not in seen_set:
                    add_to_seen_set(element)

                    yield element

            except TypeError:
                if element not in seen_list:
                    add_to_seen_list(element)

                    yield element

    else:
        seen_value_set: Set[U] = set()
        add_to_seen_value_set = seen_value_set.add
        seen_value_list: List[U] = []
        add_to_seen_value_list = seen_value_list.append

        for element in iterable:
            value = key(element)

            try:
                if value not in seen_value_set:
                    add_to_seen_value_set(value)

                    yield element

            except TypeError:
                if value not in seen_value_list:
                    add_to_seen_value_list(value)

                    yield element
