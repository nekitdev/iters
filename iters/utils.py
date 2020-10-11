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
    zip_longest,
)
from operator import add, attrgetter, mul
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
)
from typing_extensions import Protocol

__all__ = (
    "Marker",
    "MarkerOr",
    "append",
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
    "marker",
    "nth",
    "nth_or_last",
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
    "zip_longest",
)

PY_380 = hex_version == 0x30800F0  # 3.8.0

T = TypeVar("T")
U = TypeVar("U")


class SupportsAdd(Protocol):
    def __add__(self, other: Any) -> Any:
        ...


Add_T = TypeVar("Add_T", bound=SupportsAdd)


class Singleton:
    INSTANCE = None

    def __new__(cls, *args, **kwargs) -> "Singleton":
        if cls.INSTANCE is None:
            cls.INSTANCE = super().__new__(cls)

        return cls.INSTANCE

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class Marker(Singleton):
    pass


marker = Marker()

MarkerOr = Union[Marker, T]

chain_from_iterable = chain.from_iterable


def exhaust(iterator: Iterator[T], amount: Optional[int] = None) -> None:
    if amount is None:
        deque(iterator, maxlen=0)

    else:
        exhaust(take(iterator, amount))


def first(iterable: Iterable[T], default: MarkerOr[T] = marker) -> T:
    try:
        return next(iter(iterable))

    except StopIteration as error:
        if default is marker:
            raise ValueError("first() called on an empty iterable.") from error

        return cast(T, default)


def last(iterable: Iterable[T], default: MarkerOr[T] = marker) -> T:
    try:
        if isinstance(iterable, Sequence):
            return iterable[-1]

        elif isinstance(iterable, Reversible) and not PY_380:
            return next(reversed(iterable))

        else:
            return deque(iterable, maxlen=1)[-1]

    except (IndexError, ValueError, StopIteration) as error:
        if default is marker:
            raise ValueError("last() called on an empty iterable.") from error

        return cast(T, default)


def fold(iterable: Iterable[T], function: Callable[[U, Union[T, U]], U], initial: U) -> U:
    return reduce(function, iterable, initial)


def get(iterable: Iterable[T], **attrs: U) -> Iterator[T]:
    names = tuple(attr.replace("__", ".") for attr in attrs.keys())
    expected = tuple(value for value in attrs.values())

    if len(expected) == 1:
        expected = expected[0]  # type: ignore

    get_attrs = attrgetter(*names)

    def predicate(item: T) -> bool:
        return get_attrs(item) == expected

    return filter(predicate, iterable)


def nth(iterable: Iterable[T], n: int, default: MarkerOr[T] = marker) -> T:
    try:
        return next(iter_slice(iterable, n, None))

    except StopIteration as error:
        if default is marker:
            raise ValueError("nth() called with n larger than iterable length.") from error

        return cast(T, default)


def nth_or_last(iterable: Iterable[T], n: int, default: MarkerOr[T] = marker) -> T:
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


def group_longest(
    iterable: Iterable[T], n: int, fillvalue: Optional[T] = None
) -> Iterator[Tuple[Optional[T], ...]]:
    iterators = (iter(iterable),) * n

    return zip_longest(iterators, fillvalue=fillvalue)


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
    iterable: Union[T, Iterable[T]],
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


def distinct(
    iterable: Iterable[T], key: Optional[Callable[[T], U]] = None
) -> Iterator[T]:
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
