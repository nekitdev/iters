from builtins import next as standard_next
from collections import Counter as counter_dict
from collections import deque
from itertools import cycle
from operator import add, mul
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    Counter,
    Deque,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Reversible,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from named import get_type_name
from orderings import LenientOrdered, Ordering, StrictOrdered
from typing_extensions import Literal, Never, ParamSpec, TypeVarTuple, Unpack

from iters.concurrent import CONCURRENT
from iters.ordered_set import OrderedSet, ordered_set
from iters.types import marker, no_default
from iters.typing import (
    AnyExceptionType,
    AnyIterable,
    AnyIterator,
    AnySelectors,
    AsyncBinary,
    AsyncDynamicCallable,
    AsyncNullary,
    AsyncPredicate,
    AsyncQuaternary,
    AsyncTernary,
    AsyncUnary,
    Binary,
    DynamicCallable,
    DynamicTuple,
    EmptyTuple,
    Nullary,
    Predicate,
    Product,
    Quaternary,
    RecursiveAnyIterable,
    Sum,
    Ternary,
    Tuple1,
    Tuple2,
    Tuple3,
    Tuple4,
    Tuple5,
    Tuple6,
    Tuple7,
    Tuple8,
    Unary,
    is_async_iterable,
    is_async_iterator,
    is_bytes,
    is_iterable,
    is_iterator,
    is_string,
)
from iters.utils import COMPARE, repeat, repeat_with, take, unpack_binary

if CONCURRENT:
    from iters.concurrent import collect_iterable


__all__ = (
    "async_accumulate_fold",
    "async_accumulate_fold_await",
    "async_accumulate_reduce",
    "async_accumulate_reduce_await",
    "async_accumulate_product",
    "async_accumulate_sum",
    "async_all",
    "async_all_equal",
    "async_all_equal_await",
    "async_all_unique",
    "async_all_unique_await",
    "async_all_unique_fast",
    "async_all_unique_fast_await",
    "async_any",
    "async_append",
    "async_at",
    "async_at_or_last",
    "async_cartesian_power",
    "async_cartesian_product",
    "async_chain",
    "async_chain_from_iterable",
    "async_chunks",
    "async_collapse",
    "async_combine",
    "async_compare",
    "async_compare_await",
    "async_compress",
    "async_consume",
    "async_contains",
    "async_contains_identity",
    "async_copy",
    "async_copy_infinite",
    "async_copy_unsafe",
    "async_count",
    "async_count_dict",
    "async_count_dict_await",
    "async_cycle",
    "async_dict",
    "async_distribute",
    "async_distribute_infinite",
    "async_distribute_unsafe",
    "async_divide",
    "async_drop",
    "async_drop_while",
    "async_drop_while_await",
    "async_duplicates",
    "async_duplicates_await",
    "async_duplicates_fast",
    "async_duplicates_fast_await",
    "async_empty",
    "async_enumerate",
    "async_extract",
    "async_filter",
    "async_filter_await",
    "async_filter_await_map",
    "async_filter_await_map_await",
    "async_filter_except",
    "async_filter_except_await",
    "async_filter_false",
    "async_filter_false_await",
    "async_filter_false_await_map",
    "async_filter_false_await_map_await",
    "async_filter_false_map",
    "async_filter_false_map_await",
    "async_filter_map",
    "async_filter_map_await",
    "async_find_all",
    "async_find_all_await",
    "async_find",
    "async_find_await",
    "async_find_or_first",
    "async_find_or_first_await",
    "async_find_or_last",
    "async_find_or_last_await",
    "async_first",
    "async_flat_map",
    "async_flat_map_await",
    "async_flatten",
    "async_fold",
    "async_fold_await",
    "async_for_each",
    "async_for_each_await",
    "async_group",
    "async_group_await",
    "async_group_dict",
    "async_group_dict_await",
    "async_group_list",
    "async_group_list_await",
    "async_groups",
    "async_groups_longest",
    "async_has_next",
    "async_interleave",
    "async_interleave_longest",
    "async_intersperse",
    "async_intersperse_with",
    "async_intersperse_with_await",
    "async_is_empty",
    "async_is_sorted",
    "async_is_sorted_await",
    "async_iter",
    "async_iter_async_with",
    "async_iter_chunks",
    "async_iter_chunks_infinite",
    "async_iter_chunks_unsafe",
    "async_iter_except",
    "async_iter_except_await",
    "async_iter_function",
    "async_iter_function_await",
    "async_iter_length",
    "async_iter_slice",
    "async_iter_windows",
    "async_iter_with",
    "async_iterate",
    "async_iterate_await",
    "async_last",
    "async_last_with_tail",
    "async_list",
    "async_list_windows",
    "async_map",
    "async_map_await",
    "async_map_except",
    "async_map_except_await",
    "async_max",
    "async_max_await",
    "async_min",
    "async_min_await",
    "async_min_max",
    "async_min_max_await",
    "async_next",
    "async_next_of",
    "async_next_unchecked",
    "async_next_unchecked_of",
    "async_once",
    "async_once_with",
    "async_once_with_await",
    "async_ordered_set",
    "async_pad",
    "async_pad_with",
    "async_pad_with_await",
    "async_pairs",
    "async_pairs_longest",
    "async_pairs_windows",
    "async_partition",
    "async_partition_await",
    "async_partition_infinite",
    "async_partition_infinite_await",
    "async_partition_unsafe",
    "async_partition_unsafe_await",
    "async_peek",
    "async_position_all",
    "async_position_all_await",
    "async_position",
    "async_position_await",
    "async_prepend",
    "async_product",
    "async_reduce",
    "async_reduce_await",
    "async_remove",
    "async_remove_await",
    "async_remove_duplicates",
    "async_remove_duplicates_await",
    "async_repeat",
    "async_repeat_each",
    "async_repeat_last",
    "async_repeat_with",
    "async_repeat_with_await",
    "async_reverse",
    "async_reversed",
    "async_set",
    "async_side_effect",
    "async_side_effect_await",
    "async_skip",
    "async_skip_while",
    "async_skip_while_await",
    "async_sort",
    "async_sort_await",
    "async_sorted",
    "async_sorted_await",
    "async_spy",
    "async_step_by",
    "async_sum",
    "async_tail",
    "async_take",
    "async_take_while",
    "async_take_while_await",
    "async_tuple",
    "async_tuple_windows",
    "async_unique",
    "async_unique_await",
    "async_unique_fast",
    "async_unique_fast_await",
    "async_wait",
    "async_zip",
    "async_zip_equal",
    "async_zip_longest",
    "iter_to_async_iter",
    "standard_async_iter",
    "standard_async_map",
    "standard_async_map_await",
    "standard_async_next",
    "wrap_await",
)

PS = ParamSpec("PS")

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

L = TypeVar("L")
R = TypeVar("R")

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")

# AI = TypeVar("AI", bound=AnyIterable)  # AI[T]

Q = TypeVar("Q", bound=Hashable)

S = TypeVar("S", bound=Sum)
P = TypeVar("P", bound=Product)

LT = TypeVar("LT", bound=LenientOrdered)
ST = TypeVar("ST", bound=StrictOrdered)


try:
    from builtins import aiter as standard_async_iter  # type: ignore
    from builtins import anext as standard_async_next  # type: ignore

except ImportError:

    def standard_async_iter(async_iterable: AsyncIterable[T]) -> AsyncIterator[T]:  # type: ignore
        if is_async_iterable(async_iterable):
            return async_iterable.__aiter__()

        raise not_async_iterable(async_iterable)

    @overload  # type: ignore
    async def standard_async_next(async_iterator: AsyncIterator[T]) -> T:
        ...

    @overload
    async def standard_async_next(async_iterator: AsyncIterator[T], default: U) -> Union[T, U]:
        ...

    async def standard_async_next(
        async_iterator: AsyncIterator[Any], default: Any = no_default
    ) -> Any:
        if is_async_iterator(async_iterator):
            try:
                return await async_iterator.__anext__()

            except StopAsyncIteration:
                if default is no_default:
                    raise

                return default

        raise not_async_iterator(async_iterator)


@overload
async def async_compare(
    left_iterable: AnyIterable[ST], right_iterable: AnyIterable[ST], key: None = ...
) -> Ordering:
    ...


@overload
async def async_compare(
    left_iterable: AnyIterable[T], right_iterable: AnyIterable[T], key: Unary[T, ST]
) -> Ordering:
    ...


async def async_compare(
    left_iterable: AnyIterable[Any],
    right_iterable: AnyIterable[Any],
    key: Optional[Unary[Any, Any]] = None,
) -> Ordering:
    if key is None:
        return await async_compare_simple(left_iterable, right_iterable)

    return await async_compare_by(left_iterable, right_iterable, key)


async def async_compare_await(
    left_iterable: AnyIterable[T], right_iterable: AnyIterable[T], key: AsyncUnary[T, ST]
) -> Ordering:
    return await async_compare_by_await(left_iterable, right_iterable, key)


async def async_compare_by_await(
    left_iterable: AnyIterable[T], right_iterable: AnyIterable[T], key: AsyncUnary[T, ST]
) -> Ordering:
    return await async_compare_simple(
        async_map_await(key, left_iterable), async_map_await(key, right_iterable)
    )


async def async_compare_simple(
    left_iterable: AnyIterable[ST], right_iterable: AnyIterable[ST]
) -> Ordering:
    async for left, right in async_zip_longest(left_iterable, right_iterable, fill=marker):
        if left is marker:
            return Ordering.LESS

        if right is marker:
            return Ordering.GREATER

        if left < right:  # type: ignore
            return Ordering.LESS

        if left > right:  # type: ignore
            return Ordering.GREATER

    return Ordering.EQUAL


async def async_compare_by(
    left_iterable: AnyIterable[T], right_iterable: AnyIterable[T], key: Unary[T, ST]
) -> Ordering:
    return await async_compare_simple(async_map(key, left_iterable), async_map(key, right_iterable))


async def async_borrow(iterator: AsyncIterator[T]) -> AsyncGenerator[T, None]:
    async for item in iterator:
        yield item


async def async_iter_function(function: Nullary[T], sentinel: V) -> AsyncIterator[T]:
    while True:
        item = function()

        if item == sentinel:
            return

        yield item


async def async_iter_function_await(function: AsyncNullary[T], sentinel: V) -> AsyncIterator[T]:
    while True:
        item = await function()

        if item == sentinel:
            return

        yield item


async def async_empty() -> AsyncIterator[Never]:
    return
    yield  # type: ignore


async def async_once(value: T) -> AsyncIterator[T]:
    yield value


async def async_once_with(function: Nullary[T]) -> AsyncIterator[T]:
    yield function()


async def async_once_with_await(function: AsyncNullary[T]) -> AsyncIterator[T]:
    yield await function()


def wrap_await(function: Callable[PS, T]) -> Callable[PS, Awaitable[T]]:
    async def wrapped(*args: PS.args, **kwargs: PS.kwargs) -> T:
        return function(*args, **kwargs)

    return wrapped


async def async_repeat(value: T, count: Optional[int] = None) -> AsyncIterator[T]:
    if count is None:
        while True:
            yield value

    else:
        for _ in range(count):
            yield value


async def async_repeat_with(function: Nullary[T], count: Optional[int] = None) -> AsyncIterator[T]:
    if count is None:
        while True:
            yield function()

    else:
        for _ in range(count):
            yield function()


async def async_repeat_with_await(
    function: AsyncNullary[T], count: Optional[int] = None
) -> AsyncIterator[T]:
    if count is None:
        while True:
            yield await function()

    else:
        for _ in range(count):
            yield await function()


def async_repeat_factory(count: int) -> Unary[T, AsyncIterator[T]]:
    def actual_async_repeat(item: T) -> AsyncIterator[T]:
        return async_repeat(item, count)

    return actual_async_repeat


def async_repeat_each(iterable: AnyIterable[T], count: int = 2) -> AsyncIterator[T]:
    return async_flat_map(async_repeat_factory(count), iterable)


@overload
def async_repeat_last(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    ...


@overload
def async_repeat_last(
    iterable: AnyIterable[T], default: U
) -> Union[AsyncIterator[T], AsyncIterator[U]]:
    ...


async def async_repeat_last(
    iterable: AnyIterable[Any], default: Any = no_default
) -> AsyncIterator[Any]:
    item = marker

    async for item in async_iter(iterable):
        yield item

    if item is marker:
        if default is no_default:
            return

        item = default

    async for last in async_repeat(item):
        yield last


async def async_consume(iterable: AnyIterable[T]) -> None:
    async for _ in async_iter(iterable):
        pass


NOT_ASYNC_ITERABLE = "{!r} is not an async iterable"
NOT_ASYNC_ITERATOR = "{!r} is not an async iterator"


def not_async_iterable(item: T) -> TypeError:
    return TypeError(NOT_ASYNC_ITERABLE.format(get_type_name(item)))


def not_async_iterator(item: T) -> TypeError:
    return TypeError(NOT_ASYNC_ITERATOR.format(get_type_name(item)))


NOT_ANY_ITERABLE = "{!r} is not an (async) iterable"
NOT_ANY_ITERATOR = "{!r} is not an (async) iterator"


def not_any_iterable(item: T) -> TypeError:
    return TypeError(NOT_ANY_ITERABLE.format(get_type_name(item)))


def not_any_iterator(item: T) -> TypeError:
    return TypeError(NOT_ANY_ITERATOR.format(get_type_name(item)))


async def async_list(iterable: AnyIterable[T]) -> List[T]:
    return [item async for item in async_iter(iterable)]


async def async_dict(iterable: AnyIterable[Tuple[Q, T]]) -> Dict[Q, T]:
    return {key: value async for key, value in async_iter(iterable)}


async def async_set(iterable: AnyIterable[Q]) -> Set[Q]:
    return {item async for item in async_iter(iterable)}


async def async_ordered_set(iterable: AnyIterable[Q]) -> OrderedSet[Q]:
    return ordered_set(await async_list(iterable))


async def async_tuple(iterable: AnyIterable[T]) -> DynamicTuple[T]:
    return tuple(await async_list(iterable))


async def async_extract(iterable: AnyIterable[T]) -> Iterator[T]:
    return iter(await async_list(iterable))


async def async_counter_dict(iterable: AnyIterable[Q]) -> Counter[Q]:
    return counter_dict(await async_list(iterable))


@overload
async def async_sorted(
    iterable: AnyIterable[ST], *, key: None = ..., reverse: bool = ...
) -> List[ST]:
    ...


@overload
async def async_sorted(
    iterable: AnyIterable[T], *, key: Unary[T, ST], reverse: bool = ...
) -> List[T]:
    ...


async def async_sorted(
    iterable: AnyIterable[Any], *, key: Optional[Unary[Any, Any]] = None, reverse: bool = False
) -> List[Any]:
    array = await async_list(iterable)

    array.sort(key=key, reverse=reverse)

    return array


def sort_key(item: Tuple[ST, T]) -> ST:
    key, _ = item

    return key


async def async_sorted_await(
    iterable: AnyIterable[T], *, key: AsyncUnary[T, ST], reverse: bool = False
) -> List[T]:
    array = [(await key(item), item) async for item in async_iter(iterable)]

    array.sort(key=sort_key, reverse=reverse)

    return [item for _, item in array]


async def iter_to_async_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    for item in iterable:
        yield item


def async_iter(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    if is_async_iterable(iterable):
        return standard_async_iter(iterable)  # type: ignore

    if is_iterable(iterable):
        return iter_to_async_iter(iterable)

    raise not_any_iterable(iterable)


@overload
async def async_next(iterator: AnyIterator[T]) -> T:
    ...


@overload
async def async_next(iterator: AnyIterator[T], default: U) -> Union[T, U]:
    ...


async def async_next(iterator: AnyIterator[Any], default: Any = no_default) -> Any:
    if is_async_iterator(iterator):
        if default is no_default:
            return await standard_async_next(iterator)

        return await standard_async_next(iterator, default)

    if is_iterator(iterator):
        if default is no_default:
            return standard_next(iterator)

        return standard_next(iterator, default)

    raise not_any_iterator(iterator)


@overload
async def async_next_unchecked(iterator: AsyncIterator[T]) -> T:
    ...


@overload
async def async_next_unchecked(iterator: AsyncIterator[T], default: U) -> Union[T, U]:
    ...


async def async_next_unchecked(iterator: AsyncIterator[Any], default: Any = no_default) -> Any:
    if default is no_default:
        return await standard_async_next(iterator)

    return await standard_async_next(iterator, default)


async def async_for_each(function: Unary[T, Any], iterable: AnyIterable[T]) -> None:
    async for item in async_iter(iterable):
        function(item)


async def async_for_each_await(function: AsyncUnary[T, Any], iterable: AnyIterable[T]) -> None:
    async for item in async_iter(iterable):
        await function(item)


ASYNC_FIRST_ON_EMPTY = "async_first() called on an empty iterable"


@overload
async def async_first(iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_first(iterable: AnyIterable[T], default: U) -> Union[T, U]:
    ...


async def async_first(iterable: AnyIterable[Any], default: Any = no_default) -> Any:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_FIRST_ON_EMPTY)

        return default

    return result


ASYNC_LAST_ON_EMPTY = "async_last() called on an empty iterable"


@overload
async def async_last(iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_last(iterable: AnyIterable[T], default: U) -> Union[T, U]:
    ...


async def async_last(iterable: AnyIterable[Any], default: Any = no_default) -> Any:
    result = marker

    async for result in async_iter(iterable):
        pass

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_LAST_ON_EMPTY)

        return default

    return result


async def async_fold(initial: U, function: Binary[U, T, U], iterable: AnyIterable[T]) -> U:
    result = initial

    async for item in async_iter(iterable):
        result = function(result, item)

    return result


async def async_fold_await(
    initial: U, function: AsyncBinary[U, T, U], iterable: AnyIterable[T]
) -> U:
    result = initial

    async for item in async_iter(iterable):
        result = await function(result, item)

    return result


ASYNC_REDUCE_ON_EMPTY = "async_reduce() called on an empty iterable"


async def async_reduce(function: Binary[T, T, T], iterable: AnyIterable[T]) -> T:
    iterator = async_iter(iterable)

    initial = await async_next_unchecked(iterator, marker)

    if initial is marker:
        raise ValueError(ASYNC_REDUCE_ON_EMPTY)

    return await async_fold(initial, function, iterator)  # type: ignore


ASYNC_REDUCE_AWAIT_ON_EMPTY = "async_reduce_await() called on an empty iterable"


async def async_reduce_await(function: AsyncBinary[T, T, T], iterable: AnyIterable[T]) -> T:
    iterator = async_iter(iterable)

    initial = await async_next_unchecked(iterator, marker)

    if initial is marker:
        raise ValueError(ASYNC_REDUCE_AWAIT_ON_EMPTY)

    return await async_fold_await(initial, function, iterator)  # type: ignore


async def async_accumulate_fold(
    initial: U, function: Binary[U, T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    yield initial

    value = initial

    async for item in async_iter(iterable):
        value = function(value, item)

        yield value


async def async_accumulate_fold_await(
    initial: U, function: AsyncBinary[U, T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    yield initial

    value = initial

    async for item in async_iter(iterable):
        value = await function(value, item)

        yield value


ASYNC_ACCUMULATE_REDUCE_ON_EMPTY = "async_accumulate_reduce() called on an empty iterable"


async def async_accumulate_reduce(
    function: Binary[T, T, T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    initial = await async_next_unchecked(iterator, marker)

    if initial is marker:
        raise ValueError(ASYNC_ACCUMULATE_REDUCE_ON_EMPTY)

    async for value in async_accumulate_fold(initial, function, iterator):  # type: ignore
        yield value  # type: ignore


ASYNC_ACCUMULATE_REDUCE_AWAIT_ON_EMPTY = (
    "async_accumulate_reduce_await() called on an empty iterable"
)


async def async_accumulate_reduce_await(
    function: AsyncBinary[T, T, T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    initial = await async_next_unchecked(iterator, marker)

    if initial is marker:
        raise ValueError(ASYNC_ACCUMULATE_REDUCE_ON_EMPTY)

    async for value in async_accumulate_fold_await(initial, function, iterator):  # type: ignore
        yield value  # type: ignore


@overload
def async_accumulate_sum(iterable: AnyIterable[S]) -> AsyncIterator[S]:
    ...


@overload
def async_accumulate_sum(iterable: AnyIterable[S], initial: S) -> AsyncIterator[S]:
    ...


def async_accumulate_sum(
    iterable: AnyIterable[Any], initial: Any = no_default
) -> AsyncIterator[Any]:
    if initial is no_default:
        return async_accumulate_reduce(add, iterable)

    return async_accumulate_fold(initial, add, iterable)


@overload
def async_accumulate_product(iterable: AnyIterable[P]) -> AsyncIterator[P]:
    ...


@overload
def async_accumulate_product(iterable: AnyIterable[P], initial: P) -> AsyncIterator[P]:
    ...


def async_accumulate_product(
    iterable: AnyIterable[Any], initial: Any = no_default
) -> AsyncIterator[Any]:
    if initial is no_default:
        return async_accumulate_reduce(add, iterable)

    return async_accumulate_fold(initial, add, iterable)


ASYNC_AT_ON_EMPTY = "async_at() called on an empty iterable"


@overload
async def async_at(index: int, iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_at(index: int, iterable: AnyIterable[T], default: U) -> Union[T, U]:
    ...


async def async_at(index: int, iterable: AnyIterable[Any], default: Any = no_default) -> Any:
    iterator = async_drop(index, iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_AT_ON_EMPTY)

        return default

    return result


ASYNC_AT_OR_LAST_ON_EMPTY = "async_at_or_last() called on an empty iterable"


@overload
async def async_at_or_last(index: int, iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_at_or_last(index: int, iterable: AnyIterable[T], default: U) -> Union[T, U]:
    ...


async def async_at_or_last(
    index: int, iterable: AnyIterable[Any], default: Any = no_default
) -> Any:
    iterator = async_take(index + 1, iterable)

    result = await async_last(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_AT_OR_LAST_ON_EMPTY)

        return default

    return result


@overload
def async_copy_unsafe(iterable: AnyIterable[T]) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[0]) -> EmptyTuple:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[1]) -> Tuple1[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[2]) -> Tuple2[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[3]) -> Tuple3[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[4]) -> Tuple4[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[5]) -> Tuple5[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[6]) -> Tuple6[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[7]) -> Tuple7[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: Literal[8]) -> Tuple8[AsyncIterator[T]]:
    ...


@overload
def async_copy_unsafe(iterable: AnyIterable[T], copies: int) -> DynamicTuple[AsyncIterator[T]]:
    ...


def async_copy_unsafe(iterable: AnyIterable[T], copies: int = 2) -> DynamicTuple[AsyncIterator[T]]:
    iterator = async_iter(iterable)

    buffers: DynamicTuple[Deque[T]] = tuple(repeat_with(Deque[T], copies))

    async def generator(this: Deque[T]) -> AsyncIterator[T]:
        while True:
            if not this:
                item = await async_next_unchecked(iterator, marker)

                if item is marker:
                    return

                for buffer in buffers:
                    buffer.append(item)  # type: ignore

            yield this.popleft()

    return tuple(generator(buffer) for buffer in buffers)


async_copy_infinite = async_copy_unsafe


@overload
def async_copy(iterable: AnyIterable[T]) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[0]) -> EmptyTuple:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[1]) -> Tuple1[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[2]) -> Tuple2[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[3]) -> Tuple3[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[4]) -> Tuple4[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[5]) -> Tuple5[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[6]) -> Tuple6[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[7]) -> Tuple7[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: Literal[8]) -> Tuple8[AsyncIterator[T]]:
    ...


@overload
def async_copy(iterable: AnyIterable[T], copies: int) -> DynamicTuple[AsyncIterator[T]]:
    ...


def async_copy(iterable: AnyIterable[T], copies: int = 2) -> DynamicTuple[AsyncIterator[T]]:
    state: Optional[DynamicTuple[T]] = None

    async def generator() -> AsyncIterator[T]:
        nonlocal state

        if state is None:
            state = await async_tuple(iterable)

        async for value in async_iter(state):
            yield value

    return tuple(repeat_with(generator, copies))


@overload
def async_iter_slice(iterable: AnyIterable[T], __stop: Optional[int]) -> AsyncIterator[T]:
    ...


@overload
def async_iter_slice(
    iterable: AnyIterable[T],
    __start: Optional[int],
    __stop: Optional[int],
    __step: Optional[int] = ...,
) -> AsyncIterator[T]:
    ...


async def async_iter_slice(
    iterable: AnyIterable[T], *slice_args: Optional[int]
) -> AsyncIterator[T]:
    start, stop, step = extract_slice(*slice_args)

    iterator = async_iter(iterable)

    if start is None:
        pass  # no need to do anything

    else:
        async for count, _ in async_enumerate(iterator, 1):  # drop()
            if count == start:
                break

    if stop is None:
        if step is None:
            async for item in iterator:
                yield item

        else:
            async for index, item in async_enumerate(iterator):  # step_by()
                if not index % step:
                    yield item

    else:
        take = stop

        if start:
            take -= start

        if take > 0:
            if step is None:
                async for count, item in async_enumerate(iterator, 1):  # take()
                    yield item

                    if count == take:
                        break

            else:
                take -= 1

                async for index, item in async_enumerate(iterator):  # take() + step_by()
                    if not index % step:
                        yield item

                    if index == take:
                        break


NEGATIVE_START = "start can not be negative"
NEGATIVE_STOP = "stop can not be negative"
NEGATIVE_STEP = "step can not be negative"

ZERO_STEP = "step equal to 0 is ambiguous"


def extract_slice(*slice_args: Optional[int]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    slice_item = slice(*slice_args)

    start = slice_item.start
    stop = slice_item.stop
    step = slice_item.step

    if start is not None:
        if start < 0:
            raise ValueError(NEGATIVE_START)

        if not start:
            start = None

    if stop is not None:
        if stop < 0:
            raise ValueError(NEGATIVE_STOP)

    if step is not None:
        if step < 0:
            raise ValueError(NEGATIVE_STEP)

        if not step:
            raise ValueError(ZERO_STEP)

        if step == 1:
            step = None

    return (start, stop, step)


NEGATIVE_SIZE = "size can not be negative"


def async_drop(size: int, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    if size < 0:
        raise ValueError(NEGATIVE_SIZE)

    return async_drop_unchecked(size, iterable) if size else async_iter(iterable)


async def async_drop_unchecked(size: int, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    async for count, _ in async_enumerate(iterator, 1):
        if count == size:
            break

    async for item in iterator:
        yield item


async_skip = async_drop


def async_take(size: int, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    if size < 0:
        raise ValueError(NEGATIVE_SIZE)

    return async_take_unchecked(size, iterable) if size else async_empty()


async def async_take_unchecked(size: int, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    async for count, item in async_enumerate(iterable, 1):
        yield item

        if count == size:
            break


def async_step_by(step: int, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    if step < 0:
        raise ValueError(NEGATIVE_STEP)

    if not step:
        raise ValueError(ZERO_STEP)

    return async_iter(iterable) if step == 1 else async_step_by_unchecked(step, iterable)


async def async_step_by_unchecked(step: int, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    async for index, item in async_enumerate(iterable):
        if not index % step:
            yield item


@overload
def async_groups(size: Literal[0], iterable: AnyIterable[T]) -> AsyncIterator[EmptyTuple]:
    ...


@overload
def async_groups(size: Literal[1], iterable: AnyIterable[T]) -> AsyncIterator[Tuple1[T]]:
    ...


@overload
def async_groups(size: Literal[2], iterable: AnyIterable[T]) -> AsyncIterator[Tuple2[T]]:
    ...


@overload
def async_groups(size: Literal[3], iterable: AnyIterable[T]) -> AsyncIterator[Tuple3[T]]:
    ...


@overload
def async_groups(size: Literal[4], iterable: AnyIterable[T]) -> AsyncIterator[Tuple4[T]]:
    ...


@overload
def async_groups(size: Literal[5], iterable: AnyIterable[T]) -> AsyncIterator[Tuple5[T]]:
    ...


@overload
def async_groups(size: Literal[6], iterable: AnyIterable[T]) -> AsyncIterator[Tuple6[T]]:
    ...


@overload
def async_groups(size: Literal[7], iterable: AnyIterable[T]) -> AsyncIterator[Tuple7[T]]:
    ...


@overload
def async_groups(size: Literal[8], iterable: AnyIterable[T]) -> AsyncIterator[Tuple8[T]]:
    ...


@overload
def async_groups(size: int, iterable: AnyIterable[T]) -> AsyncIterator[DynamicTuple[T]]:
    ...


def async_groups(size: int, iterable: AnyIterable[T]) -> AsyncIterator[DynamicTuple[T]]:
    return async_zip(*repeat(async_iter(iterable), size))


@overload
def async_groups_longest(size: Literal[0], iterable: AnyIterable[T]) -> AsyncIterator[EmptyTuple]:
    ...


@overload
def async_groups_longest(size: Literal[1], iterable: AnyIterable[T]) -> AsyncIterator[Tuple1[T]]:
    ...


@overload
def async_groups_longest(
    size: Literal[2], iterable: AnyIterable[T]
) -> AsyncIterator[Tuple2[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[3], iterable: AnyIterable[T]
) -> AsyncIterator[Tuple3[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[4], iterable: AnyIterable[T]
) -> AsyncIterator[Tuple4[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[5], iterable: AnyIterable[T]
) -> AsyncIterator[Tuple5[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[6], iterable: AnyIterable[T]
) -> AsyncIterator[Tuple6[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[7], iterable: AnyIterable[T]
) -> AsyncIterator[Tuple7[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[8], iterable: AnyIterable[T]
) -> AsyncIterator[Tuple8[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: int, iterable: AnyIterable[T]
) -> AsyncIterator[DynamicTuple[Optional[T]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[0], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[EmptyTuple]:
    ...


@overload
def async_groups_longest(
    size: Literal[1], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple1[T]]:
    ...


@overload
def async_groups_longest(
    size: Literal[2], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple2[Union[T, U]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[3], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple3[Union[T, U]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[4], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple4[Union[T, U]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[5], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple5[Union[T, U]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[6], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple6[Union[T, U]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[7], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple7[Union[T, U]]]:
    ...


@overload
def async_groups_longest(
    size: Literal[8], iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple8[Union[T, U]]]:
    ...


@overload
def async_groups_longest(
    size: int, iterable: AnyIterable[T], fill: U
) -> AsyncIterator[DynamicTuple[Union[T, U]]]:
    ...


def async_groups_longest(
    size: int, iterable: AnyIterable[Any], fill: Optional[Any] = None
) -> AsyncIterator[DynamicTuple[Any]]:
    return async_zip_longest(*repeat(async_iter(iterable), size), fill=fill)


@overload
def async_pairs_longest(iterable: AnyIterable[T]) -> AsyncIterator[Tuple[Optional[T], Optional[T]]]:
    ...


@overload
def async_pairs_longest(
    iterable: AnyIterable[T], fill: U
) -> AsyncIterator[Tuple[Union[T, U], Union[T, U]]]:
    ...


def async_pairs_longest(
    iterable: AnyIterable[Any], fill: Optional[Any] = None
) -> AsyncIterator[Tuple[Any, Any]]:
    return async_groups_longest(2, iterable, fill)


def async_pairs(iterable: AnyIterable[T]) -> AsyncIterator[Tuple[T, T]]:
    return async_groups(2, iterable)


def async_flatten(nested: AnyIterable[AnyIterable[T]]) -> AsyncIterator[T]:
    return async_chain_from_iterable(nested)


# TODO: change `function` type?


def async_flat_map(
    function: Unary[T, AnyIterable[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    return async_flatten(async_map(function, iterable))


def async_flat_map_await(
    function: AsyncUnary[T, AnyIterable[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    return async_flatten(async_map_await(function, iterable))


async def async_filter_map(
    predicate: Optional[Predicate[T]], function: Unary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    if predicate is None:
        async for item in async_iter(iterable):
            if item:
                yield function(item)

    else:
        async for item in async_iter(iterable):
            if predicate(item):
                yield function(item)


async def async_filter_await_map(
    predicate: AsyncPredicate[T], function: Unary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        if await predicate(item):
            yield function(item)


async def async_filter_map_await(
    predicate: Optional[Predicate[T]], function: AsyncUnary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    if predicate is None:
        async for item in async_iter(iterable):
            if item:
                yield await function(item)

    else:
        async for item in async_iter(iterable):
            if predicate(item):
                yield await function(item)


async def async_filter_await_map_await(
    predicate: AsyncPredicate[T], function: AsyncUnary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        if await predicate(item):
            yield await function(item)


async def async_filter_false_map(
    predicate: Optional[Predicate[T]], function: Unary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    if predicate is None:
        async for item in async_iter(iterable):
            if not item:
                yield function(item)

    else:
        async for item in async_iter(iterable):
            if not predicate(item):
                yield function(item)


async def async_filter_false_await_map(
    predicate: AsyncPredicate[T], function: Unary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        if not await predicate(item):
            yield function(item)


async def async_filter_false_map_await(
    predicate: Optional[Predicate[T]], function: AsyncUnary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    if predicate is None:
        async for item in async_iter(iterable):
            if not item:
                yield await function(item)

    else:
        async for item in async_iter(iterable):
            if not predicate(item):
                yield await function(item)


async def async_filter_false_await_map_await(
    predicate: AsyncPredicate[T], function: AsyncUnary[T, U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        if not await predicate(item):
            yield await function(item)


def async_partition_unsafe(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T]
) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    for_true, for_false = async_copy_unsafe(iterable)

    return async_filter(predicate, for_true), async_filter_false(predicate, for_false)


async_partition_infinite = async_partition_unsafe


def async_partition_unsafe_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    for_true, for_false = async_copy_unsafe(iterable)

    return async_filter_await(predicate, for_true), async_filter_false_await(predicate, for_false)


async_partition_infinite_await = async_partition_unsafe_await


def async_partition(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T]
) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    for_true, for_false = async_copy(iterable)

    return async_filter(predicate, for_true), async_filter_false(predicate, for_false)


def async_partition_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    for_true, for_false = async_copy(iterable)

    return async_filter_await(predicate, for_true), async_filter_false_await(predicate, for_false)


def async_prepend(item: T, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    return async_chain(async_once(item), iterable)


def async_append(item: T, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    return async_chain(iterable, async_once(item))


class StopAsyncGroup(Exception):
    pass


async def async_next_or_stop_async_group(iterator: AsyncIterator[T]) -> T:
    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        raise StopAsyncGroup

    return result  # type: ignore


async def async_group_simple(
    iterable: AnyIterable[Any],
) -> AsyncIterator[Tuple[Any, AsyncIterator[Any]]]:
    exhausted = False
    count = 0

    current_key = current_item = nothing = marker

    iterator = async_iter(iterable)

    async def seek_group() -> AsyncIterator[Any]:
        nonlocal count, current_item, current_key, exhausted

        item = current_item

        if not exhausted:
            previous_key = current_key

            while previous_key == current_key:
                item = await async_next_or_stop_async_group(iterator)

                current_key = item

        current_item = nothing
        exhausted = False

        count += 1

        return group(current_key, item, count)  # type: ignore

    async def group(desired_key: Any, item: Any, group_count: int) -> AsyncIterator[Any]:
        nonlocal count, current_item, current_key, exhausted

        if count == group_count:
            yield item

        async for item in iterator:
            if item == desired_key:
                yield item

            else:
                exhausted = True
                current_item = item
                current_key = item
                break

    try:
        while True:
            next_group = await seek_group()

            yield current_key, next_group

    except StopAsyncGroup:
        current_key = current_item = nothing
        return


async def async_group_by(
    iterable: AnyIterable[Any], key: Unary[Any, Any]
) -> AsyncIterator[Tuple[Any, AsyncIterator[Any]]]:
    exhausted = False
    count = 0

    current_key = current_item = nothing = marker

    iterator = async_iter(iterable)

    async def seek_group() -> AsyncIterator[Any]:
        nonlocal count, current_item, current_key, exhausted

        item = current_item

        if not exhausted:
            previous_key = current_key

            while previous_key == current_key:
                item = await async_next_or_stop_async_group(iterator)

                current_key = key(item)

        current_item = nothing
        exhausted = False

        count += 1

        return group(current_key, item, count)  # type: ignore

    async def group(desired_key: Any, item: Any, group_count: int) -> AsyncIterator[Any]:
        nonlocal count, current_item, current_key, exhausted

        if count == group_count:
            yield item

        async for item in iterator:
            if item == desired_key:
                yield item

            else:
                exhausted = True
                current_item = item
                current_key = key(item)
                break

    try:
        while True:
            next_group = await seek_group()

            yield current_key, next_group

    except StopAsyncGroup:
        current_key = current_item = nothing
        return


async def async_group_by_await(
    iterable: AnyIterable[Any], key: AsyncUnary[Any, Any]
) -> AsyncIterator[Tuple[Any, AsyncIterator[Any]]]:
    exhausted = False
    count = 0

    current_key = current_item = nothing = marker

    iterator = async_iter(iterable)

    async def seek_group() -> AsyncIterator[Any]:
        nonlocal count, current_item, current_key, exhausted

        item = current_item

        if not exhausted:
            previous_key = current_key

            while previous_key == current_key:
                item = await async_next_or_stop_async_group(iterator)

                current_key = await key(item)

        current_item = nothing
        exhausted = False

        count += 1

        return group(current_key, item, count)  # type: ignore

    async def group(desired_key: Any, item: Any, group_count: int) -> AsyncIterator[Any]:
        nonlocal count, current_item, current_key, exhausted

        if count == group_count:
            yield item

        async for item in iterator:
            if item == desired_key:
                yield item

            else:
                exhausted = True
                current_item = item
                current_key = await key(item)
                break

    try:
        while True:
            next_group = await seek_group()

            yield current_key, next_group

    except StopAsyncGroup:
        current_key = current_item = nothing
        return


@overload
def async_group(
    iterable: AnyIterable[T], key: None = ...
) -> AsyncIterator[Tuple[T, AsyncIterator[T]]]:
    ...


@overload
def async_group(
    iterable: AnyIterable[T], key: Unary[T, U]
) -> AsyncIterator[Tuple[U, AsyncIterator[T]]]:
    ...


def async_group(
    iterable: AnyIterable[Any], key: Optional[Unary[Any, Any]] = None
) -> AsyncIterator[Tuple[Any, AsyncIterator[Any]]]:
    return async_group_simple(iterable) if key is None else async_group_by(iterable, key)


def async_group_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, U]
) -> AsyncIterator[Tuple[U, AsyncIterator[T]]]:
    return async_group_by_await(iterable, key)


@overload
def async_group_list(iterable: AnyIterable[T], key: None = ...) -> AsyncIterator[Tuple[T, List[T]]]:
    ...


@overload
def async_group_list(
    iterable: AnyIterable[T], key: Unary[T, U]
) -> AsyncIterator[Tuple[U, List[T]]]:
    ...


async def async_group_list(
    iterable: AnyIterable[Any], key: Optional[Unary[Any, Any]] = None
) -> AsyncIterator[Tuple[Any, List[Any]]]:
    async for group_key, group_iterator in async_group(iterable, key):
        yield (group_key, await async_list(group_iterator))


async def async_group_list_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, U]
) -> AsyncIterator[Tuple[U, List[T]]]:
    async for group_key, group_iterator in async_group_await(iterable, key):
        yield (group_key, await async_list(group_iterator))


@overload
async def async_group_dict(iterable: AnyIterable[Q], key: None = ...) -> Dict[Q, List[Q]]:
    ...


@overload
async def async_group_dict(iterable: AnyIterable[T], key: Unary[T, Q]) -> Dict[Q, List[T]]:
    ...


async def async_group_dict(
    iterable: AnyIterable[Any], key: Optional[Unary[Any, Any]] = None
) -> Dict[Any, List[Any]]:
    result: Dict[Any, List[Any]] = {}

    async for group_key, group_list in async_group_list(iterable, key):
        result.setdefault(group_key, []).extend(group_list)

    return result


async def async_group_dict_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, Q]
) -> Dict[Q, List[T]]:
    result: Dict[Q, List[T]] = {}

    async for group_key, group_list in async_group_list_await(iterable, key):
        result.setdefault(group_key, []).extend(group_list)

    return result


@overload
async def async_count_dict(iterable: AnyIterable[Q], key: None = ...) -> Counter[Q]:
    ...


@overload
async def async_count_dict(iterable: AnyIterable[T], key: Unary[T, Q]) -> Counter[Q]:
    ...


async def async_count_dict(
    iterable: AnyIterable[Any], key: Optional[Unary[Any, Any]] = None
) -> Counter[Any]:
    return await async_counter_dict(iterable if key is None else async_map(key, iterable))


async def async_count_dict_await(iterable: AnyIterable[T], key: AsyncUnary[T, Q]) -> Counter[Q]:
    return await async_counter_dict(async_map_await(key, iterable))


async def async_chunks(size: int, iterable: AnyIterable[T]) -> AsyncIterator[List[T]]:
    iterator = async_iter(iterable)

    while True:
        chunk = await async_list(async_take(size, iterator))

        if not chunk:
            return

        yield chunk


async def async_iter_chunks_unsafe(
    size: int, iterable: AnyIterable[T]
) -> AsyncIterator[AsyncIterator[T]]:
    source = async_iter(iterable)

    while True:
        empty, source = await async_is_empty(source)

        if empty:
            return

        source, iterator = async_copy_unsafe(source)

        yield async_take(size, iterator)

        await async_consume(async_take(size, source))


async_iter_chunks_infinite = async_iter_chunks_unsafe


async def async_iter_chunks(size: int, iterable: AnyIterable[T]) -> AsyncIterator[AsyncIterator[T]]:
    source = async_iter(iterable)

    while True:
        empty, source = await async_is_empty(source)

        if empty:
            return

        source, iterator = async_copy(source)

        yield async_take(size, iterator)

        await async_consume(async_take(size, source))


async def async_iter_length(iterable: AnyIterable[T]) -> int:
    length = 0

    async for _ in async_iter(iterable):
        length += 1

    return length


@overload
async def async_sum(iterable: AnyIterable[S]) -> S:
    ...


@overload
async def async_sum(iterable: AnyIterable[S], initial: S) -> S:
    ...


async def async_sum(iterable: AnyIterable[Any], initial: Any = no_default) -> Any:
    if initial is no_default:
        return await async_reduce(add, iterable)

    return await async_fold(initial, add, iterable)


@overload
async def async_product(iterable: AnyIterable[P]) -> P:
    ...


@overload
async def async_product(iterable: AnyIterable[P], initial: P) -> P:
    ...


async def async_product(iterable: AnyIterable[Any], initial: Any = no_default) -> Any:
    if initial is no_default:
        return await async_reduce(mul, iterable)

    return await async_fold(initial, mul, iterable)


async def async_iterate(
    function: Unary[T, T], value: T, count: Optional[int] = None
) -> AsyncIterator[T]:
    if count is None:
        while True:
            yield value
            value = function(value)

    else:
        for _ in range(count):
            yield value
            value = function(value)


async def async_iterate_await(
    function: AsyncUnary[T, T], value: T, count: Optional[int] = None
) -> AsyncIterator[T]:
    if count is None:
        while True:
            yield value
            value = await function(value)

    else:
        for _ in range(count):
            yield value
            value = await function(value)


async def async_walk(node: RecursiveAnyIterable[T]) -> AsyncIterator[T]:
    if is_string(node) or is_bytes(node):
        yield node  # type: ignore
        return

    try:
        tree = async_iter(node)  # type: ignore

    except TypeError:
        yield node  # type: ignore
        return

    else:
        async for child in tree:
            async for nested in async_walk(child):  # type: ignore
                yield nested


def async_collapse(iterable: AnyIterable[RecursiveAnyIterable[T]]) -> AsyncIterator[T]:
    return async_walk(iterable)


async def async_iter_with(context_manager: ContextManager[AnyIterable[T]]) -> AsyncIterator[T]:
    with context_manager as iterable:
        async for item in async_iter(iterable):
            yield item


async def async_iter_async_with(
    async_context_manager: AsyncContextManager[AnyIterable[T]],
) -> AsyncIterator[T]:
    async with async_context_manager as iterable:
        async for item in async_iter(iterable):
            yield item


async def async_pad(
    value: U,
    iterable: AnyIterable[T],
    size: Optional[int] = None,
    *,
    multiple: bool = False,
) -> AsyncIterator[Union[T, U]]:
    if size is None:
        async for item in async_chain(iterable, async_repeat(value)):  # type: ignore
            yield item  # type: ignore

    else:
        count = 0

        async for item in async_iter(iterable):
            count += 1
            yield item

        length = (size - count) % size if multiple else (size - count)

        if length > 0:
            async for item in async_repeat(value, length):
                yield item


async def async_pad_with(
    function: Unary[int, U],
    iterable: AnyIterable[T],
    size: Optional[int] = None,
    *,
    multiple: bool = False,
) -> AsyncIterator[Union[T, U]]:
    index = 0

    async for item in async_iter(iterable):
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


async def async_pad_with_await(
    function: AsyncUnary[int, U],
    iterable: AnyIterable[T],
    size: Optional[int] = None,
    *,
    multiple: bool = False,
) -> AsyncIterator[Union[T, U]]:
    index = 0

    async for item in async_iter(iterable):
        yield item
        index += 1

    if size is None:
        while True:
            yield await function(index)
            index += 1

    else:
        length = (size - index) % size if multiple else (size - index)

        if length > 0:
            for index in range(index, index + length):
                yield await function(index)


async def async_contains(value: U, iterable: AnyIterable[T]) -> bool:
    return await async_any(item == value async for item in async_iter(iterable))


async def async_contains_identity(value: T, iterable: AnyIterable[T]) -> bool:
    return await async_any(item is value async for item in async_iter(iterable))


@overload
async def async_all_unique_fast(iterable: AnyIterable[Q], key: None = ...) -> bool:
    ...


@overload
async def async_all_unique_fast(iterable: AnyIterable[T], key: Unary[T, Q]) -> bool:
    ...


async def async_all_unique_fast(
    iterable: AnyIterable[Any], key: Optional[Unary[Any, Any]] = None
) -> bool:
    is_unique, _ = await async_is_empty(async_duplicates_fast(iterable, key))

    return is_unique


async def async_all_unique_fast_await(iterable: AnyIterable[T], key: AsyncUnary[T, Q]) -> bool:
    is_unique, _ = await async_is_empty(async_duplicates_fast_await(iterable, key))

    return is_unique


async def async_all_unique(iterable: AnyIterable[T], key: Optional[Unary[T, U]] = None) -> bool:
    is_unique, _ = await async_is_empty(async_duplicates(iterable, key))

    return is_unique


async def async_all_unique_await(iterable: AnyIterable[T], key: AsyncUnary[T, U]) -> bool:
    is_unique, _ = await async_is_empty(async_duplicates_await(iterable, key))

    return is_unique


async def async_all_equal(iterable: AnyIterable[T], key: Optional[Unary[T, U]] = None) -> bool:
    return await length_at_most_one(async_group(iterable, key))


async def async_all_equal_await(iterable: AnyIterable[T], key: AsyncUnary[T, U]) -> bool:
    return await length_at_most_one(async_group_await(iterable, key))


async def length_at_most_one(iterable: AnyIterable[T]) -> bool:
    iterator = async_iter(iterable)

    none = await async_next_unchecked(iterator, marker) is marker
    only = await async_next_unchecked(iterator, marker) is marker

    return none or only


def async_remove(predicate: Optional[Predicate[T]], iterable: AnyIterable[T]) -> AsyncIterator[T]:
    return async_filter_false(predicate, iterable)


def async_remove_await(predicate: AsyncPredicate[T], iterable: AnyIterable[T]) -> AsyncIterator[T]:
    return async_filter_false_await(predicate, iterable)


async def async_remove_duplicates(
    iterable: AnyIterable[T], key: Optional[Unary[T, U]] = None
) -> AsyncIterator[T]:
    async for item in async_group(iterable, key):
        _, iterator = item

        yield await async_next(iterator)


async def async_remove_duplicates_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, U]
) -> AsyncIterator[T]:
    async for item in async_group_await(iterable, key):
        _, iterator = item

        yield await async_next(iterator)


async def async_spy(size: int, iterable: AnyIterable[T]) -> Tuple[List[T], AsyncIterator[T]]:
    iterator = async_iter(iterable)

    head = await async_list(async_take(size, iterator))

    return head.copy(), async_chain(head, iterator)


ASYNC_PEEK_ON_EMPTY = "async_peek() called on an empty iterable"


@overload
async def async_peek(iterable: AnyIterable[T]) -> Tuple[T, AsyncIterator[T]]:
    ...


@overload
async def async_peek(iterable: AnyIterable[T], default: U) -> Tuple[Union[T, U], AsyncIterator[T]]:
    ...


async def async_peek(
    iterable: AnyIterable[Any], default: Any = no_default
) -> Tuple[Any, AsyncIterator[Any]]:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_PEEK_ON_EMPTY)

        return (default, iterator)

    return (result, async_prepend(result, iterator))


async def async_has_next(iterable: AnyIterable[T]) -> Tuple[bool, AsyncIterator[T]]:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        return (False, iterator)

    return (True, async_prepend(result, iterator))  # type: ignore


async def async_is_empty(iterable: AnyIterable[T]) -> Tuple[bool, AsyncIterator[T]]:
    has_some, iterator = await async_has_next(iterable)

    return (not has_some, iterator)


def async_next_of(iterator: AnyIterator[T]) -> AsyncNullary[T]:
    async def call() -> T:
        return await async_next(iterator)

    return call


def async_next_unchecked_of(iterator: AsyncIterator[T]) -> AsyncNullary[T]:
    async def call() -> T:
        return await async_next_unchecked(iterator)

    return call


async def async_combine(*iterables: AnyIterable[T]) -> AsyncIterator[T]:
    pending = len(iterables)
    async_nexts = cycle(async_next_unchecked_of(async_iter(iterable)) for iterable in iterables)

    while pending:
        try:
            for async_next in async_nexts:
                yield await async_next()

        except StopIteration:
            pending -= 1
            async_nexts = cycle(take(pending, async_nexts))


@overload
def async_distribute_unsafe(count: Literal[0], iterable: AnyIterable[T]) -> EmptyTuple:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[1], iterable: AnyIterable[T]
) -> Tuple1[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[2], iterable: AnyIterable[T]
) -> Tuple2[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[3], iterable: AnyIterable[T]
) -> Tuple3[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[4], iterable: AnyIterable[T]
) -> Tuple4[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[5], iterable: AnyIterable[T]
) -> Tuple5[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[6], iterable: AnyIterable[T]
) -> Tuple6[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[7], iterable: AnyIterable[T]
) -> Tuple7[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(
    count: Literal[8], iterable: AnyIterable[T]
) -> Tuple8[AsyncIterator[T]]:
    ...


@overload
def async_distribute_unsafe(count: int, iterable: AnyIterable[T]) -> DynamicTuple[AsyncIterator[T]]:
    ...


def async_distribute_unsafe(count: int, iterable: AnyIterable[T]) -> DynamicTuple[AsyncIterator[T]]:
    iterators = async_copy_unsafe(iterable, count)

    return tuple(
        async_step_by(count, async_drop(index, iterator))
        for index, iterator in enumerate(iterators)
    )


async_distribute_infinite = async_distribute_unsafe


@overload
def async_distribute(count: Literal[0], iterable: AnyIterable[T]) -> EmptyTuple:
    ...


@overload
def async_distribute(count: Literal[1], iterable: AnyIterable[T]) -> Tuple1[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: Literal[2], iterable: AnyIterable[T]) -> Tuple2[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: Literal[3], iterable: AnyIterable[T]) -> Tuple3[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: Literal[4], iterable: AnyIterable[T]) -> Tuple4[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: Literal[5], iterable: AnyIterable[T]) -> Tuple5[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: Literal[6], iterable: AnyIterable[T]) -> Tuple6[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: Literal[7], iterable: AnyIterable[T]) -> Tuple7[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: Literal[8], iterable: AnyIterable[T]) -> Tuple8[AsyncIterator[T]]:
    ...


@overload
def async_distribute(count: int, iterable: AnyIterable[T]) -> DynamicTuple[AsyncIterator[T]]:
    ...


def async_distribute(count: int, iterable: AnyIterable[T]) -> DynamicTuple[AsyncIterator[T]]:
    iterators = async_copy(iterable, count)

    return tuple(
        async_step_by(count, async_drop(index, iterator))
        for index, iterator in enumerate(iterators)
    )


async def async_divide(count: int, iterable: AnyIterable[T]) -> AsyncIterator[AsyncIterator[T]]:
    array = await async_list(iterable)

    size, last = divmod(len(array), count)

    stop = 0

    for index in range(count):
        start = stop
        stop += size

        if index < last:
            stop += 1

        yield async_iter(array[start:stop])


# async_interleave(async_repeat(value), iterable) -> (value, item_1, ..., value, item_n)
# async_drop(1, ...) -> (item_1, ..., value, item_n)


def async_intersperse(value: T, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    return async_drop(1, async_interleave(async_repeat(value), iterable))


def async_intersperse_with(function: Nullary[T], iterable: AnyIterable[T]) -> AsyncIterator[T]:
    return async_drop(1, async_interleave(async_repeat_with(function), iterable))


def async_intersperse_with_await(
    function: AsyncNullary[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    return async_drop(1, async_interleave(async_repeat_with_await(function), iterable))


def async_interleave(*iterables: AnyIterable[T]) -> AsyncIterator[T]:
    return async_flatten(async_zip(*iterables))


def async_interleave_longest(*iterables: AnyIterable[T]) -> AsyncIterator[T]:
    iterator = async_flatten(async_zip_longest(*iterables, fill=marker))
    return (item async for item in iterator if item is not marker)  # type: ignore


async def async_position_all(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T]
) -> AsyncIterator[int]:
    if predicate is None:
        async for index, item in async_enumerate(iterable):
            if item:
                yield index

    else:
        async for index, item in async_enumerate(iterable):
            if predicate(item):
                yield index


async def async_position_all_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[int]:
    async for index, item in async_enumerate(iterable):
        if await predicate(item):
            yield index


ASYNC_POSITION_NO_MATCH = "async_position() has not found any matches"


@overload
async def async_position(predicate: Optional[Predicate[T]], iterable: AnyIterable[T]) -> int:
    ...


@overload
async def async_position(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T], default: U
) -> Union[int, U]:
    ...


async def async_position(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T], default: Any = no_default
) -> Any:
    index = await async_next_unchecked(async_position_all(predicate, iterable), None)

    if index is None:
        if default is no_default:
            raise ValueError(ASYNC_POSITION_NO_MATCH)

        return default

    return index


@overload
async def async_position_await(predicate: AsyncPredicate[T], iterable: AnyIterable[T]) -> int:
    ...


@overload
async def async_position_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T], default: U
) -> Union[int, U]:
    ...


async def async_position_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T], default: Any = no_default
) -> Any:
    index = await async_next_unchecked(async_position_all_await(predicate, iterable), None)

    if index is None:
        if default is no_default:
            raise ValueError(ASYNC_POSITION_NO_MATCH)

        return default

    return index


def async_find_all(predicate: Optional[Predicate[T]], iterable: AnyIterable[T]) -> AsyncIterator[T]:
    return async_filter(predicate, iterable)


def async_find_all_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    return async_filter_await(predicate, iterable)


ASYNC_FIND_NO_MATCH = "async_find() has not found any matches"
ASYNC_FIND_ON_EMPTY = "async_find() called on an empty iterable"


@overload
async def async_find(predicate: Optional[Predicate[T]], iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_find(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T], default: U
) -> Union[T, U]:
    ...


async def async_find(
    predicate: Optional[Predicate[Any]], iterable: AnyIterable[Any], default: Any = no_default
) -> Any:
    item = marker

    iterator = async_iter(iterable)

    if predicate is None:
        async for item in iterator:
            if item:
                return item

    else:
        async for item in iterator:
            if predicate(item):
                return item

    if default is no_default:
        raise ValueError(ASYNC_FIND_ON_EMPTY if item is marker else ASYNC_FIND_NO_MATCH)

    return default


ASYNC_FIND_AWAIT_NO_MATCH = "async_find_await() has not found any matches"
ASYNC_FIND_AWAIT_ON_EMPTY = "async_find_await() called on an empty iterable"


@overload
async def async_find_await(predicate: AsyncPredicate[T], iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_find_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T], default: U
) -> Union[T, U]:
    ...


async def async_find_await(
    predicate: AsyncPredicate[Any], iterable: AnyIterable[Any], default: Any = no_default
) -> Any:
    item = marker

    async for item in async_iter(iterable):
        if await predicate(item):
            return item

    if default is no_default:
        raise ValueError(ASYNC_FIND_ON_EMPTY if item is marker else ASYNC_FIND_NO_MATCH)

    return default


ASYNC_FIND_OR_FIRST_ON_EMPTY = "async_find_or_first() called on an empty iterable"


@overload
async def async_find_or_first(predicate: Optional[Predicate[T]], iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_find_or_first(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T], default: U
) -> Union[T, U]:
    ...


async def async_find_or_first(
    predicate: Optional[Predicate[Any]], iterable: AnyIterable[Any], default: Any = no_default
) -> Any:
    iterator = async_iter(iterable)

    first = await async_next_unchecked(iterator, marker)

    if first is marker:
        if default is no_default:
            raise ValueError(ASYNC_FIND_OR_FIRST_ON_EMPTY)

        first = default

    iterator = async_prepend(first, iterator)

    if predicate is None:
        async for item in iterator:
            if item:
                return item

    else:
        async for item in iterator:
            if predicate(item):
                return item

    return first


ASYNC_FIND_OR_FIRST_AWAIT_ON_EMPTY = "async_find_or_first_await() called on an empty iterable"


@overload
async def async_find_or_first_await(predicate: AsyncPredicate[T], iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_find_or_first_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T], default: U
) -> Union[T, U]:
    ...


async def async_find_or_first_await(
    predicate: AsyncPredicate[Any], iterable: AnyIterable[Any], default: Any = no_default
) -> Any:
    iterator = async_iter(iterable)

    first = await async_next_unchecked(iterator, marker)

    if first is marker:
        if default is no_default:
            raise ValueError(ASYNC_FIND_OR_FIRST_ON_EMPTY)

        first = default

    async for item in async_prepend(first, iterator):
        if await predicate(item):
            return item

    return first


ASYNC_FIND_OR_LAST_ON_EMPTY = "async_find_or_last() called on an empty iterable"


@overload
async def async_find_or_last(predicate: Optional[Predicate[T]], iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_find_or_last(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T], default: U
) -> Union[T, U]:
    ...


async def async_find_or_last(
    predicate: Optional[Predicate[Any]], iterable: AnyIterable[Any], default: Any = no_default
) -> Any:
    item = marker

    iterator = async_iter(iterable)

    if predicate is None:
        async for item in iterator:
            if item:
                return item

    else:
        async for item in iterator:
            if predicate(item):
                return item

    if item is marker:
        if default is no_default:
            raise ValueError(ASYNC_FIND_OR_LAST_ON_EMPTY)

        return default

    return item


ASYNC_FIND_OR_LAST_AWAIT_ON_EMPTY = "async_find_or_last_await() called on an empty iterable"


@overload
async def async_find_or_last_await(predicate: AsyncPredicate[T], iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_find_or_last_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T], default: U
) -> Union[T, U]:
    ...


async def async_find_or_last_await(
    predicate: AsyncPredicate[Any], iterable: AnyIterable[Any], default: Any = no_default
) -> Any:
    item = marker

    async for item in async_iter(iterable):
        if await predicate(item):
            return item

    if item is marker:
        if default is no_default:
            raise ValueError(ASYNC_FIND_OR_LAST_ON_EMPTY)

        return default

    return item


ASYNC_MIN_MAX_ON_EMPTY = "async_min_max() called on an empty iterable"


@overload
async def async_min_max(iterable: AnyIterable[ST], *, key: None = ...) -> Tuple[ST, ST]:
    ...


@overload
async def async_min_max(iterable: AnyIterable[T], *, key: Unary[T, ST]) -> Tuple[T, T]:
    ...


@overload
async def async_min_max(
    iterable: AnyIterable[ST], *, key: None = ..., default: Tuple[U, V]
) -> Union[Tuple[ST, ST], Tuple[U, V]]:
    ...


@overload
async def async_min_max(
    iterable: AnyIterable[T], *, key: Unary[T, ST], default: Tuple[U, V]
) -> Union[Tuple[T, T], Tuple[U, V]]:
    ...


async def async_min_max(
    iterable: AnyIterable[Any], *, key: Optional[Unary[Any, Any]] = None, default: Any = no_default
) -> Tuple[Any, Any]:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_MIN_MAX_ON_EMPTY)

        default_min, default_max = default

        return (default_min, default_max)

    if key is None:
        return await async_min_max_simple(iterator, result)

    return await async_min_max_by(iterator, result, key)


async def async_min_max_simple(iterable: AnyIterable[ST], value: ST) -> Tuple[ST, ST]:
    low = high = value

    async for item in async_iter(iterable):
        if item < low:  # type: ignore  # investigate
            low = item

        if high < item:  # type: ignore  # investigate
            high = item

    return (low, high)


async def async_min_max_by(iterable: AnyIterable[T], value: T, key: Unary[T, ST]) -> Tuple[T, T]:
    low = high = value
    low_key = high_key = key(value)

    async for item in async_iter(iterable):
        item_key = key(item)

        if item_key < low_key:  # type: ignore  # investigate
            low_key = item_key
            low = item

        if high_key < item_key:  # type: ignore  # investigate
            high_key = item_key
            high = item

    return (low, high)


ASYNC_MIN_MAX_AWAIT_ON_EMPTY = "async_min_max_await() called on an empty iterable"


@overload
async def async_min_max_await(iterable: AnyIterable[T], *, key: AsyncUnary[T, ST]) -> Tuple[T, T]:
    ...


@overload
async def async_min_max_await(
    iterable: AnyIterable[T], *, key: AsyncUnary[T, ST], default: Tuple[U, V]
) -> Union[Tuple[T, T], Tuple[U, V]]:
    ...


async def async_min_max_await(
    iterable: AnyIterable[Any], *, key: AsyncUnary[Any, Any], default: Any = no_default
) -> Tuple[Any, Any]:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_MIN_MAX_AWAIT_ON_EMPTY)

        default_min, default_max = default

        return (default_min, default_max)

    return await async_min_max_by_await(iterator, result, key)


async def async_min_max_by_await(
    iterable: AnyIterable[T], value: T, key: AsyncUnary[T, ST]
) -> Tuple[T, T]:
    low = high = value
    low_key = high_key = await key(value)

    async for item in async_iter(iterable):
        item_key = await key(item)

        if item_key < low_key:  # type: ignore  # investigate
            low_key = item_key
            low = item

        if high_key < item_key:  # type: ignore  # investigate
            high_key = item_key
            high = item

    return (low, high)


async def async_filter_except(
    validate: Unary[T, Any], iterable: AnyIterable[T], *errors: AnyExceptionType
) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        try:
            validate(item)

        except errors:
            pass

        else:
            yield item


async def async_filter_except_await(
    validate: AsyncUnary[T, Any], iterable: AnyIterable[T], *errors: AnyExceptionType
) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        try:
            await validate(item)

        except errors:
            pass

        else:
            yield item


async def async_map_except(
    function: Unary[T, U], iterable: AnyIterable[T], *errors: AnyExceptionType
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        try:
            yield function(item)

        except errors:
            pass


async def async_map_except_await(
    function: AsyncUnary[T, U], iterable: AnyIterable[T], *errors: AnyExceptionType
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        try:
            yield await function(item)

        except errors:
            pass


async def async_iter_except(function: Nullary[T], *errors: AnyExceptionType) -> AsyncIterator[T]:
    try:
        while True:
            yield function()

    except errors:
        pass


async def async_iter_except_await(
    function: AsyncNullary[T], *errors: AnyExceptionType
) -> AsyncIterator[T]:
    try:
        while True:
            yield await function()

    except errors:
        pass


ASYNC_LAST_WITH_TAIL_ON_EMPTY = "async_last_with_tail() called on an empty iterable"


@overload
async def async_last_with_tail(iterable: AnyIterable[T]) -> T:
    ...


@overload
async def async_last_with_tail(iterable: AnyIterable[T], default: U) -> Union[T, U]:
    ...


async def async_last_with_tail(iterable: AnyIterable[Any], default: Any = no_default) -> Any:
    iterator = async_tail(1, iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_LAST_WITH_TAIL_ON_EMPTY)

        return default

    return result


async def async_tail(size: int, iterable: AnyIterable[T]) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    window = deque(await async_list(async_take(size, iterator)), size)

    async for item in iterator:
        window.append(item)

    for item in window:
        yield item


@overload
async def async_is_sorted(
    iterable: AnyIterable[LT],
    key: None = ...,
    *,
    strict: Literal[False] = ...,
    reverse: bool = ...,
) -> bool:
    ...


@overload
async def async_is_sorted(
    iterable: AnyIterable[ST],
    key: None = ...,
    *,
    strict: Literal[True],
    reverse: bool = ...,
) -> bool:
    ...


@overload
async def async_is_sorted(
    iterable: AnyIterable[T],
    key: Unary[T, LT],
    *,
    strict: Literal[False] = ...,
    reverse: bool = ...,
) -> bool:
    ...


@overload
async def async_is_sorted(
    iterable: AnyIterable[T],
    key: Unary[T, ST],
    *,
    strict: Literal[True],
    reverse: bool = ...,
) -> bool:
    ...


async def async_is_sorted(
    iterable: AnyIterable[Any],
    key: Optional[Unary[Any, Any]] = None,
    *,
    strict: bool = False,
    reverse: bool = False,
) -> bool:
    if key is None:
        return await async_is_sorted_simple(iterable, strict=strict, reverse=reverse)

    return await async_is_sorted_by(iterable, key, strict=strict, reverse=reverse)


@overload
async def async_is_sorted_await(
    iterable: AnyIterable[T],
    key: AsyncUnary[T, LT],
    *,
    strict: Literal[False] = ...,
    reverse: bool = ...,
) -> bool:
    ...


@overload
async def async_is_sorted_await(
    iterable: AnyIterable[T],
    key: AsyncUnary[T, ST],
    *,
    strict: Literal[True],
    reverse: bool = ...,
) -> bool:
    ...


async def async_is_sorted_await(
    iterable: AnyIterable[Any],
    key: AsyncUnary[Any, Any],
    *,
    strict: bool = False,
    reverse: bool = False,
) -> bool:
    return await async_is_sorted_by_await(iterable, key, strict=strict, reverse=reverse)


async def async_is_sorted_simple(
    iterable: AnyIterable[Any], *, strict: bool = False, reverse: bool = False
) -> bool:
    compare = COMPARE[(strict, reverse)]
    return await async_all(async_map(unpack_binary(compare), async_pairs_windows(iterable)))


async def async_is_sorted_by(
    iterable: AnyIterable[Any],
    key: Unary[Any, Any],
    *,
    strict: bool = False,
    reverse: bool = False,
) -> bool:
    return await async_is_sorted_simple(async_map(key, iterable), strict=strict, reverse=reverse)


async def async_is_sorted_by_await(
    iterable: AnyIterable[Any],
    key: AsyncUnary[Any, Any],
    *,
    strict: bool = False,
    reverse: bool = False,
) -> bool:
    return await async_is_sorted_simple(
        async_map_await(key, iterable), strict=strict, reverse=reverse
    )


@overload
def async_sort(
    iterable: AnyIterable[ST], *, key: None = ..., reverse: bool = ...
) -> AsyncIterator[ST]:
    ...


@overload
def async_sort(
    iterable: AnyIterable[T], *, key: Unary[T, ST], reverse: bool = ...
) -> AsyncIterator[T]:
    ...


async def async_sort(
    iterable: AnyIterable[Any], *, key: Optional[Unary[Any, Any]] = None, reverse: bool = False
) -> AsyncIterator[Any]:
    for item in await async_sorted(iterable, key=key, reverse=reverse):
        yield item


async def async_sort_await(
    iterable: AnyIterable[T], *, key: AsyncUnary[T, ST], reverse: bool = False
) -> AsyncIterator[T]:
    for item in await async_sorted_await(iterable, key=key, reverse=reverse):
        yield item


async def async_list_windows(size: int, iterable: AnyIterable[T]) -> AsyncIterator[List[T]]:
    iterator = async_iter(iterable)

    window = deque(await async_list(async_take(size, iterator)), size)

    if len(window) == size:
        yield list(window)

    async for item in iterator:
        window.append(item)
        yield list(window)


@overload
def async_tuple_windows(size: Literal[0], iterable: AnyIterable[T]) -> AsyncIterator[EmptyTuple]:
    ...


@overload
def async_tuple_windows(size: Literal[1], iterable: AnyIterable[T]) -> AsyncIterator[Tuple1[T]]:
    ...


@overload
def async_tuple_windows(size: Literal[2], iterable: AnyIterable[T]) -> AsyncIterator[Tuple2[T]]:
    ...


@overload
def async_tuple_windows(size: Literal[3], iterable: AnyIterable[T]) -> AsyncIterator[Tuple3[T]]:
    ...


@overload
def async_tuple_windows(size: Literal[4], iterable: AnyIterable[T]) -> AsyncIterator[Tuple4[T]]:
    ...


@overload
def async_tuple_windows(size: Literal[5], iterable: AnyIterable[T]) -> AsyncIterator[Tuple5[T]]:
    ...


@overload
def async_tuple_windows(size: Literal[6], iterable: AnyIterable[T]) -> AsyncIterator[Tuple6[T]]:
    ...


@overload
def async_tuple_windows(size: Literal[7], iterable: AnyIterable[T]) -> AsyncIterator[Tuple7[T]]:
    ...


@overload
def async_tuple_windows(size: Literal[8], iterable: AnyIterable[T]) -> AsyncIterator[Tuple8[T]]:
    ...


@overload
def async_tuple_windows(size: int, iterable: AnyIterable[T]) -> AsyncIterator[DynamicTuple[T]]:
    ...


async def async_tuple_windows(
    size: int, iterable: AnyIterable[T]
) -> AsyncIterator[DynamicTuple[T]]:
    iterator = async_iter(iterable)

    window = deque(await async_list(async_take(size, iterator)), size)

    if len(window) == size:
        yield tuple(window)

    async for item in iterator:
        window.append(item)
        yield tuple(window)


def async_pairs_windows(iterable: AnyIterable[T]) -> AsyncIterator[Tuple[T, T]]:
    return async_tuple_windows(2, iterable)


async def async_iter_windows(
    size: int, iterable: AnyIterable[T]
) -> AsyncIterator[AsyncIterator[T]]:
    async for window in async_list_windows(size, iterable):
        yield iter_to_async_iter(window)


async def async_set_windows(size: int, iterable: AnyIterable[T]) -> AsyncIterator[Set[T]]:
    iterator = async_iter(iterable)

    window = deque(await async_list(async_take(size, iterator)), size)

    if len(window) == size:
        yield set(window)

    async for item in iterator:
        window.append(item)
        yield set(window)


async def async_side_effect(function: Unary[T, Any], iterable: AnyIterable[T]) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        function(item)
        yield item


async def async_side_effect_await(
    function: AsyncUnary[T, Any], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        await function(item)
        yield item


async def async_duplicates_fast_simple(iterable: AnyIterable[Q]) -> AsyncIterator[Q]:
    seen: Set[Q] = set()
    add_to_seen = seen.add

    async for item in async_iter(iterable):
        if item in seen:
            yield item

        else:
            add_to_seen(item)


async def async_duplicates_fast_by(iterable: AnyIterable[T], key: Unary[T, Q]) -> AsyncIterator[T]:
    seen_values: Set[Q] = set()
    add_to_seen_values = seen_values.add

    async for item in async_iter(iterable):
        value = key(item)

        if value in seen_values:
            yield item

        else:
            add_to_seen_values(value)


async def async_duplicates_fast_by_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, Q]
) -> AsyncIterator[T]:
    seen_values: Set[Q] = set()
    add_to_seen_values = seen_values.add

    async for item in async_iter(iterable):
        value = await key(item)

        if value in seen_values:
            yield item

        else:
            add_to_seen_values(value)


@overload
def async_duplicates_fast(iterable: AnyIterable[Q], key: None = ...) -> AsyncIterator[Q]:
    ...


@overload
def async_duplicates_fast(iterable: AnyIterable[T], key: Unary[T, Q]) -> AsyncIterator[T]:
    ...


def async_duplicates_fast(
    iterable: AnyIterable[Any], key: Optional[Unary[Any, Any]] = None
) -> AsyncIterator[Any]:
    return (
        async_duplicates_fast_simple(iterable)
        if key is None
        else async_duplicates_fast_by(iterable, key)
    )


def async_duplicates_fast_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, Q]
) -> AsyncIterator[T]:
    return async_duplicates_fast_by_await(iterable, key)


async def async_duplicates_simple(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    seen_set: Set[T] = set()
    add_to_seen_set = seen_set.add
    seen_list: List[T] = []
    add_to_seen_list = seen_list.append

    async for item in async_iter(iterable):
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


async def async_duplicates_by(iterable: AnyIterable[T], key: Unary[T, U]) -> AsyncIterator[T]:
    seen_values_set: Set[U] = set()
    add_to_seen_values_set = seen_values_set.add
    seen_values_list: List[U] = []
    add_to_seen_values_list = seen_values_list.append

    async for item in async_iter(iterable):
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


async def async_duplicates_by_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, U]
) -> AsyncIterator[T]:
    seen_values_set: Set[U] = set()
    add_to_seen_values_set = seen_values_set.add
    seen_values_list: List[U] = []
    add_to_seen_values_list = seen_values_list.append

    async for item in async_iter(iterable):
        value = await key(item)

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


def async_duplicates(
    iterable: AnyIterable[T], key: Optional[Unary[T, U]] = None
) -> AsyncIterator[T]:
    return async_duplicates_simple(iterable) if key is None else async_duplicates_by(iterable, key)


def async_duplicates_await(iterable: AnyIterable[T], key: AsyncUnary[T, U]) -> AsyncIterator[T]:
    return async_duplicates_by_await(iterable, key)


async def async_unique_fast_simple(iterable: AnyIterable[Q]) -> AsyncIterator[Q]:
    seen: Set[Q] = set()
    add_to_seen = seen.add

    async for item in async_iter(iterable):
        if item not in seen:
            add_to_seen(item)

            yield item


async def async_unique_fast_by(iterable: AnyIterable[T], key: Unary[T, Q]) -> AsyncIterator[T]:
    seen_values: Set[Q] = set()
    add_to_seen_values = seen_values.add

    async for item in async_iter(iterable):
        value = key(item)

        if value not in seen_values:
            add_to_seen_values(value)

            yield item


async def async_unique_fast_by_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, Q]
) -> AsyncIterator[T]:
    seen_values: Set[Q] = set()
    add_to_seen_values = seen_values.add

    async for item in async_iter(iterable):
        value = await key(item)

        if value not in seen_values:
            add_to_seen_values(value)

            yield item


@overload
def async_unique_fast(iterable: AnyIterable[Q], key: None = ...) -> AsyncIterator[Q]:
    ...


@overload
def async_unique_fast(iterable: AnyIterable[T], key: Unary[T, Q]) -> AsyncIterator[T]:
    ...


def async_unique_fast(
    iterable: AnyIterable[Any], key: Optional[Unary[Any, Any]] = None
) -> AsyncIterator[Any]:
    return (
        async_unique_fast_simple(iterable) if key is None else async_unique_fast_by(iterable, key)
    )


def async_unique_fast_await(iterable: AnyIterable[T], key: AsyncUnary[T, Q]) -> AsyncIterator[T]:
    return async_unique_fast_by_await(iterable, key)


async def async_unique_simple(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    seen_set: Set[T] = set()
    add_to_seen_set = seen_set.add
    seen_list: List[T] = []
    add_to_seen_list = seen_list.append

    async for item in async_iter(iterable):
        try:
            if item not in seen_set:
                add_to_seen_set(item)

                yield item

        except TypeError:
            if item not in seen_list:
                add_to_seen_list(item)

                yield item


async def async_unique_by(iterable: AnyIterable[T], key: Unary[T, U]) -> AsyncIterator[T]:
    seen_values_set: Set[U] = set()
    add_to_seen_values_set = seen_values_set.add
    seen_values_list: List[U] = []
    add_to_seen_values_list = seen_values_list.append

    async for item in async_iter(iterable):
        value = key(item)

        try:
            if value not in seen_values_set:
                add_to_seen_values_set(value)

                yield item

        except TypeError:
            if value not in seen_values_list:
                add_to_seen_values_list(value)

                yield item


async def async_unique_by_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, U]
) -> AsyncIterator[T]:
    seen_values_set: Set[U] = set()
    add_to_seen_values_set = seen_values_set.add
    seen_values_list: List[U] = []
    add_to_seen_values_list = seen_values_list.append

    async for item in async_iter(iterable):
        value = await key(item)

        try:
            if value not in seen_values_set:
                add_to_seen_values_set(value)

                yield item

        except TypeError:
            if value not in seen_values_list:
                add_to_seen_values_list(value)

                yield item


def async_unique(iterable: AnyIterable[T], key: Optional[Unary[T, U]] = None) -> AsyncIterator[T]:
    return async_unique_simple(iterable) if key is None else async_unique_by(iterable, key)


def async_unique_await(iterable: AnyIterable[T], key: AsyncUnary[T, U]) -> AsyncIterator[T]:
    return async_unique_by_await(iterable, key)


class StopAsyncZip(Exception):
    pass


async def async_next_or_stop_async_zip(iterator: AsyncIterator[T]) -> T:
    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        raise StopAsyncZip

    return result  # type: ignore


@overload
def async_zip() -> AsyncIterator[Never]:
    ...


@overload
def async_zip(__iterable_a: AnyIterable[A]) -> AsyncIterator[Tuple[A]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
) -> AsyncIterator[Tuple[A, B]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B], __iterable_c: AnyIterable[C]
) -> AsyncIterator[Tuple[A, B, C]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
) -> AsyncIterator[Tuple[A, B, C, D]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
) -> AsyncIterator[Tuple[A, B, C, D, E]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
) -> AsyncIterator[Tuple[A, B, C, D, E, F]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
) -> AsyncIterator[Tuple[A, B, C, D, E, F, G]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
    __iterable_h: AnyIterable[H],
) -> AsyncIterator[Tuple[A, B, C, D, E, F, G, H]]:
    ...


@overload
def async_zip(
    __iterable_a: AnyIterable[Any],
    __iterable_b: AnyIterable[Any],
    __iterable_c: AnyIterable[Any],
    __iterable_d: AnyIterable[Any],
    __iterable_e: AnyIterable[Any],
    __iterable_f: AnyIterable[Any],
    __iterable_g: AnyIterable[Any],
    __iterable_h: AnyIterable[Any],
    __iterable_next: AnyIterable[Any],
    *iterables: Iterable[Any],
) -> AsyncIterator[DynamicTuple[Any]]:
    ...


async def async_zip(*iterables: AnyIterable[Any]) -> AsyncIterator[DynamicTuple[Any]]:
    if not iterables:
        return

    iterators = tuple(map(async_iter, iterables))

    while True:
        try:
            yield await async_tuple(async_map_await(async_next_or_stop_async_zip, iterators))

        except StopAsyncZip:
            return


SINGULAR = " "
PLURAL = "s 1-"

SHORTER = "async_zip_equal() argument {short} is shorter than argument{plural}{index}"
LONGER = "async_zip_equal() argument {long} is longer than argument{plural}{index}"


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


@overload
def async_zip_equal() -> AsyncIterator[Never]:
    ...


@overload
def async_zip_equal(__iterable_a: AnyIterable[A]) -> AsyncIterator[Tuple[A]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
) -> AsyncIterator[Tuple[A, B]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B], __iterable_c: AnyIterable[C]
) -> AsyncIterator[Tuple[A, B, C]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
) -> AsyncIterator[Tuple[A, B, C, D]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
) -> AsyncIterator[Tuple[A, B, C, D, E]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
) -> AsyncIterator[Tuple[A, B, C, D, E, F]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
) -> AsyncIterator[Tuple[A, B, C, D, E, F, G]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
    __iterable_h: AnyIterable[H],
) -> AsyncIterator[Tuple[A, B, C, D, E, F, G, H]]:
    ...


@overload
def async_zip_equal(
    __iterable_a: AnyIterable[Any],
    __iterable_b: AnyIterable[Any],
    __iterable_c: AnyIterable[Any],
    __iterable_d: AnyIterable[Any],
    __iterable_e: AnyIterable[Any],
    __iterable_f: AnyIterable[Any],
    __iterable_g: AnyIterable[Any],
    __iterable_h: AnyIterable[Any],
    __iterable_next: AnyIterable[Any],
    *iterables: AnyIterable[Any],
) -> AsyncIterator[DynamicTuple[Any]]:
    ...


async def async_zip_equal(*iterables: AnyIterable[Any]) -> AsyncIterator[DynamicTuple[Any]]:
    if not iterables:
        return  # early return

    async for item in async_zip_longest(*iterables, fill=marker):  # check for length
        head, *tail = item

        if head is marker:  # argument longer than previous arguments
            for index, value in enumerate(tail, 1):
                if value is not marker:
                    raise ValueError(format_longer(index))

        else:  # argument shorter than previous ones
            for index, value in enumerate(tail, 1):
                if value is marker:
                    raise ValueError(format_shorter(index))

        yield item  # simply yield if everything is alright


class StopAsyncZipLongest(Exception):
    pass


@overload
def async_zip_longest() -> AsyncIterator[Never]:
    ...


@overload
def async_zip_longest(__iterable_a: AnyIterable[A]) -> AsyncIterator[Tuple[A]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
) -> AsyncIterator[Tuple[Optional[A], Optional[B]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B], __iterable_c: AnyIterable[C]
) -> AsyncIterator[Tuple[Optional[A], Optional[B], Optional[C]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
) -> AsyncIterator[Tuple[Optional[A], Optional[B], Optional[C], Optional[D]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
) -> AsyncIterator[Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
) -> AsyncIterator[
    Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E], Optional[F]]
]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
) -> AsyncIterator[
    Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E], Optional[F], Optional[G]]
]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
    __iterable_h: AnyIterable[H],
) -> AsyncIterator[
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
def async_zip_longest(
    __iterable_a: AnyIterable[Any],
    __iterable_b: AnyIterable[Any],
    __iterable_c: AnyIterable[Any],
    __iterable_d: AnyIterable[Any],
    __iterable_e: AnyIterable[Any],
    __iterable_f: AnyIterable[Any],
    __iterable_g: AnyIterable[Any],
    __iterable_h: AnyIterable[Any],
    __iterable_next: AnyIterable[Any],
    *iterables: AnyIterable[Any],
) -> AsyncIterator[DynamicTuple[Optional[Any]]]:
    ...


@overload
def async_zip_longest(*, fill: T) -> AsyncIterator[Never]:
    ...


@overload
def async_zip_longest(__iterable_a: AnyIterable[A], *, fill: T) -> AsyncIterator[Tuple[A]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B], *, fill: T
) -> AsyncIterator[Tuple[Union[A, T], Union[B, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    *,
    fill: T,
) -> AsyncIterator[Tuple[Union[A, T], Union[B, T], Union[C, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    *,
    fill: T,
) -> AsyncIterator[Tuple[Union[A, T], Union[B, T], Union[C, T], Union[D, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    *,
    fill: T,
) -> AsyncIterator[Tuple[Union[A, T], Union[B, T], Union[C, T], Union[D, T], Union[E, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    *,
    fill: T,
) -> AsyncIterator[
    Tuple[Union[A, T], Union[B, T], Union[C, T], Union[D, T], Union[E, T], Union[F, T]]
]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
    *,
    fill: T,
) -> AsyncIterator[
    Tuple[Union[A, T], Union[B, T], Union[C, T], Union[D, T], Union[E, T], Union[F, T], Union[G, T]]
]:
    ...


@overload
def async_zip_longest(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
    __iterable_h: AnyIterable[H],
    *,
    fill: T,
) -> AsyncIterator[
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
def async_zip_longest(
    __iterable_a: AnyIterable[Any],
    __iterable_b: AnyIterable[Any],
    __iterable_c: AnyIterable[Any],
    __iterable_d: AnyIterable[Any],
    __iterable_e: AnyIterable[Any],
    __iterable_f: AnyIterable[Any],
    __iterable_g: AnyIterable[Any],
    __iterable_h: AnyIterable[Any],
    __iterable_next: AnyIterable[Any],
    *iterables: AnyIterable[Any],
    fill: T,
) -> AsyncIterator[DynamicTuple[Union[Any, T]]]:
    ...


async def async_zip_longest(
    *iterables: AnyIterable[Any], fill: Optional[Any] = None
) -> AsyncIterator[DynamicTuple[Any]]:
    if not iterables:
        return

    remain = len(iterables) - 1

    async def sentinel(fill: Optional[Any]) -> AsyncIterator[Optional[Any]]:
        nonlocal remain

        if not remain:
            raise StopAsyncZipLongest

        remain -= 1

        yield fill

    fillers = async_repeat(fill)

    iterators = [async_chain(iterable, sentinel(fill), fillers) for iterable in iterables]

    try:
        while True:
            yield await async_tuple(async_map_await(async_next_unchecked, iterators))

    except StopAsyncZipLongest:
        pass


async def async_chain(*iterables: AnyIterable[T]) -> AsyncIterator[T]:
    for iterable in iterables:
        async for item in async_iter(iterable):
            yield item


async def async_chain_from_iterable(nested: AnyIterable[AnyIterable[T]]) -> AsyncIterator[T]:
    async for iterable in async_iter(nested):
        async for item in async_iter(iterable):
            yield item


async def async_wait(iterable: AnyIterable[Awaitable[T]]) -> AsyncIterator[T]:
    if is_async_iterable(iterable):
        async for awaitable in iterable:
            yield await awaitable  # type: ignore

    elif is_iterable(iterable):
        for awaitable in iterable:
            yield await awaitable  # type: ignore

    else:
        raise not_any_iterable(iterable)


if CONCURRENT:

    async def async_wait_concurrent(iterable: AnyIterable[Awaitable[T]]) -> AsyncIterator[T]:
        awaitables: Iterable[Awaitable[T]]

        if is_async_iterable(iterable):
            awaitables = await async_extract(iterable)  # type: ignore

        elif is_iterable(iterable):
            awaitables = iterable  # type: ignore

        else:
            raise not_any_iterable(iterable)

        results = await collect_iterable(awaitables)

        for result in results:
            yield result

    def async_wait_concurrent_bound(
        bound: int, iterable: AnyIterable[Awaitable[T]]
    ) -> AsyncIterator[T]:
        return async_flat_map(async_wait_concurrent, async_chunks(bound, iterable))

    def async_map_concurrent(
        function: AsyncUnary[T, U], iterable: AnyIterable[T]
    ) -> AsyncIterator[U]:
        return async_wait_concurrent(async_map(function, iterable))

    def async_map_concurrent_bound(
        bound: int, function: AsyncUnary[T, U], iterable: AnyIterable[T]
    ) -> AsyncIterator[U]:
        return async_wait_concurrent_bound(bound, async_map(function, iterable))


async def async_all(iterable: AnyIterable[T]) -> bool:
    async for item in async_iter(iterable):
        if not item:
            return False

    return True


async def async_any(iterable: AnyIterable[T]) -> bool:
    async for item in async_iter(iterable):
        if item:
            return True

    return False


async def async_count(start: int = 0, step: int = 1) -> AsyncIterator[int]:
    value = start

    while True:
        yield value

        value += step


async def async_enumerate(iterable: AnyIterable[T], start: int = 0) -> AsyncIterator[Tuple[int, T]]:
    value = start

    async for item in async_iter(iterable):
        yield (value, item)

        value += 1


async def async_filter(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    if predicate is None:
        async for item in iterator:
            if item:
                yield item

    else:
        async for item in iterator:
            if predicate(item):
                yield item


async def async_filter_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        if await predicate(item):
            yield item


async def async_filter_false(
    predicate: Optional[Predicate[T]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    if predicate is None:
        async for item in iterator:
            if not item:
                yield item

    else:
        async for item in iterator:
            if not predicate(item):
                yield item


async def async_filter_false_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        if not await predicate(item):
            yield item


ASYNC_MAX_ON_EMPTY = "async_max() called on an empty iterable"


@overload
async def async_max(iterable: AnyIterable[ST], *, key: None = ...) -> ST:
    ...


@overload
async def async_max(iterable: AnyIterable[T], *, key: Unary[T, ST]) -> T:
    ...


@overload
async def async_max(iterable: AnyIterable[ST], *, key: None = ..., default: U) -> Union[ST, U]:
    ...


@overload
async def async_max(iterable: AnyIterable[T], *, key: Unary[T, ST], default: U) -> Union[T, U]:
    ...


async def async_max(
    iterable: AnyIterable[Any],
    *,
    key: Optional[Unary[Any, Any]] = None,
    default: Any = no_default,
) -> Any:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_MAX_ON_EMPTY)

        return default

    if key is None:
        return await async_max_simple(iterator, result)

    return await async_max_by(iterator, result, key)


async def async_max_simple(iterable: AnyIterable[ST], value: ST) -> ST:
    async for item in async_iter(iterable):
        if value < item:  # type: ignore  # investigate
            value = item

    return value


async def async_max_by(iterable: AnyIterable[T], value: T, key: Unary[T, ST]) -> T:
    value_key = key(value)

    async for item in async_iter(iterable):
        item_key = key(item)

        if value_key < item_key:  # type: ignore  # investigate
            value = item
            value_key = item_key

    return value


ASYNC_MAX_AWAIT_ON_EMPTY = "async_max_await() called on an empty iterable"


@overload
async def async_max_await(iterable: AnyIterable[T], key: AsyncUnary[T, ST]) -> T:
    ...


@overload
async def async_max_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, ST], default: U
) -> Unary[T, U]:
    ...


async def async_max_await(
    iterable: AnyIterable[Any], key: AsyncUnary[Any, Any], default: Any = no_default
) -> Any:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_MAX_AWAIT_ON_EMPTY)

        return default

    return await async_max_by_await(iterator, result, key)


async def async_max_by_await(iterable: AnyIterable[T], value: T, key: AsyncUnary[T, ST]) -> T:
    value_key = await key(value)

    async for item in async_iter(iterable):
        item_key = await key(item)

        if value_key < item_key:  # type: ignore  # investigate
            value = item
            value_key = item_key

    return value


ASYNC_MIN_ON_EMPTY = "async_min() called on an empty iterable"


@overload
async def async_min(iterable: AnyIterable[ST], *, key: None = ...) -> ST:
    ...


@overload
async def async_min(iterable: AnyIterable[T], *, key: Unary[T, ST]) -> T:
    ...


@overload
async def async_min(iterable: AnyIterable[ST], *, key: None = ..., default: U) -> Union[ST, U]:
    ...


@overload
async def async_min(iterable: AnyIterable[T], *, key: Unary[T, ST], default: U) -> Union[T, U]:
    ...


async def async_min(
    iterable: AnyIterable[Any],
    *,
    key: Optional[Unary[Any, Any]] = None,
    default: Any = no_default,
) -> Any:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_MIN_ON_EMPTY)

        return default

    if key is None:
        return await async_min_simple(iterator, result)

    return await async_min_by(iterator, result, key)


async def async_min_simple(iterable: AnyIterable[ST], value: ST) -> ST:
    async for item in async_iter(iterable):
        if item < value:  # type: ignore  # investigate
            value = item

    return value


async def async_min_by(iterable: AnyIterable[T], value: T, key: Unary[T, ST]) -> T:
    value_key = key(value)

    async for item in async_iter(iterable):
        item_key = key(item)

        if item_key < value_key:  # type: ignore  # investigate
            value = item
            value_key = item_key

    return value


ASYNC_MIN_AWAIT_ON_EMPTY = "async_min_await() called on an empty iterable"


@overload
async def async_min_await(iterable: AnyIterable[T], key: AsyncUnary[T, ST]) -> T:
    ...


@overload
async def async_min_await(
    iterable: AnyIterable[T], key: AsyncUnary[T, ST], default: U
) -> Union[T, U]:
    ...


async def async_min_await(
    iterable: AnyIterable[Any], key: AsyncUnary[Any, Any], default: Any = no_default
) -> Any:
    iterator = async_iter(iterable)

    result = await async_next_unchecked(iterator, marker)

    if result is marker:
        if default is no_default:
            raise ValueError(ASYNC_MIN_AWAIT_ON_EMPTY)

        return default

    return await async_min_by_await(iterator, result, key)


async def async_min_by_await(iterable: AnyIterable[T], value: T, key: AsyncUnary[T, ST]) -> T:
    value_key = await key(value)

    async for item in async_iter(iterable):
        item_key = await key(item)

        if item_key < value_key:  # type: ignore  # investigate
            value = item
            value_key = item_key

    return value


@overload
def standard_async_map(function: Unary[T, R], __iterable_t: AnyIterable[T]) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map(
    function: Binary[T, U, R], __iterable_t: AnyIterable[T], __iterable_u: AnyIterable[U]
) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map(
    function: Ternary[T, U, V, R],
    __iterable_t: AnyIterable[T],
    __iterable_u: AnyIterable[U],
    __iterable_v: AnyIterable[V],
) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map(
    function: Quaternary[T, U, V, W, R],
    __iterable_t: AnyIterable[T],
    __iterable_u: AnyIterable[U],
    __iterable_v: AnyIterable[V],
    __iterable_w: AnyIterable[W],
) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map(
    function: DynamicCallable[R],
    __iterable_t: AnyIterable[Any],
    __iterable_u: AnyIterable[Any],
    __iterable_v: AnyIterable[Any],
    __iterable_w: AnyIterable[Any],
    __iterable_next: AnyIterable[Any],
    *iterables: AnyIterable[Any],
) -> AsyncIterator[R]:
    ...


async def standard_async_map(
    function: DynamicCallable[R], *iterables: AnyIterable[Any]
) -> AsyncIterator[R]:
    async for args in async_zip(*iterables):
        yield function(*args)


@overload
def standard_async_map_await(
    function: AsyncUnary[T, R], __iterable_t: AnyIterable[T]
) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map_await(
    function: AsyncBinary[T, U, R],
    __iterable_t: AnyIterable[T],
    __iterable_u: AnyIterable[U],
) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map_await(
    function: AsyncTernary[T, U, V, R],
    __iterable_t: AnyIterable[T],
    __iterable_u: AnyIterable[U],
    __iterable_v: AnyIterable[V],
) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map_await(
    function: AsyncQuaternary[T, U, V, W, R],
    __iterable_t: AnyIterable[T],
    __iterable_u: AnyIterable[U],
    __iterable_v: AnyIterable[V],
    __iterable_w: AnyIterable[W],
) -> AsyncIterator[R]:
    ...


@overload
def standard_async_map_await(
    function: AsyncDynamicCallable[R],
    __iterable_t: AnyIterable[Any],
    __iterable_u: AnyIterable[Any],
    __iterable_v: AnyIterable[Any],
    __iterable_w: AnyIterable[Any],
    __iterable_next: AnyIterable[Any],
    *iterables: AnyIterable[Any],
) -> AsyncIterator[R]:
    ...


async def standard_async_map_await(
    function: AsyncDynamicCallable[R], *iterables: AnyIterable[Any]
) -> AsyncIterator[R]:
    async for args in async_zip(*iterables):
        yield await function(*args)


async def async_map(function: Unary[T, U], iterable: AnyIterable[T]) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        yield function(item)


async def async_map_await(function: AsyncUnary[T, U], iterable: AnyIterable[T]) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        yield await function(item)


async def async_compress(iterable: AnyIterable[T], selectors: AnySelectors) -> AsyncIterator[T]:
    async for item, keep in async_zip(iterable, selectors):
        if keep:
            yield item


async def async_cycle(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    saved: List[T] = []

    async for item in async_iter(iterable):
        yield item
        saved.append(item)

    if not saved:
        return

    while True:
        for item in saved:
            yield item


async def async_drop_while(predicate: Predicate[T], iterable: AnyIterable[T]) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    async for item in iterator:
        if not predicate(item):
            yield item
            break

    async for item in iterator:
        yield item


async_skip_while = async_drop_while


async def async_drop_while_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    iterator = async_iter(iterable)

    async for item in iterator:
        if not await predicate(item):
            yield item
            break

    async for item in iterator:
        yield item


async_skip_while_await = async_drop_while_await


async def async_take_while(predicate: Predicate[T], iterable: AnyIterable[T]) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        if predicate(item):
            yield item

        else:
            break


async def async_take_while_await(
    predicate: AsyncPredicate[T], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for item in async_iter(iterable):
        if await predicate(item):
            yield item

        else:
            break


def async_reverse(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    try:
        return async_reversed(iterable)  # type: ignore

    except TypeError:
        return async_reverse_with_list(iterable)


def async_reversed(iterable: Reversible[T]) -> AsyncIterator[T]:
    return iter_to_async_iter(reversed(iterable))


async def async_reverse_with_list(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    array = await async_list(async_iter(iterable))

    for item in reversed(array):
        yield item


@overload
def async_cartesian_product() -> AsyncIterator[EmptyTuple]:
    ...


@overload
def async_cartesian_product(__iterable_a: AnyIterable[A]) -> AsyncIterator[Tuple[A]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
) -> AsyncIterator[Tuple[A, B]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B], __iterable_c: AnyIterable[C]
) -> AsyncIterator[Tuple[A, B, C]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
) -> AsyncIterator[Tuple[A, B, C, D]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
) -> AsyncIterator[Tuple[A, B, C, D, E]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
) -> AsyncIterator[Tuple[A, B, C, D, E, F]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
) -> AsyncIterator[Tuple[A, B, C, D, E, F, G]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[A],
    __iterable_b: AnyIterable[B],
    __iterable_c: AnyIterable[C],
    __iterable_d: AnyIterable[D],
    __iterable_e: AnyIterable[E],
    __iterable_f: AnyIterable[F],
    __iterable_g: AnyIterable[G],
    __iterable_h: AnyIterable[H],
) -> AsyncIterator[Tuple[A, B, C, D, E, F, G, H]]:
    ...


@overload
def async_cartesian_product(
    __iterable_a: AnyIterable[Any],
    __iterable_b: AnyIterable[Any],
    __iterable_c: AnyIterable[Any],
    __iterable_d: AnyIterable[Any],
    __iterable_e: AnyIterable[Any],
    __iterable_f: AnyIterable[Any],
    __iterable_g: AnyIterable[Any],
    __iterable_h: AnyIterable[Any],
    __iterable_next: AnyIterable[Any],
    *iterables: AnyIterable[Any],
) -> AsyncIterator[DynamicTuple[Any]]:
    ...


def async_cartesian_product(*iterables: AnyIterable[Any]) -> AsyncIterator[DynamicTuple[Any]]:
    stack = async_once(())

    for iterable in iterables:
        stack = async_cartesian_product_step(stack, iterable)  # type: ignore

    return stack


Ts = TypeVarTuple("Ts")  # type: ignore


async def async_cartesian_product_step(
    stack: AnyIterable[Tuple[Unpack[Ts]]], iterable: AnyIterable[T]  # type: ignore
) -> AsyncIterator[Tuple[Unpack[Ts], T]]:  # type: ignore
    array = await async_list(iterable)

    async for items in async_iter(stack):
        for item in array:
            yield items + tuple_args(item)


@overload
def async_cartesian_power(power: Literal[0], iterable: AnyIterable[T]) -> AsyncIterator[EmptyTuple]:
    ...


@overload
def async_cartesian_power(power: Literal[1], iterable: AnyIterable[T]) -> AsyncIterator[Tuple1[T]]:
    ...


@overload
def async_cartesian_power(power: Literal[2], iterable: AnyIterable[T]) -> AsyncIterator[Tuple2[T]]:
    ...


@overload
def async_cartesian_power(power: Literal[3], iterable: AnyIterable[T]) -> AsyncIterator[Tuple3[T]]:
    ...


@overload
def async_cartesian_power(power: Literal[4], iterable: AnyIterable[T]) -> AsyncIterator[Tuple4[T]]:
    ...


@overload
def async_cartesian_power(power: Literal[5], iterable: AnyIterable[T]) -> AsyncIterator[Tuple5[T]]:
    ...


@overload
def async_cartesian_power(power: Literal[6], iterable: AnyIterable[T]) -> AsyncIterator[Tuple6[T]]:
    ...


@overload
def async_cartesian_power(power: Literal[7], iterable: AnyIterable[T]) -> AsyncIterator[Tuple7[T]]:
    ...


@overload
def async_cartesian_power(power: Literal[8], iterable: AnyIterable[T]) -> AsyncIterator[Tuple8[T]]:
    ...


@overload
def async_cartesian_power(power: int, iterable: AnyIterable[T]) -> AsyncIterator[DynamicTuple[T]]:
    ...


def async_cartesian_power(power: int, iterable: AnyIterable[T]) -> AsyncIterator[DynamicTuple[T]]:
    state = None

    async def async_cartesian_power_step(
        stack: AnyIterable[Tuple[Unpack[Ts]]],  # type: ignore
    ) -> AsyncIterator[Tuple[Unpack[Ts], T]]:  # type: ignore
        nonlocal state

        if state is None:
            state = await async_list(iterable)

        async for items in async_iter(stack):
            for item in state:
                yield items + tuple_args(item)

    stack = async_once(())

    for _ in range(power):
        stack = async_cartesian_power_step(stack)  # type: ignore

    return stack


Args = TypeVarTuple("Args")  # type: ignore


def tuple_args(*args: Unpack[Args]) -> Tuple[Unpack[Args]]:  # type: ignore
    return args  # type: ignore
