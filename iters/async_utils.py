import asyncio

from inspect import isawaitable as is_awaitable, iscoroutinefunction as is_coroutine_function
from operator import add, attrgetter as get_attr_factory, mul
from sys import maxsize as max_size
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Reversible,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
    overload,
)

from typing_extensions import AsyncContextManager

from iters.types import Marker, MarkerOr, Order, marker
from iters.utils import first, last

__all__ = (
    "AnyIterable",
    "AnyIterator",
    "MaybeAwaitable",
    "async_all",
    "async_any",
    "async_append",
    "async_at",
    "async_at_or_last",
    "async_chain",
    "async_chain_from_iterable",
    "async_collapse",
    "async_compress",
    "async_copy",
    "async_copy_infinite",
    "async_copy_safe",
    "async_count",
    "async_cycle",
    "async_dict",
    "async_distinct",
    "async_drop",
    "async_drop_while",
    "async_enumerate",
    "async_exhaust",
    "async_filter",
    "async_filter_nowait",
    "async_filter_false",
    "async_filter_false_nowait",
    "async_first",
    "async_flatten",
    "async_fold",
    "async_get",
    "async_group",
    "async_group_longest",
    "async_iter",
    "async_iter_any_iter",
    "async_iter_chunk",
    "async_iter_len",
    "async_iter_slice",
    "async_iterate",
    "async_last",
    "async_list",
    "async_list_chunk",
    "async_map",
    "async_map_nowait",
    "async_max",
    "async_min",
    "async_next",
    "async_next_unchecked",
    "async_parallel_filter",
    "async_parallel_filter_false",
    "async_parallel_flatten",
    "async_parallel_map",
    "async_parallel_star_map",
    "async_parallel_wait",
    "async_partition",
    "async_partition_infinite",
    "async_partition_safe",
    "async_prepend",
    "async_product",
    "async_repeat",
    "async_reversed",
    "async_set",
    "async_side_effect",
    "async_star_map",
    "async_star_map_nowait",
    "async_step_by",
    "async_sum",
    "async_take",
    "async_tuple",
    "async_tuple_chunk",
    "async_wait",
    "async_with_async_iter",
    "async_with_iter",
    "async_zip",
    "async_zip_longest",
    "iter_async_function",
    "iter_async_iter",
    "iter_sync_function",
    "iter_to_async_iter",
    "maybe_await",
    "reverse_to_async",
    "run_iterators",
)

KT = TypeVar("KT")
VT = TypeVar("VT")

N = TypeVar("N", int, float)

R = TypeVar("R")
T = TypeVar("T")
U = TypeVar("U")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

AnyIterable = Union[AsyncIterable[T], Iterable[T]]
AnyIterator = Union[AsyncIterator[T], Iterator[T]]
MaybeAwaitable = Union[T, Awaitable[T]]

OrderT = TypeVar("OrderT", bound=Order)


async def maybe_await(value: MaybeAwaitable[R]) -> R:
    if is_awaitable(value):
        return await cast(Awaitable[R], value)

    return cast(R, value)


async def async_wait(iterable: AnyIterable[MaybeAwaitable[T]]) -> AsyncIterator[T]:
    async for element in async_iter_any_iter(iterable):
        yield await maybe_await(element)


async def async_parallel_wait(iterable: AnyIterable[MaybeAwaitable[T]]) -> AsyncIterator[T]:
    coroutines = [maybe_await(element) async for element in async_iter_any_iter(iterable)]

    results = await asyncio.gather(*coroutines)

    for result in results:
        yield result


@overload
def async_iter(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    ...


@overload
def async_iter(function: Callable[[], T], sentinel: T) -> AsyncIterator[T]:
    ...


@overload
def async_iter(async_function: Callable[[], Awaitable[T]], sentinel: T) -> AsyncIterator[T]:
    ...


@no_type_check
def async_iter(
    something: Union[AsyncIterable[T], Iterable[T], Callable[[], T], Callable[[], Awaitable[T]]],
    sentinel: MarkerOr[T] = marker,
) -> AsyncIterator[T]:
    if isinstance(something, AsyncIterable):
        return iter_async_iter(something)

    elif isinstance(something, Iterable):
        return iter_to_async_iter(something)

    elif callable(something):
        if is_coroutine_function(something):
            return iter_async_function(something, sentinel)

        return iter_sync_function(something, sentinel)

    else:
        raise TypeError(
            "Expected iterable, async iterable, function or async function, "
            f"got {type(something).__name__!r}."
        )


def async_iter_any_iter(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    if isinstance(iterable, AsyncIterable):
        return iter_async_iter(iterable)

    elif isinstance(iterable, Iterable):
        return iter_to_async_iter(iterable)

    else:
        raise TypeError(f"Expected iterable or async iterable, got {type(iterable).__name__!r}.")


async def async_next(iterator: AnyIterator[T], default: MarkerOr[T] = marker) -> T:
    if isinstance(iterator, AsyncIterator):
        try:
            return await iterator.__anext__()

        except StopAsyncIteration:
            if default is marker:
                raise

            return cast(T, default)

    return next(iterator)


async def iter_to_async_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    for element in iterable:
        yield element


async def async_next_unchecked(
    async_iterator: AsyncIterator[T], default: MarkerOr[T] = marker
) -> T:
    try:
        return await async_iterator.__anext__()

    except StopAsyncIteration:
        if default is marker:
            raise

        return cast(T, default)


def iter_async_iter(async_iterable: AsyncIterable[T]) -> AsyncIterator[T]:
    return async_iterable.__aiter__()


async def iter_sync_function(function: Callable[[], T], sentinel: T) -> AsyncIterator[T]:
    return iter_to_async_iter(iter(function, sentinel))


async def iter_async_function(
    async_function: Callable[[], Awaitable[T]], sentinel: T
) -> AsyncIterator[T]:
    while True:
        value = await async_function()

        if value == sentinel:
            break

        yield value


async def async_all(iterable: AnyIterable[T]) -> bool:
    async for element in async_iter_any_iter(iterable):
        if not element:
            return False
    return True


async def async_any(iterable: AnyIterable[T]) -> bool:
    async for element in async_iter_any_iter(iterable):
        if element:
            return True
    return False


async def async_count(start: N = 0, step: N = 1) -> AsyncIterator[N]:
    value = start

    while True:
        yield value

        value += step


async def async_with_async_iter(
    async_context_manager: AsyncContextManager[AnyIterable[T]],
) -> AsyncIterator[T]:
    async with async_context_manager as iterable:
        async for element in async_iter_any_iter(iterable):
            yield element


def async_with_iter(context_manager: ContextManager[AnyIterable[T]]) -> AsyncIterator[T]:
    with context_manager as iterable:
        return async_iter_any_iter(iterable)


def reverse_to_async(iterable: Reversible[T]) -> AsyncIterator[T]:
    return iter_to_async_iter(reversed(iterable))


async def async_reversed(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    # async because we need to await on async_list(iterable) in order to be able to reverse

    array = await async_list(async_iter_any_iter(iterable))

    async for element in reverse_to_async(array):
        yield element


@overload
async def async_reduce(
    function: Callable[[T, T], MaybeAwaitable[T]],
    iterable: AnyIterable[T],
    initial: Marker = marker,
) -> T:
    ...


@overload
async def async_reduce(
    function: Callable[[T, U], MaybeAwaitable[T]],
    iterable: AnyIterable[U],
    initial: T,
) -> T:
    ...


async def async_reduce(
    function: Callable[[Any, Any], MaybeAwaitable[Any]],
    iterable: AnyIterable[Any],
    initial: MarkerOr[Any] = marker,
) -> Any:
    async_iterator = async_iter_any_iter(iterable)

    if initial is marker:
        try:
            value = await async_next_unchecked(async_iterator)

        except StopAsyncIteration:
            raise TypeError("reduce() of empty iterable with no initial value") from None

    async for element in async_iterator:
        value = await maybe_await(function(value, element))

    return value


async def async_product(iterable: AnyIterable[T], start: MarkerOr[T] = marker) -> T:
    if start is marker:
        start = cast(T, 0)

    return await async_reduce(mul, iterable, start)


async def async_sum(iterable: AnyIterable[T], start: MarkerOr[T] = marker) -> T:
    if start is marker:
        start = cast(T, 0)

    return await async_reduce(add, iterable, start)


async def async_iter_len(iterable: AnyIterable[T]) -> int:
    length = 0

    async for _ in async_iter_any_iter(iterable):
        length += 1

    return length


async def async_dict(iterable: AnyIterable[Tuple[KT, VT]]) -> Dict[KT, VT]:
    return {key: value async for key, value in async_iter_any_iter(iterable)}


async def async_list(iterable: AnyIterable[T]) -> List[T]:
    return [element async for element in async_iter_any_iter(iterable)]


async def async_set(iterable: AnyIterable[T]) -> Set[T]:
    return {element async for element in async_iter_any_iter(iterable)}


async def async_tuple(iterable: AnyIterable[T]) -> Tuple[T, ...]:
    return tuple(await async_list(iterable))


async def async_enumerate(iterable: AnyIterable[T], start: int = 0) -> AsyncIterator[Tuple[int, T]]:
    if not isinstance(start, int):
        raise TypeError(f"Expected start to be integer, got {type(start).__name__!r}.")

    number = start

    async for element in async_iter_any_iter(iterable):
        yield number, element

        number += 1


async def async_filter(
    predicate: Callable[[T], MaybeAwaitable[Any]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for element in async_iter_any_iter(iterable):
        if await maybe_await(predicate(element)):
            yield element


async def async_filter_nowait(
    predicate: Callable[[T], Any], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for element in async_iter_any_iter(iterable):
        if predicate(element):
            yield element


async def async_parallel_filter(
    predicate: Callable[[T], MaybeAwaitable[Any]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    elements = await async_list(async_iter_any_iter(iterable))

    coroutines = (maybe_await(predicate(element)) for element in elements)
    results = await asyncio.gather(*coroutines)

    async for element, result in async_zip(elements, results):
        if result:
            yield element


async def async_filter_false(
    predicate: Callable[[T], MaybeAwaitable[Any]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for element in async_iter_any_iter(iterable):
        if not await maybe_await(predicate(element)):
            yield element


async def async_filter_false_nowait(
    predicate: Callable[[T], Any], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for element in async_iter_any_iter(iterable):
        if not predicate(element):
            yield element


async def async_parallel_filter_false(
    predicate: Callable[[T], MaybeAwaitable[Any]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    elements = await async_list(async_iter_any_iter(iterable))

    coroutines = (maybe_await(predicate(element)) for element in elements)
    results = await asyncio.gather(*coroutines)

    async for element, result in async_zip(elements, results):
        if not result:
            yield element


async def async_map(
    function: Callable[[T], MaybeAwaitable[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for element in async_iter_any_iter(iterable):
        yield await maybe_await(function(element))


async def async_map_nowait(
    function: Callable[[T], U], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for element in async_iter_any_iter(iterable):
        yield function(element)


async def async_parallel_map(
    function: Callable[[T], MaybeAwaitable[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    coroutines = [maybe_await(function(element)) async for element in async_iter_any_iter(iterable)]
    results = await asyncio.gather(*coroutines)

    for result in results:
        yield result


async def async_star_map(
    function: Callable[..., MaybeAwaitable[T]], iterable: AnyIterable[AnyIterable[Any]]
) -> AsyncIterator[T]:
    async for args_iterable in async_iter_any_iter(iterable):
        args = await async_list(async_iter_any_iter(args_iterable))

        yield await maybe_await(function(*args))


async def async_star_map_nowait(
    function: Callable[..., T], iterable: AnyIterable[AnyIterable[Any]]
) -> AsyncIterator[T]:
    async for args_iterable in async_iter_any_iter(iterable):
        args = await async_list(async_iter_any_iter(args_iterable))

        yield function(*args)


async def async_parallel_star_map(
    function: Callable[..., MaybeAwaitable[T]], iterable: AnyIterable[AnyIterable[Any]]
) -> AsyncIterator[T]:
    args_list = [
        await async_list(async_iter_any_iter(args_iterable))
        async for args_iterable in async_iter_any_iter(iterable)
    ]
    coroutines = (maybe_await(function(*args)) for args in args_list)

    results = await asyncio.gather(*coroutines)

    for result in results:
        yield result


async def async_cycle(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    saved = []

    async for element in async_iter_any_iter(iterable):
        yield element
        saved.append(element)

    if not saved:
        return

    while True:
        for element in saved:
            yield element


async def async_repeat(value: T, times: Optional[int] = None) -> AsyncIterator[T]:
    if times is None:
        while True:
            yield value

    else:
        for _ in range(times):
            yield value


async def async_chain(*iterables: AnyIterable[T]) -> AsyncIterator[T]:
    for iterable in iterables:
        async for element in async_iter_any_iter(iterable):
            yield element


async def async_chain_from_iterable(iterables: AnyIterable[AnyIterable[T]]) -> AsyncIterator[T]:
    async for iterable in async_iter_any_iter(iterables):
        async for element in async_iter_any_iter(iterable):
            yield element


async def async_compress(iterable: AnyIterable[T], selectors: AnyIterable[U]) -> AsyncIterator[T]:
    async for element, selector in async_zip(iterable, selectors):
        if selector:
            yield element


async def async_drop_while(
    predicate: Callable[[T], MaybeAwaitable[Any]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async_iterator = async_iter_any_iter(iterable)

    async for element in async_iterator:
        if not await maybe_await(predicate(element)):
            yield element
            break

    async for element in async_iterator:
        yield element


async def async_take_while(
    predicate: Callable[[T], MaybeAwaitable[Any]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    async for element in async_iter_any_iter(iterable):
        if await maybe_await(predicate(element)):
            yield element

        else:
            break


def async_copy(iterable: AnyIterable[T], n: int = 2) -> Tuple[AsyncIterator[T], ...]:
    async_iterator = async_iter_any_iter(iterable)

    queues: List["asyncio.Queue[T]"] = [asyncio.Queue() for _ in range(n)]

    async def generator(this_queue: "asyncio.Queue[T]") -> AsyncIterator[T]:
        while True:
            if not this_queue.qsize():
                try:
                    value = await async_next_unchecked(async_iterator)

                except StopAsyncIteration:
                    return

                await asyncio.gather(*(queue.put(value) for queue in queues))

            yield await this_queue.get()

    return tuple(generator(this_queue) for this_queue in queues)


async_copy_infinite = async_copy


def async_copy_safe(iterable: AnyIterable[T], n: int = 2) -> Tuple[AsyncIterator[T], ...]:
    state: Optional[Tuple[T, ...]] = None

    async def generator() -> AsyncIterator[T]:
        nonlocal state

        if state is None:
            state = await async_tuple(iterable)

        async for value in async_iter_any_iter(state):
            yield value

    return tuple(generator() for _ in range(n))


@overload
def async_zip(__iterable_1: AnyIterable[T1]) -> AsyncIterator[Tuple[T1]]:
    ...


@overload
def async_zip(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
) -> AsyncIterator[Tuple[T1, T2]]:
    ...


@overload
def async_zip(
    __iterable_1: AnyIterable[T1], __iterable_2: AnyIterable[T2], __iterable_3: AnyIterable[T3]
) -> AsyncIterator[Tuple[T1, T2, T3]]:
    ...


@overload
def async_zip(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
    __iterable_4: AnyIterable[T4],
) -> AsyncIterator[Tuple[T1, T2, T3, T4]]:
    ...


@overload
def async_zip(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
    __iterable_4: AnyIterable[T4],
    __iterable_5: AnyIterable[T5],
) -> AsyncIterator[Tuple[T1, T2, T3, T4, T5]]:
    ...


@overload
def async_zip(
    __iterable_1: AnyIterable[Any],
    __iterable_2: AnyIterable[Any],
    __iterable_3: AnyIterable[Any],
    __iterable_4: AnyIterable[Any],
    __iterable_5: AnyIterable[Any],
    __iterable_6: AnyIterable[Any],
    *iterables: AnyIterable[Any],
) -> AsyncIterator[Tuple[Any, ...]]:
    ...


async def async_zip(*iterables: AnyIterable[Any]) -> AsyncIterator[Tuple[Any, ...]]:
    iterators: List[AsyncIterator[Any]] = [async_iter_any_iter(iterable) for iterable in iterables]

    if not iterators:
        return

    while True:
        try:
            values: Tuple[Any, ...] = tuple(
                await asyncio.gather(*map(async_next_unchecked, iterators))
            )

            yield values

        except StopAsyncIteration:
            return


@overload
def async_iter_slice(iterable: AnyIterable[T], __stop: Optional[int]) -> AsyncIterator[T]:
    ...


@overload
def async_iter_slice(
    iterable: AnyIterable[T],
    __start: Optional[int],
    __stop: Optional[int],
) -> AsyncIterator[T]:
    ...


@overload
def async_iter_slice(
    iterable: AnyIterable[T],
    __start: Optional[int],
    __stop: Optional[int],
    __step: Optional[int],
) -> AsyncIterator[T]:
    ...


async def async_iter_slice(
    iterable: AnyIterable[T], *slice_args: Optional[int]
) -> AsyncIterator[T]:
    real_slice = slice(*slice_args)

    start, stop, step = real_slice.start or 0, real_slice.stop or max_size, real_slice.step or 1

    if start < 0:
        raise ValueError("Slice start can not be lower than 0.")

    if stop < 0:
        raise ValueError("Slice stop can not be lower than 0.")

    if step < 1:
        raise ValueError("Slice step can not be lower than 1.")

    async for index, element in async_enumerate(iterable):
        if index >= start and not (index - start) % step:
            yield element

        if index + 1 >= stop:
            break


@overload
async def async_max(
    iterable: AnyIterable[OrderT],
    *,
    default: Marker = marker,
) -> OrderT:
    ...


@overload
async def async_max(
    iterable: AnyIterable[T],
    *,
    key: Callable[[T], MaybeAwaitable[OrderT]],
    default: Marker = marker,
) -> T:
    ...


@overload
async def async_max(
    iterable: AnyIterable[OrderT],
    *,
    default: T,
) -> Union[OrderT, T]:
    ...


@overload
async def async_max(
    iterable: AnyIterable[T],
    *,
    key: Callable[[T], MaybeAwaitable[OrderT]],
    default: U,
) -> Union[T, U]:
    ...


async def async_max(
    iterable: AnyIterable[Any],
    *,
    key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None,
    default: MarkerOr[Any] = marker,
) -> Any:
    async_iterator = async_iter_any_iter(iterable)

    try:
        value = await async_next_unchecked(async_iterator)

        if key is not None:
            value_key = await maybe_await(key(value))

    except StopAsyncIteration:
        if default is marker:
            raise TypeError("max() of empty iterable with no default") from None

        return default

    if key is None:
        async for element in async_iterator:
            if element > value:
                value = element

    else:
        async for element in async_iterator:
            element_key = await maybe_await(key(value))

            if element_key > value_key:
                value = element
                value_key = element_key

    return value


@overload
async def async_min(
    iterable: AnyIterable[OrderT],
    *,
    default: Marker = marker,
) -> OrderT:
    ...


@overload
async def async_min(
    iterable: AnyIterable[T],
    *,
    key: Callable[[T], MaybeAwaitable[OrderT]],
    default: Marker = marker,
) -> T:
    ...


@overload
async def async_min(
    iterable: AnyIterable[OrderT],
    *,
    default: T,
) -> Union[OrderT, T]:
    ...


@overload
async def async_min(
    iterable: AnyIterable[T],
    *,
    key: Callable[[T], MaybeAwaitable[OrderT]],
    default: U,
) -> Union[T, U]:
    ...


async def async_min(
    iterable: AnyIterable[Any],
    *,
    key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None,
    default: MarkerOr[Any] = marker,
) -> Any:
    async_iterator = async_iter_any_iter(iterable)

    try:
        value = await async_next_unchecked(async_iterator)

        if key is not None:
            value_key = await maybe_await(key(value))

    except StopAsyncIteration:
        if default is marker:
            raise TypeError("min() of empty iterable with no default") from None

        return default

    if key is None:
        async for element in async_iterator:
            if element < value:
                value = element

    else:
        async for element in async_iterator:
            element_key = await maybe_await(key(value))

            if element_key < value_key:
                value = element
                value_key = element_key

    return value


async def async_exhaust(iterable: AnyIterable[T], amount: Optional[int] = None) -> None:
    if amount is None:
        async for _ in async_iter_any_iter(iterable):
            pass

    else:
        await async_exhaust(async_take(iterable, amount))


@overload
async def async_first(iterable: AnyIterable[T], default: Marker = marker) -> T:
    ...


@overload
async def async_first(iterable: AnyIterable[T], default: U) -> Union[T, U]:
    ...


async def async_first(
    iterable: AnyIterable[T], default: MarkerOr[U] = marker
) -> Union[T, U]:
    try:
        return await async_next_unchecked(async_iter_any_iter(iterable))

    except StopAsyncIteration as error:
        if default is marker:
            raise ValueError("async_first() called on an empty async iterable.") from error

        return cast(U, default)


@overload
async def async_last(iterable: AnyIterable[T], default: Marker = marker) -> T:
    ...


@overload
async def async_last(iterable: AnyIterable[T], default: U) -> Union[T, U]:
    ...


async def async_last(
    iterable: AnyIterable[T], default: MarkerOr[U] = marker
) -> Union[T, U]:
    if isinstance(iterable, AsyncIterable):
        no_iter_sentinel = object()

        element: Union[Any, T] = no_iter_sentinel

        async for element in iterable:
            pass

        if element is no_iter_sentinel:
            if default is marker:
                raise ValueError("async_last() called on an empty async iterable.")

            return cast(T, default)

        return element

    else:
        return last(iterable)


async def async_fold(
    iterable: AnyIterable[T], function: Callable[[U, T], MaybeAwaitable[U]], initial: U
) -> U:
    return await async_reduce(function, iterable, initial)


DOT = "."
DUNDER = "__"


def async_get(iterable: AnyIterable[T], **attrs: U) -> AsyncIterator[T]:
    names = tuple(attr.replace(DUNDER, DOT) for attr in attrs.keys())
    expected = tuple(attrs.values())

    # special case for one attribute -> we recieve pure values instead of tuples

    if len(expected) == 1:
        expected = first(expected)  # type: ignore

    get_attrs = get_attr_factory(*names)

    def predicate(item: T) -> bool:
        return get_attrs(item) == expected

    return async_filter(predicate, iterable)


@overload
async def async_at(iterable: AnyIterable[T], n: int, default: Marker = marker) -> T:
    ...


@overload
async def async_at(iterable: AnyIterable[T], n: int, default: U) -> Union[T, U]:
    ...


async def async_at(
    iterable: AnyIterable[T], n: int, default: MarkerOr[U] = marker
) -> Union[T, U]:
    try:
        return await async_next_unchecked(async_iter_slice(iterable, n, None))

    except StopIteration as error:
        if default is marker:
            raise ValueError("async_at() called with n larger than iterable length.") from error

        return cast(U, default)


@overload
async def async_at_or_last(
    iterable: AnyIterable[T], n: int, default: Marker = marker
) -> T:
    ...


@overload
async def async_at_or_last(
    iterable: AnyIterable[T], n: int, default: U
) -> Union[T, U]:
    ...


async def async_at_or_last(
    iterable: AnyIterable[T], n: int, default: MarkerOr[U] = marker
) -> Union[T, U]:
    return await async_last(async_iter_slice(iterable, n + 1), default=default)


def async_drop(iterable: AnyIterable[T], n: int) -> AsyncIterator[T]:
    return async_iter_slice(iterable, n, None)


def async_take(iterable: AnyIterable[T], n: int) -> AsyncIterator[T]:
    return async_iter_slice(iterable, n)


def async_step_by(iterable: AnyIterable[T], step: int) -> AsyncIterator[T]:
    return async_iter_slice(iterable, None, None, step)


def async_group(iterable: AnyIterable[T], n: int) -> AsyncIterator[Tuple[T, ...]]:
    iterators = (async_iter_any_iter(iterable),) * n

    return async_zip(*iterators)


@overload
def async_group_longest(
    iterable: AnyIterable[T], n: int
) -> AsyncIterator[Tuple[Optional[T], ...]]:
    ...


@overload
def async_group_longest(
    iterable: AnyIterable[T], n: int, fill: T
) -> AsyncIterator[Tuple[T, ...]]:
    ...


def async_group_longest(
    iterable: AnyIterable[T], n: int, fill: Optional[T] = None
) -> AsyncIterator[Tuple[Optional[T], ...]]:
    iterators = (async_iter_any_iter(iterable),) * n

    return async_zip_longest(*iterators, fill=fill)


class ZipExhausted(Exception):
    pass


@overload
def async_zip_longest(__iterable_1: AnyIterable[T1]) -> AsyncIterator[Tuple[Optional[T1]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
) -> AsyncIterator[Tuple[Optional[T1], Optional[T2]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
) -> AsyncIterator[Tuple[Optional[T1], Optional[T2], Optional[T3]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
    __iterable_4: AnyIterable[T4],
) -> AsyncIterator[Tuple[Optional[T1], Optional[T2], Optional[T3], Optional[T4]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
    __iterable_4: AnyIterable[T4],
    __iterable_5: AnyIterable[T5],
) -> AsyncIterator[Tuple[Optional[T1], Optional[T2], Optional[T3], Optional[T4], Optional[T5]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[Any],
    __iterable_2: AnyIterable[Any],
    __iterable_3: AnyIterable[Any],
    __iterable_4: AnyIterable[Any],
    __iterable_5: AnyIterable[Any],
    __iterable_6: AnyIterable[Any],
    *iterables: AnyIterable[Any],
) -> AsyncIterator[Tuple[Optional[Any], ...]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1], *, fill: T
) -> AsyncIterator[Tuple[Union[T1, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1], __iterable_2: AnyIterable[T2], *, fill: T
) -> AsyncIterator[Tuple[Union[T1, T], Union[T2, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
    *,
    fill: T,
) -> AsyncIterator[Tuple[Union[T1, T], Union[T2, T], Union[T3, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
    __iterable_4: AnyIterable[T4],
    *,
    fill: T,
) -> AsyncIterator[Tuple[Union[T1, T], Union[T2, T], Union[T3, T], Union[T4, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[T1],
    __iterable_2: AnyIterable[T2],
    __iterable_3: AnyIterable[T3],
    __iterable_4: AnyIterable[T4],
    __iterable_5: AnyIterable[T5],
    *,
    fill: T,
) -> AsyncIterator[Tuple[Union[T1, T], Union[T2, T], Union[T3, T], Union[T4, T], Union[T5, T]]]:
    ...


@overload
def async_zip_longest(
    __iterable_1: AnyIterable[Any],
    __iterable_2: AnyIterable[Any],
    __iterable_3: AnyIterable[Any],
    __iterable_4: AnyIterable[Any],
    __iterable_5: AnyIterable[Any],
    __iterable_6: AnyIterable[Any],
    *iterables: AnyIterable[Any],
    fill: T,
) -> AsyncIterator[Tuple[Union[Any, T], ...]]:
    ...


async def async_zip_longest(
    *iterables: AnyIterable[Any], fill: Optional[T] = None
) -> AsyncIterator[Tuple[Union[Any, Optional[T]], ...]]:
    if not iterables:
        return

    remain = len(iterables) - 1

    async def sentinel() -> AsyncIterator[Optional[T]]:
        nonlocal remain

        if not remain:
            raise ZipExhausted

        remain -= 1

        yield fill

    fillers = async_repeat(fill)

    iterators = [async_chain(iterable, sentinel(), fillers) for iterable in iterables]

    try:
        while True:
            yield await async_tuple(async_map(async_next_unchecked, iterators))

    except ZipExhausted:
        pass


def async_flatten(iterable: AnyIterable[AnyIterable[T]]) -> AsyncIterator[T]:
    return async_chain_from_iterable(iterable)


async def async_parallel_flatten(iterable: AnyIterable[AnyIterable[T]]) -> AsyncIterator[T]:
    coroutines = [
        async_list(async_iter_any_iter(element_iterable))
        async for element_iterable in async_iter_any_iter(iterable)
    ]

    flatten_elements = await asyncio.gather(*coroutines)

    for element in flatten_elements:
        yield element


async def run_iterators(
    iterators: AnyIterable[AnyIterable[T]],
    *ignore_exceptions: Type[BaseException],
    concurrent: bool = True,
) -> AsyncIterator[T]:
    if concurrent:
        coroutines = [
            async_list(async_iter_any_iter(iterator))
            async for iterator in async_iter_any_iter(iterators)
        ]

        results: List[Union[List[T], BaseException]] = await asyncio.gather(
            *coroutines, return_exceptions=True
        )

        filtered_results = (
            result for result in results if not isinstance(result, ignore_exceptions)
        )

        for result in filtered_results:
            if isinstance(result, BaseException):
                raise result  # was not handled -> raise

            for element in result:
                yield element

    else:
        async for iterator in async_iter_any_iter(iterators):
            try:
                async for element in async_iter_any_iter(iterator):
                    yield element

            except ignore_exceptions:
                pass


def async_prepend(iterable: AnyIterable[T], item: T) -> AsyncIterator[T]:
    return async_chain((item,), iterable)


def async_append(iterable: AnyIterable[T], item: T) -> AsyncIterator[T]:
    return async_chain(iterable, (item,))


def async_partition(
    iterable: AnyIterable[T], predicate: Callable[[T], MaybeAwaitable[Any]] = bool
) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    for_true, for_false = async_copy(iterable)

    return async_filter(predicate, for_true), async_filter_false(predicate, for_false)


async_partition_infinite = async_partition


def async_partition_safe(
    iterable: AnyIterable[T], predicate: Callable[[T], MaybeAwaitable[Any]] = bool
) -> Tuple[AsyncIterator[T], AsyncIterator[T]]:
    for_true, for_false = async_copy_safe(iterable)

    return async_filter(predicate, for_true), async_filter_false(predicate, for_false)


async def async_list_chunk(iterable: AnyIterable[T], n: int) -> AsyncIterator[List[T]]:
    iterator = async_iter_any_iter(iterable)

    while True:
        part = await async_list(async_take(iterator, n))

        if not part:
            break

        yield part


async def async_tuple_chunk(iterable: AnyIterable[T], n: int) -> AsyncIterator[Tuple[T, ...]]:
    iterator = async_iter_any_iter(iterable)

    while True:
        part = await async_tuple(async_take(iterator, n))

        if not part:
            break

        yield part


async def async_iter_chunk(iterable: AnyIterable[T], n: int) -> AsyncIterator[AsyncIterator[T]]:
    source = async_iter_any_iter(iterable)

    while True:
        try:
            item = await async_next_unchecked(source)

        except StopAsyncIteration:
            return

        source, iterator = async_copy(async_prepend(source, item))

        yield async_take(iterator, n)

        await async_exhaust(source, n)


async def async_iterate(function: Callable[[T], MaybeAwaitable[T]], value: T) -> AsyncIterator[T]:
    while True:
        yield value
        value = await maybe_await(function(value))


def async_collapse(
    iterable: AnyIterable[T],
    base_type: Optional[Type[Any]] = None,
    levels: Optional[int] = None,
) -> AsyncIterator[T]:
    async def walk(node: Union[T, AnyIterable[T]], level: int) -> AsyncIterator[T]:
        if (
            ((levels is not None) and (level > levels))
            or isinstance(node, (str, bytes))
            or ((base_type is not None) and isinstance(node, base_type))
        ):
            yield cast(T, node)
            return

        try:
            tree = async_iter_any_iter(node)  # type: ignore

        except TypeError:
            yield cast(T, node)
            return

        else:
            async for child in tree:
                async for item in walk(child, level + 1):
                    yield item

    return walk(iterable, 0)


async def async_side_effect(
    iterable: AnyIterable[T],
    function: Callable[[T], MaybeAwaitable[None]],
    before: Optional[Callable[[], MaybeAwaitable[None]]] = None,
    after: Optional[Callable[[], MaybeAwaitable[None]]] = None,
) -> AsyncIterator[T]:
    try:
        if before is not None:
            await maybe_await(before())

        async for item in async_iter_any_iter(iterable):
            await maybe_await(function(item))
            yield item

    finally:
        if after is not None:
            await maybe_await(after())


async def async_distinct(
    iterable: AnyIterable[T], key: Optional[Callable[[T], U]] = None
) -> AsyncIterator[T]:
    if key is None:
        seen_set: Set[T] = set()
        add_to_seen_set = seen_set.add
        seen_list: List[T] = []
        add_to_seen_list = seen_list.append

        async for element in async_iter_any_iter(iterable):
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

        async for element in async_iter_any_iter(iterable):
            value = await maybe_await(key(element))

            try:
                if value not in seen_value_set:
                    add_to_seen_value_set(value)
                    yield element

            except TypeError:
                if value not in seen_value_list:
                    add_to_seen_value_list(value)
                    yield element
