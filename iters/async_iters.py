from functools import wraps
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    Dict,
    Generic,
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
    overload,
    no_type_check,
)
from typing_extensions import AsyncContextManager

from iters.async_utils import (
    AnyIterable,
    MarkerOr,
    MaybeAwaitable,
    marker,
    async_all,
    async_any,
    async_append,
    async_chain,
    async_chain_from_iterable,
    async_collapse,
    async_compress,
    async_copy,
    async_copy_infinite,
    async_copy_safe,
    async_count,
    async_cycle,
    async_dict,
    async_drop,
    async_drop_while,
    async_enumerate,
    async_exhaust,
    async_filter,
    async_filter_false,
    async_first,
    async_flatten,
    async_fold,
    async_get,
    async_group,
    async_group_longest,
    async_iter as async_std_iter,
    async_iter_chunk,
    async_iter_len,
    async_iter_slice,
    async_iterate,
    async_last,
    async_list,
    async_list_chunk,
    async_map,
    async_max,
    async_min,
    async_next,
    async_next_unchecked,
    async_nth,
    async_nth_or_last,
    async_partition,
    async_partition_infinite,
    async_partition_safe,
    async_prepend,
    async_product,
    async_reduce,
    async_repeat,
    async_set,
    async_side_effect,
    async_star_map,
    async_std_reversed,
    async_step_by,
    async_sum,
    async_take,
    async_take_while,
    async_tuple,
    async_tuple_chunk,
    async_with_async_iter,
    async_with_iter,
    async_zip,
    async_zip_longest,
    maybe_await,
)

__all__ = (
    "AsyncIter",
    "async_iter",
    "async_next",
    "async_reversed",
    "return_async_iter",
    "async_std_iter",
    "async_std_reversed",
)

T = TypeVar("T")
U = TypeVar("U")


class AsyncIter(Generic[T]):
    @overload
    def __init__(self, iterator: Iterator[T]) -> None:  # noqa
        ...

    @overload
    def __init__(self, iterable: Iterable[T]) -> None:  # noqa
        ...

    @overload
    def __init__(self, async_iterator: AsyncIterator[T]) -> None:  # noqa
        ...

    @overload
    def __init__(self, async_iterable: AsyncIterable[T]) -> None:  # noqa
        ...

    @overload
    def __init__(self, function: Callable[[], T], sentinel: T) -> None:  # noqa
        ...

    @overload
    def __init__(self, async_function: Callable[[], Awaitable[T]], sentinel: T) -> None:  # noqa
        ...

    @no_type_check
    def __init__(  # noqa
        self,
        something: Union[
            Iterator[T],
            Iterable[T],
            AsyncIterator[T],
            AsyncIterable[T],
            Callable[[], T],
            Callable[[], Awaitable[T]],
        ],
        sentinel: MarkerOr[T] = marker,
    ) -> None:
        self._iterator: AsyncIterator[T] = async_std_iter(something, sentinel)

    def __aiter__(self) -> "AsyncIter[T]":
        return self

    async def __next__(self) -> T:
        return await async_next_unchecked(self._iterator)  # type: ignore

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[T] at 0x{id(self):016x}>"

    def unwrap(self) -> AsyncIterator[T]:
        return self._iterator  # type: ignore

    @classmethod
    def count(cls, start: T = 0, step: T = 1) -> "AsyncIter[T]":  # type: ignore
        return cls(cast(AsyncIterator[T], async_count(start, step)))

    @classmethod
    def repeat(cls, to_repeat: T, times: Optional[int] = None) -> "AsyncIter[T]":
        return cls(async_repeat(to_repeat, times))

    @classmethod
    def reversed(cls, iterable: Reversible[T]) -> "AsyncIter[T]":
        return cls(async_std_reversed(iterable))

    @classmethod
    def with_iter(cls, context_manager: ContextManager[Iterable[T]]) -> "AsyncIter[T]":
        return cls(async_with_iter(context_manager))

    @classmethod
    def async_with_iter(
        cls, context_manager: AsyncContextManager[AsyncIterable[T]]
    ) -> "AsyncIter[T]":
        return cls(async_with_async_iter(context_manager))

    @classmethod
    def iterate(cls, function: Callable[[T], MaybeAwaitable[T]], value: T) -> "AsyncIter[T]":
        return cls(async_iterate(function, value))

    def iter(self) -> "AsyncIter[T]":
        return self

    async def next(self) -> T:
        return await async_next_unchecked(self._iterator)  # type: ignore

    async def next_or(self, default: Optional[T]) -> Optional[T]:
        return await async_next_unchecked(self._iterator, default)  # type: ignore

    async def all(self) -> bool:
        return await async_all(self._iterator)  # type: ignore

    async def any(self) -> bool:
        return await async_any(self._iterator)  # type: ignore

    def append(self, item: T) -> "AsyncIter[T]":
        return self.__class__(async_append(self._iterator, item))  # type: ignore

    def prepend(self, item: T) -> "AsyncIter[T]":
        return self.__class__(async_prepend(self._iterator, item))  # type: ignore

    def chain(self, *iterables: AnyIterable[T]) -> "AsyncIter[T]":
        return self.__class__(async_chain(self._iterator, *iterables))  # type: ignore

    def chain_with(self, iterables: AnyIterable[AnyIterable[T]]) -> "AsyncIter[T]":
        return self.__class__(
            async_chain(self._iterator, async_chain_from_iterable(iterables))  # type: ignore
        )

    def collapse(
        self, base_type: Optional[Type[Any]] = None, levels: Optional[int] = None
    ) -> "AsyncIter[T]":
        return self.__class__(
            async_collapse(self._iterator, base_type=base_type, levels=levels)  # type: ignore
        )

    def reverse(self) -> "AsyncIter[T]":
        return self.__class__(
            cast(AsyncIterator[T], async_reversed(self._iterator))  # type: ignore
        )

    def slice(self, *slice_args) -> "AsyncIter[T]":
        return self.__class__(async_iter_slice(self._iterator, *slice_args))  # type: ignore

    async def exhaust(self) -> None:
        await async_exhaust(self._iterator)  # type: ignore

    async def for_each(self, function: Callable[[T], MaybeAwaitable[None]]) -> None:
        await self.map(function).exhaust()

    async def collect(
        self, function: Callable[[AsyncIterator[T]], MaybeAwaitable[AnyIterable[T]]],
    ) -> AnyIterable[T]:
        return await maybe_await(function(self._iterator))  # type: ignore

    async def dict(self) -> Dict[T, T]:
        return await async_dict(self._iterator)  # type: ignore

    async def list(self) -> List[T]:
        return await async_list(self._iterator)  # type: ignore

    async def set(self) -> Set[T]:
        return await async_set(self._iterator)  # type: ignore

    async def tuple(self) -> Tuple[T, ...]:
        return await async_tuple(self._iterator)  # type: ignore

    def compress(self, selectors: AnyIterable[U]) -> "AsyncIter[T]":
        return self.__class__(async_compress(self._iterator, selectors))  # type: ignore

    def copy(self) -> "AsyncIter[T]":
        for_self, to_return = async_copy(self._iterator)  # type: ignore

        self._iterator = for_self

        return self.__class__(to_return)

    def copy_infinite(self) -> "AsyncIter[T]":
        for_self, to_return = async_copy_infinite(self._iterator)

        self._iterator = for_self

        return self.__class__(to_return)

    def copy_safe(self) -> "AsyncIter[T]":
        for_self, to_return = async_copy_safe(self._iterator)

        self._iterator = for_self

        return self.__class__(to_return)

    def cycle(self) -> "AsyncIter[T]":
        return self.__class__(async_cycle(self._iterator))

    def drop(self, n: int) -> "AsyncIter[T]":
        return self.__class__(async_drop(self._iterator, n))

    skip = drop

    def drop_while(self, predicate: Callable[[T], MaybeAwaitable[bool]]) -> "AsyncIter[T]":
        return self.__class__(async_drop_while(predicate, self._iterator))

    skip_while = drop_while

    def take(self, n: int) -> "AsyncIter[T]":
        return self.__class__(async_take(self._iterator, n))

    def take_while(self, predicate: Callable[[T], MaybeAwaitable[bool]]) -> "AsyncIter[T]":
        return self.__class__(async_take_while(predicate, self._iterator))

    def step_by(self, step: int) -> "AsyncIter[T]":
        return self.__class__(async_step_by(self._iterator, step))

    def enumerate(self, start: int = 0) -> "AsyncIter[Tuple[int, T]]":
        return self.__class__(async_enumerate(self._iterator))

    def filter(self, predicate: Callable[[T], MaybeAwaitable[bool]]) -> "AsyncIter[T]":
        return self.__class__(async_filter(predicate, self._iterator))

    def filter_false(self, predicate: Callable[[T], MaybeAwaitable[bool]]) -> "AsyncIter[T]":
        return self.__class__(async_filter_false(predicate, self._iterator))

    def find_all(self, predicate: Callable[[T], MaybeAwaitable[bool]]) -> "AsyncIter[T]":
        return self.filter(predicate)

    async def find(
        self, predicate: Callable[[T], bool], default: Optional[T] = None
    ) -> Optional[T]:
        return await self.find_all(predicate).next_or(default)

    async def first(self) -> T:
        return await async_first(self._iterator)

    async def first_or(self, default: Optional[T]) -> Optional[T]:
        return await async_first(self._iterator, default)

    async def fold(self, function: Callable[[U, Union[T, U]], MaybeAwaitable[U]], initial: U) -> U:
        return await async_fold(self._iterator, function, initial)

    async def reduce(self, function: Callable[[U, Union[T, U]], MaybeAwaitable[U]]) -> U:
        return await async_reduce(function, self._iterator)

    async def max(self, *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None) -> T:
        if key is None:
            return await async_max(self._iterator)

        return await async_max(self._iterator, key=key)

    async def min(self, *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None) -> T:
        if key is None:
            return await async_min(self._iterator)

        return await async_min(self._iterator, key=key)

    async def max_or(
        self, default: Optional[T], *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None
    ) -> Optional[T]:
        if key is None:
            return await async_max(self._iterator, default=default)

        return await async_max(self._iterator, key=key, default=default)

    async def min_or(
        self, default: Optional[T], *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None
    ) -> Optional[T]:
        if key is None:
            return await async_min(self._iterator, default=default)

        return await async_min(self._iterator, key=key, default=default)

    async def sum(self, start: MarkerOr[U] = marker) -> U:
        return await async_sum(self._iterator, start)

    async def product(self, start: MarkerOr[U] = marker) -> U:
        return await async_product(self._iterator, start)

    def get_all(self, **attrs) -> "AsyncIter[T]":
        return self.__class__(async_get(self._iterator, **attrs))

    async def get(self, *, default: Optional[T] = None, **attrs) -> Optional[T]:
        return await self.get_all(**attrs).next_or(default)

    async def last(self) -> T:
        return await async_last(self._iterator)

    async def last_or(self, default: Optional[T]) -> Optional[T]:
        return await async_last(self._iterator, default)

    def flatten(self) -> "AsyncIter[T]":
        return self.__class__(async_flatten(self._iterator))

    def group(self, n: int) -> "AsyncIter[Tuple[T, ...]]":
        return self.__class__(async_group(self._iterator, n))

    def group_longest(
        self, n: int, fill: Optional[T] = None
    ) -> "AsyncIter[Tuple[Optional[T], ...]]":
        return self.__class__(async_group_longest(self._iterator, n, fill))

    def iter_chunk(self, n: int) -> "AsyncIter[AsyncIter[T]]":
        return self.__class__(async_map(self.__class__, async_iter_chunk(self._iterator, n)))

    def list_chunk(self, n: int) -> "AsyncIter[List[T]]":
        return self.__class__(async_list_chunk(self._iterator, n))

    def tuple_chunk(self, n: int) -> "AsyncIter[Tuple[T, ...]]":
        return self.__class__(async_tuple_chunk(self._iterator, n))

    async def nth(self, n: int) -> T:
        return await async_nth(self._iterator, n)

    async def nth_or(self, n: int, default: T) -> T:
        return await async_nth(self._iterator, n, default)

    async def nth_or_last(self, n: int) -> T:
        return await async_nth_or_last(self._iterator, n)

    async def length(self) -> int:
        return await async_iter_len(self._iterator)

    def map(self, function: Callable[[T], MaybeAwaitable[U]]) -> "AsyncIter[U]":
        return self.__class__(async_map(function, self._iterator))

    def star_map(self, function: Callable[..., MaybeAwaitable[U]]) -> "AsyncIter[U]":
        return self.__class__(async_star_map(function, self._iterator))

    def partition(
        self, predicate: Callable[[T], MaybeAwaitable[bool]]
    ) -> "Tuple[AsyncIter[T], AsyncIter[T]]":
        with_true, with_false = async_partition(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_infinite(
        self, predicate: Callable[[T], MaybeAwaitable[bool]]
    ) -> "Tuple[AsyncIter[T], AsyncIter[T]]":
        with_true, with_false = async_partition_infinite(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_safe(
        self, predicate: Callable[[T], MaybeAwaitable[bool]]
    ) -> "Tuple[AsyncIter[T], AsyncIter[T]]":
        with_true, with_false = async_partition_safe(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def zip(self, *iterables: Iterable[T]) -> "AsyncIter[T]":
        return self.__class__(async_zip(self._iterator, *iterables))

    def zip_longest(self, *iterables: Iterable[T], fill: T = None) -> "AsyncIter[T]":
        return self.__class__(async_zip_longest(self._iterator, *iterables, fillvalue=fill))

    def side_effect(
        self,
        function: Callable[[T], MaybeAwaitable[None]],
        before: Optional[Callable[[], MaybeAwaitable[None]]] = None,
        after: Optional[Callable[[], MaybeAwaitable[None]]] = None,
    ) -> "AsyncIter[T]":
        return self.__class__(async_side_effect(self._iterator, function, before, after))

    def inspect(self, string: str) -> "AsyncIter[T]":
        def print_and_string(item: T) -> None:
            print(string, item)

        return self.side_effect(print_and_string)

    def inspect_format(self, format_string: str) -> "AsyncIter[T]":
        def print_format(item: T) -> None:
            print(format_string.format(item))

        return self.side_effect(print_format)


async_iter = AsyncIter
async_reversed = async_iter.reversed


def return_async_iter(function: Callable[..., AsyncIterable[T]]) -> Callable[..., AsyncIter[T]]:
    @wraps(function)
    def wrapper(*args, **kwargs) -> AsyncIter[T]:
        return async_iter(function(*args, **kwargs))

    return wrapper
