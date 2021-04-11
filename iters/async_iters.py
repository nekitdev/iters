from functools import wraps
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
    overload,
    no_type_check,
)
from typing_extensions import AsyncContextManager

from iters.async_utils import (
    AnyIterable,
    MaybeAwaitable,
    async_all,
    async_any,
    async_append,
    async_at,
    async_at_or_last,
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
    async_distinct,
    async_drop,
    async_drop_while,
    async_enumerate,
    async_exhaust,
    async_filter,
    async_filter_nowait,
    async_filter_false,
    async_filter_false_nowait,
    async_first,
    async_flatten,
    async_fold,
    async_get,
    async_group,
    async_group_longest,
    async_iter as std_async_iter,
    async_iter_chunk,
    async_iter_len,
    async_iter_slice,
    async_iterate,
    async_last,
    async_list,
    async_list_chunk,
    async_map,
    async_map_nowait,
    async_max,
    async_min,
    async_next,
    async_next_unchecked,
    async_parallel_filter,
    async_parallel_filter_false,
    async_parallel_flatten,
    async_parallel_map,
    async_parallel_star_map,
    async_parallel_wait,
    async_partition,
    async_partition_infinite,
    async_partition_safe,
    async_prepend,
    async_product,
    async_reduce,
    async_repeat,
    async_reversed as std_async_reversed,
    async_set,
    async_side_effect,
    async_star_map,
    async_star_map_nowait,
    async_step_by,
    async_sum,
    async_take,
    async_take_while,
    async_tuple,
    async_tuple_chunk,
    async_wait,
    async_with_async_iter,
    async_with_iter,
    async_zip,
    async_zip_longest,
    maybe_await,
    reverse_to_async,
    run_iterators,
)
from iters.types import MarkerOr, Order, marker

__all__ = (
    # async iterator class
    "AsyncIter",
    # convenient functions to get an async iterator
    "async_iter",
    "async_reversed",
    # next functions; checked version works on any iterator, unchecked assumes an async iterator
    "async_next",
    "async_next_unchecked",
    # since we are shadowing standard functions, export them as <std>
    "std_async_iter",
    "std_async_reversed",
    # decorator to wrap return value of the function into an async iterator
    "return_async_iter",
)

__all__ = (
    "AsyncIter",
    "async_iter",
    "async_next",
    "async_next_unchecked",
    "async_reversed",
    "return_async_iter",
    "std_async_iter",
    "std_async_reversed",
)

KT = TypeVar("KT")
VT = TypeVar("VT")

N = TypeVar("N", int, float)

T = TypeVar("T", covariant=True)
U = TypeVar("U")
V = TypeVar("V")

T1 = TypeVar("T1", covariant=True)
T2 = TypeVar("T2", covariant=True)
T3 = TypeVar("T3", covariant=True)
T4 = TypeVar("T4", covariant=True)
T5 = TypeVar("T5", covariant=True)

OrderT = TypeVar("OrderT", bound=Order)


class AsyncIter(AsyncIterator[T]):
    _iterator: AsyncIterator[T]

    @overload
    def __init__(self, iterator: Iterator[T]) -> None:
        ...

    @overload
    def __init__(self, iterable: Iterable[T]) -> None:
        ...

    @overload
    def __init__(self, async_iterator: AsyncIterator[T]) -> None:
        ...

    @overload
    def __init__(self, async_iterable: AsyncIterable[T]) -> None:
        ...

    @overload
    def __init__(self, function: Callable[[], T], sentinel: T) -> None:
        ...

    @overload
    def __init__(self, async_function: Callable[[], Awaitable[T]], sentinel: T) -> None:
        ...

    @no_type_check
    def __init__(
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
        self._iterator: AsyncIterator[T] = std_async_iter(something, sentinel)

    def __aiter__(self) -> "AsyncIter[T]":
        return self

    async def __anext__(self) -> T:
        return await async_next_unchecked(self._iterator)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}<T> at 0x{id(self):016x}>"

    def unwrap(self) -> AsyncIterator[T]:
        return self._iterator

    @classmethod
    def count(cls, start: N = 0, step: N = 1) -> "AsyncIter[N]":
        return cls(async_count(start, step))

    @classmethod
    def repeat(cls, to_repeat: V, times: Optional[int] = None) -> "AsyncIter[V]":
        return cls(async_repeat(to_repeat, times))

    @classmethod
    def reversed(cls, iterable: Reversible[T]) -> "AsyncIter[T]":
        return cls(reverse_to_async(iterable))

    @classmethod
    def with_iter(cls, context_manager: ContextManager[Iterable[T]]) -> "AsyncIter[T]":
        return cls(async_with_iter(context_manager))

    @classmethod
    def async_with_iter(
        cls, context_manager: AsyncContextManager[AsyncIterable[T]]
    ) -> "AsyncIter[T]":
        return cls(async_with_async_iter(context_manager))

    @classmethod
    def iterate(cls, function: Callable[[V], MaybeAwaitable[V]], value: V) -> "AsyncIter[V]":
        return cls(async_iterate(function, value))

    def iter(self) -> "AsyncIter[T]":
        return self

    async def next(self) -> T:
        return await async_next_unchecked(self._iterator)

    async def next_or(self, default: U) -> Union[T, U]:
        return await async_next_unchecked(self._iterator, default)

    async def next_or_null(self) -> Optional[T]:
        return await self.next_or(None)

    async def all(self) -> bool:
        return await async_all(self._iterator)

    async def any(self) -> bool:
        return await async_any(self._iterator)

    def append(self: "AsyncIter[V]", item: V) -> "AsyncIter[V]":
        return self.__class__(async_append(self._iterator, item))

    def prepend(self: "AsyncIter[V]", item: V) -> "AsyncIter[V]":
        return self.__class__(async_prepend(self._iterator, item))

    def chain(self, *iterables: AnyIterable[T]) -> "AsyncIter[T]":
        return self.__class__(async_chain(self._iterator, *iterables))

    def chain_with(self, iterables: AnyIterable[AnyIterable[T]]) -> "AsyncIter[T]":
        return self.__class__(async_chain(self._iterator, async_chain_from_iterable(iterables)))

    def collapse(
        self, base_type: Optional[Type[Any]] = None, levels: Optional[int] = None
    ) -> "AsyncIter[T]":
        return self.__class__(async_collapse(self._iterator, base_type=base_type, levels=levels))

    def reverse(self) -> "AsyncIter[T]":
        return self.__class__(std_async_reversed(self._iterator))

    def slice(self, *slice_args: int) -> "AsyncIter[T]":
        return self.__class__(async_iter_slice(self._iterator, *slice_args))

    def wait(self: "AsyncIter[MaybeAwaitable[V]]") -> "AsyncIter[V]":
        return self.__class__(async_wait(self._iterator))  # type: ignore

    def parallel_wait(self: "AsyncIter[MaybeAwaitable[V]]") -> "AsyncIter[V]":
        return self.__class__(async_parallel_wait(self._iterator))  # type: ignore

    async def exhaust(self) -> None:
        await async_exhaust(self._iterator)

    async def for_each(self, function: Callable[[T], MaybeAwaitable[None]]) -> None:
        await self.map(function).exhaust()

    async def join(self: "AsyncIter[str]", delim: str) -> str:
        return delim.join(await self.list())

    async def collect(
        self,
        function: Callable[[AsyncIterator[T]], MaybeAwaitable[AnyIterable[T]]],
    ) -> AnyIterable[T]:
        return await maybe_await(function(self._iterator))

    def distinct(self, key: Optional[Callable[[T], U]] = None) -> "AsyncIter[T]":
        return self.__class__(async_distinct(self._iterator, key))

    unique = distinct

    async def dict(self: "AsyncIter[Tuple[KT, VT]]") -> Dict[KT, VT]:
        return await async_dict(self._iterator)

    async def list(self) -> List[T]:
        return await async_list(self._iterator)

    async def set(self) -> Set[T]:
        return await async_set(self._iterator)

    async def tuple(self) -> Tuple[T, ...]:
        return await async_tuple(self._iterator)

    def compress(self, selectors: AnyIterable[U]) -> "AsyncIter[T]":
        return self.__class__(async_compress(self._iterator, selectors))

    def copy(self) -> "AsyncIter[T]":
        for_self, to_return = async_copy(self._iterator)

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

    def drop_while(self, predicate: Callable[[T], MaybeAwaitable[Any]]) -> "AsyncIter[T]":
        return self.__class__(async_drop_while(predicate, self._iterator))

    skip_while = drop_while

    def take(self, n: int) -> "AsyncIter[T]":
        return self.__class__(async_take(self._iterator, n))

    def take_while(self, predicate: Callable[[T], MaybeAwaitable[Any]]) -> "AsyncIter[T]":
        return self.__class__(async_take_while(predicate, self._iterator))

    def step_by(self, step: int) -> "AsyncIter[T]":
        return self.__class__(async_step_by(self._iterator, step))

    def enumerate(self, start: int = 0) -> "AsyncIter[Tuple[int, T]]":
        return self.__class__(async_enumerate(self._iterator, start))

    def filter(self, predicate: Callable[[T], MaybeAwaitable[Any]]) -> "AsyncIter[T]":
        return self.__class__(async_filter(predicate, self._iterator))

    def filter_nowait(self, predicate: Callable[[T], Any]) -> "AsyncIter[T]":
        return self.__class__(async_filter_nowait(predicate, self._iterator))

    def parallel_filter(self, predicate: Callable[[T], MaybeAwaitable[Any]]) -> "AsyncIter[T]":
        return self.__class__(async_parallel_filter(predicate, self._iterator))

    def filter_false(self, predicate: Callable[[T], MaybeAwaitable[Any]]) -> "AsyncIter[T]":
        return self.__class__(async_filter_false(predicate, self._iterator))

    def filter_false_nowait(self, predicate: Callable[[T], MaybeAwaitable[Any]]) -> "AsyncIter[T]":
        return self.__class__(async_filter_false_nowait(predicate, self._iterator))

    def parallel_filter_false(
        self, predicate: Callable[[T], MaybeAwaitable[Any]]
    ) -> "AsyncIter[T]":
        return self.__class__(async_parallel_filter_false(predicate, self._iterator))

    def find_all(self, predicate: Callable[[T], MaybeAwaitable[Any]]) -> "AsyncIter[T]":
        return self.filter(predicate)

    async def find(self, predicate: Callable[[T], Any]) -> T:
        return await self.find_all(predicate).next()

    async def find_or(self, predicate: Callable[[T], Any], default: U) -> Union[T, U]:
        return await self.find_all(predicate).next_or(default)

    async def find_or_null(self, predicate: Callable[[T], Any]) -> Optional[T]:
        return await self.find_all(predicate).next_or_null()

    async def first(self) -> T:
        return await async_first(self._iterator)

    async def first_or(self, default: U) -> Union[T, U]:
        return await async_first(self._iterator, default)

    async def first_or_null(self) -> Optional[T]:
        return await self.first_or(None)

    async def fold(self, function: Callable[[U, Union[T, U]], MaybeAwaitable[U]], initial: U) -> U:
        return await async_fold(self._iterator, function, initial)

    async def reduce(self, function: Callable[[T, T], MaybeAwaitable[T]]) -> T:
        return await async_reduce(function, self._iterator)

    @overload
    async def max(self: "AsyncIter[OrderT]") -> OrderT:
        ...

    @overload
    async def max(self: "AsyncIter[T]", *, key: Callable[[T], MaybeAwaitable[OrderT]]) -> T:
        ...

    async def max(
        self, *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None
    ) -> Any:
        if key is None:
            return await async_max(self._iterator)

        return await async_max(self._iterator, key=key)

    @overload
    async def min(self: "AsyncIter[OrderT]") -> OrderT:
        ...

    @overload
    async def min(self: "AsyncIter[T]", *, key: Callable[[T], MaybeAwaitable[OrderT]]) -> T:
        ...

    async def min(
        self, *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None
    ) -> Any:
        if key is None:
            return await async_min(self._iterator)

        return await async_min(self._iterator, key=key)

    @overload
    async def max_or(
        self: "AsyncIter[OrderT]", default: U
    ) -> Union[OrderT, U]:
        ...

    @overload
    async def max_or(
        self: "AsyncIter[T]",
        default: U,
        *,
        key: Callable[[T], MaybeAwaitable[OrderT]],
    ) -> Union[T, U]:
        ...

    async def max_or(
        self: "AsyncIter[Any]",
        default: Any,
        *,
        key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None,
    ) -> Any:
        if key is None:
            return await async_max(self._iterator, default=default)

        return await async_max(self._iterator, key=key, default=default)

    @overload
    async def max_or_null(self: "AsyncIter[OrderT]") -> Optional[OrderT]:
        ...

    @overload
    async def max_or_null(
        self: "AsyncIter[T]", *, key: Callable[[T], MaybeAwaitable[OrderT]]
    ) -> Optional[T]:
        ...

    async def max_or_null(
        self: "AsyncIter[Any]", *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None
    ) -> Optional[Any]:
        if key is None:
            return await self.max_or(None)

        return await self.max_or(None, key=key)

    @overload
    async def min_or(self: "AsyncIter[OrderT]", default: U) -> Union[OrderT, U]:
        ...

    @overload
    async def min_or(
        self: "AsyncIter[T]",
        default: U,
        *,
        key: Callable[[T], MaybeAwaitable[OrderT]],
    ) -> Union[T, U]:
        ...

    async def min_or(
        self: "AsyncIter[Any]",
        default: Any,
        *,
        key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None,
    ) -> Any:
        if key is None:
            return await async_min(self._iterator, default=default)

        return await async_min(self._iterator, key=key, default=default)

    @overload
    async def min_or_null(self: "AsyncIter[OrderT]") -> Optional[OrderT]:
        ...

    @overload
    async def min_or_null(
        self: "AsyncIter[T]", *, key: Callable[[T], MaybeAwaitable[OrderT]]
    ) -> Optional[T]:
        ...

    async def min_or_null(
        self: "AsyncIter[Any]", *, key: Optional[Callable[[Any], MaybeAwaitable[Any]]] = None
    ) -> Optional[Any]:
        if key is None:
            return await self.min_or(None)

        return await self.min_or(None, key=key)

    async def sum(self, start: MarkerOr[T] = marker) -> T:
        return await async_sum(self._iterator, start)

    async def product(self, start: MarkerOr[T] = marker) -> T:
        return await async_product(self._iterator, start)

    def get_all(self, **attrs: Any) -> "AsyncIter[T]":
        return self.__class__(async_get(self._iterator, **attrs))

    async def get(self, **attrs: Any) -> T:
        return await self.get_all(**attrs).next()

    async def get_or(self, *, default: U, **attrs: Any) -> Union[T, U]:
        return await self.get_all(**attrs).next_or(default)

    async def get_or_null(self, **attrs: Any) -> Optional[T]:
        return await self.get_all(**attrs).next_or_null()

    async def last(self) -> T:
        return await async_last(self._iterator)

    async def last_or(self, default: U) -> Union[T, U]:
        return await async_last(self._iterator, default)

    async def last_or_null(self) -> Optional[T]:
        return await self.last_or(None)

    def flatten(self: "AsyncIter[AnyIterable[V]]") -> "AsyncIter[V]":
        return self.__class__(async_flatten(self._iterator))  # type: ignore

    def parallel_flatten(self: "AsyncIter[AnyIterable[V]]") -> "AsyncIter[V]":
        return self.__class__(async_parallel_flatten(self._iterator))  # type: ignore

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

    async def at(self, n: int) -> T:
        return await async_at(self._iterator, n)

    async def at_or(self, n: int, default: U) -> Union[T, U]:
        return await async_at(self._iterator, n, default)

    async def at_or_null(self, n: int) -> Optional[T]:
        return await self.at_or(n, None)

    async def at_or_last(self, n: int) -> T:
        return await async_at_or_last(self._iterator, n)

    async def length(self) -> int:
        return await async_iter_len(self._iterator)

    def run_iterators(
        self: "AsyncIter[AnyIterable[V]]",
        *ignore_exceptions: Type[BaseException],
        concurrent: bool = True,
    ) -> "AsyncIter[V]":
        return self.__class__(  # type: ignore
            run_iterators(
                self._iterator, *ignore_exceptions, concurrent=concurrent  # type: ignore
            )
        )

    def map(self, function: Callable[[T], MaybeAwaitable[U]]) -> "AsyncIter[U]":
        return self.__class__(async_map(function, self._iterator))

    def map_nowait(self, function: Callable[[T], U]) -> "AsyncIter[U]":
        return self.__class__(async_map_nowait(function, self._iterator))

    def parallel_map(self, function: Callable[[T], MaybeAwaitable[U]]) -> "AsyncIter[U]":
        return self.__class__(async_parallel_map(function, self._iterator))

    def star_map(
        self: "AsyncIter[AnyIterable[Any]]", function: Callable[..., MaybeAwaitable[V]]
    ) -> "AsyncIter[V]":
        return self.__class__(async_star_map(function, self._iterator))  # type: ignore

    def star_map_nowait(
        self: "AsyncIter[AnyIterable[Any]]", function: Callable[..., V]
    ) -> "AsyncIter[V]":
        return self.__class__(async_star_map_nowait(function, self._iterator))  # type: ignore

    def parallel_star_map(
        self: "AsyncIter[AnyIterable[Any]]", function: Callable[..., MaybeAwaitable[V]]
    ) -> "AsyncIter[V]":
        return self.__class__(async_parallel_star_map(function, self._iterator))  # type: ignore

    def partition(
        self, predicate: Callable[[T], MaybeAwaitable[Any]]
    ) -> "Tuple[AsyncIter[T], AsyncIter[T]]":
        with_true, with_false = async_partition(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_infinite(
        self, predicate: Callable[[T], MaybeAwaitable[Any]]
    ) -> "Tuple[AsyncIter[T], AsyncIter[T]]":
        with_true, with_false = async_partition_infinite(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_safe(
        self, predicate: Callable[[T], MaybeAwaitable[Any]]
    ) -> "Tuple[AsyncIter[T], AsyncIter[T]]":
        with_true, with_false = async_partition_safe(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    @overload
    def zip(self, __iterable_1: AnyIterable[T1]) -> "AsyncIter[Tuple[T, T1]]":
        ...

    @overload
    def zip(
        self, __iterable_1: AnyIterable[T1], __iterable_2: AnyIterable[T2]
    ) -> "AsyncIter[Tuple[T, T1, T2]]":
        ...

    @overload
    def zip(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
    ) -> "AsyncIter[Tuple[T, T1, T2, T3]]":
        ...

    @overload
    def zip(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
        __iterable_4: AnyIterable[T4],
    ) -> "AsyncIter[Tuple[T, T1, T2, T3, T4]]":
        ...

    @overload
    def zip(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
        __iterable_4: AnyIterable[T4],
        __iterable_5: AnyIterable[T5],
    ) -> "AsyncIter[Tuple[T, T1, T2, T3, T4, T5]]":
        ...

    @overload
    def zip(
        self,
        __iterable_1: AnyIterable[Any],
        __iterable_2: AnyIterable[Any],
        __iterable_3: AnyIterable[Any],
        __iterable_4: AnyIterable[Any],
        __iterable_5: AnyIterable[Any],
        __iterable_6: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> "AsyncIter[Tuple[Any, ...]]":
        ...

    @no_type_check
    def zip(self, *iterables: AnyIterable[Any]) -> "AsyncIter[Tuple[Any, ...]]":
        return self.__class__(async_zip(self._iterator, *iterables))

    @overload
    def zip_longest(
        self, __iterable_1: AnyIterable[T1]
    ) -> "AsyncIter[Tuple[Optional[T], Optional[T1]]]":
        ...

    @overload
    def zip_longest(
        self, __iterable_1: AnyIterable[T1], __iterable_2: AnyIterable[T2]
    ) -> "AsyncIter[Tuple[Optional[T], Optional[T1], Optional[T2]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
    ) -> "AsyncIter[Tuple[Optional[T], Optional[T1], Optional[T2], Optional[T3]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
        __iterable_4: AnyIterable[T4],
    ) -> "AsyncIter[Tuple[Optional[T], Optional[T1], Optional[T2], Optional[T3], Optional[T4]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
        __iterable_4: AnyIterable[T4],
        __iterable_5: AnyIterable[T5],
    ) -> "AsyncIter[Tuple[Optional[T], Optional[T1], Optional[T2], Optional[T3], Optional[T4], Optional[T5]]]":  # noqa
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[Any],
        __iterable_2: AnyIterable[Any],
        __iterable_3: AnyIterable[Any],
        __iterable_4: AnyIterable[Any],
        __iterable_5: AnyIterable[Any],
        __iterable_6: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> "AsyncIter[Tuple[Optional[Any], ...]]":
        ...

    @overload
    def zip_longest(
        self, __iterable_1: AnyIterable[T1], *, fill: U
    ) -> "AsyncIter[Tuple[Union[T, U], Union[T1, U]]]":
        ...

    @overload
    def zip_longest(
        self, __iterable_1: AnyIterable[T1], __iterable_2: AnyIterable[T2], *, fill: U
    ) -> "AsyncIter[Tuple[Union[T, U], Union[T1, U], Union[T2, U]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
        *,
        fill: U
    ) -> "AsyncIter[Tuple[Union[T, U], Union[T1, U], Union[T2, U], Union[T3, U]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
        __iterable_4: AnyIterable[T4],
        *,
        fill: U
    ) -> "AsyncIter[Tuple[Union[T, U], Union[T1, U], Union[T2, U], Union[T3, U], Union[T4, U]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[T1],
        __iterable_2: AnyIterable[T2],
        __iterable_3: AnyIterable[T3],
        __iterable_4: AnyIterable[T4],
        __iterable_5: AnyIterable[T5],
        *,
        fill: U
    ) -> "AsyncIter[Tuple[Union[T, U], Union[T1, U], Union[T2, U], Union[T3, U], Union[T4, U], Union[T5, U]]]":  # noqa
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: AnyIterable[Any],
        __iterable_2: AnyIterable[Any],
        __iterable_3: AnyIterable[Any],
        __iterable_4: AnyIterable[Any],
        __iterable_5: AnyIterable[Any],
        __iterable_6: AnyIterable[Any],
        *iterables: AnyIterable[Any],
        fill: U,
    ) -> "AsyncIter[Tuple[Union[Any, U], ...]]":
        ...

    @no_type_check
    def zip_longest(
        self, *iterables: AnyIterable[Any], fill: Optional[U] = None
    ) -> "AsyncIter[Tuple[Union[Optional[Any], U], ...]]":
        return self.__class__(async_zip_longest(self._iterator, *iterables, fill=fill))

    def side_effect(
        self,
        function: Callable[[T], MaybeAwaitable[None]],
        before: Optional[Callable[[], MaybeAwaitable[None]]] = None,
        after: Optional[Callable[[], MaybeAwaitable[None]]] = None,
    ) -> "AsyncIter[T]":
        return self.__class__(async_side_effect(self._iterator, function, before, after))

    def inspect(self, string: str) -> "AsyncIter[T]":
        def print_and_string(item: Any) -> None:
            print(string, item)

        return self.side_effect(print_and_string)

    def inspect_format(self, format_string: str) -> "AsyncIter[T]":
        def print_format(item: Any) -> None:
            print(format_string.format(item))

        return self.side_effect(print_format)


async_iter = AsyncIter
async_reversed = async_iter.reversed


def return_async_iter(function: Callable[..., AsyncIterable[T]]) -> Callable[..., AsyncIter[T]]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> AsyncIter[T]:
        return async_iter(function(*args, **kwargs))

    return wrapper
