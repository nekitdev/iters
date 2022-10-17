from __future__ import annotations

from functools import wraps
from typing import (
    Any,
    AnyStr,
    AsyncContextManager,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    Counter,
    Dict,
    Generator,
    Hashable,
    Iterator,
    List,
    Optional,
    Reversible,
    Set,
    Tuple,
    TypeVar,
    Union,
    no_type_check,
    overload,
)

from typing_extensions import Literal, Never, ParamSpec

from iters.async_utils import (
    async_accumulate_fold,
    async_accumulate_fold_await,
    async_accumulate_product,
    async_accumulate_reduce,
    async_accumulate_reduce_await,
    async_accumulate_sum,
    async_all,
    async_all_equal,
    async_all_equal_await,
    async_all_unique,
    async_all_unique_await,
    async_all_unique_fast,
    async_all_unique_fast_await,
    async_any,
    async_append,
    async_at,
    async_at_or_last,
    async_cartesian_power,
    async_cartesian_product,
    async_chain,
    async_chain_from_iterable,
    async_chunks,
    async_collapse,
    async_combine,
    async_compare,
    async_compare_await,
    async_compress,
    async_consume,
    async_contains,
    async_contains_identity,
    async_copy,
    async_copy_unsafe,
    async_count,
    async_count_dict,
    async_count_dict_await,
    async_cycle,
    async_dict,
    async_distribute,
    async_distribute_unsafe,
    async_divide,
    async_drop,
    async_drop_while,
    async_drop_while_await,
    async_duplicates,
    async_duplicates_await,
    async_duplicates_fast,
    async_duplicates_fast_await,
    async_empty,
    async_enumerate,
    async_extract,
    async_filter,
    async_filter_await,
    async_filter_await_map,
    async_filter_await_map_await,
    async_filter_except,
    async_filter_except_await,
    async_filter_false,
    async_filter_false_await,
    async_filter_false_await_map,
    async_filter_false_await_map_await,
    async_filter_false_map,
    async_filter_false_map_await,
    async_filter_map,
    async_filter_map_await,
    async_find,
    async_find_all,
    async_find_all_await,
    async_find_await,
    async_find_or_first,
    async_find_or_first_await,
    async_find_or_last,
    async_find_or_last_await,
    async_first,
    async_flat_map,
    async_flat_map_await,
    async_flatten,
    async_fold,
    async_fold_await,
    async_for_each,
    async_for_each_await,
    async_group,
    async_group_await,
    async_group_dict,
    async_group_dict_await,
    async_group_list,
    async_group_list_await,
    async_groups,
    async_groups_longest,
    async_has_next,
    async_interleave,
    async_interleave_longest,
    async_intersperse,
    async_intersperse_with,
    async_intersperse_with_await,
    async_is_empty,
    async_is_sorted,
    async_is_sorted_await,
)
from iters.async_utils import async_iter as async_iter_any_iter
from iters.async_utils import (
    async_iter_async_with,
    async_iter_chunks,
    async_iter_chunks_unsafe,
    async_iter_except,
    async_iter_except_await,
    async_iter_function,
    async_iter_function_await,
    async_iter_length,
    async_iter_slice,
    async_iter_windows,
    async_iter_with,
    async_iterate,
    async_iterate_await,
    async_last,
    async_last_with_tail,
    async_list,
    async_list_windows,
    async_map,
    async_map_await,
    async_map_except,
    async_map_except_await,
    async_max,
    async_max_await,
    async_min,
    async_min_await,
    async_min_max,
    async_min_max_await,
    async_next,
    async_next_unchecked,
    async_once,
    async_once_with,
    async_once_with_await,
    async_pad,
    async_pad_with,
    async_pad_with_await,
    async_pairs,
    async_pairs_longest,
    async_pairs_windows,
    async_partition,
    async_partition_await,
    async_partition_unsafe,
    async_partition_unsafe_await,
    async_peek,
    async_position,
    async_position_all,
    async_position_all_await,
    async_position_await,
    async_prepend,
    async_product,
    async_reduce,
    async_reduce_await,
    async_remove,
    async_remove_await,
    async_remove_duplicates,
    async_remove_duplicates_await,
    async_repeat,
    async_repeat_each,
    async_repeat_last,
    async_repeat_with,
    async_repeat_with_await,
    async_reverse,
    async_reversed,
    async_set,
    async_side_effect,
    async_side_effect_await,
    async_sort,
    async_sort_await,
    async_sorted,
    async_sorted_await,
    async_spy,
    async_step_by,
    async_sum,
    async_tail,
    async_take,
    async_take_while,
    async_take_while_await,
    async_tuple,
    async_tuple_windows,
    async_unique,
    async_unique_await,
    async_unique_fast,
    async_unique_fast_await,
    async_wait,
    async_zip,
    async_zip_equal,
    async_zip_longest,
    standard_async_iter,
    standard_async_next,
)
from iters.concurrent import CONCURRENT
from iters.types import Ordering
from iters.typing import (
    AnyExceptionType,
    AnyIterable,
    AnySelectors,
    AsyncBinary,
    AsyncNullary,
    AsyncPredicate,
    AsyncUnary,
    Binary,
    DynamicTuple,
    EitherLenientOrdered,
    EitherStrictOrdered,
    EmptyTuple,
    Nullary,
    Predicate,
    Product,
    RecursiveAnyIterable,
    Sum,
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

if CONCURRENT:
    from iters.async_utils import (
        async_map_concurrent,
        async_map_concurrent_bound,
        async_wait_concurrent,
        async_wait_concurrent_bound,
    )

__all__ = (
    # the async iterator type
    "AsyncIter",
    # the alias of the previous type
    "async_iter",
    # next functions; checked version works on any iterator, unchecked assumes async iteration
    "async_next",
    "async_next_unchecked",
    # since we are "shadowing" standard functions
    "standard_async_iter",
    "standard_async_next",
    # wrap results of function calls into async iterators
    "wrap_async_iter",
)

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)

V = TypeVar("V")
W = TypeVar("W")

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")

Q = TypeVar("Q", bound=Hashable)

S = TypeVar("S", bound=Sum)
P = TypeVar("P", bound=Product)

LT = TypeVar("LT", bound=EitherLenientOrdered)
ST = TypeVar("ST", bound=EitherStrictOrdered)

PS = ParamSpec("PS")

EMPTY_BYTES = bytes()
EMPTY_STRING = str()

DEFAULT_START = 0
DEFAULT_STEP = 1


class AsyncIter(AsyncIterator[T]):
    _iterator: AsyncIterator[T]

    @property
    def iterator(self) -> AsyncIterator[T]:
        return self._iterator

    def _replace(self, iterator: AsyncIterator[T]) -> None:
        self._iterator = iterator

    @classmethod
    def empty(cls) -> AsyncIter[T]:
        return cls.create(async_empty())

    @classmethod
    def once(cls, value: V) -> AsyncIter[V]:
        return cls.create(async_once(value))

    @classmethod
    def once_with(cls, function: Nullary[V]) -> AsyncIter[V]:
        return cls.create(async_once_with(function))

    @classmethod
    def once_with_await(cls, function: AsyncNullary[V]) -> AsyncIter[V]:
        return cls.create(async_once_with_await(function))

    @classmethod
    def repeat(cls, value: V) -> AsyncIter[V]:
        return cls.create(async_repeat(value))

    @classmethod
    def repeat_exactly(cls, value: V, count: int) -> AsyncIter[V]:
        return cls.create(async_repeat(value, count))

    @classmethod
    def repeat_with(cls, function: Nullary[V]) -> AsyncIter[V]:
        return cls.create(async_repeat_with(function))

    @classmethod
    def repeat_with_await(cls, function: AsyncNullary[V]) -> AsyncIter[V]:
        return cls.create(async_repeat_with_await(function))

    @classmethod
    def repeat_exactly_with(cls, function: Nullary[V], count: int) -> AsyncIter[V]:
        return cls.create(async_repeat_with(function, count))

    @classmethod
    def repeat_exactly_with_await(cls, function: AsyncNullary[V], count: int) -> AsyncIter[V]:
        return cls.create(async_repeat_with_await(function, count))

    @classmethod
    def count_from_by(cls, start: int, step: int) -> AsyncIter[int]:
        return cls.create(async_count(start, step))

    @classmethod
    def count_from(cls, start: int) -> AsyncIter[int]:
        return cls.count_from_by(start, DEFAULT_STEP)

    @classmethod
    def count_by(cls, step: int) -> AsyncIter[int]:
        return cls.count_from_by(DEFAULT_START, step)

    @classmethod
    def count(cls) -> AsyncIter[int]:
        return cls.count_from_by(DEFAULT_START, DEFAULT_STEP)

    @classmethod
    def iterate(cls, function: Unary[V, V], value: V) -> AsyncIter[V]:
        return cls.create(async_iterate(function, value))

    @classmethod
    def iterate_exactly(cls, function: Unary[V, V], value: V, count: int) -> AsyncIter[V]:
        return cls.create(async_iterate(function, value, count))

    @classmethod
    def iterate_await(cls, function: AsyncUnary[V, V], value: V) -> AsyncIter[V]:
        return cls.create(async_iterate_await(function, value))

    @classmethod
    def iterate_exactly_await(
        cls, function: AsyncUnary[V, V], value: V, count: int
    ) -> AsyncIter[V]:
        return cls.create(async_iterate_await(function, value, count))

    @classmethod
    def iter_except(cls, function: Nullary[T], *errors: AnyExceptionType) -> AsyncIter[T]:
        return cls.create(async_iter_except(function, *errors))

    @classmethod
    def iter_except_await(
        cls, function: AsyncNullary[T], *errors: AnyExceptionType
    ) -> AsyncIter[T]:
        return cls.create(async_iter_except_await(function, *errors))

    @classmethod
    def iter_with(cls, context_manager: ContextManager[AnyIterable[T]]) -> AsyncIter[T]:
        return cls.create(async_iter_with(context_manager))

    @classmethod
    def iter_async_with(
        cls, async_context_manager: AsyncContextManager[AnyIterable[T]]
    ) -> AsyncIter[T]:
        return cls.create(async_iter_async_with(async_context_manager))

    @classmethod
    def create_chain(cls, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return cls.create(async_chain(*iterables))

    @classmethod
    def create_chain_from_iterable(cls, iterable: AnyIterable[AnyIterable[T]]) -> AsyncIter[T]:
        return cls.create(async_chain_from_iterable(iterable))

    @classmethod
    def create_combine(cls, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return cls.create(async_combine(*iterables))

    @classmethod
    def create_interleave(cls, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return cls.create(async_interleave(*iterables))

    @classmethod
    def create_interleave_longest(cls, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return cls.create(async_interleave_longest(*iterables))

    @overload
    @classmethod
    def create_zip(cls) -> AsyncIter[T]:
        ...

    @overload
    @classmethod
    def create_zip(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[A]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[A, B]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[A, B, C]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[A, B, C, D]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[A, B, C, D, E]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G, H]]:
        ...

    @overload
    @classmethod
    def create_zip(
        cls,
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
    ) -> AsyncIter[DynamicTuple[Any]]:
        ...

    @no_type_check
    @classmethod
    def create_zip(cls, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return cls.create(async_zip(*iterables))

    @overload
    @classmethod
    def create_zip_equal(cls) -> AsyncIter[T]:
        ...

    @overload
    @classmethod
    def create_zip_equal(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[A]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[A, B]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[A, B, C]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[A, B, C, D]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[A, B, C, D, E]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G, H]]:
        ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
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
    ) -> AsyncIter[DynamicTuple[Any]]:
        ...

    @no_type_check
    @classmethod
    def create_zip_equal(cls, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return cls.create(async_zip_equal(*iterables))

    @overload
    @classmethod
    def create_zip_longest(cls) -> AsyncIter[T]:
        ...

    @overload
    @classmethod
    def create_zip_longest(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[A]]:
        ...

    @overload
    @classmethod
    def create_zip_longest(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[Optional[A], Optional[B]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[Optional[A], Optional[B], Optional[C]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[Optional[A], Optional[B], Optional[C], Optional[D]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[
        Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E], Optional[F]]
    ]:
        ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[
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
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[
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
    @classmethod
    def create_zip_longest(
        cls,
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
    ) -> AsyncIter[DynamicTuple[Optional[Any]]]:
        ...

    @no_type_check
    @classmethod
    def create_zip_longest(
        cls, *iterables: AnyIterable[Any]
    ) -> AsyncIter[DynamicTuple[Optional[Any]]]:
        return cls.create(async_zip_longest(*iterables))

    @overload
    @classmethod
    def create_zip_longest_with(cls, *, fill: V) -> AsyncIter[T]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls, __iterable_a: AnyIterable[A], *, fill: V
    ) -> AsyncIter[Tuple[A]]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B], *, fill: V
    ) -> AsyncIter[Tuple[Union[A, V], Union[B, V]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        *,
        fill: V,
    ) -> AsyncIter[Tuple[Union[A, V], Union[B, V], Union[C, V]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        *,
        fill: V,
    ) -> AsyncIter[Tuple[Union[A, V], Union[B, V], Union[C, V], Union[D, V]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        *,
        fill: V,
    ) -> AsyncIter[Tuple[Union[A, V], Union[B, V], Union[C, V], Union[D, V], Union[E, V]]]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        *,
        fill: V,
    ) -> AsyncIter[
        Tuple[Union[A, V], Union[B, V], Union[C, V], Union[D, V], Union[E, V], Union[F, V]]
    ]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        *,
        fill: V,
    ) -> AsyncIter[
        Tuple[
            Union[A, V],
            Union[B, V],
            Union[C, V],
            Union[D, V],
            Union[E, V],
            Union[F, V],
            Union[G, V],
        ]
    ]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
        *,
        fill: V,
    ) -> AsyncIter[
        Tuple[
            Union[A, V],
            Union[B, V],
            Union[C, V],
            Union[D, V],
            Union[E, V],
            Union[F, V],
            Union[G, V],
            Union[H, V],
        ]
    ]:
        ...

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
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
        fill: V,
    ) -> AsyncIter[DynamicTuple[Union[Any, V]]]:
        ...

    @no_type_check
    @classmethod
    def create_zip_longest_with(
        cls, *iterables: AnyIterable[Any], fill: V
    ) -> AsyncIter[DynamicTuple[Union[Any, V]]]:
        return cls.create(async_zip_longest(*iterables, fill=fill))

    @overload
    @classmethod
    def create_cartesian_product(cls) -> AsyncIter[EmptyTuple]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[A]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[A, B]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[A, B, C]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[A, B, C, D]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[A, B, C, D, E]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G, H]]:
        ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
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
    ) -> AsyncIter[DynamicTuple[Any]]:
        ...

    @no_type_check
    @classmethod
    def create_cartesian_product(cls, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return cls.create(async_cartesian_product(*iterables))

    @classmethod
    def reversed(cls, reversible: Reversible[T]) -> AsyncIter[T]:
        return cls.create(async_reversed(reversible))

    @classmethod
    def function(cls, function: Nullary[T], sentinel: V) -> AsyncIter[T]:
        return cls.create(async_iter_function(function, sentinel))

    @classmethod
    def function_await(cls, function: AsyncNullary[T], sentinel: V) -> AsyncIter[T]:
        return cls.create(async_iter_function_await(function, sentinel))

    @classmethod
    def create(cls, iterable: AnyIterable[U]) -> AsyncIter[U]:
        return cls(iterable)  # type: ignore

    @classmethod
    def create_tuple(cls, iterables: DynamicTuple[AnyIterable[U]]) -> DynamicTuple[AsyncIter[U]]:
        return tuple(map(cls, iterables))  # type: ignore

    @classmethod
    def create_nested(cls, nested: AnyIterable[AnyIterable[U]]) -> AsyncIter[AsyncIter[U]]:
        return cls(map(cls, nested))  # type: ignore

    def __init__(self, iterable: AnyIterable[T]) -> None:
        self._iterator = async_iter_any_iter(iterable)

    def __aiter__(self) -> AsyncIter[T]:
        return self.iter()

    async def __anext__(self) -> T:
        return await self.next()

    def __await__(self) -> Generator[None, None, List[T]]:
        return self.list().__await__()

    def unwrap(self) -> AsyncIterator[T]:
        return self.iterator

    def iter(self) -> AsyncIter[T]:
        return self

    async def next(self) -> T:
        return await async_next_unchecked(self.iterator)

    async def next_or(self, default: V) -> Union[T, V]:
        return await async_next_unchecked(self.iterator, default)

    async def next_or_none(self) -> Optional[T]:
        return await self.next_or(None)

    async def compare(self: AsyncIter[ST], other: AnyIterable[ST]) -> Ordering:
        return await async_compare(self.iterator, other)

    async def compare_by(self, other: AnyIterable[T], key: Unary[T, ST]) -> Ordering:
        return await async_compare(self.iterator, other, key)

    async def compare_by_await(self, other: AnyIterable[T], key: AsyncUnary[T, ST]) -> Ordering:
        return await async_compare_await(self.iterator, other, key)

    async def length(self) -> int:
        return await async_iter_length(self.iterator)

    async def first(self) -> T:
        return await async_first(self.iterator)

    async def first_or(self, default: V) -> Union[T, V]:
        return await async_first(self.iterator, default)

    async def first_or_none(self) -> Optional[T]:
        return await self.first_or(None)

    async def last(self) -> T:
        return await async_last(self.iterator)

    async def last_or(self, default: V) -> Union[T, V]:
        return await async_last(self.iterator, default)

    async def last_or_none(self) -> Optional[T]:
        return await self.last_or(None)

    async def last_with_tail(self) -> T:
        return await async_last_with_tail(self.iterator)

    async def last_with_tail_or(self, default: V) -> Union[T, V]:
        return await async_last_with_tail(self.iterator, default)

    async def last_with_tail_or_none(self) -> Optional[T]:
        return await self.last_with_tail_or(None)

    def collect(self, function: Unary[AsyncIterable[T], U]) -> U:
        return function(self.iterator)

    async def collect_await(self, function: AsyncUnary[AsyncIterable[T], U]) -> U:
        return await function(self.iterator)

    async def list(self) -> List[T]:
        return await async_list(self.iterator)

    async def set(self: AsyncIter[Q]) -> Set[Q]:
        return await async_set(self.iterator)

    async def tuple(self) -> DynamicTuple[T]:
        return await async_tuple(self.iterator)

    async def dict(self: AsyncIter[Tuple[Q, V]]) -> Dict[Q, V]:
        return await async_dict(self.iterator)

    async def extract(self) -> Iterator[T]:
        return await async_extract(self.iterator)

    async def join(self: AsyncIter[AnyStr], string: AnyStr) -> AnyStr:
        return string.join(await self.list())

    async def string(self: AsyncIter[str]) -> str:
        return await self.join(EMPTY_STRING)

    async def bytes(self: AsyncIter[bytes]) -> bytes:
        return await self.join(EMPTY_BYTES)

    async def count_dict(self: AsyncIter[Q]) -> Counter[Q]:
        return await async_count_dict(self.iterator)

    async def count_dict_by(self, key: Unary[T, Q]) -> Counter[Q]:
        return await async_count_dict(self.iterator, key)

    async def count_dict_by_await(self, key: AsyncUnary[T, Q]) -> Counter[Q]:
        return await async_count_dict_await(self.iterator, key)

    async def group_dict(self: AsyncIter[Q]) -> Dict[Q, List[Q]]:
        return await async_group_dict(self.iterator)

    async def group_dict_by(self, key: Unary[T, Q]) -> Dict[Q, List[T]]:
        return await async_group_dict(self.iterator, key)

    async def group_dict_by_await(self, key: AsyncUnary[T, Q]) -> Dict[Q, List[T]]:
        return await async_group_dict_await(self.iterator, key)

    def group(self) -> AsyncIter[Tuple[T, AsyncIter[T]]]:
        return self.create(
            (group_key, self.create(group_iterator))
            async for group_key, group_iterator in async_group(self.iterator)
        )

    def group_by(self, key: Unary[T, U]) -> AsyncIter[Tuple[U, AsyncIter[T]]]:
        return self.create(
            (group_key, self.create(group_iterator))
            async for group_key, group_iterator in async_group(self.iterator, key)
        )

    def group_by_await(self, key: AsyncUnary[T, U]) -> AsyncIter[Tuple[U, AsyncIter[T]]]:
        return self.create(
            (group_key, self.create(group_iterator))
            async for group_key, group_iterator in async_group_await(self.iterator, key)
        )

    def group_list(self) -> AsyncIter[Tuple[T, List[T]]]:
        return self.create(async_group_list(self.iterator))

    def group_list_by(self, key: Unary[T, U]) -> AsyncIter[Tuple[U, List[T]]]:
        return self.create(async_group_list(self.iterator, key))

    def group_list_by_await(self, key: AsyncUnary[T, U]) -> AsyncIter[Tuple[U, List[T]]]:
        return self.create(async_group_list_await(self.iterator, key))

    async def all(self) -> bool:
        return await async_all(self.iterator)

    async def all_by(self, predicate: Predicate[T]) -> bool:
        return await self.map(predicate).all()

    async def all_by_await(self, predicate: AsyncPredicate[T]) -> bool:
        return await self.map_await(predicate).all()

    async def any(self) -> bool:
        return await async_any(self.iterator)

    async def any_by(self, predicate: Predicate[T]) -> bool:
        return await self.map(predicate).any()

    async def any_by_await(self, predicate: AsyncPredicate[T]) -> bool:
        return await self.map_await(predicate).any()

    async def all_equal(self) -> bool:
        return await async_all_equal(self.iterator)

    async def all_equal_by(self, key: Unary[T, U]) -> bool:
        return await async_all_equal(self.iterator, key)

    async def all_equal_by_await(self, key: AsyncUnary[T, U]) -> bool:
        return await async_all_equal_await(self.iterator, key)

    async def all_unique(self) -> bool:
        return await async_all_unique(self.iterator)

    async def all_unique_by(self, key: Unary[T, U]) -> bool:
        return await async_all_unique(self.iterator, key)

    async def all_unique_by_await(self, key: AsyncUnary[T, U]) -> bool:
        return await async_all_unique_await(self.iterator, key)

    async def all_unique_fast(self: AsyncIter[Q]) -> bool:
        return await async_all_unique_fast(self.iterator)

    async def all_unique_fast_by(self, key: Unary[T, Q]) -> bool:
        return await async_all_unique_fast(self.iterator, key)

    async def all_unique_fast_by_await(self, key: AsyncUnary[T, Q]) -> bool:
        return await async_all_unique_fast_await(self.iterator, key)

    def remove(self, predicate: Optional[Predicate[T]]) -> AsyncIter[T]:
        return self.create(async_remove(predicate, self.iterator))

    def remove_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_remove_await(predicate, self.iterator))

    def remove_duplicates(self) -> AsyncIter[T]:
        return self.create(async_remove_duplicates(self.iterator))

    def remove_duplicates_by(self, key: Unary[T, U]) -> AsyncIter[T]:
        return self.create(async_remove_duplicates(self.iterator, key))

    def remove_duplicates_by_await(self, key: AsyncUnary[T, U]) -> AsyncIter[T]:
        return self.create(async_remove_duplicates_await(self.iterator, key))

    def filter(self, predicate: Optional[Predicate[T]]) -> AsyncIter[T]:
        return self.create(async_filter(predicate, self.iterator))

    def filter_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_filter_await(predicate, self.iterator))

    def filter_false(self, predicate: Optional[Predicate[T]]) -> AsyncIter[T]:
        return self.create(async_filter_false(predicate, self.iterator))

    def filter_false_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_filter_false_await(predicate, self.iterator))

    def filter_except(self, validate: Unary[T, Any], *errors: AnyExceptionType) -> AsyncIter[T]:
        return self.create(async_filter_except(validate, self.iterator, *errors))

    def filter_except_await(
        self, validate: AsyncUnary[T, Any], *errors: AnyExceptionType
    ) -> AsyncIter[T]:
        return self.create(async_filter_except_await(validate, self.iterator, *errors))

    def compress(self, selectors: AnySelectors) -> AsyncIter[T]:
        return self.create(async_compress(self.iterator, selectors))

    def position_all(self, predicate: Optional[Predicate[T]]) -> AsyncIter[int]:
        return self.create(async_position_all(predicate, self.iterator))

    def position_all_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[int]:
        return self.create(async_position_all_await(predicate, self.iterator))

    async def position(self, predicate: Optional[Predicate[T]]) -> int:
        return await async_position(predicate, self.iterator)

    async def position_or(self, predicate: Optional[Predicate[T]], default: V) -> Union[int, V]:
        return await async_position(predicate, self.iterator, default)

    async def position_or_none(self, predicate: Optional[Predicate[T]]) -> Optional[int]:
        return await self.position_or(predicate, None)

    async def position_await(self, predicate: AsyncPredicate[T]) -> int:
        return await async_position_await(predicate, self.iterator)

    async def position_await_or(self, predicate: AsyncPredicate[T], default: V) -> Union[int, V]:
        return await async_position_await(predicate, self.iterator, default)

    async def position_await_or_none(self, predicate: AsyncPredicate[T]) -> Optional[int]:
        return await self.position_await_or(predicate, None)

    def find_all(self, predicate: Optional[Predicate[T]]) -> AsyncIter[T]:
        return self.create(async_find_all(predicate, self.iterator))

    def find_all_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_find_all_await(predicate, self.iterator))

    async def find(self, predicate: Optional[Predicate[T]]) -> T:
        return await async_find(predicate, self.iterator)

    async def find_or(self, predicate: Optional[Predicate[T]], default: V) -> Union[T, V]:
        return await async_find(predicate, self.iterator, default)  # type: ignore  # strange

    async def find_or_none(self, predicate: Optional[Predicate[T]]) -> Optional[T]:
        return await self.find_or(predicate, None)

    async def find_await(self, predicate: AsyncPredicate[T]) -> T:
        return await async_find_await(predicate, self.iterator)

    async def find_await_or(self, predicate: AsyncPredicate[T], default: V) -> Union[T, V]:
        return await async_find_await(predicate, self.iterator, default)  # type: ignore  # strange

    async def find_await_or_none(self, predicate: AsyncPredicate[T]) -> Optional[T]:
        return await self.find_await_or(predicate, None)

    async def find_or_first(self, predicate: Optional[Predicate[T]]) -> T:
        return await async_find_or_first(predicate, self.iterator)

    async def find_or_first_or(self, predicate: Optional[Predicate[T]], default: V) -> Union[T, V]:
        return await async_find_or_first(predicate, self.iterator, default)  # type: ignore  # strange

    async def find_or_first_or_none(self, predicate: Optional[Predicate[T]]) -> Optional[T]:
        return await self.find_or_first_or(predicate, None)

    async def find_or_first_await(self, predicate: AsyncPredicate[T]) -> T:
        return await async_find_or_first_await(predicate, self.iterator)

    async def find_or_first_await_or(self, predicate: AsyncPredicate[T], default: V) -> Union[T, V]:
        return await async_find_or_first_await(predicate, self.iterator, default)  # type: ignore  # strange

    async def find_or_first_await_or_none(self, predicate: AsyncPredicate[T]) -> Optional[T]:
        return await self.find_or_first_await_or(predicate, None)

    async def find_or_last(self, predicate: Optional[Predicate[T]]) -> T:
        return await async_find_or_last(predicate, self.iterator)

    async def find_or_last_or(self, predicate: Optional[Predicate[T]], default: V) -> Union[T, V]:
        return await async_find_or_last(predicate, self.iterator, default)  # type: ignore  # strange

    async def find_or_last_or_none(self, predicate: Optional[Predicate[T]]) -> Optional[T]:
        return await self.find_or_last_or(predicate, None)

    async def find_or_last_await(self, predicate: AsyncPredicate[T]) -> T:
        return await async_find_or_last_await(predicate, self.iterator)

    async def find_or_last_await_or(self, predicate: AsyncPredicate[T], default: V) -> Union[T, V]:
        return await async_find_or_last_await(predicate, self.iterator, default)  # type: ignore  # strange

    async def find_or_last_await_or_none(self, predicate: AsyncPredicate[T]) -> Optional[T]:
        return await self.find_or_last_await_or(predicate, None)

    async def contains(self, item: V) -> bool:
        return await async_contains(item, self.iterator)

    async def contains_identity(self: AsyncIter[V], item: V) -> bool:
        return await async_contains_identity(item, self.iterator)

    async def reduce(self, function: Binary[T, T, T]) -> T:
        return await async_reduce(function, self.iterator)

    async def reduce_await(self, function: AsyncBinary[T, T, T]) -> T:
        return await async_reduce_await(function, self.iterator)

    async def fold(self, initial: V, function: Binary[V, T, V]) -> V:
        return await async_fold(initial, function, self.iterator)

    async def fold_await(self, initial: V, function: AsyncBinary[V, T, V]) -> V:
        return await async_fold_await(initial, function, self.iterator)

    async def sum(self: AsyncIter[S]) -> S:
        return await async_sum(self.iterator)

    async def sum_with(self: AsyncIter[S], initial: S) -> S:
        return await async_sum(self.iterator, initial)

    async def product(self: AsyncIter[P]) -> P:
        return await async_product(self.iterator)

    async def product_with(self: AsyncIter[P], initial: P) -> P:
        return await async_product(self.iterator, initial)

    def accumulate_reduce(self, function: Binary[T, T, T]) -> AsyncIter[T]:
        return self.create(async_accumulate_reduce(function, self.iterator))

    def accumulate_reduce_await(self, function: AsyncBinary[T, T, T]) -> AsyncIter[T]:
        return self.create(async_accumulate_reduce_await(function, self.iterator))

    def accumulate_fold(self, initial: V, function: Binary[V, T, V]) -> AsyncIter[V]:
        return self.create(async_accumulate_fold(initial, function, self.iterator))

    def accumulate_fold_await(self, initial: V, function: AsyncBinary[V, T, V]) -> AsyncIter[V]:
        return self.create(async_accumulate_fold_await(initial, function, self.iterator))

    def accumulate_sum(self: AsyncIter[S]) -> AsyncIter[S]:
        return self.create(async_accumulate_sum(self.iterator))

    def accumulate_sum_with(self: AsyncIter[S], initial: S) -> AsyncIter[S]:
        return self.create(async_accumulate_sum(self.iterator, initial))

    def accumulate_product(self: AsyncIter[P]) -> AsyncIter[P]:
        return self.create(async_accumulate_product(self.iterator))

    def accumulate_product_with(self: AsyncIter[P], initial: P) -> AsyncIter[P]:
        return self.create(async_accumulate_product(self.iterator, initial))

    async def min(self: AsyncIter[ST]) -> ST:
        return await async_min(self.iterator)

    async def min_or(self: AsyncIter[ST], default: V) -> Union[ST, V]:
        return await async_min(self.iterator, default=default)

    async def min_or_none(self: AsyncIter[ST]) -> Optional[ST]:
        return await self.min_or(None)

    async def min_by(self, key: Unary[T, ST]) -> T:
        return await async_min(self.iterator, key=key)

    async def min_by_or(self, key: Unary[T, ST], default: V) -> Union[T, V]:
        return await async_min(self.iterator, key=key, default=default)  # type: ignore  # strange

    async def min_by_or_none(self, key: Unary[T, ST]) -> Optional[T]:
        return await self.min_by_or(key, None)

    async def min_by_await(self, key: AsyncUnary[T, ST]) -> T:
        return await async_min_await(self.iterator, key=key)

    async def min_by_await_or(self, key: AsyncUnary[T, ST], default: V) -> Union[T, V]:
        return await async_min_await(self.iterator, key=key, default=default)  # type: ignore  # strange

    async def min_by_await_or_none(self, key: AsyncUnary[T, ST]) -> Optional[T]:
        return await self.min_by_await_or(key, None)

    async def max(self: AsyncIter[ST]) -> ST:
        return await async_max(self.iterator)

    async def max_or(self: AsyncIter[ST], default: V) -> Union[ST, V]:
        return await async_max(self.iterator, default=default)

    async def max_or_none(self: AsyncIter[ST]) -> Optional[ST]:
        return await self.max_or(None)

    async def max_by(self, key: Unary[T, ST]) -> T:
        return await async_max(self.iterator, key=key)

    async def max_by_or(self, key: Unary[T, ST], default: V) -> Union[T, V]:
        return await async_max(self.iterator, key=key, default=default)  # type: ignore  # strange

    async def max_by_or_none(self, key: Unary[T, ST]) -> Optional[T]:
        return await self.max_by_or(key, None)

    async def max_by_await(self, key: AsyncUnary[T, ST]) -> T:
        return await async_max_await(self.iterator, key=key)

    async def max_by_await_or(self, key: AsyncUnary[T, ST], default: V) -> Union[T, V]:
        return await async_max_await(self.iterator, key=key, default=default)  # type: ignore  # strange

    async def max_by_await_or_none(self, key: AsyncUnary[T, ST]) -> Optional[T]:
        return await self.max_by_await_or(key, None)

    async def min_max(self: AsyncIter[ST]) -> Tuple[ST, ST]:
        return await async_min_max(self.iterator)

    async def min_max_or(
        self: AsyncIter[ST], default: Tuple[V, W]
    ) -> Union[Tuple[ST, ST], Tuple[V, W]]:
        return await async_min_max(self.iterator, default=default)

    async def min_max_by(self, key: Unary[T, ST]) -> Tuple[T, T]:
        return await async_min_max(self.iterator, key=key)

    async def min_max_by_or(
        self, key: Unary[T, ST], default: Tuple[V, W]
    ) -> Tuple[Union[T, V], Union[T, W]]:
        return await async_min_max(self.iterator, key=key, default=default)

    async def min_max_by_await(self, key: AsyncUnary[T, ST]) -> Tuple[T, T]:
        return await async_min_max_await(self.iterator, key=key)

    async def min_max_by_await_or(
        self, key: AsyncUnary[T, ST], default: Tuple[V, W]
    ) -> Tuple[Union[T, V], Union[T, W]]:
        return await async_min_max_await(self.iterator, key=key, default=default)

    def map(self, function: Unary[T, U]) -> AsyncIter[U]:
        return self.create(async_map(function, self.iterator))

    def map_await(self, function: AsyncUnary[T, U]) -> AsyncIter[U]:
        return self.create(async_map_await(function, self.iterator))

    def map_except(self, function: Unary[T, U], *errors: AnyExceptionType) -> AsyncIter[U]:
        return self.create(async_map_except(function, self.iterator, *errors))

    def map_except_await(
        self, function: AsyncUnary[T, U], *errors: AnyExceptionType
    ) -> AsyncIter[U]:
        return self.create(async_map_except_await(function, self.iterator, *errors))

    if CONCURRENT:

        def map_concurrent(self, function: AsyncUnary[T, U]) -> AsyncIter[U]:
            return self.create(async_map_concurrent(function, self.iterator))

        def map_concurrent_bound(self, bound: int, function: AsyncUnary[T, U]) -> AsyncIter[U]:
            return self.create(async_map_concurrent_bound(bound, function, self.iterator))

    def flat_map(self, function: Unary[T, AnyIterable[U]]) -> AsyncIter[U]:
        return self.create(async_flat_map(function, self.iterator))

    def flat_map_await(self, function: AsyncUnary[T, AnyIterable[U]]) -> AsyncIter[U]:
        return self.create(async_flat_map_await(function, self.iterator))

    def filter_map(self, predicate: Optional[Predicate[T]], function: Unary[T, U]) -> AsyncIter[U]:
        return self.create(async_filter_map(predicate, function, self.iterator))

    def filter_await_map(self, predicate: AsyncPredicate[T], function: Unary[T, U]) -> AsyncIter[U]:
        return self.create(async_filter_await_map(predicate, function, self.iterator))

    def filter_map_await(
        self, predicate: Optional[Predicate[T]], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_map_await(predicate, function, self.iterator))

    def filter_await_map_await(
        self, predicate: AsyncPredicate[T], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_await_map_await(predicate, function, self.iterator))

    def filter_false_map(
        self, predicate: Optional[Predicate[T]], function: Unary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_map(predicate, function, self.iterator))

    def filter_false_await_map(
        self, predicate: AsyncPredicate[T], function: Unary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_await_map(predicate, function, self.iterator))

    def filter_false_map_await(
        self, predicate: Optional[Predicate[T]], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_map_await(predicate, function, self.iterator))

    def filter_false_await_map_await(
        self, predicate: AsyncPredicate[T], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_await_map_await(predicate, function, self.iterator))

    def flatten(self: AsyncIter[AnyIterable[U]]) -> AsyncIter[U]:
        return self.create(async_flatten(self.iterator))

    def collapse(self: AsyncIter[RecursiveAnyIterable[U]]) -> AsyncIter[U]:
        return self.create(async_collapse(self.iterator))

    def enumerate(self) -> AsyncIter[Tuple[int, T]]:
        return self.create(async_enumerate(self.iterator))

    def enumerate_from(self, start: int) -> AsyncIter[Tuple[int, T]]:
        return self.create(async_enumerate(self.iterator, start))

    async def consume(self) -> None:
        await async_consume(self.iterator)

    async def for_each(self, function: Unary[T, Any]) -> None:
        await async_for_each(function, self.iterator)

    async def for_each_await(self, function: AsyncUnary[T, Any]) -> None:
        await async_for_each_await(function, self.iterator)

    def append(self: AsyncIter[V], item: V) -> AsyncIter[V]:
        return self.create(async_append(item, self.iterator))

    def prepend(self: AsyncIter[V], item: V) -> AsyncIter[V]:
        return self.create(async_prepend(item, self.iterator))

    async def at(self, index: int) -> T:
        return await async_at(index, self.iterator)

    async def at_or(self, index: int, default: V) -> Union[T, V]:
        return await async_at(index, self.iterator, default)

    async def at_or_none(self, index: int) -> Optional[T]:
        return await self.at_or(index, None)

    async def at_or_last(self, index: int) -> T:
        return await async_at_or_last(index, self.iterator)

    async def at_or_last_or(self, index: int, default: V) -> Union[T, V]:
        return await async_at_or_last(index, self.iterator, default)

    async def at_or_last_or_none(self, index: int) -> Optional[T]:
        return await self.at_or_last_or(index, None)

    @overload
    def slice(self, __stop: Optional[int]) -> AsyncIter[T]:
        ...

    @overload
    def slice(
        self, __start: Optional[int], __stop: Optional[int], __step: Optional[int] = ...
    ) -> AsyncIter[T]:
        ...

    def slice(self, *slice_args: Optional[int]) -> AsyncIter[T]:
        return self.create(async_iter_slice(self.iterator, *slice_args))

    def drop(self, size: int) -> AsyncIter[T]:
        return self.create(async_drop(size, self.iterator))

    skip = drop

    def drop_while(self, predicate: Predicate[T]) -> AsyncIter[T]:
        return self.create(async_drop_while(predicate, self.iterator))

    skip_while = drop_while

    def drop_while_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_drop_while_await(predicate, self.iterator))

    skip_while_await = drop_while_await

    def take(self, size: int) -> AsyncIter[T]:
        return self.create(async_take(size, self.iterator))

    def take_while(self, predicate: Predicate[T]) -> AsyncIter[T]:
        return self.create(async_take_while(predicate, self.iterator))

    def take_while_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_take_while_await(predicate, self.iterator))

    def step_by(self, step: int) -> AsyncIter[T]:
        return self.create(async_step_by(step, self.iterator))

    def tail(self, size: int) -> AsyncIter[T]:
        return self.create(async_tail(size, self.iterator))

    def chain(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_chain(self.iterator, *iterables))

    def chain_with(self, iterables: AnyIterable[AnyIterable[T]]) -> AsyncIter[T]:
        return self.chain(async_chain_from_iterable(iterables))

    def cycle(self) -> AsyncIter[T]:
        return self.create(async_cycle(self.iterator))

    def intersperse(self: AsyncIter[V], value: V) -> AsyncIter[V]:
        return self.create(async_intersperse(value, self.iterator))

    def intersperse_with(self, function: Nullary[T]) -> AsyncIter[T]:
        return self.create(async_intersperse_with(function, self.iterator))

    def intersperse_with_await(self, function: AsyncNullary[T]) -> AsyncIter[T]:
        return self.create(async_intersperse_with_await(function, self.iterator))

    def interleave(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_interleave(self.iterator, *iterables))

    def interleave_longest(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_interleave_longest(self.iterator, *iterables))

    def combine(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_combine(*iterables))

    @overload
    def distribute_unsafe(self, count: Literal[0]) -> EmptyTuple:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[1]) -> Tuple1[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[2]) -> Tuple2[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[3]) -> Tuple3[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[4]) -> Tuple4[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[5]) -> Tuple5[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[6]) -> Tuple6[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[7]) -> Tuple7[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: Literal[8]) -> Tuple8[AsyncIter[T]]:
        ...

    @overload
    def distribute_unsafe(self, count: int) -> DynamicTuple[AsyncIter[T]]:
        ...

    def distribute_unsafe(self, count: int) -> DynamicTuple[AsyncIter[T]]:
        return self.create_tuple(async_distribute_unsafe(count, self.iterator))

    distribute_infinite = distribute_unsafe

    @overload
    def distribute(self, count: Literal[0]) -> EmptyTuple:
        ...

    @overload
    def distribute(self, count: Literal[1]) -> Tuple1[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: Literal[2]) -> Tuple2[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: Literal[3]) -> Tuple3[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: Literal[4]) -> Tuple4[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: Literal[5]) -> Tuple5[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: Literal[6]) -> Tuple6[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: Literal[7]) -> Tuple7[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: Literal[8]) -> Tuple8[AsyncIter[T]]:
        ...

    @overload
    def distribute(self, count: int) -> DynamicTuple[AsyncIter[T]]:
        ...

    def distribute(self, count: int) -> DynamicTuple[AsyncIter[T]]:
        return self.create_tuple(async_distribute(count, self.iterator))

    def divide(self, count: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_divide(count, self.iterator))

    def pad(self, value: V) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad(value, self.iterator))

    def pad_exactly(self, value: V, size: int) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad(value, self.iterator, size))

    def pad_multiple(self, value: V, size: int) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad(value, self.iterator, size, multiple=True))

    def pad_none(self) -> AsyncIter[Optional[T]]:
        return self.pad(None)

    def pad_none_exactly(self, size: int) -> AsyncIter[Optional[T]]:
        return self.pad_exactly(None, size)

    def pad_none_multiple(self, size: int) -> AsyncIter[Optional[T]]:
        return self.pad_multiple(None, size)

    def pad_with(self, function: Unary[int, V]) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad_with(function, self.iterator))

    def pad_exactly_with(self, function: Unary[int, V], size: int) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad_with(function, self.iterator, size))

    def pad_multiple_with(self, function: Unary[int, V], size: int) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad_with(function, self.iterator, size, multiple=True))

    def pad_with_await(self, function: AsyncUnary[int, V]) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad_with_await(function, self.iterator))

    def pad_exactly_with_await(
        self, function: AsyncUnary[int, V], size: int
    ) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad_with_await(function, self.iterator, size))

    def pad_multiple_with_await(
        self, function: AsyncUnary[int, V], size: int
    ) -> AsyncIter[Union[T, V]]:
        return self.create(async_pad_with_await(function, self.iterator, size, multiple=True))

    def chunks(self, size: int) -> AsyncIter[List[T]]:
        return self.create(async_chunks(size, self.iterator))

    def iter_chunks(self, size: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_iter_chunks(size, self.iterator))

    def iter_chunks_unsafe(self, size: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_iter_chunks_unsafe(size, self.iterator))

    iter_chunks_infinite = iter_chunks_unsafe

    @overload
    def groups(self, size: Literal[0]) -> AsyncIter[Never]:
        ...

    @overload
    def groups(self, size: Literal[1]) -> AsyncIter[Tuple1[T]]:
        ...

    @overload
    def groups(self, size: Literal[2]) -> AsyncIter[Tuple2[T]]:
        ...

    @overload
    def groups(self, size: Literal[3]) -> AsyncIter[Tuple3[T]]:
        ...

    @overload
    def groups(self, size: Literal[4]) -> AsyncIter[Tuple4[T]]:
        ...

    @overload
    def groups(self, size: Literal[5]) -> AsyncIter[Tuple5[T]]:
        ...

    @overload
    def groups(self, size: Literal[6]) -> AsyncIter[Tuple6[T]]:
        ...

    @overload
    def groups(self, size: Literal[7]) -> AsyncIter[Tuple7[T]]:
        ...

    @overload
    def groups(self, size: Literal[8]) -> AsyncIter[Tuple8[T]]:
        ...

    @overload
    def groups(self, size: int) -> AsyncIter[DynamicTuple[T]]:
        ...

    def groups(self, size: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_groups(size, self.iterator))

    @overload
    def groups_longest(self, size: Literal[0]) -> AsyncIter[Never]:
        ...

    @overload
    def groups_longest(self, size: Literal[1]) -> AsyncIter[Tuple1[T]]:
        ...

    @overload
    def groups_longest(self, size: Literal[2]) -> AsyncIter[Tuple2[Optional[T]]]:
        ...

    @overload
    def groups_longest(self, size: Literal[3]) -> AsyncIter[Tuple3[Optional[T]]]:
        ...

    @overload
    def groups_longest(self, size: Literal[4]) -> AsyncIter[Tuple4[Optional[T]]]:
        ...

    @overload
    def groups_longest(self, size: Literal[5]) -> AsyncIter[Tuple5[Optional[T]]]:
        ...

    @overload
    def groups_longest(self, size: Literal[6]) -> AsyncIter[Tuple6[Optional[T]]]:
        ...

    @overload
    def groups_longest(self, size: Literal[7]) -> AsyncIter[Tuple7[Optional[T]]]:
        ...

    @overload
    def groups_longest(self, size: Literal[8]) -> AsyncIter[Tuple8[Optional[T]]]:
        ...

    @overload
    def groups_longest(self, size: int) -> AsyncIter[DynamicTuple[Optional[T]]]:
        ...

    def groups_longest(self, size: int) -> AsyncIter[DynamicTuple[Optional[T]]]:
        return self.create(async_groups_longest(size, self.iterator))

    @overload
    def groups_longest_with(self, size: Literal[0], fill: V) -> AsyncIter[Never]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[1], fill: V) -> AsyncIter[Tuple1[T]]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[2], fill: V) -> AsyncIter[Tuple2[Union[T, V]]]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[3], fill: V) -> AsyncIter[Tuple3[Union[T, V]]]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[4], fill: V) -> AsyncIter[Tuple4[Union[T, V]]]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[5], fill: V) -> AsyncIter[Tuple5[Union[T, V]]]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[6], fill: V) -> AsyncIter[Tuple6[Union[T, V]]]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[7], fill: V) -> AsyncIter[Tuple7[Union[T, V]]]:
        ...

    @overload
    def groups_longest_with(self, size: Literal[8], fill: V) -> AsyncIter[Tuple8[Union[T, V]]]:
        ...

    @overload
    def groups_longest_with(self, size: int, fill: V) -> AsyncIter[DynamicTuple[Union[T, V]]]:
        ...

    def groups_longest_with(self, size: int, fill: V) -> AsyncIter[DynamicTuple[Union[T, V]]]:
        return self.create(async_groups_longest(size, self.iterator, fill))

    def pairs(self) -> AsyncIter[Tuple[T, T]]:
        return self.create(async_pairs(self.iterator))

    def pairs_longest(self) -> AsyncIter[Tuple[Optional[T], Optional[T]]]:
        return self.create(async_pairs_longest(self.iterator))

    def pairs_longest_with(self, fill: V) -> AsyncIter[Tuple[Union[T, V], Union[T, V]]]:
        return self.create(async_pairs_longest(self.iterator, fill))

    def iter_windows(self, size: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_iter_windows(size, self.iterator))

    def list_windows(self, size: int) -> AsyncIter[List[T]]:
        return self.create(async_list_windows(size, self.iterator))

    def pairs_windows(self) -> AsyncIter[Tuple[T, T]]:
        return self.create(async_pairs_windows(self.iterator))

    @overload
    def tuple_windows(self, size: Literal[0]) -> AsyncIter[EmptyTuple]:
        ...

    @overload
    def tuple_windows(self, size: Literal[1]) -> AsyncIter[Tuple1[T]]:
        ...

    @overload
    def tuple_windows(self, size: Literal[2]) -> AsyncIter[Tuple2[T]]:
        ...

    @overload
    def tuple_windows(self, size: Literal[3]) -> AsyncIter[Tuple3[T]]:
        ...

    @overload
    def tuple_windows(self, size: Literal[4]) -> AsyncIter[Tuple4[T]]:
        ...

    @overload
    def tuple_windows(self, size: Literal[5]) -> AsyncIter[Tuple5[T]]:
        ...

    @overload
    def tuple_windows(self, size: Literal[6]) -> AsyncIter[Tuple6[T]]:
        ...

    @overload
    def tuple_windows(self, size: Literal[7]) -> AsyncIter[Tuple7[T]]:
        ...

    @overload
    def tuple_windows(self, size: Literal[8]) -> AsyncIter[Tuple8[T]]:
        ...

    @overload
    def tuple_windows(self, size: int) -> AsyncIter[DynamicTuple[T]]:
        ...

    def tuple_windows(self, size: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_tuple_windows(size, self.iterator))

    @overload
    def zip(self) -> AsyncIter[Tuple[T]]:
        ...

    @overload
    def zip(self, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[T, A]]:
        ...

    @overload
    def zip(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[T, A, B]]:
        ...

    @overload
    def zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[T, A, B, C]]:
        ...

    @overload
    def zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[T, A, B, C, D]]:
        ...

    @overload
    def zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E]]:
        ...

    @overload
    def zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F]]:
        ...

    @overload
    def zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G]]:
        ...

    @overload
    def zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G, H]]:
        ...

    @overload
    def zip(
        self,
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
    ) -> AsyncIter[DynamicTuple[Any]]:
        ...

    def zip(self, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return self.create(async_zip(self.iterator, *iterables))

    @overload
    def zip_equal(self) -> AsyncIter[Tuple[T]]:
        ...

    @overload
    def zip_equal(self, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[T, A]]:
        ...

    @overload
    def zip_equal(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[T, A, B]]:
        ...

    @overload
    def zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[T, A, B, C]]:
        ...

    @overload
    def zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[T, A, B, C, D]]:
        ...

    @overload
    def zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E]]:
        ...

    @overload
    def zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F]]:
        ...

    @overload
    def zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G]]:
        ...

    @overload
    def zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G, H]]:
        ...

    @overload
    def zip_equal(
        self,
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
    ) -> AsyncIter[DynamicTuple[Any]]:
        ...

    def zip_equal(self, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return self.create(async_zip_equal(self.iterator, *iterables))

    @overload
    def zip_longest(self) -> AsyncIter[Tuple[T]]:
        ...

    @overload
    def zip_longest(
        self, __iterable_a: AnyIterable[A]
    ) -> AsyncIter[Tuple[Optional[T], Optional[A]]]:
        ...

    @overload
    def zip_longest(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[Optional[T], Optional[A], Optional[B]]]:
        ...

    @overload
    def zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[Optional[T], Optional[A], Optional[B], Optional[C]]]:
        ...

    @overload
    def zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[Optional[T], Optional[A], Optional[B], Optional[C], Optional[D]]]:
        ...

    @overload
    def zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[
        Tuple[Optional[T], Optional[A], Optional[B], Optional[C], Optional[D], Optional[E]]
    ]:
        ...

    @overload
    def zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[
        Tuple[
            Optional[T],
            Optional[A],
            Optional[B],
            Optional[C],
            Optional[D],
            Optional[E],
            Optional[F],
        ]
    ]:
        ...

    @overload
    def zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[
        Tuple[
            Optional[T],
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
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[
        Tuple[
            Optional[T],
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
        self,
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
    ) -> AsyncIter[DynamicTuple[Optional[Any]]]:
        ...

    def zip_longest(self, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Optional[Any]]]:
        return self.create(async_zip_longest(self.iterator, *iterables))

    @overload
    def zip_longest_with(self, *, fill: V) -> AsyncIter[Tuple[T]]:
        ...

    @overload
    def zip_longest_with(
        self, __iterable_a: AnyIterable[A], *, fill: V
    ) -> AsyncIter[Tuple[Union[T, V], Union[A, V]]]:
        ...

    @overload
    def zip_longest_with(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B], *, fill: V
    ) -> AsyncIter[Tuple[Union[T, V], Union[A, V], Union[B, V]]]:
        ...

    @overload
    def zip_longest_with(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        *,
        fill: V,
    ) -> AsyncIter[Tuple[Union[T, V], Union[A, V], Union[B, V], Union[C, V]]]:
        ...

    @overload
    def zip_longest_with(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        *,
        fill: V,
    ) -> AsyncIter[Tuple[Union[T, V], Union[A, V], Union[B, V], Union[C, V], Union[D, V]]]:
        ...

    @overload
    def zip_longest_with(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        *,
        fill: V,
    ) -> AsyncIter[
        Tuple[Union[T, V], Union[A, V], Union[B, V], Union[C, V], Union[D, V], Union[E, V]]
    ]:
        ...

    @overload
    def zip_longest_with(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        *,
        fill: V,
    ) -> AsyncIter[
        Tuple[
            Union[T, V],
            Union[A, V],
            Union[B, V],
            Union[C, V],
            Union[D, V],
            Union[E, V],
            Union[F, V],
        ]
    ]:
        ...

    @overload
    def zip_longest_with(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        *,
        fill: V,
    ) -> AsyncIter[
        Tuple[
            Union[T, V],
            Union[A, V],
            Union[B, V],
            Union[C, V],
            Union[D, V],
            Union[E, V],
            Union[F, V],
            Union[G, V],
        ]
    ]:
        ...

    @overload
    def zip_longest_with(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
        *,
        fill: V,
    ) -> AsyncIter[
        Tuple[
            Union[T, V],
            Union[A, V],
            Union[B, V],
            Union[C, V],
            Union[D, V],
            Union[E, V],
            Union[F, V],
            Union[G, V],
            Union[H, V],
        ]
    ]:
        ...

    @overload
    def zip_longest_with(
        self,
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
        fill: V,
    ) -> AsyncIter[DynamicTuple[Union[Any, V]]]:
        ...

    @no_type_check  # strange
    def zip_longest_with(
        self, *iterables: AnyIterable[Any], fill: V
    ) -> AsyncIter[DynamicTuple[Union[Any, V]]]:
        return self.create(async_zip_longest(self.iterator, *iterables, fill=fill))

    @overload
    def cartesian_product(self) -> AsyncIter[Tuple[T]]:
        ...

    @overload
    def cartesian_product(self, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[T, A]]:
        ...

    @overload
    def cartesian_product(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[T, A, B]]:
        ...

    @overload
    def cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[T, A, B, C]]:
        ...

    @overload
    def cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[T, A, B, C, D]]:
        ...

    @overload
    def cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E]]:
        ...

    @overload
    def cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F]]:
        ...

    @overload
    def cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G]]:
        ...

    @overload
    def cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G, H]]:
        ...

    @overload
    def cartesian_product(
        self,
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
    ) -> AsyncIter[DynamicTuple[Any]]:
        ...

    def cartesian_product(self, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return self.create(async_cartesian_product(self.iterator, *iterables))

    @overload
    def cartesian_power(self, power: Literal[0]) -> AsyncIter[EmptyTuple]:
        ...

    @overload
    def cartesian_power(self, power: Literal[1]) -> AsyncIter[Tuple1[T]]:
        ...

    @overload
    def cartesian_power(self, power: Literal[2]) -> AsyncIter[Tuple2[T]]:
        ...

    @overload
    def cartesian_power(self, power: Literal[3]) -> AsyncIter[Tuple3[T]]:
        ...

    @overload
    def cartesian_power(self, power: Literal[4]) -> AsyncIter[Tuple4[T]]:
        ...

    @overload
    def cartesian_power(self, power: Literal[5]) -> AsyncIter[Tuple5[T]]:
        ...

    @overload
    def cartesian_power(self, power: Literal[6]) -> AsyncIter[Tuple6[T]]:
        ...

    @overload
    def cartesian_power(self, power: Literal[7]) -> AsyncIter[Tuple7[T]]:
        ...

    @overload
    def cartesian_power(self, power: Literal[8]) -> AsyncIter[Tuple8[T]]:
        ...

    def cartesian_power(self, power: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_cartesian_power(power, self.iterator))

    def reverse(self) -> AsyncIter[T]:
        return self.create(async_reverse(self.iterator))

    async def sorted(self: AsyncIter[ST]) -> List[ST]:
        return await async_sorted(self.iterator)

    async def sorted_by(self, key: Unary[T, ST]) -> List[T]:
        return await async_sorted(self.iterator, key=key)

    async def sorted_reverse(self: AsyncIter[ST]) -> List[ST]:
        return await async_sorted(self.iterator, reverse=True)

    async def sorted_reverse_by(self, key: Unary[T, ST]) -> List[T]:
        return await async_sorted(self.iterator, key=key, reverse=True)

    async def sorted_by_await(self, key: AsyncUnary[T, ST]) -> List[T]:
        return await async_sorted_await(self.iterator, key=key)

    async def sorted_reverse_by_await(self, key: AsyncUnary[T, ST]) -> List[T]:
        return await async_sorted_await(self.iterator, key=key, reverse=True)

    def sort(self: AsyncIter[ST]) -> AsyncIter[ST]:
        return self.create(async_sort(self.iterator))

    def sort_by(self, key: Unary[T, ST]) -> AsyncIter[T]:
        return self.create(async_sort(self.iterator, key=key))

    def sort_reverse(self: AsyncIter[ST]) -> AsyncIter[ST]:
        return self.create(async_sort(self.iterator, reverse=True))

    def sort_reverse_by(self, key: Unary[T, ST]) -> AsyncIter[T]:
        return self.create(async_sort(self.iterator, key=key, reverse=True))

    def sort_by_await(self, key: AsyncUnary[T, ST]) -> AsyncIter[T]:
        return self.create(async_sort_await(self.iterator, key=key))

    def sort_reverse_by_await(self, key: AsyncUnary[T, ST]) -> AsyncIter[T]:
        return self.create(async_sort_await(self.iterator, key=key, reverse=True))

    async def is_sorted(self: AsyncIter[LT]) -> bool:
        return await async_is_sorted(self.iterator)

    async def is_sorted_by(self, key: Unary[T, LT]) -> bool:
        return await async_is_sorted(self.iterator, key)

    async def is_sorted_reverse(self: AsyncIter[LT]) -> bool:
        return await async_is_sorted(self.iterator, reverse=True)

    async def is_sorted_reverse_by(self, key: Unary[T, LT]) -> bool:
        return await async_is_sorted(self.iterator, key, reverse=True)

    async def is_sorted_strict(self: AsyncIter[ST]) -> bool:
        return await async_is_sorted(self.iterator, strict=True)

    async def is_sorted_strict_by(self, key: Unary[T, ST]) -> bool:
        return await async_is_sorted(self.iterator, key, strict=True)

    async def is_sorted_reverse_strict(self: AsyncIter[ST]) -> bool:
        return await async_is_sorted(self.iterator, strict=True, reverse=True)

    async def is_sorted_reverse_strict_by(self, key: Unary[T, ST]) -> bool:
        return await async_is_sorted(self.iterator, key, strict=True, reverse=True)

    async def is_sorted_by_await(self, key: AsyncUnary[T, LT]) -> bool:
        return await async_is_sorted_await(self.iterator, key)

    async def is_sorted_reverse_by_await(self, key: AsyncUnary[T, LT]) -> bool:
        return await async_is_sorted_await(self.iterator, key, reverse=True)

    async def is_sorted_strict_by_await(self, key: AsyncUnary[T, ST]) -> bool:
        return await async_is_sorted_await(self.iterator, key, strict=True)

    async def is_sorted_reverse_strict_by_await(self, key: AsyncUnary[T, ST]) -> bool:
        return await async_is_sorted_await(self.iterator, key, strict=True, reverse=True)

    def duplicates_fast(self: AsyncIter[Q]) -> AsyncIter[Q]:
        return self.create(async_duplicates_fast(self.iterator))

    def duplicates_fast_by(self, key: Unary[T, Q]) -> AsyncIter[T]:
        return self.create(async_duplicates_fast(self.iterator, key))

    def duplicates_fast_by_await(self, key: AsyncUnary[T, Q]) -> AsyncIter[T]:
        return self.create(async_duplicates_fast_await(self.iterator, key))

    def duplicates(self) -> AsyncIter[T]:
        return self.create(async_duplicates(self.iterator))

    def duplicates_by(self, key: Unary[T, V]) -> AsyncIter[T]:
        return self.create(async_duplicates(self.iterator, key))

    def duplicates_by_await(self, key: AsyncUnary[T, V]) -> AsyncIter[T]:
        return self.create(async_duplicates_await(self.iterator, key))

    def unique_fast(self: AsyncIter[Q]) -> AsyncIter[Q]:
        return self.create(async_unique_fast(self.iterator))

    def unique_fast_by(self, key: Unary[T, Q]) -> AsyncIter[T]:
        return self.create(async_unique_fast(self.iterator, key))

    def unique_fast_by_await(self, key: AsyncUnary[T, Q]) -> AsyncIter[T]:
        return self.create(async_unique_fast_await(self.iterator, key))

    def unique(self) -> AsyncIter[T]:
        return self.create(async_unique(self.iterator))

    def unique_by(self, key: Unary[T, V]) -> AsyncIter[T]:
        return self.create(async_unique(self.iterator, key))

    def unique_by_await(self, key: AsyncUnary[T, V]) -> AsyncIter[T]:
        return self.create(async_unique_await(self.iterator, key))

    def partition(self, predicate: Optional[Predicate[T]]) -> Tuple[AsyncIter[T], AsyncIter[T]]:
        true, false = async_partition(predicate, self.iterator)

        return (self.create(true), self.create(false))

    def partition_await(self, predicate: AsyncPredicate[T]) -> Tuple[AsyncIter[T], AsyncIter[T]]:
        true, false = async_partition_await(predicate, self.iterator)

        return (self.create(true), self.create(false))

    def partition_unsafe(
        self, predicate: Optional[Predicate[T]]
    ) -> Tuple[AsyncIter[T], AsyncIter[T]]:
        true, false = async_partition_unsafe(predicate, self.iterator)

        return (self.create(true), self.create(false))

    partition_infinite = partition_unsafe

    def partition_unsafe_await(
        self, predicate: AsyncPredicate[T]
    ) -> Tuple[AsyncIter[T], AsyncIter[T]]:
        true, false = async_partition_unsafe_await(predicate, self.iterator)

        return (self.create(true), self.create(false))

    partition_infinite_await = partition_unsafe_await

    def copy(self) -> AsyncIter[T]:
        iterator, result = async_copy(self.iterator)

        self._replace(iterator)

        return self.create(result)

    def copy_unsafe(self) -> AsyncIter[T]:
        iterator, result = async_copy_unsafe(self.iterator)

        self._replace(iterator)

        return self.create(result)

    copy_infinite = copy_unsafe

    async def spy(self, size: int) -> List[T]:
        result, iterator = await async_spy(size, self.iterator)

        self._replace(iterator)

        return result

    async def peek(self) -> T:
        item, iterator = await async_peek(self.iterator)

        self._replace(iterator)

        return item

    async def peek_or(self, default: V) -> Union[T, V]:
        item, iterator = await async_peek(self.iterator, default)

        self._replace(iterator)

        return item

    async def peek_or_none(self) -> Optional[T]:
        return await self.peek_or(None)

    async def has_next(self) -> bool:
        result, iterator = await async_has_next(self.iterator)

        self._replace(iterator)

        return result

    async def is_empty(self) -> bool:
        result, iterator = await async_is_empty(self.iterator)

        self._replace(iterator)

        return result

    def repeat_last(self) -> AsyncIter[T]:
        return self.create(async_repeat_last(self.iterator))

    def repeat_last_or(self, default: V) -> AsyncIter[Union[T, V]]:
        return self.create(async_repeat_last(self.iterator, default))

    def repeat_last_or_none(self) -> AsyncIter[Optional[T]]:
        return self.repeat_last_or(None)

    def repeat_each(self, count: int) -> AsyncIter[T]:
        return self.create(async_repeat_each(self.iterator, count))

    def side_effect(self, function: Unary[T, Any]) -> AsyncIter[T]:
        return self.create(async_side_effect(function, self.iterator))

    def side_effect_await(self, function: AsyncUnary[T, Any]) -> AsyncIter[T]:
        return self.create(async_side_effect_await(function, self.iterator))

    def wait(self: AsyncIter[Awaitable[U]]) -> AsyncIter[U]:
        return self.create(async_wait(self.iterator))

    if CONCURRENT:

        def wait_concurrent(self: AsyncIter[Awaitable[U]]) -> AsyncIter[U]:
            return self.create(async_wait_concurrent(self.iterator))

        def wait_concurrent_bound(self: AsyncIter[Awaitable[U]], bound: int) -> AsyncIter[U]:
            return self.create(async_wait_concurrent_bound(bound, self.iterator))


async_iter = AsyncIter


def wrap_async_iter(function: Callable[PS, AnyIterable[T]]) -> Callable[PS, AsyncIter[T]]:
    @wraps(function)
    def wrap(*args: PS.args, **kwargs: PS.kwargs) -> AsyncIter[T]:
        return async_iter(function(*args, **kwargs))

    return wrap
