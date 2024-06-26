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
    Literal,
    Optional,
    Reversible,
    Set,
    Tuple,
    TypeVar,
    no_type_check,
    overload,
)

from mixed_methods import mixed_method
from orderings import LenientOrdered, Ordering, StrictOrdered
from typing_aliases import (
    AnyErrorType,
    AnyIterable,
    AnySelectors,
    AsyncBinary,
    AsyncForEach,
    AsyncInspect,
    AsyncNullary,
    AsyncPredicate,
    AsyncUnary,
    AsyncValidate,
    Binary,
    DynamicTuple,
    EmptyTuple,
    ForEach,
    Inspect,
    Nullary,
    Pair,
    Predicate,
    RecursiveAnyIterable,
    Tuple1,
    Tuple2,
    Tuple3,
    Tuple4,
    Tuple5,
    Tuple6,
    Tuple7,
    Tuple8,
    Unary,
    Validate,
)
from typing_extensions import Never, ParamSpec
from wraps.early.decorators import early_option_await
from wraps.primitives.option import Option, Some
from wraps.primitives.result import Result
from wraps.wraps.futures import wrap_future, wrap_future_option, wrap_future_result

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
    async_combinations,
    async_combinations_with_replacement,
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
    async_has_next,
    async_inspect,
    async_inspect_await,
    async_interleave,
    async_interleave_longest,
    async_intersperse,
    async_intersperse_with,
    async_intersperse_with_await,
    async_is_empty,
    async_is_sorted,
    async_is_sorted_await,
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
    async_map_concurrent,
    async_map_concurrent_bound,
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
    async_of,
    async_once,
    async_once_with,
    async_once_with_await,
    async_ordered_set,
    async_pad,
    async_pad_with,
    async_pad_with_await,
    async_pairs,
    async_pairs_windows,
    async_partition,
    async_partition_await,
    async_partition_unsafe,
    async_partition_unsafe_await,
    async_peek,
    async_permutations,
    async_permute,
    async_position,
    async_position_all,
    async_position_all_await,
    async_position_await,
    async_power_set,
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
    async_rest,
    async_reverse,
    async_reversed,
    async_set,
    async_set_windows,
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
    async_transpose,
    async_tuple,
    async_tuple_windows,
    async_unique,
    async_unique_await,
    async_unique_fast,
    async_unique_fast_await,
    async_wait,
    async_wait_concurrent,
    async_wait_concurrent_bound,
    async_zip,
    async_zip_equal,
    standard_async_iter,
    standard_async_next,
)
from iters.async_utils import async_groups_longest as standard_async_groups_longest
from iters.async_utils import async_iter as async_iter_any_iter
from iters.async_utils import async_pairs_longest as standard_async_pairs_longest
from iters.async_utils import async_zip_longest as standard_async_zip_longest
from iters.async_wraps import (
    async_at_most_one,
    async_exactly_one,
    async_filter_map_option,
    async_filter_map_option_await,
    async_scan,
    async_scan_await,
)
from iters.constants import DEFAULT_START, DEFAULT_STEP, EMPTY_BYTES, EMPTY_STRING
from iters.ordered_set import OrderedSet
from iters.types import MarkerOr, marker, wrap_marked
from iters.typing import OptionalPredicate, Product, Sum

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

LT = TypeVar("LT", bound=LenientOrdered)
ST = TypeVar("ST", bound=StrictOrdered)

PS = ParamSpec("PS")


def wrap_marked_item(item: DynamicTuple[MarkerOr[T]]) -> DynamicTuple[Option[T]]:
    return tuple(map(wrap_marked, item))


def wrap_marked_iterable(
    items: AnyIterable[DynamicTuple[MarkerOr[T]]],
) -> AsyncIterator[DynamicTuple[Option[T]]]:
    return async_map(wrap_marked_item, items)


def async_zip_longest(*iterables: AnyIterable[Any]) -> AsyncIterator[DynamicTuple[Option[Any]]]:
    return wrap_marked_iterable(standard_async_zip_longest(*iterables, fill=marker))


def async_groups_longest(
    size: int, iterable: AnyIterable[T]
) -> AsyncIterator[DynamicTuple[Option[T]]]:
    return wrap_marked_iterable(standard_async_groups_longest(size, iterable, marker))


def async_pairs_longest(iterable: AnyIterable[T]) -> AsyncIterator[Pair[Option[T]]]:
    return wrap_marked_iterable(  # type: ignore[return-value]
        standard_async_pairs_longest(iterable, marker)
    )


class AsyncIter(AsyncIterator[T]):
    # internals

    _iterator: AsyncIterator[T]

    def __init__(self, iterable: AnyIterable[T]) -> None:
        self._iterator = async_iter_any_iter(iterable)

    def _replace(self, iterator: AsyncIterator[T]) -> None:
        self._iterator = iterator

    # implementation

    @property
    def iterator(self) -> AsyncIterator[T]:
        """The underlying iterator."""
        return self._iterator

    @classmethod
    def empty(cls) -> AsyncIter[T]:
        return cls.create(async_empty())

    @classmethod
    def of(cls, *items: V) -> AsyncIter[V]:
        return cls.create(async_of(*items))

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
    def iter_except(cls, function: Nullary[T], *errors: AnyErrorType) -> AsyncIter[T]:
        return cls.create(async_iter_except(function, *errors))

    @classmethod
    def iter_except_await(cls, function: AsyncNullary[T], *errors: AnyErrorType) -> AsyncIter[T]:
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
    def create_chain_with(cls, iterable: AnyIterable[AnyIterable[T]]) -> AsyncIter[T]:
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
    def create_zip(cls) -> AsyncIter[T]: ...

    @overload
    @classmethod
    def create_zip(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[A]]: ...

    @overload
    @classmethod
    def create_zip(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[A, B]]: ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[A, B, C]]: ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[A, B, C, D]]: ...

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[A, B, C, D, E]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G, H]]: ...

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
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Any]]: ...

    @no_type_check
    @classmethod
    def create_zip(cls, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return cls.create(async_zip(*iterables))

    @overload
    @classmethod
    def create_zip_equal(cls) -> AsyncIter[T]: ...

    @overload
    @classmethod
    def create_zip_equal(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[A]]: ...

    @overload
    @classmethod
    def create_zip_equal(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[A, B]]: ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[A, B, C]]: ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[A, B, C, D]]: ...

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[A, B, C, D, E]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G, H]]: ...

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
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Any]]: ...

    @no_type_check
    @classmethod
    def create_zip_equal(cls, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return cls.create(async_zip_equal(*iterables))

    @overload
    @classmethod
    def create_zip_longest(cls) -> AsyncIter[T]: ...

    @overload
    @classmethod
    def create_zip_longest(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[Option[A]]]: ...

    @overload
    @classmethod
    def create_zip_longest(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[Option[A], Option[B]]]: ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[Option[A], Option[B], Option[C]]]: ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[Option[A], Option[B], Option[C], Option[D]]]: ...

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[Option[A], Option[B], Option[C], Option[D], Option[E]]]: ...

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
    ) -> AsyncIter[Tuple[Option[A], Option[B], Option[C], Option[D], Option[E], Option[F]]]: ...

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
            Option[A],
            Option[B],
            Option[C],
            Option[D],
            Option[E],
            Option[F],
            Option[G],
        ]
    ]: ...

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
            Option[A],
            Option[B],
            Option[C],
            Option[D],
            Option[E],
            Option[F],
            Option[G],
            Option[H],
        ]
    ]: ...

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
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Option[Any]]]: ...

    @no_type_check
    @classmethod
    def create_zip_longest(
        cls, *iterables: AnyIterable[Any]
    ) -> AsyncIter[DynamicTuple[Option[Any]]]:
        return cls.create(async_zip_longest(*iterables))

    @overload
    @classmethod
    def create_cartesian_product(cls) -> AsyncIter[EmptyTuple]: ...

    @overload
    @classmethod
    def create_cartesian_product(cls, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[A]]: ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[A, B]]: ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[A, B, C]]: ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[A, B, C, D]]: ...

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[A, B, C, D, E]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G]]: ...

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
    ) -> AsyncIter[Tuple[A, B, C, D, E, F, G, H]]: ...

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
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Any]]: ...

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
        return cls(iterable)  # type: ignore[arg-type, return-value]

    @classmethod
    def create_tuple(cls, iterables: DynamicTuple[AnyIterable[U]]) -> DynamicTuple[AsyncIter[U]]:
        return tuple(map(cls.create, iterables))

    @classmethod
    def create_nested(cls, nested: AnyIterable[AnyIterable[U]]) -> AsyncIter[AsyncIter[U]]:
        return cls.create(map(cls.create, nested))  # type: ignore[arg-type]

    @classmethod
    def create_option(cls, option: Option[AnyIterable[U]]) -> Option[AsyncIter[U]]:
        return option.map(cls.create)

    def __aiter__(self) -> AsyncIter[T]:
        return self

    async def __anext__(self) -> T:
        return await async_next_unchecked(self.iterator)

    def __await__(self) -> Generator[None, None, List[T]]:
        return self.list().__await__()

    def unwrap(self) -> AsyncIterator[T]:
        return self.iterator

    def async_iter(self) -> AsyncIter[T]:
        return self

    @wrap_future_option
    async def next(self) -> Option[T]:
        return wrap_marked(await async_next_unchecked(self.iterator, marker))

    @wrap_future
    async def compare(self: AsyncIter[ST], other: AnyIterable[ST]) -> Ordering:
        return await async_compare(self.iterator, other)

    @wrap_future
    async def compare_by(self, other: AnyIterable[T], key: Unary[T, ST]) -> Ordering:
        return await async_compare(self.iterator, other, key)

    @wrap_future
    async def compare_by_await(self, other: AnyIterable[T], key: AsyncUnary[T, ST]) -> Ordering:
        return await async_compare_await(self.iterator, other, key)

    @wrap_future
    async def length(self) -> int:
        return await async_iter_length(self.iterator)

    @wrap_future_option
    async def first(self) -> Option[T]:
        return wrap_marked(await async_first(self.iterator, marker))

    @wrap_future_option
    async def last(self) -> Option[T]:
        return wrap_marked(await async_last(self.iterator, marker))

    @wrap_future_option
    async def last_with_tail(self) -> Option[T]:
        return wrap_marked(await async_last_with_tail(self.iterator, marker))

    def collect(self, function: Unary[AsyncIterable[T], U]) -> U:
        return function(self.iterator)

    @wrap_future
    async def collect_await(self, function: AsyncUnary[AsyncIterable[T], U]) -> U:
        return await function(self.iterator)

    def collect_iter(self, function: Unary[AsyncIterable[T], AnyIterable[U]]) -> AsyncIter[U]:
        return self.create(self.collect(function))

    @wrap_future
    async def list(self) -> List[T]:
        return await async_list(self.iterator)

    @wrap_future
    async def set(self: AsyncIter[Q]) -> Set[Q]:
        return await async_set(self.iterator)

    @wrap_future
    async def ordered_set(self: AsyncIter[Q]) -> OrderedSet[Q]:
        return await async_ordered_set(self.iterator)

    @wrap_future
    async def tuple(self) -> DynamicTuple[T]:
        return await async_tuple(self.iterator)

    @wrap_future
    async def dict(self: AsyncIter[Tuple[Q, V]]) -> Dict[Q, V]:
        return await async_dict(self.iterator)

    @wrap_future
    async def extract(self) -> Iterator[T]:
        return await async_extract(self.iterator)

    @wrap_future
    async def join(self: AsyncIter[AnyStr], string: AnyStr) -> AnyStr:
        return string.join(await self.list())

    @wrap_future
    async def string(self: AsyncIter[str]) -> str:
        return await self.join(EMPTY_STRING)

    @wrap_future
    async def bytes(self: AsyncIter[bytes]) -> bytes:
        return await self.join(EMPTY_BYTES)

    @wrap_future
    async def count_dict(self: AsyncIter[Q]) -> Counter[Q]:
        return await async_count_dict(self.iterator)

    @wrap_future
    async def count_dict_by(self, key: Unary[T, Q]) -> Counter[Q]:
        return await async_count_dict(self.iterator, key)

    @wrap_future
    async def count_dict_by_await(self, key: AsyncUnary[T, Q]) -> Counter[Q]:
        return await async_count_dict_await(self.iterator, key)

    @wrap_future
    async def group_dict(self: AsyncIter[Q]) -> Dict[Q, List[Q]]:
        return await async_group_dict(self.iterator)

    @wrap_future
    async def group_dict_by(self, key: Unary[T, Q]) -> Dict[Q, List[T]]:
        return await async_group_dict(self.iterator, key)

    @wrap_future
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

    @wrap_future
    async def all(self) -> bool:
        return await async_all(self.iterator)

    @wrap_future
    async def all_by(self, predicate: Predicate[T]) -> bool:
        return await self.map(predicate).all()

    @wrap_future
    async def all_by_await(self, predicate: AsyncPredicate[T]) -> bool:
        return await self.map_await(predicate).all()

    @wrap_future
    async def any(self) -> bool:
        return await async_any(self.iterator)

    @wrap_future
    async def any_by(self, predicate: Predicate[T]) -> bool:
        return await self.map(predicate).any()

    @wrap_future
    async def any_by_await(self, predicate: AsyncPredicate[T]) -> bool:
        return await self.map_await(predicate).any()

    @wrap_future
    async def all_equal(self) -> bool:
        return await async_all_equal(self.iterator)

    @wrap_future
    async def all_equal_by(self, key: Unary[T, U]) -> bool:
        return await async_all_equal(self.iterator, key)

    @wrap_future
    async def all_equal_by_await(self, key: AsyncUnary[T, U]) -> bool:
        return await async_all_equal_await(self.iterator, key)

    @wrap_future
    async def all_unique(self) -> bool:
        return await async_all_unique(self.iterator)

    @wrap_future
    async def all_unique_by(self, key: Unary[T, U]) -> bool:
        return await async_all_unique(self.iterator, key)

    @wrap_future
    async def all_unique_by_await(self, key: AsyncUnary[T, U]) -> bool:
        return await async_all_unique_await(self.iterator, key)

    @wrap_future
    async def all_unique_fast(self: AsyncIter[Q]) -> bool:
        return await async_all_unique_fast(self.iterator)

    @wrap_future
    async def all_unique_fast_by(self, key: Unary[T, Q]) -> bool:
        return await async_all_unique_fast(self.iterator, key)

    @wrap_future
    async def all_unique_fast_by_await(self, key: AsyncUnary[T, Q]) -> bool:
        return await async_all_unique_fast_await(self.iterator, key)

    def remove(self, predicate: OptionalPredicate[T]) -> AsyncIter[T]:
        return self.create(async_remove(predicate, self.iterator))

    def remove_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_remove_await(predicate, self.iterator))

    def remove_duplicates(self) -> AsyncIter[T]:
        return self.create(async_remove_duplicates(self.iterator))

    def remove_duplicates_by(self, key: Unary[T, U]) -> AsyncIter[T]:
        return self.create(async_remove_duplicates(self.iterator, key))

    def remove_duplicates_by_await(self, key: AsyncUnary[T, U]) -> AsyncIter[T]:
        return self.create(async_remove_duplicates_await(self.iterator, key))

    def filter(self, predicate: OptionalPredicate[T]) -> AsyncIter[T]:
        return self.create(async_filter(predicate, self.iterator))

    def filter_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_filter_await(predicate, self.iterator))

    def filter_false(self, predicate: OptionalPredicate[T]) -> AsyncIter[T]:
        return self.create(async_filter_false(predicate, self.iterator))

    def filter_false_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_filter_false_await(predicate, self.iterator))

    def filter_except(self, validate: Validate[T], *errors: AnyErrorType) -> AsyncIter[T]:
        return self.create(async_filter_except(validate, self.iterator, *errors))

    def filter_except_await(
        self, validate: AsyncValidate[T], *errors: AnyErrorType
    ) -> AsyncIter[T]:
        return self.create(async_filter_except_await(validate, self.iterator, *errors))

    def compress(self, selectors: AnySelectors) -> AsyncIter[T]:
        return self.create(async_compress(self.iterator, selectors))

    def position_all(self, predicate: OptionalPredicate[T]) -> AsyncIter[int]:
        return self.create(async_position_all(predicate, self.iterator))

    def position_all_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[int]:
        return self.create(async_position_all_await(predicate, self.iterator))

    @wrap_future_option
    async def position(self, predicate: OptionalPredicate[T]) -> Option[int]:
        return wrap_marked(await async_position(predicate, self.iterator, marker))

    @wrap_future_option
    async def position_await(self, predicate: AsyncPredicate[T]) -> Option[int]:
        return wrap_marked(await async_position_await(predicate, self.iterator, marker))

    def find_all(self, predicate: OptionalPredicate[T]) -> AsyncIter[T]:
        return self.create(async_find_all(predicate, self.iterator))

    def find_all_await(self, predicate: AsyncPredicate[T]) -> AsyncIter[T]:
        return self.create(async_find_all_await(predicate, self.iterator))

    @wrap_future_option
    async def find(self, predicate: OptionalPredicate[T]) -> Option[T]:
        return wrap_marked(
            await async_find(predicate, self.iterator, marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def find_await(self, predicate: AsyncPredicate[T]) -> Option[T]:
        return wrap_marked(
            await async_find_await(predicate, self.iterator, marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def find_or_first(self, predicate: OptionalPredicate[T]) -> Option[T]:
        return wrap_marked(
            await async_find_or_first(predicate, self.iterator, marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def find_or_first_await(self, predicate: AsyncPredicate[T]) -> Option[T]:
        return wrap_marked(
            await async_find_or_first_await(
                predicate,  # type: ignore[arg-type]
                self.iterator,
                marker,
            )
        )

    @wrap_future_option
    async def find_or_last(self, predicate: OptionalPredicate[T]) -> Option[T]:
        return wrap_marked(
            await async_find_or_last(predicate, self.iterator, marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def find_or_last_await(self, predicate: AsyncPredicate[T]) -> Option[T]:
        return wrap_marked(
            await async_find_or_last_await(
                predicate,  # type: ignore[arg-type]
                self.iterator,
                marker,
            )
        )

    @wrap_future
    async def contains(self, item: V) -> bool:
        return await async_contains(item, self.iterator)

    @wrap_future
    async def contains_identity(self: AsyncIter[V], item: V) -> bool:
        return await async_contains_identity(item, self.iterator)

    @wrap_future_option
    async def reduce(self, function: Binary[T, T, T]) -> Option[T]:
        return wrap_marked(
            await async_reduce(function, self.iterator, marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def reduce_await(self, function: AsyncBinary[T, T, T]) -> Option[T]:
        return wrap_marked(
            await async_reduce_await(function, self.iterator, marker)  # type: ignore[arg-type]
        )

    @wrap_future
    async def fold(self, initial: V, function: Binary[V, T, V]) -> V:
        return await async_fold(initial, function, self.iterator)

    @wrap_future
    async def fold_await(self, initial: V, function: AsyncBinary[V, T, V]) -> V:
        return await async_fold_await(initial, function, self.iterator)

    @wrap_future
    @early_option_await
    async def sum(self: AsyncIter[S]) -> Option[S]:
        return Some(await self.sum_with(await self.next().early()))

    @wrap_future
    async def sum_with(self: AsyncIter[S], initial: S) -> S:
        return await async_sum(self.iterator, initial)

    @wrap_future
    @early_option_await
    async def product(self: AsyncIter[P]) -> Option[P]:
        return Some(await self.product_with(await self.next().early()))

    @wrap_future
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

    @wrap_future_option
    async def min(self: AsyncIter[ST]) -> Option[ST]:
        return wrap_marked(await async_min(self.iterator, default=marker))

    @wrap_future_option
    async def min_by(self, key: Unary[T, ST]) -> Option[T]:
        return wrap_marked(
            await async_min(self.iterator, key=key, default=marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def min_by_await(self, key: AsyncUnary[T, ST]) -> Option[T]:
        return wrap_marked(
            await async_min_await(self.iterator, key=key, default=marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def max(self: AsyncIter[ST]) -> Option[ST]:
        return wrap_marked(await async_max(self.iterator, default=marker))

    @wrap_future_option
    async def max_by(self, key: Unary[T, ST]) -> Option[T]:
        return wrap_marked(
            await async_max(self.iterator, key=key, default=marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def max_by_await(self, key: AsyncUnary[T, ST]) -> Option[T]:
        return wrap_marked(
            await async_max_await(self.iterator, key=key, default=marker)  # type: ignore[arg-type]
        )

    @wrap_future_option
    async def min_max(self: AsyncIter[ST]) -> Option[Pair[ST]]:
        return wrap_marked(await async_min_max(self.iterator, default=marker))

    @wrap_future_option
    async def min_max_by(self, key: Unary[T, ST]) -> Option[Pair[T]]:
        return wrap_marked(await async_min_max(self.iterator, key=key, default=marker))

    @wrap_future_option
    async def min_max_by_await(self, key: AsyncUnary[T, ST]) -> Option[Pair[T]]:
        return wrap_marked(await async_min_max_await(self.iterator, key=key, default=marker))

    def map(self, function: Unary[T, U]) -> AsyncIter[U]:
        return self.create(async_map(function, self.iterator))

    def map_await(self, function: AsyncUnary[T, U]) -> AsyncIter[U]:
        return self.create(async_map_await(function, self.iterator))

    def map_except(self, function: Unary[T, U], *errors: AnyErrorType) -> AsyncIter[U]:
        return self.create(async_map_except(function, self.iterator, *errors))

    def map_except_await(self, function: AsyncUnary[T, U], *errors: AnyErrorType) -> AsyncIter[U]:
        return self.create(async_map_except_await(function, self.iterator, *errors))

    def map_concurrent(self, function: AsyncUnary[T, U]) -> AsyncIter[U]:
        return self.create(async_map_concurrent(function, self.iterator))

    def map_concurrent_bound(self, bound: int, function: AsyncUnary[T, U]) -> AsyncIter[U]:
        return self.create(async_map_concurrent_bound(bound, function, self.iterator))

    def flat_map(self, function: Unary[T, AnyIterable[U]]) -> AsyncIter[U]:
        return self.create(async_flat_map(function, self.iterator))

    def flat_map_await(self, function: AsyncUnary[T, AnyIterable[U]]) -> AsyncIter[U]:
        return self.create(async_flat_map_await(function, self.iterator))

    def filter_map(self, predicate: OptionalPredicate[T], function: Unary[T, U]) -> AsyncIter[U]:
        return self.create(async_filter_map(predicate, function, self.iterator))

    def filter_await_map(self, predicate: AsyncPredicate[T], function: Unary[T, U]) -> AsyncIter[U]:
        return self.create(async_filter_await_map(predicate, function, self.iterator))

    def filter_map_await(
        self, predicate: OptionalPredicate[T], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_map_await(predicate, function, self.iterator))

    def filter_await_map_await(
        self, predicate: AsyncPredicate[T], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_await_map_await(predicate, function, self.iterator))

    def filter_false_map(
        self, predicate: OptionalPredicate[T], function: Unary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_map(predicate, function, self.iterator))

    def filter_false_await_map(
        self, predicate: AsyncPredicate[T], function: Unary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_await_map(predicate, function, self.iterator))

    def filter_false_map_await(
        self, predicate: OptionalPredicate[T], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_map_await(predicate, function, self.iterator))

    def filter_false_await_map_await(
        self, predicate: AsyncPredicate[T], function: AsyncUnary[T, U]
    ) -> AsyncIter[U]:
        return self.create(async_filter_false_await_map_await(predicate, function, self.iterator))

    def flatten(self: AsyncIter[AnyIterable[U]]) -> AsyncIter[U]:
        return self.create(async_flatten(self.iterator))

    @overload
    def collapse(self: AsyncIter[RecursiveAnyIterable[AnyStr]]) -> AsyncIter[AnyStr]: ...

    @overload
    def collapse(self: AsyncIter[RecursiveAnyIterable[U]]) -> AsyncIter[U]: ...

    def collapse(self: AsyncIter[RecursiveAnyIterable[Any]]) -> AsyncIter[Any]:
        return self.create(async_collapse(self.iterator))

    def enumerate(self) -> AsyncIter[Tuple[int, T]]:
        return self.create(async_enumerate(self.iterator))

    def enumerate_from(self, start: int) -> AsyncIter[Tuple[int, T]]:
        return self.create(async_enumerate(self.iterator, start))

    @wrap_future
    async def consume(self) -> None:
        await async_consume(self.iterator)

    @wrap_future
    async def for_each(self, function: ForEach[T]) -> None:
        await async_for_each(function, self.iterator)

    @wrap_future
    async def for_each_await(self, function: AsyncForEach[T]) -> None:
        await async_for_each_await(function, self.iterator)

    def append(self: AsyncIter[V], item: V) -> AsyncIter[V]:
        return self.create(async_append(item, self.iterator))

    def prepend(self: AsyncIter[V], item: V) -> AsyncIter[V]:
        return self.create(async_prepend(item, self.iterator))

    @wrap_future_option
    async def at(self, index: int) -> Option[T]:
        return wrap_marked(await async_at(index, self.iterator, marker))

    @wrap_future_option
    async def at_or_last(self, index: int) -> Option[T]:
        return wrap_marked(await async_at_or_last(index, self.iterator, marker))

    @overload
    def slice(self, __stop: Optional[int]) -> AsyncIter[T]: ...

    @overload
    def slice(
        self, __start: Optional[int], __stop: Optional[int], __step: Optional[int] = ...
    ) -> AsyncIter[T]: ...

    def slice(self, *slice_args: Optional[int]) -> AsyncIter[T]:
        return self.create(async_iter_slice(self.iterator, *slice_args))

    def drop(self, size: int) -> AsyncIter[T]:
        return self.create(async_drop(size, self.iterator))

    skip = drop

    def rest(self) -> AsyncIter[T]:
        return self.create(async_rest(self.iterator))

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

    def apply_chain(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_chain(self.iterator, *iterables))

    chain = mixed_method(create_chain, apply_chain)

    def apply_chain_with(self, iterables: AnyIterable[AnyIterable[T]]) -> AsyncIter[T]:
        return self.chain(async_chain_from_iterable(iterables))

    chain_with = mixed_method(create_chain_with, apply_chain_with)

    def cycle(self) -> AsyncIter[T]:
        return self.create(async_cycle(self.iterator))

    def intersperse(self: AsyncIter[V], value: V) -> AsyncIter[V]:
        return self.create(async_intersperse(value, self.iterator))

    def intersperse_with(self, function: Nullary[T]) -> AsyncIter[T]:
        return self.create(async_intersperse_with(function, self.iterator))

    def intersperse_with_await(self, function: AsyncNullary[T]) -> AsyncIter[T]:
        return self.create(async_intersperse_with_await(function, self.iterator))

    def apply_interleave(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_interleave(self.iterator, *iterables))

    interleave = mixed_method(create_interleave, apply_interleave)

    def apply_interleave_longest(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_interleave_longest(self.iterator, *iterables))

    interleave_longest = mixed_method(create_interleave_longest, apply_interleave_longest)

    def apply_combine(self, *iterables: AnyIterable[T]) -> AsyncIter[T]:
        return self.create(async_combine(*iterables))

    combine = mixed_method(create_combine, apply_combine)

    @overload
    def distribute_unsafe(self, count: Literal[0]) -> EmptyTuple: ...

    @overload
    def distribute_unsafe(self, count: Literal[1]) -> Tuple1[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: Literal[2]) -> Tuple2[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: Literal[3]) -> Tuple3[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: Literal[4]) -> Tuple4[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: Literal[5]) -> Tuple5[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: Literal[6]) -> Tuple6[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: Literal[7]) -> Tuple7[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: Literal[8]) -> Tuple8[AsyncIter[T]]: ...

    @overload
    def distribute_unsafe(self, count: int) -> DynamicTuple[AsyncIter[T]]: ...

    def distribute_unsafe(self, count: int) -> DynamicTuple[AsyncIter[T]]:
        return self.create_tuple(async_distribute_unsafe(count, self.iterator))

    distribute_infinite = distribute_unsafe

    @overload
    def distribute(self, count: Literal[0]) -> EmptyTuple: ...

    @overload
    def distribute(self, count: Literal[1]) -> Tuple1[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: Literal[2]) -> Tuple2[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: Literal[3]) -> Tuple3[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: Literal[4]) -> Tuple4[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: Literal[5]) -> Tuple5[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: Literal[6]) -> Tuple6[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: Literal[7]) -> Tuple7[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: Literal[8]) -> Tuple8[AsyncIter[T]]: ...

    @overload
    def distribute(self, count: int) -> DynamicTuple[AsyncIter[T]]: ...

    def distribute(self, count: int) -> DynamicTuple[AsyncIter[T]]:
        return self.create_tuple(async_distribute(count, self.iterator))

    def divide(self, count: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_divide(count, self.iterator))

    def pad(self: AsyncIter[V], value: V) -> AsyncIter[V]:
        return self.create(async_pad(value, self.iterator))

    def pad_exactly(self: AsyncIter[V], value: V, size: int) -> AsyncIter[V]:
        return self.create(async_pad(value, self.iterator, size))

    def pad_multiple(self: AsyncIter[V], value: V, size: int) -> AsyncIter[V]:
        return self.create(async_pad(value, self.iterator, size, multiple=True))

    def pad_with(self: AsyncIter[V], function: Unary[int, V]) -> AsyncIter[V]:
        return self.create(async_pad_with(function, self.iterator))

    def pad_exactly_with(self: AsyncIter[V], function: Unary[int, V], size: int) -> AsyncIter[V]:
        return self.create(async_pad_with(function, self.iterator, size))

    def pad_multiple_with(self: AsyncIter[V], function: Unary[int, V], size: int) -> AsyncIter[V]:
        return self.create(async_pad_with(function, self.iterator, size, multiple=True))

    def pad_with_await(self: AsyncIter[V], function: AsyncUnary[int, V]) -> AsyncIter[V]:
        return self.create(async_pad_with_await(function, self.iterator))

    def pad_exactly_with_await(
        self: AsyncIter[V], function: AsyncUnary[int, V], size: int
    ) -> AsyncIter[V]:
        return self.create(async_pad_with_await(function, self.iterator, size))

    def pad_multiple_with_await(
        self: AsyncIter[V], function: AsyncUnary[int, V], size: int
    ) -> AsyncIter[V]:
        return self.create(async_pad_with_await(function, self.iterator, size, multiple=True))

    def chunks(self, size: int) -> AsyncIter[List[T]]:
        return self.create(async_chunks(size, self.iterator))

    def iter_chunks(self, size: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_iter_chunks(size, self.iterator))

    def iter_chunks_unsafe(self, size: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_iter_chunks_unsafe(size, self.iterator))

    iter_chunks_infinite = iter_chunks_unsafe

    @overload
    def groups(self, size: Literal[0]) -> AsyncIter[Never]: ...

    @overload
    def groups(self, size: Literal[1]) -> AsyncIter[Tuple1[T]]: ...

    @overload
    def groups(self, size: Literal[2]) -> AsyncIter[Tuple2[T]]: ...

    @overload
    def groups(self, size: Literal[3]) -> AsyncIter[Tuple3[T]]: ...

    @overload
    def groups(self, size: Literal[4]) -> AsyncIter[Tuple4[T]]: ...

    @overload
    def groups(self, size: Literal[5]) -> AsyncIter[Tuple5[T]]: ...

    @overload
    def groups(self, size: Literal[6]) -> AsyncIter[Tuple6[T]]: ...

    @overload
    def groups(self, size: Literal[7]) -> AsyncIter[Tuple7[T]]: ...

    @overload
    def groups(self, size: Literal[8]) -> AsyncIter[Tuple8[T]]: ...

    @overload
    def groups(self, size: int) -> AsyncIter[DynamicTuple[T]]: ...

    def groups(self, size: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_groups(size, self.iterator))

    @overload
    def groups_longest(self, size: Literal[0]) -> AsyncIter[Never]: ...

    @overload
    def groups_longest(self, size: Literal[1]) -> AsyncIter[Tuple1[Option[T]]]: ...

    @overload
    def groups_longest(self, size: Literal[2]) -> AsyncIter[Tuple2[Option[T]]]: ...

    @overload
    def groups_longest(self, size: Literal[3]) -> AsyncIter[Tuple3[Option[T]]]: ...

    @overload
    def groups_longest(self, size: Literal[4]) -> AsyncIter[Tuple4[Option[T]]]: ...

    @overload
    def groups_longest(self, size: Literal[5]) -> AsyncIter[Tuple5[Option[T]]]: ...

    @overload
    def groups_longest(self, size: Literal[6]) -> AsyncIter[Tuple6[Option[T]]]: ...

    @overload
    def groups_longest(self, size: Literal[7]) -> AsyncIter[Tuple7[Option[T]]]: ...

    @overload
    def groups_longest(self, size: Literal[8]) -> AsyncIter[Tuple8[Option[T]]]: ...

    @overload
    def groups_longest(self, size: int) -> AsyncIter[DynamicTuple[Option[T]]]: ...

    def groups_longest(self, size: int) -> AsyncIter[DynamicTuple[Option[T]]]:
        return self.create(async_groups_longest(size, self.iterator))

    def pairs(self) -> AsyncIter[Pair[T]]:
        return self.create(async_pairs(self.iterator))

    def pairs_longest(self) -> AsyncIter[Pair[Option[T]]]:
        return self.create(async_pairs_longest(self.iterator))

    def iter_windows(self, size: int) -> AsyncIter[AsyncIter[T]]:
        return self.create_nested(async_iter_windows(size, self.iterator))

    def list_windows(self, size: int) -> AsyncIter[List[T]]:
        return self.create(async_list_windows(size, self.iterator))

    def pairs_windows(self) -> AsyncIter[Pair[T]]:
        return self.create(async_pairs_windows(self.iterator))

    @overload
    def tuple_windows(self, size: Literal[0]) -> AsyncIter[EmptyTuple]: ...

    @overload
    def tuple_windows(self, size: Literal[1]) -> AsyncIter[Tuple1[T]]: ...

    @overload
    def tuple_windows(self, size: Literal[2]) -> AsyncIter[Tuple2[T]]: ...

    @overload
    def tuple_windows(self, size: Literal[3]) -> AsyncIter[Tuple3[T]]: ...

    @overload
    def tuple_windows(self, size: Literal[4]) -> AsyncIter[Tuple4[T]]: ...

    @overload
    def tuple_windows(self, size: Literal[5]) -> AsyncIter[Tuple5[T]]: ...

    @overload
    def tuple_windows(self, size: Literal[6]) -> AsyncIter[Tuple6[T]]: ...

    @overload
    def tuple_windows(self, size: Literal[7]) -> AsyncIter[Tuple7[T]]: ...

    @overload
    def tuple_windows(self, size: Literal[8]) -> AsyncIter[Tuple8[T]]: ...

    @overload
    def tuple_windows(self, size: int) -> AsyncIter[DynamicTuple[T]]: ...

    def tuple_windows(self, size: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_tuple_windows(size, self.iterator))

    def set_windows(self: AsyncIter[Q], size: int) -> AsyncIter[Set[Q]]:
        return self.create(async_set_windows(size, self.iterator))

    @overload
    def apply_zip(self) -> AsyncIter[Tuple[T]]: ...

    @overload
    def apply_zip(self, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[T, A]]: ...

    @overload
    def apply_zip(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[T, A, B]]: ...

    @overload
    def apply_zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[T, A, B, C]]: ...

    @overload
    def apply_zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[T, A, B, C, D]]: ...

    @overload
    def apply_zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E]]: ...

    @overload
    def apply_zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F]]: ...

    @overload
    def apply_zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G]]: ...

    @overload
    def apply_zip(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G, H]]: ...

    @overload
    def apply_zip(
        self,
        __iterable_a: AnyIterable[Any],
        __iterable_b: AnyIterable[Any],
        __iterable_c: AnyIterable[Any],
        __iterable_d: AnyIterable[Any],
        __iterable_e: AnyIterable[Any],
        __iterable_f: AnyIterable[Any],
        __iterable_g: AnyIterable[Any],
        __iterable_h: AnyIterable[Any],
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Any]]: ...

    def apply_zip(self, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return self.create(async_zip(self.iterator, *iterables))

    zip = mixed_method(create_zip, apply_zip)

    @overload
    def apply_zip_equal(self) -> AsyncIter[Tuple[T]]: ...

    @overload
    def apply_zip_equal(self, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[T, A]]: ...

    @overload
    def apply_zip_equal(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[T, A, B]]: ...

    @overload
    def apply_zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[T, A, B, C]]: ...

    @overload
    def apply_zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[T, A, B, C, D]]: ...

    @overload
    def apply_zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E]]: ...

    @overload
    def apply_zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F]]: ...

    @overload
    def apply_zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G]]: ...

    @overload
    def apply_zip_equal(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G, H]]: ...

    @overload
    def apply_zip_equal(
        self,
        __iterable_a: AnyIterable[Any],
        __iterable_b: AnyIterable[Any],
        __iterable_c: AnyIterable[Any],
        __iterable_d: AnyIterable[Any],
        __iterable_e: AnyIterable[Any],
        __iterable_f: AnyIterable[Any],
        __iterable_g: AnyIterable[Any],
        __iterable_h: AnyIterable[Any],
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Any]]: ...

    def apply_zip_equal(self, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return self.create(async_zip_equal(self.iterator, *iterables))

    zip_equal = mixed_method(create_zip_equal, apply_zip_equal)

    @overload
    def apply_zip_longest(self) -> AsyncIter[Tuple[Option[T]]]: ...

    @overload
    def apply_zip_longest(
        self, __iterable_a: AnyIterable[A]
    ) -> AsyncIter[Tuple[Option[T], Option[A]]]: ...

    @overload
    def apply_zip_longest(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[Option[T], Option[A], Option[B]]]: ...

    @overload
    def apply_zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[Option[T], Option[A], Option[B], Option[C]]]: ...

    @overload
    def apply_zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[Option[T], Option[A], Option[B], Option[C], Option[D]]]: ...

    @overload
    def apply_zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[Option[T], Option[A], Option[B], Option[C], Option[D], Option[E]]]: ...

    @overload
    def apply_zip_longest(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[
        Tuple[
            Option[T],
            Option[A],
            Option[B],
            Option[C],
            Option[D],
            Option[E],
            Option[F],
        ]
    ]: ...

    @overload
    def apply_zip_longest(
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
            Option[T],
            Option[A],
            Option[B],
            Option[C],
            Option[D],
            Option[E],
            Option[F],
            Option[G],
        ]
    ]: ...

    @overload
    def apply_zip_longest(
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
            Option[T],
            Option[A],
            Option[B],
            Option[C],
            Option[D],
            Option[E],
            Option[F],
            Option[G],
            Option[H],
        ]
    ]: ...

    @overload
    def apply_zip_longest(
        self,
        __iterable_a: AnyIterable[Any],
        __iterable_b: AnyIterable[Any],
        __iterable_c: AnyIterable[Any],
        __iterable_d: AnyIterable[Any],
        __iterable_e: AnyIterable[Any],
        __iterable_f: AnyIterable[Any],
        __iterable_g: AnyIterable[Any],
        __iterable_h: AnyIterable[Any],
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Option[Any]]]: ...

    def apply_zip_longest(
        self, *iterables: AnyIterable[Any]
    ) -> AsyncIter[DynamicTuple[Option[Any]]]:
        return self.create(async_zip_longest(self.iterator, *iterables))

    zip_longest = mixed_method(create_zip_longest, apply_zip_longest)

    def transpose(self: AsyncIter[AnyIterable[T]]) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_transpose(self.iterator))

    @overload
    def apply_cartesian_product(self) -> AsyncIter[Tuple[T]]: ...

    @overload
    def apply_cartesian_product(self, __iterable_a: AnyIterable[A]) -> AsyncIter[Tuple[T, A]]: ...

    @overload
    def apply_cartesian_product(
        self, __iterable_a: AnyIterable[A], __iterable_b: AnyIterable[B]
    ) -> AsyncIter[Tuple[T, A, B]]: ...

    @overload
    def apply_cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
    ) -> AsyncIter[Tuple[T, A, B, C]]: ...

    @overload
    def apply_cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
    ) -> AsyncIter[Tuple[T, A, B, C, D]]: ...

    @overload
    def apply_cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E]]: ...

    @overload
    def apply_cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F]]: ...

    @overload
    def apply_cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G]]: ...

    @overload
    def apply_cartesian_product(
        self,
        __iterable_a: AnyIterable[A],
        __iterable_b: AnyIterable[B],
        __iterable_c: AnyIterable[C],
        __iterable_d: AnyIterable[D],
        __iterable_e: AnyIterable[E],
        __iterable_f: AnyIterable[F],
        __iterable_g: AnyIterable[G],
        __iterable_h: AnyIterable[H],
    ) -> AsyncIter[Tuple[T, A, B, C, D, E, F, G, H]]: ...

    @overload
    def apply_cartesian_product(
        self,
        __iterable_a: AnyIterable[Any],
        __iterable_b: AnyIterable[Any],
        __iterable_c: AnyIterable[Any],
        __iterable_d: AnyIterable[Any],
        __iterable_e: AnyIterable[Any],
        __iterable_f: AnyIterable[Any],
        __iterable_g: AnyIterable[Any],
        __iterable_h: AnyIterable[Any],
        __iterable_n: AnyIterable[Any],
        *iterables: AnyIterable[Any],
    ) -> AsyncIter[DynamicTuple[Any]]: ...

    def apply_cartesian_product(self, *iterables: AnyIterable[Any]) -> AsyncIter[DynamicTuple[Any]]:
        return self.create(async_cartesian_product(self.iterator, *iterables))

    cartesian_product = mixed_method(create_cartesian_product, apply_cartesian_product)

    @overload
    def cartesian_power(self, power: Literal[0]) -> AsyncIter[EmptyTuple]: ...

    @overload
    def cartesian_power(self, power: Literal[1]) -> AsyncIter[Tuple1[T]]: ...

    @overload
    def cartesian_power(self, power: Literal[2]) -> AsyncIter[Tuple2[T]]: ...

    @overload
    def cartesian_power(self, power: Literal[3]) -> AsyncIter[Tuple3[T]]: ...

    @overload
    def cartesian_power(self, power: Literal[4]) -> AsyncIter[Tuple4[T]]: ...

    @overload
    def cartesian_power(self, power: Literal[5]) -> AsyncIter[Tuple5[T]]: ...

    @overload
    def cartesian_power(self, power: Literal[6]) -> AsyncIter[Tuple6[T]]: ...

    @overload
    def cartesian_power(self, power: Literal[7]) -> AsyncIter[Tuple7[T]]: ...

    @overload
    def cartesian_power(self, power: Literal[8]) -> AsyncIter[Tuple8[T]]: ...

    def cartesian_power(self, power: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_cartesian_power(power, self.iterator))

    @overload
    def combinations(self, count: Literal[0]) -> AsyncIter[EmptyTuple]: ...

    @overload
    def combinations(self, count: Literal[1]) -> AsyncIter[Tuple1[T]]: ...

    @overload
    def combinations(self, count: Literal[2]) -> AsyncIter[Tuple2[T]]: ...

    @overload
    def combinations(self, count: Literal[3]) -> AsyncIter[Tuple3[T]]: ...

    @overload
    def combinations(self, count: Literal[4]) -> AsyncIter[Tuple4[T]]: ...

    @overload
    def combinations(self, count: Literal[5]) -> AsyncIter[Tuple5[T]]: ...

    @overload
    def combinations(self, count: Literal[6]) -> AsyncIter[Tuple6[T]]: ...

    @overload
    def combinations(self, count: Literal[7]) -> AsyncIter[Tuple7[T]]: ...

    @overload
    def combinations(self, count: Literal[8]) -> AsyncIter[Tuple8[T]]: ...

    @overload
    def combinations(self, count: int) -> AsyncIter[DynamicTuple[T]]: ...

    def combinations(self, count: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_combinations(count, self.iterator))

    @overload
    def combinations_with_replacement(self, count: Literal[0]) -> AsyncIter[EmptyTuple]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[1]) -> AsyncIter[Tuple1[T]]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[2]) -> AsyncIter[Tuple2[T]]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[3]) -> AsyncIter[Tuple3[T]]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[4]) -> AsyncIter[Tuple4[T]]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[5]) -> AsyncIter[Tuple5[T]]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[6]) -> AsyncIter[Tuple6[T]]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[7]) -> AsyncIter[Tuple7[T]]: ...

    @overload
    def combinations_with_replacement(self, count: Literal[8]) -> AsyncIter[Tuple8[T]]: ...

    @overload
    def combinations_with_replacement(self, count: int) -> AsyncIter[DynamicTuple[T]]: ...

    def combinations_with_replacement(self, count: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_combinations_with_replacement(count, self.iterator))

    def permute(self) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_permute(self.iterator))

    @overload
    def permutations(self, count: Literal[0]) -> AsyncIter[EmptyTuple]: ...

    @overload
    def permutations(self, count: Literal[1]) -> AsyncIter[Tuple1[T]]: ...

    @overload
    def permutations(self, count: Literal[2]) -> AsyncIter[Tuple2[T]]: ...

    @overload
    def permutations(self, count: Literal[3]) -> AsyncIter[Tuple3[T]]: ...

    @overload
    def permutations(self, count: Literal[4]) -> AsyncIter[Tuple4[T]]: ...

    @overload
    def permutations(self, count: Literal[5]) -> AsyncIter[Tuple5[T]]: ...

    @overload
    def permutations(self, count: Literal[6]) -> AsyncIter[Tuple6[T]]: ...

    @overload
    def permutations(self, count: Literal[7]) -> AsyncIter[Tuple7[T]]: ...

    @overload
    def permutations(self, count: Literal[8]) -> AsyncIter[Tuple8[T]]: ...

    @overload
    def permutations(self, count: int) -> AsyncIter[DynamicTuple[T]]: ...

    def permutations(self, count: int) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_permutations(count, self.iterator))

    def power_set(self) -> AsyncIter[DynamicTuple[T]]:
        return self.create(async_power_set(self.iterator))

    def reverse(self) -> AsyncIter[T]:
        return self.create(async_reverse(self.iterator))

    @wrap_future
    async def sorted(self: AsyncIter[ST]) -> List[ST]:
        return await async_sorted(self.iterator)

    @wrap_future
    async def sorted_by(self, key: Unary[T, ST]) -> List[T]:
        return await async_sorted(self.iterator, key=key)

    @wrap_future
    async def sorted_reverse(self: AsyncIter[ST]) -> List[ST]:
        return await async_sorted(self.iterator, reverse=True)

    @wrap_future
    async def sorted_reverse_by(self, key: Unary[T, ST]) -> List[T]:
        return await async_sorted(self.iterator, key=key, reverse=True)

    @wrap_future
    async def sorted_by_await(self, key: AsyncUnary[T, ST]) -> List[T]:
        return await async_sorted_await(self.iterator, key=key)

    @wrap_future
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

    @wrap_future
    async def is_sorted(self: AsyncIter[LT]) -> bool:
        return await async_is_sorted(self.iterator)

    @wrap_future
    async def is_sorted_by(self, key: Unary[T, LT]) -> bool:
        return await async_is_sorted(self.iterator, key)

    @wrap_future
    async def is_sorted_reverse(self: AsyncIter[LT]) -> bool:
        return await async_is_sorted(self.iterator, reverse=True)

    @wrap_future
    async def is_sorted_reverse_by(self, key: Unary[T, LT]) -> bool:
        return await async_is_sorted(self.iterator, key, reverse=True)

    @wrap_future
    async def is_sorted_strict(self: AsyncIter[ST]) -> bool:
        return await async_is_sorted(self.iterator, strict=True)

    @wrap_future
    async def is_sorted_strict_by(self, key: Unary[T, ST]) -> bool:
        return await async_is_sorted(self.iterator, key, strict=True)

    @wrap_future
    async def is_sorted_reverse_strict(self: AsyncIter[ST]) -> bool:
        return await async_is_sorted(self.iterator, strict=True, reverse=True)

    @wrap_future
    async def is_sorted_reverse_strict_by(self, key: Unary[T, ST]) -> bool:
        return await async_is_sorted(self.iterator, key, strict=True, reverse=True)

    @wrap_future
    async def is_sorted_by_await(self, key: AsyncUnary[T, LT]) -> bool:
        return await async_is_sorted_await(self.iterator, key)

    @wrap_future
    async def is_sorted_reverse_by_await(self, key: AsyncUnary[T, LT]) -> bool:
        return await async_is_sorted_await(self.iterator, key, reverse=True)

    @wrap_future
    async def is_sorted_strict_by_await(self, key: AsyncUnary[T, ST]) -> bool:
        return await async_is_sorted_await(self.iterator, key, strict=True)

    @wrap_future
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

    def partition(self, predicate: OptionalPredicate[T]) -> Pair[AsyncIter[T]]:
        true, false = async_partition(predicate, self.iterator)

        return (self.create(true), self.create(false))

    def partition_await(self, predicate: AsyncPredicate[T]) -> Pair[AsyncIter[T]]:
        true, false = async_partition_await(predicate, self.iterator)

        return (self.create(true), self.create(false))

    def partition_unsafe(self, predicate: OptionalPredicate[T]) -> Pair[AsyncIter[T]]:
        true, false = async_partition_unsafe(predicate, self.iterator)

        return (self.create(true), self.create(false))

    partition_infinite = partition_unsafe

    def partition_unsafe_await(self, predicate: AsyncPredicate[T]) -> Pair[AsyncIter[T]]:
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

    @wrap_future
    async def spy(self, size: int) -> List[T]:
        result, iterator = await async_spy(size, self.iterator)

        self._replace(iterator)

        return result

    @wrap_future_option
    async def peek(self) -> Option[T]:
        item, iterator = await async_peek(self.iterator, marker)

        self._replace(iterator)

        return wrap_marked(item)

    @wrap_future
    async def has_next(self) -> bool:
        result, iterator = await async_has_next(self.iterator)

        self._replace(iterator)

        return result

    @wrap_future
    async def is_empty(self) -> bool:
        result, iterator = await async_is_empty(self.iterator)

        self._replace(iterator)

        return result

    def repeat_last(self) -> AsyncIter[T]:
        return self.create(async_repeat_last(self.iterator))

    def repeat_each(self, count: int) -> AsyncIter[T]:
        return self.create(async_repeat_each(count, self.iterator))

    def inspect(self, function: Inspect[T]) -> AsyncIter[T]:
        return self.create(async_inspect(function, self.iterator))

    def inspect_await(self, function: AsyncInspect[T]) -> AsyncIter[T]:
        return self.create(async_inspect_await(function, self.iterator))

    def scan(self, state: V, function: Binary[V, T, Option[U]]) -> AsyncIter[U]:
        return self.create(async_scan(state, function, self.iterator))

    def scan_await(self, state: V, function: AsyncBinary[V, T, Option[U]]) -> AsyncIter[U]:
        return self.create(async_scan_await(state, function, self.iterator))

    def filter_map_option(self, function: Unary[T, Option[U]]) -> AsyncIter[U]:
        return self.create(async_filter_map_option(function, self.iterator))

    def filter_map_option_await(self, function: AsyncUnary[T, Option[U]]) -> AsyncIter[U]:
        return self.create(async_filter_map_option_await(function, self.iterator))

    def wait(self: AsyncIter[Awaitable[U]]) -> AsyncIter[U]:
        return self.create(async_wait(self.iterator))

    def wait_concurrent(self: AsyncIter[Awaitable[U]]) -> AsyncIter[U]:
        return self.create(async_wait_concurrent(self.iterator))

    def wait_concurrent_bound(self: AsyncIter[Awaitable[U]], bound: int) -> AsyncIter[U]:
        return self.create(async_wait_concurrent_bound(bound, self.iterator))

    @wrap_future_result
    async def at_most_one(self) -> Result[Option[T], AsyncIter[T]]:
        result = await async_at_most_one(self.iterator)

        return result.map_error(self.create)

    @wrap_future_result
    async def exactly_one(self) -> Result[T, Option[AsyncIter[T]]]:
        result = await async_exactly_one(self.iterator)

        return result.map_error(self.create_option)

    @wrap_future
    async def into_iter(self) -> Iter[T]:
        return iter(await self.extract())


async_iter = AsyncIter


def wrap_async_iter(function: Callable[PS, AnyIterable[T]]) -> Callable[PS, AsyncIter[T]]:
    @wraps(function)
    def wrap(*args: PS.args, **kwargs: PS.kwargs) -> AsyncIter[T]:
        return async_iter(function(*args, **kwargs))

    return wrap


from iters.iters import Iter, iter
