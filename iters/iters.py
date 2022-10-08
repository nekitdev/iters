from __future__ import annotations

from builtins import iter as standard_iter
from builtins import reversed as standard_reversed
from functools import wraps
from typing import (
    Any,
    AnyStr,
    Callable,
    ContextManager,
    Counter,
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
    no_type_check,
    overload,
)

from typing_extensions import Literal, Never, ParamSpec

from iters.types import Ordering
from iters.typing import (
    AnyExceptionType,
    Binary,
    DynamicTuple,
    EitherLenientOrdered,
    EitherStrictOrdered,
    EmptyTuple,
    Nullary,
    Predicate,
    Product,
    RecursiveIterable,
    Selectors,
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
from iters.utils import (
    accumulate_fold,
    accumulate_product,
    accumulate_reduce,
    accumulate_sum,
    all_equal,
    all_unique,
    all_unique_fast,
    append,
    at,
    at_or_last,
    cartesian_power,
    cartesian_product,
    chain,
    chain_from_iterable,
    chunks,
    collapse,
    combine,
    compare,
    compress,
    consume,
    contains,
    contains_identity,
    copy,
    copy_unsafe,
    count,
    count_dict,
    cycle,
    distribute,
    distribute_unsafe,
    divide,
    drop,
    drop_while,
    duplicates,
    duplicates_fast,
    empty,
    filter_except,
    filter_false,
    filter_false_map,
    filter_map,
    find,
    find_all,
    find_or_first,
    find_or_last,
    first,
    flat_map,
    flatten,
    fold,
    for_each,
    group,
    group_dict,
    group_list,
    groups,
    groups_longest,
    has_next,
    interleave,
    interleave_longest,
    intersperse,
    intersperse_with,
    is_empty,
    is_sorted,
    iter_chunks,
    iter_chunks_unsafe,
    iter_except,
    iter_function,
    iter_length,
    iter_slice,
    iter_windows,
    iter_with,
    iterate,
    last,
    last_with_tail,
    list_windows,
    map_except,
    min_max,
    once,
    once_with,
    pad,
    pad_with,
    pairs,
    pairs_longest,
    pairs_windows,
    partition,
    partition_unsafe,
    peek,
    position,
    position_all,
    prepend,
    product,
    reduce,
    remove,
    remove_duplicates,
    repeat,
    repeat_each,
    repeat_last,
    repeat_with,
    reverse,
    side_effect,
    sort,
    spy,
    step_by,
    sum,
    tail,
    take,
    take_while,
    tuple_windows,
    unique,
    unique_fast,
    zip,
    zip_equal,
    zip_longest,
)

__all__ = (
    # the iterator type
    "Iter",
    # the alias of the previous type
    "iter",
    # the alias of `iter.reversed`
    "reversed",
    # since we are shadowing standard functions
    "standard_iter",
    "standard_reversed",
    # wrap results of function calls into iterators
    "wrap_iter",
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


class Iter(Iterator[T]):
    _iterator: Iterator[T]

    @property
    def iterator(self) -> Iterator[T]:
        """The underlying iterator."""
        return self._iterator

    def _replace(self, iterator: Iterator[T]) -> None:
        self._iterator = iterator

    @classmethod
    def empty(cls) -> Iter[T]:
        """Creates an empty iterator.

        Returns:
            An empty [`Iter[T]`][iters.iters.Iter].
        """
        return cls.create(empty())

    @classmethod
    def once(cls, value: V) -> Iter[V]:
        """Creates an iterator that yields a `value` exactly once.

        This is commonly used to adapt a single value into a [`chain`][iters.iters.Iter.chain]
        of other kinds of iteration. Maybe you have an iterator that covers almost everything,
        but you need an extra special case. Maybe you have a function which works on iterators,
        but you only need to process one value.

        Arguments:
            value: The value to yield.

        Returns:
            An [`Iter[V]`][iters.iters.Iter] with `value` of type `V`.
        """
        return cls.create(once(value))

    @classmethod
    def once_with(cls, function: Nullary[V]) -> Iter[V]:
        """Creates an iterator that lazily generates a value exactly once
        by invoking the `function` provided.

        This is commonly used to adapt a single value into a [`chain`][iters.iters.Iter.chain]
        of other kinds of iteration. Maybe you have an iterator that covers almost everything,
        but you need an extra special case. Maybe you have a function which works on iterators,
        but you only need to process one value.

        Unlike [`once`][iters.iters.Iter.once], this function will
        lazily generate the value on request.

        Arguments:
            function: The value-generating function to use.

        Returns:
            An [`Iter[V]`][iters.iters.Iter] with the generated `value` of type `V`.
        """
        return cls.create(once_with(function))

    @classmethod
    def repeat(cls, value: V) -> Iter[V]:
        """Creates an iterator that endlessly repeats a single `value`.

        This function repeats a single value over and over again.

        Infinite iterators like [`repeat`][iters.iters.Iter.repeat]
        are often used with adapters like [`take`][iters.iters.Iter.take],
        in order to make them finite.

        Arguments:
            value: The value to repeat.

        Returns:
            An infinite [`Iter[V]`][iters.iters.Iter] with repeated `value` of type `V`.
        """
        return cls.create(repeat(value))

    @classmethod
    def repeat_exactly(cls, value: V, count: int) -> Iter[V]:
        """Creates an iterator that repeats a single `value` exactly `count` times.

        This function is a shorthand for [`iter.repeat(value).take(count)`][iters.iters.Iter.take].

        Arguments:
            value: The value to repeat.
            count: The number of times to repeat the `value`.

        Returns:
            An [`Iter[V]`][iters.iters.Iter] with `value` of type `V` repeated `count` times.
        """
        return cls.create(repeat(value, count))

    @classmethod
    def repeat_with(cls, function: Nullary[V]) -> Iter[V]:
        """Creates an iterator that endlessly generates values of type `V`.

        This function repeats values over and over again.

        Infinite iterators like [`repeat_with`][iters.iters.Iter.repeat_with]
        are often used with adapters like [`take`][iters.iters.Iter.take],
        in order to make them finite.

        Arguments:
            function: The value-generating function to use.

        Returns:
            An infinite [`Iter[V]`][iters.iters.Iter] with repeated `value` of type `V`.
        """
        return cls.create(repeat_with(function))

    @classmethod
    def repeat_exactly_with(cls, function: Nullary[V], count: int) -> Iter[V]:
        """Creates an iterator that generates values of type `V` exactly `count` times.

        This function is a shorthand for
        [`iter.repeat_with(function).take(count)`][iters.iters.Iter.take].

        Arguments:
            function: The value-generating function to use.
            count: The number of times to repeat values.

        Returns:
            An [`Iter[V]`][iters.iters.Iter] with repeated
                `value` of type `V` exactly `count` times.
        """
        return cls.create(repeat_with(function, count))

    @classmethod
    def count_from_by(cls, start: int, step: int) -> Iter[int]:
        return cls.create(count(start, step))

    @classmethod
    def count_from(cls, start: int) -> Iter[int]:
        return cls.count_from_by(start, DEFAULT_STEP)

    @classmethod
    def count_by(cls, step: int) -> Iter[int]:
        return cls.count_from_by(DEFAULT_START, step)

    @classmethod
    def count(cls) -> Iter[int]:
        return cls.count_from_by(DEFAULT_START, DEFAULT_STEP)

    @classmethod
    def iterate(cls, function: Unary[V, V], value: V) -> Iter[V]:
        return cls.create(iterate(function, value))

    @classmethod
    def iterate_exactly(cls, function: Unary[V, V], value: V, count: int) -> Iter[V]:
        return cls.create(iterate(function, value, count))

    @classmethod
    def iter_except(cls, function: Nullary[T], *errors: AnyExceptionType) -> Iter[T]:
        return cls.create(iter_except(function, *errors))

    @classmethod
    def iter_with(cls, context_manager: ContextManager[Iterable[T]]) -> Iter[T]:
        return cls.create(iter_with(context_manager))

    @classmethod
    def create_chain(cls, *iterables: Iterable[T]) -> Iter[T]:
        return cls.create(chain(*iterables))

    @classmethod
    def create_chain_from_iterable(cls, iterable: Iterable[Iterable[T]]) -> Iter[T]:
        return cls.create(chain_from_iterable(iterable))

    @classmethod
    def create_combine(cls, *iterables: Iterable[T]) -> Iter[T]:
        return cls.create(combine(*iterables))

    @classmethod
    def create_interleave(cls, *iterables: Iterable[T]) -> Iter[T]:
        return cls.create(interleave(*iterables))

    @classmethod
    def create_interleave_longest(cls, *iterables: Iterable[T]) -> Iter[T]:
        return cls.create(interleave_longest(*iterables))

    @overload
    @classmethod
    def create_zip(cls) -> Iter[T]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(cls, __iterable_a: Iterable[A]) -> Iter[Tuple[A]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B]) -> Iter[Tuple[A, B]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[A, B, C]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[A, B, C, D]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[A, B, C, D, E]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[Tuple[A, B, C, D, E, F]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[Tuple[A, B, C, D, E, F, G]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[Tuple[A, B, C, D, E, F, G, H]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip(
        cls,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Any]]:
        ...  # pragma: overload

    @no_type_check
    @classmethod
    def create_zip(cls, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Any]]:
        return cls.create(zip(*iterables))

    @overload
    @classmethod
    def create_zip_equal(cls) -> Iter[T]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(cls, __iterable_a: Iterable[A]) -> Iter[Tuple[A]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B]
    ) -> Iter[Tuple[A, B]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[A, B, C]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[A, B, C, D]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[A, B, C, D, E]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[Tuple[A, B, C, D, E, F]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[Tuple[A, B, C, D, E, F, G]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[Tuple[A, B, C, D, E, F, G, H]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_equal(
        cls,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Any]]:
        ...  # pragma: overload

    @no_type_check
    @classmethod
    def create_zip_equal(cls, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Any]]:
        return cls.create(zip_equal(*iterables))

    @overload
    @classmethod
    def create_zip_longest(cls) -> Iter[T]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(cls, __iterable_a: Iterable[A]) -> Iter[Tuple[A]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B]
    ) -> Iter[Tuple[Optional[A], Optional[B]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[Optional[A], Optional[B], Optional[C]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[Optional[A], Optional[B], Optional[C], Optional[D]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[Tuple[Optional[A], Optional[B], Optional[C], Optional[D], Optional[E], Optional[F]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest(
        cls,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Optional[Any]]]:
        ...  # pragma: overload

    @no_type_check
    @classmethod
    def create_zip_longest(cls, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Optional[Any]]]:
        return cls.create(zip_longest(*iterables))

    @overload
    @classmethod
    def create_zip_longest_with(cls, *, fill: V) -> Iter[T]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(cls, __iterable_a: Iterable[A], *, fill: V) -> Iter[Tuple[A]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B], *, fill: V
    ) -> Iter[Tuple[Union[A, V], Union[B, V]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        *,
        fill: V,
    ) -> Iter[Tuple[Union[A, V], Union[B, V], Union[C, V]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        *,
        fill: V,
    ) -> Iter[Tuple[Union[A, V], Union[B, V], Union[C, V], Union[D, V]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        *,
        fill: V,
    ) -> Iter[Tuple[Union[A, V], Union[B, V], Union[C, V], Union[D, V], Union[E, V]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        *,
        fill: V,
    ) -> Iter[Tuple[Union[A, V], Union[B, V], Union[C, V], Union[D, V], Union[E, V], Union[F, V]]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        *,
        fill: V,
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
        *,
        fill: V,
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    @classmethod
    def create_zip_longest_with(
        cls,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
        fill: V,
    ) -> Iter[DynamicTuple[Union[Any, V]]]:
        ...  # pragma: overload

    @no_type_check
    @classmethod
    def create_zip_longest_with(
        cls, *iterables: Iterable[Any], fill: V
    ) -> Iter[DynamicTuple[Union[Any, V]]]:
        return cls.create(zip_longest(*iterables, fill=fill))

    @overload
    @classmethod
    def create_cartesian_product(cls) -> Iter[EmptyTuple]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(cls, __iterable_a: Iterable[A]) -> Iter[Tuple[A]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B]
    ) -> Iter[Tuple[A, B]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[A, B, C]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[A, B, C, D]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[A, B, C, D, E]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[Tuple[A, B, C, D, E, F]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[Tuple[A, B, C, D, E, F, G]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[Tuple[A, B, C, D, E, F, G, H]]:
        ...  # pragma: overload

    @overload
    @classmethod
    def create_cartesian_product(
        cls,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Any]]:
        ...  # pragma: overload

    @no_type_check
    @classmethod
    def create_cartesian_product(cls, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Any]]:
        return cls.create(cartesian_product(*iterables))

    @classmethod
    def reversed(cls, reversible: Reversible[T]) -> Iter[T]:
        return cls.create(standard_reversed(reversible))

    @classmethod
    def function(cls, function: Nullary[T], sentinel: V) -> Iter[T]:
        return cls.create(iter_function(function, sentinel))

    @classmethod
    def create(cls, iterable: Iterable[U]) -> Iter[U]:
        return cls(iterable)  # type: ignore

    @classmethod
    def create_tuple(cls, iterables: DynamicTuple[Iterable[U]]) -> DynamicTuple[Iter[U]]:
        return tuple(map(cls, iterables))  # type: ignore

    @classmethod
    def create_nested(cls, nested: Iterable[Iterable[U]]) -> Iter[Iter[U]]:
        return cls(map(cls, nested))  # type: ignore

    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterator = standard_iter(iterable)

    def __iter__(self) -> Iter[T]:
        return self.iter()

    def __next__(self) -> T:
        return self.next()

    def unwrap(self) -> Iterator[T]:
        return self.iterator

    def iter(self) -> Iter[T]:
        return self

    def next(self) -> T:
        return next(self.iterator)

    def next_or(self, default: V) -> Union[T, V]:
        return next(self.iterator, default)

    def next_or_none(self) -> Optional[T]:
        return self.next_or(None)

    def compare(self: Iter[ST], other: Iterable[ST]) -> Ordering:
        return compare(self.iterator, other)

    def compare_by(self, other: Iterable[T], key: Unary[T, ST]) -> Ordering:
        return compare(self.iterator, other, key)

    def length(self) -> int:
        return iter_length(self.iterator)

    def first(self) -> T:
        return first(self.iterator)

    def first_or(self, default: V) -> Union[T, V]:
        return first(self.iterator, default)

    def first_or_none(self) -> Optional[T]:
        return self.first_or(None)

    def last(self) -> T:
        return last(self.iterator)

    def last_or(self, default: V) -> Union[T, V]:
        return last(self.iterator, default)

    def last_or_none(self) -> Optional[T]:
        return self.last_or(None)

    def last_with_tail(self) -> T:
        return last_with_tail(self.iterator)

    def last_with_tail_or(self, default: V) -> Union[T, V]:
        return last_with_tail(self.iterator, default)

    def last_with_tail_or_none(self) -> Optional[T]:
        return self.last_with_tail_or(None)

    def collect(self, function: Unary[Iterable[T], U]) -> U:
        return function(self.iterator)

    def list(self) -> List[T]:
        return list(self.iterator)

    def set(self: Iter[Q]) -> Set[Q]:
        return set(self.iterator)

    def tuple(self) -> DynamicTuple[T]:
        return tuple(self.iterator)

    def dict(self: Iter[Tuple[Q, V]]) -> Dict[Q, V]:
        return dict(self.iterator)

    def join(self: Iter[AnyStr], string: AnyStr) -> AnyStr:
        return string.join(self.iterator)

    def string(self: Iter[str]) -> str:
        return self.join(EMPTY_STRING)

    def bytes(self: Iter[bytes]) -> bytes:
        return self.join(EMPTY_BYTES)

    def count_dict(self: Iter[Q]) -> Counter[Q]:
        return count_dict(self.iterator)

    def count_dict_by(self, key: Unary[T, Q]) -> Counter[Q]:
        return count_dict(self.iterator, key)

    def group_dict(self: Iter[Q]) -> Dict[Q, List[Q]]:
        return group_dict(self.iterator)

    def group_dict_by(self, key: Unary[T, Q]) -> Dict[Q, List[T]]:
        return group_dict(self.iterator, key)

    def group(self) -> Iter[Tuple[T, Iter[T]]]:
        return self.create(
            (group_key, self.create(group_iterator))
            for group_key, group_iterator in group(self.iterator)
        )

    def group_by(self, key: Unary[T, U]) -> Iter[Tuple[U, Iter[T]]]:
        return self.create(
            (group_key, self.create(group_iterator))
            for group_key, group_iterator in group(self.iterator, key)
        )

    def group_list(self) -> Iter[Tuple[T, List[T]]]:
        return self.create(group_list(self.iterator))

    def group_list_by(self, key: Unary[T, U]) -> Iter[Tuple[U, List[T]]]:
        return self.create(group_list(self.iterator, key))

    def all(self) -> bool:
        return all(self.iterator)

    def all_by(self, predicate: Predicate[T]) -> bool:
        return self.map(predicate).all()

    def any(self) -> bool:
        return any(self.iterator)

    def any_by(self, predicate: Predicate[T]) -> bool:
        return self.map(predicate).any()

    def all_equal(self) -> bool:
        return all_equal(self.iterator)

    def all_equal_by(self, key: Unary[T, U]) -> bool:
        return all_equal(self.iterator, key)

    def all_unique(self) -> bool:
        return all_unique(self.iterator)

    def all_unique_by(self, key: Unary[T, U]) -> bool:
        return all_unique(self.iterator, key)

    def all_unique_fast(self: Iter[Q]) -> bool:
        return all_unique_fast(self.iterator)

    def all_unique_fast_by(self, key: Unary[T, Q]) -> bool:
        return all_unique_fast(self.iterator, key)

    def remove(self, predicate: Predicate[T]) -> Iter[T]:
        return self.create(remove(predicate, self.iterator))

    def remove_duplicates(self) -> Iter[T]:
        return self.create(remove_duplicates(self.iterator))

    def remove_duplicates_by(self, key: Unary[T, U]) -> Iter[T]:
        return self.create(remove_duplicates(self.iterator, key))

    def filter(self, predicate: Predicate[T]) -> Iter[T]:
        return self.create(filter(predicate, self.iterator))

    def filter_false(self, predicate: Predicate[T]) -> Iter[T]:
        return self.create(filter_false(predicate, self.iterator))

    def filter_except(self, validate: Unary[T, Any], *errors: AnyExceptionType) -> Iter[T]:
        return self.create(filter_except(validate, self.iterator, *errors))

    def compress(self, selectors: Selectors) -> Iter[T]:
        return self.create(compress(self.iterator, selectors))

    def position_all(self, predicate: Predicate[T]) -> Iter[int]:
        return self.create(position_all(predicate, self.iterator))

    def position(self, predicate: Predicate[T]) -> int:
        return position(predicate, self.iterator)

    def position_or(self, predicate: Predicate[T], default: V) -> Union[int, V]:
        return position(predicate, self.iterator, default)

    def position_or_none(self, predicate: Predicate[T]) -> Optional[int]:
        return self.position_or(predicate, None)

    def find_all(self, predicate: Predicate[T]) -> Iter[T]:
        return self.create(find_all(predicate, self.iterator))

    def find(self, predicate: Predicate[T]) -> T:
        return find(predicate, self.iterator)

    def find_or(self, predicate: Predicate[T], default: V) -> Union[T, V]:
        return find(predicate, self.iterator, default)  # type: ignore  # strange

    def find_or_none(self, predicate: Predicate[T]) -> Optional[T]:
        return self.find_or(predicate, None)

    def find_or_first(self, predicate: Predicate[T]) -> T:
        return find_or_first(predicate, self.iterator)

    def find_or_first_or(self, predicate: Predicate[T], default: V) -> Union[T, V]:
        return find_or_first(predicate, self.iterator, default)  # type: ignore  # strange

    def find_or_first_or_none(self, predicate: Predicate[T]) -> Optional[T]:
        return self.find_or_first_or(predicate, None)

    def find_or_last(self, predicate: Predicate[T]) -> T:
        return find_or_last(predicate, self.iterator)

    def find_or_last_or(self, predicate: Predicate[T], default: V) -> Union[T, V]:
        return find_or_last(predicate, self.iterator, default)  # type: ignore  # strange

    def find_or_last_or_none(self, predicate: Predicate[T]) -> Optional[T]:
        return self.find_or_last_or(predicate, None)

    def contains(self, item: V) -> bool:
        return contains(item, self.iterator)

    def contains_identity(self: Iter[V], item: V) -> bool:
        return contains_identity(item, self.iterator)

    def reduce(self, function: Binary[T, T, T]) -> T:
        return reduce(function, self.iterator)

    def fold(self, initial: V, function: Binary[V, T, V]) -> V:
        return fold(initial, function, self.iterator)

    def sum(self: Iter[S]) -> S:
        return sum(self.iterator)

    def sum_with(self: Iter[S], initial: S) -> S:
        return sum(self.iterator, initial)

    def product(self: Iter[P]) -> P:
        return product(self.iterator)

    def product_with(self: Iter[P], initial: P) -> P:
        return product(self.iterator, initial)

    def accumulate_reduce(self, function: Binary[T, T, T]) -> Iter[T]:
        return self.create(accumulate_reduce(function, self.iterator))

    def accumulate_fold(self, initial: V, function: Binary[V, T, V]) -> Iter[V]:
        return self.create(accumulate_fold(initial, function, self.iterator))

    def accumulate_sum(self: Iter[S]) -> Iter[S]:
        return self.create(accumulate_sum(self.iterator))

    def accumulate_sum_with(self: Iter[S], initial: S) -> Iter[S]:
        return self.create(accumulate_sum(self.iterator, initial))

    def accumulate_product(self: Iter[P]) -> Iter[P]:
        return self.create(accumulate_product(self.iterator))

    def accumulate_product_with(self: Iter[P], initial: P) -> Iter[P]:
        return self.create(accumulate_product(self.iterator, initial))

    def min(self: Iter[ST]) -> ST:
        return min(self.iterator)

    def min_or(self: Iter[ST], default: V) -> Union[ST, V]:
        return min(self.iterator, default=default)

    def min_or_none(self: Iter[ST]) -> Optional[ST]:
        return self.min_or(None)

    def min_by(self, key: Unary[T, ST]) -> T:
        return min(self.iterator, key=key)

    def min_by_or(self, key: Unary[T, ST], default: V) -> Union[T, V]:
        return min(self.iterator, key=key, default=default)  # type: ignore  # strange

    def min_by_or_none(self, key: Unary[T, ST]) -> Optional[T]:
        return self.min_by_or(key, None)

    def max(self: Iter[ST]) -> ST:
        return max(self.iterator)

    def max_or(self: Iter[ST], default: V) -> Union[ST, V]:
        return max(self.iterator, default=default)

    def max_or_none(self: Iter[ST]) -> Optional[ST]:
        return self.max_or(None)

    def max_by(self, key: Unary[T, ST]) -> T:
        return max(self.iterator, key=key)

    def max_by_or(self, key: Unary[T, ST], default: V) -> Union[T, V]:
        return max(self.iterator, key=key, default=default)  # type: ignore  # strange

    def max_by_or_none(self, key: Unary[T, ST]) -> Optional[T]:
        return self.max_by_or(key, None)

    def min_max(self: Iter[ST]) -> Tuple[ST, ST]:
        return min_max(self.iterator)

    def min_max_or(self: Iter[ST], default: Tuple[V, W]) -> Union[Tuple[ST, ST], Tuple[V, W]]:
        return min_max(self.iterator, default=default)

    def min_max_by(self, key: Unary[T, ST]) -> Tuple[T, T]:
        return min_max(self.iterator, key=key)

    def min_max_by_or(
        self, key: Unary[T, ST], default: Tuple[V, W]
    ) -> Tuple[Union[T, V], Union[T, W]]:
        return min_max(self.iterator, key=key, default=default)

    def map(self, function: Unary[T, U]) -> Iter[U]:
        return self.create(map(function, self.iterator))

    def map_except(self, function: Unary[T, U], *errors: AnyExceptionType) -> Iter[U]:
        return self.create(map_except(function, self.iterator, *errors))

    def flat_map(self, function: Unary[T, Iterable[U]]) -> Iter[U]:
        return self.create(flat_map(function, self.iterator))

    def filter_map(self, predicate: Predicate[T], function: Unary[T, U]) -> Iter[U]:
        return self.create(filter_map(predicate, function, self.iterator))

    def filter_false_map(self, predicate: Predicate[T], function: Unary[T, U]) -> Iter[U]:
        return self.create(filter_false_map(predicate, function, self.iterator))

    def flatten(self: Iter[Iterable[U]]) -> Iter[U]:
        return self.create(flatten(self.iterator))

    def collapse(self: Iter[RecursiveIterable[U]]) -> Iter[U]:
        return self.create(collapse(self.iterator))

    def enumerate(self) -> Iter[Tuple[int, T]]:
        return self.create(enumerate(self.iterator))

    def enumerate_from(self, start: int) -> Iter[Tuple[int, T]]:
        return self.create(enumerate(self.iterator, start))

    def consume(self) -> None:
        consume(self.iterator)

    def for_each(self, function: Unary[T, Any]) -> None:
        for_each(function, self.iterator)

    def append(self: Iter[V], item: V) -> Iter[V]:
        return self.create(append(item, self.iterator))

    def prepend(self: Iter[V], item: V) -> Iter[V]:
        return self.create(prepend(item, self.iterator))

    def at(self, index: int) -> T:
        return at(index, self.iterator)

    def at_or(self, index: int, default: V) -> Union[T, V]:
        return at(index, self.iterator, default)

    def at_or_none(self, index: int) -> Optional[T]:
        return self.at_or(index, None)

    def at_or_last(self, index: int) -> T:
        return at_or_last(index, self.iterator)

    def at_or_last_or(self, index: int, default: V) -> Union[T, V]:
        return at_or_last(index, self.iterator, default)

    def at_or_last_or_none(self, index: int) -> Optional[T]:
        return self.at_or_last_or(index, None)

    @overload
    def slice(self, __stop: Optional[int]) -> Iter[T]:
        ...  # pragma: overload

    @overload
    def slice(
        self, __start: Optional[int], __stop: Optional[int], __step: Optional[int] = ...
    ) -> Iter[T]:
        ...  # pragma: overload

    def slice(self, *slice_args: Optional[int]) -> Iter[T]:
        return self.create(iter_slice(self.iterator, *slice_args))

    def drop(self, size: int) -> Iter[T]:
        return self.create(drop(size, self.iterator))

    skip = drop

    def drop_while(self, predicate: Predicate[T]) -> Iter[T]:
        return self.create(drop_while(predicate, self.iterator))

    skip_while = drop_while

    def take(self, size: int) -> Iter[T]:
        return self.create(take(size, self.iterator))

    def take_while(self, predicate: Predicate[T]) -> Iter[T]:
        return self.create(take_while(predicate, self.iterator))

    def step_by(self, step: int) -> Iter[T]:
        return self.create(step_by(step, self.iterator))

    def tail(self, size: int) -> Iter[T]:
        return self.create(tail(size, self.iterator))

    def chain(self, *iterables: Iterable[T]) -> Iter[T]:
        return self.create(chain(self.iterator, *iterables))

    def chain_with(self, iterables: Iterable[Iterable[T]]) -> Iter[T]:
        return self.chain(chain_from_iterable(iterables))

    def cycle(self) -> Iter[T]:
        return self.create(cycle(self.iterator))

    def intersperse(self: Iter[V], value: V) -> Iter[V]:
        return self.create(intersperse(value, self.iterator))

    def intersperse_with(self, function: Nullary[T]) -> Iter[T]:
        return self.create(intersperse_with(function, self.iterator))

    def interleave(self, *iterables: Iterable[T]) -> Iter[T]:
        return self.create(interleave(self.iterator, *iterables))

    def interleave_longest(self, *iterables: Iterable[T]) -> Iter[T]:
        return self.create(interleave_longest(self.iterator, *iterables))

    def combine(self, *iterables: Iterable[T]) -> Iter[T]:
        return self.create(combine(*iterables))

    @overload
    def distribute_unsafe(self, count: Literal[0]) -> EmptyTuple:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[1]) -> Tuple1[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[2]) -> Tuple2[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[3]) -> Tuple3[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[4]) -> Tuple4[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[5]) -> Tuple5[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[6]) -> Tuple6[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[7]) -> Tuple7[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: Literal[8]) -> Tuple8[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute_unsafe(self, count: int) -> DynamicTuple[Iter[T]]:
        ...  # pragma: overload

    def distribute_unsafe(self, count: int) -> DynamicTuple[Iter[T]]:
        return self.create_tuple(distribute_unsafe(count, self.iterator))

    distribute_infinite = distribute_unsafe

    @overload
    def distribute(self, count: Literal[0]) -> EmptyTuple:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[1]) -> Tuple1[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[2]) -> Tuple2[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[3]) -> Tuple3[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[4]) -> Tuple4[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[5]) -> Tuple5[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[6]) -> Tuple6[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[7]) -> Tuple7[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: Literal[8]) -> Tuple8[Iter[T]]:
        ...  # pragma: overload

    @overload
    def distribute(self, count: int) -> DynamicTuple[Iter[T]]:
        ...  # pragma: overload

    def distribute(self, count: int) -> DynamicTuple[Iter[T]]:
        return self.create_tuple(distribute(count, self.iterator))

    def divide(self, count: int) -> Iter[Iter[T]]:
        return self.create_nested(divide(count, self.iterator))

    def pad(self, value: V) -> Iter[Union[T, V]]:
        return self.create(pad(value, self.iterator))

    def pad_exactly(self, value: V, size: int) -> Iter[Union[T, V]]:
        return self.create(pad(value, self.iterator, size))

    def pad_multiple(self, value: V, size: int) -> Iter[Union[T, V]]:
        return self.create(pad(value, self.iterator, size, multiple=True))

    def pad_none(self) -> Iter[Optional[T]]:
        return self.pad(None)

    def pad_none_exactly(self, size: int) -> Iter[Optional[T]]:
        return self.pad_exactly(None, size)

    def pad_none_multiple(self, size: int) -> Iter[Optional[T]]:
        return self.pad_multiple(None, size)

    def pad_with(self, function: Unary[int, V]) -> Iter[Union[T, V]]:
        return self.create(pad_with(function, self.iterator))

    def pad_exactly_with(self, function: Unary[int, V], size: int) -> Iter[Union[T, V]]:
        return self.create(pad_with(function, self.iterator, size))

    def pad_multiple_with(self, function: Unary[int, V], size: int) -> Iter[Union[T, V]]:
        return self.create(pad_with(function, self.iterator, size, multiple=True))

    def chunks(self, size: int) -> Iter[List[T]]:
        return self.create(chunks(size, self.iterator))

    def iter_chunks(self, size: int) -> Iter[Iter[T]]:
        return self.create_nested(iter_chunks(size, self.iterator))

    def iter_chunks_unsafe(self, size: int) -> Iter[Iter[T]]:
        return self.create_nested(iter_chunks_unsafe(size, self.iterator))

    iter_chunks_infinite = iter_chunks_unsafe

    @overload
    def groups(self, size: Literal[0]) -> Iter[Never]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[1]) -> Iter[Tuple1[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[2]) -> Iter[Tuple2[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[3]) -> Iter[Tuple3[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[4]) -> Iter[Tuple4[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[5]) -> Iter[Tuple5[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[6]) -> Iter[Tuple6[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[7]) -> Iter[Tuple7[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: Literal[8]) -> Iter[Tuple8[T]]:
        ...  # pragma: overload

    @overload
    def groups(self, size: int) -> Iter[DynamicTuple[T]]:
        ...  # pragma: overload

    def groups(self, size: int) -> Iter[DynamicTuple[T]]:
        return self.create(groups(size, self.iterator))

    @overload
    def groups_longest(self, size: Literal[0]) -> Iter[Never]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[1]) -> Iter[Tuple1[T]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[2]) -> Iter[Tuple2[Optional[T]]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[3]) -> Iter[Tuple3[Optional[T]]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[4]) -> Iter[Tuple4[Optional[T]]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[5]) -> Iter[Tuple5[Optional[T]]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[6]) -> Iter[Tuple6[Optional[T]]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[7]) -> Iter[Tuple7[Optional[T]]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: Literal[8]) -> Iter[Tuple8[Optional[T]]]:
        ...  # pragma: overload

    @overload
    def groups_longest(self, size: int) -> Iter[DynamicTuple[Optional[T]]]:
        ...  # pragma: overload

    def groups_longest(self, size: int) -> Iter[DynamicTuple[Optional[T]]]:
        return self.create(groups_longest(size, self.iterator))

    @overload
    def groups_longest_with(self, size: Literal[0], fill: V) -> Iter[Never]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[1], fill: V) -> Iter[Tuple1[T]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[2], fill: V) -> Iter[Tuple2[Union[T, V]]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[3], fill: V) -> Iter[Tuple3[Union[T, V]]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[4], fill: V) -> Iter[Tuple4[Union[T, V]]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[5], fill: V) -> Iter[Tuple5[Union[T, V]]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[6], fill: V) -> Iter[Tuple6[Union[T, V]]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[7], fill: V) -> Iter[Tuple7[Union[T, V]]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: Literal[8], fill: V) -> Iter[Tuple8[Union[T, V]]]:
        ...  # pragma: overload

    @overload
    def groups_longest_with(self, size: int, fill: V) -> Iter[DynamicTuple[Union[T, V]]]:
        ...  # pragma: overload

    def groups_longest_with(self, size: int, fill: V) -> Iter[DynamicTuple[Union[T, V]]]:
        return self.create(groups_longest(size, self.iterator, fill))

    def pairs(self) -> Iter[Tuple[T, T]]:
        return self.create(pairs(self.iterator))

    def pairs_longest(self) -> Iter[Tuple[Optional[T], Optional[T]]]:
        return self.create(pairs_longest(self.iterator))

    def pairs_longest_with(self, fill: V) -> Iter[Tuple[Union[T, V], Union[T, V]]]:
        return self.create(pairs_longest(self.iterator, fill))

    def iter_windows(self, size: int) -> Iter[Iter[T]]:
        return self.create_nested(iter_windows(size, self.iterator))

    def list_windows(self, size: int) -> Iter[List[T]]:
        return self.create(list_windows(size, self.iterator))

    def pairs_windows(self) -> Iter[Tuple[T, T]]:
        return self.create(pairs_windows(self.iterator))

    @overload
    def tuple_windows(self, size: Literal[0]) -> Iter[EmptyTuple]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[1]) -> Iter[Tuple1[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[2]) -> Iter[Tuple2[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[3]) -> Iter[Tuple3[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[4]) -> Iter[Tuple4[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[5]) -> Iter[Tuple5[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[6]) -> Iter[Tuple6[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[7]) -> Iter[Tuple7[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: Literal[8]) -> Iter[Tuple8[T]]:
        ...  # pragma: overload

    @overload
    def tuple_windows(self, size: int) -> Iter[DynamicTuple[T]]:
        ...  # pragma: overload

    def tuple_windows(self, size: int) -> Iter[DynamicTuple[T]]:
        return self.create(tuple_windows(size, self.iterator))

    @overload
    def zip(self) -> Iter[Tuple[T]]:
        ...  # pragma: overload

    @overload
    def zip(self, __iterable_a: Iterable[A]) -> Iter[Tuple[T, A]]:
        ...  # pragma: overload

    @overload
    def zip(self, __iterable_a: Iterable[A], __iterable_b: Iterable[B]) -> Iter[Tuple[T, A, B]]:
        ...  # pragma: overload

    @overload
    def zip(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[T, A, B, C]]:
        ...  # pragma: overload

    @overload
    def zip(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[T, A, B, C, D]]:
        ...  # pragma: overload

    @overload
    def zip(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[T, A, B, C, D, E]]:
        ...  # pragma: overload

    @overload
    def zip(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[Tuple[T, A, B, C, D, E, F]]:
        ...  # pragma: overload

    @overload
    def zip(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[Tuple[T, A, B, C, D, E, F, G]]:
        ...  # pragma: overload

    @overload
    def zip(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[Tuple[T, A, B, C, D, E, F, G, H]]:
        ...  # pragma: overload

    @overload
    def zip(
        self,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Any]]:
        ...  # pragma: overload

    def zip(self, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Any]]:
        return self.create(zip(self.iterator, *iterables))

    @overload
    def zip_equal(self) -> Iter[Tuple[T]]:
        ...  # pragma: overload

    @overload
    def zip_equal(self, __iterable_a: Iterable[A]) -> Iter[Tuple[T, A]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B]
    ) -> Iter[Tuple[T, A, B]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[T, A, B, C]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[T, A, B, C, D]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[T, A, B, C, D, E]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[Tuple[T, A, B, C, D, E, F]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[Tuple[T, A, B, C, D, E, F, G]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[Tuple[T, A, B, C, D, E, F, G, H]]:
        ...  # pragma: overload

    @overload
    def zip_equal(
        self,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Any]]:
        ...  # pragma: overload

    def zip_equal(self, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Any]]:
        return self.create(zip_equal(self.iterator, *iterables))

    @overload
    def zip_longest(self) -> Iter[Tuple[T]]:
        ...  # pragma: overload

    @overload
    def zip_longest(self, __iterable_a: Iterable[A]) -> Iter[Tuple[Optional[T], Optional[A]]]:
        ...  # pragma: overload

    @overload
    def zip_longest(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B]
    ) -> Iter[Tuple[Optional[T], Optional[A], Optional[B]]]:
        ...  # pragma: overload

    @overload
    def zip_longest(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[Optional[T], Optional[A], Optional[B], Optional[C]]]:
        ...  # pragma: overload

    @overload
    def zip_longest(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[Optional[T], Optional[A], Optional[B], Optional[C], Optional[D]]]:
        ...  # pragma: overload

    @overload
    def zip_longest(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[Optional[T], Optional[A], Optional[B], Optional[C], Optional[D], Optional[E]]]:
        ...  # pragma: overload

    @overload
    def zip_longest(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    def zip_longest(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    def zip_longest(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    def zip_longest(
        self,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Optional[Any]]]:
        ...  # pragma: overload

    def zip_longest(self, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Optional[Any]]]:
        return self.create(zip_longest(self.iterator, *iterables))

    @overload
    def zip_longest_with(self, *, fill: V) -> Iter[Tuple[T]]:
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self, __iterable_a: Iterable[A], *, fill: V
    ) -> Iter[Tuple[Union[T, V], Union[A, V]]]:
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B], *, fill: V
    ) -> Iter[Tuple[Union[T, V], Union[A, V], Union[B, V]]]:
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        *,
        fill: V,
    ) -> Iter[Tuple[Union[T, V], Union[A, V], Union[B, V], Union[C, V]]]:
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        *,
        fill: V,
    ) -> Iter[Tuple[Union[T, V], Union[A, V], Union[B, V], Union[C, V], Union[D, V]]]:
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        *,
        fill: V,
    ) -> Iter[Tuple[Union[T, V], Union[A, V], Union[B, V], Union[C, V], Union[D, V], Union[E, V]]]:
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        *,
        fill: V,
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        *,
        fill: V,
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
        *,
        fill: V,
    ) -> Iter[
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
        ...  # pragma: overload

    @overload
    def zip_longest_with(
        self,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
        fill: V,
    ) -> Iter[DynamicTuple[Union[Any, V]]]:
        ...  # pragma: overload

    @no_type_check  # strange
    def zip_longest_with(
        self, *iterables: Iterable[Any], fill: V
    ) -> Iter[DynamicTuple[Union[Any, V]]]:
        return self.create(zip_longest(self.iterator, *iterables, fill=fill))

    @overload
    def cartesian_product(self) -> Iter[Tuple[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(self, __iterable_a: Iterable[A]) -> Iter[Tuple[T, A]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B]
    ) -> Iter[Tuple[T, A, B]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self, __iterable_a: Iterable[A], __iterable_b: Iterable[B], __iterable_c: Iterable[C]
    ) -> Iter[Tuple[T, A, B, C]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
    ) -> Iter[Tuple[T, A, B, C, D]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
    ) -> Iter[Tuple[T, A, B, C, D, E]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
    ) -> Iter[Tuple[T, A, B, C, D, E, F]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
    ) -> Iter[Tuple[T, A, B, C, D, E, F, G]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self,
        __iterable_a: Iterable[A],
        __iterable_b: Iterable[B],
        __iterable_c: Iterable[C],
        __iterable_d: Iterable[D],
        __iterable_e: Iterable[E],
        __iterable_f: Iterable[F],
        __iterable_g: Iterable[G],
        __iterable_h: Iterable[H],
    ) -> Iter[Tuple[T, A, B, C, D, E, F, G, H]]:
        ...  # pragma: overload

    @overload
    def cartesian_product(
        self,
        __iterable_a: Iterable[Any],
        __iterable_b: Iterable[Any],
        __iterable_c: Iterable[Any],
        __iterable_d: Iterable[Any],
        __iterable_e: Iterable[Any],
        __iterable_f: Iterable[Any],
        __iterable_g: Iterable[Any],
        __iterable_h: Iterable[Any],
        __iterable_next: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> Iter[DynamicTuple[Any]]:
        ...  # pragma: overload

    def cartesian_product(self, *iterables: Iterable[Any]) -> Iter[DynamicTuple[Any]]:
        return self.create(cartesian_product(self.iterator, *iterables))

    @overload
    def cartesian_power(self, power: Literal[0]) -> Iter[EmptyTuple]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[1]) -> Iter[Tuple1[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[2]) -> Iter[Tuple2[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[3]) -> Iter[Tuple3[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[4]) -> Iter[Tuple4[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[5]) -> Iter[Tuple5[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[6]) -> Iter[Tuple6[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[7]) -> Iter[Tuple7[T]]:
        ...  # pragma: overload

    @overload
    def cartesian_power(self, power: Literal[8]) -> Iter[Tuple8[T]]:
        ...  # pragma: overload

    def cartesian_power(self, power: int) -> Iter[DynamicTuple[T]]:
        return self.create(cartesian_power(power, self.iterator))

    def reverse(self) -> Iter[T]:
        return self.create(reverse(self.iterator))

    def sorted(self: Iter[ST]) -> List[ST]:
        return sorted(self.iterator)

    def sorted_by(self, key: Unary[T, ST]) -> List[T]:
        return sorted(self.iterator, key=key)

    def sorted_reverse(self: Iter[ST]) -> List[ST]:
        return sorted(self.iterator, reverse=True)

    def sorted_reverse_by(self, key: Unary[T, ST]) -> List[T]:
        return sorted(self.iterator, key=key, reverse=True)

    def sort(self: Iter[ST]) -> Iter[ST]:
        return self.create(sort(self.iterator))

    def sort_by(self, key: Unary[T, ST]) -> Iter[T]:
        return self.create(sort(self.iterator, key=key))

    def sort_reverse(self: Iter[ST]) -> Iter[ST]:
        return self.create(sort(self.iterator, reverse=True))

    def sort_reverse_by(self, key: Unary[T, ST]) -> Iter[T]:
        return self.create(sort(self.iterator, key=key, reverse=True))

    def is_sorted(self: Iter[LT]) -> bool:
        return is_sorted(self.iterator)

    def is_sorted_by(self, key: Unary[T, LT]) -> bool:
        return is_sorted(self.iterator, key)

    def is_sorted_reverse(self: Iter[LT]) -> bool:
        return is_sorted(self.iterator, reverse=True)

    def is_sorted_reverse_by(self, key: Unary[T, LT]) -> bool:
        return is_sorted(self.iterator, key, reverse=True)

    def is_sorted_strict(self: Iter[ST]) -> bool:
        return is_sorted(self.iterator, strict=True)

    def is_sorted_strict_by(self, key: Unary[T, ST]) -> bool:
        return is_sorted(self.iterator, key, strict=True)

    def is_sorted_reverse_strict(self: Iter[ST]) -> bool:
        return is_sorted(self.iterator, strict=True, reverse=True)

    def is_sorted_reverse_strict_by(self, key: Unary[T, ST]) -> bool:
        return is_sorted(self.iterator, key, strict=True, reverse=True)

    def duplicates_fast(self: Iter[Q]) -> Iter[Q]:
        return self.create(duplicates_fast(self.iterator))

    def duplicates_fast_by(self, key: Unary[T, Q]) -> Iter[T]:
        return self.create(duplicates_fast(self.iterator, key))

    def duplicates(self) -> Iter[T]:
        return self.create(duplicates(self.iterator))

    def duplicates_by(self, key: Unary[T, V]) -> Iter[T]:
        return self.create(duplicates(self.iterator, key))

    def unique_fast(self: Iter[Q]) -> Iter[Q]:
        return self.create(unique_fast(self.iterator))

    def unique_fast_by(self, key: Unary[T, Q]) -> Iter[T]:
        return self.create(unique_fast(self.iterator, key))

    def unique(self) -> Iter[T]:
        return self.create(unique(self.iterator))

    def unique_by(self, key: Unary[T, V]) -> Iter[T]:
        return self.create(unique(self.iterator, key))

    def partition(self, predicate: Predicate[T]) -> Tuple[Iter[T], Iter[T]]:
        true, false = partition(predicate, self.iterator)

        return (self.create(true), self.create(false))

    def partition_unsafe(self, predicate: Predicate[T]) -> Tuple[Iter[T], Iter[T]]:
        true, false = partition_unsafe(predicate, self.iterator)

        return (self.create(true), self.create(false))

    partition_infinite = partition_unsafe

    def copy(self) -> Iter[T]:
        iterator, result = copy(self.iterator)

        self._replace(iterator)

        return self.create(result)

    def copy_unsafe(self) -> Iter[T]:
        iterator, result = copy_unsafe(self.iterator)

        self._replace(iterator)

        return self.create(result)

    copy_infinite = copy_unsafe

    def spy(self, size: int) -> List[T]:
        result, iterator = spy(size, self.iterator)

        self._replace(iterator)

        return result

    def peek(self) -> T:
        item, iterator = peek(self.iterator)

        self._replace(iterator)

        return item

    def peek_or(self, default: V) -> Union[T, V]:
        item, iterator = peek(self.iterator, default)

        self._replace(iterator)

        return item

    def peek_or_none(self) -> Optional[T]:
        return self.peek_or(None)

    def has_next(self) -> bool:
        result, iterator = has_next(self.iterator)

        self._replace(iterator)

        return result

    def is_empty(self) -> bool:
        result, iterator = is_empty(self.iterator)

        self._replace(iterator)

        return result

    def repeat_last(self) -> Iter[T]:
        return self.create(repeat_last(self.iterator))

    def repeat_last_or(self, default: V) -> Iter[Union[T, V]]:
        return self.create(repeat_last(self.iterator, default))

    def repeat_last_or_none(self) -> Iter[Optional[T]]:
        return self.repeat_last_or(None)

    def repeat_each(self, count: int) -> Iter[T]:
        return self.create(repeat_each(self.iterator, count))

    def side_effect(self, function: Unary[T, Any]) -> Iter[T]:
        return self.create(side_effect(function, self.iterator))


iter = Iter
reversed = iter.reversed


def wrap_iter(function: Callable[PS, Iterable[T]]) -> Callable[PS, Iter[T]]:
    @wraps(function)
    def wrap(*args: PS.args, **kwargs: PS.kwargs) -> Iter[T]:
        return iter(function(*args, **kwargs))

    return wrap
