from functools import wraps
from typing import (
    Any,
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

from iters.types import MarkerOr, Order, marker
from iters.utils import (
    append,
    at,
    at_or_last,
    chain,
    chain_from_iterable,
    collapse,
    compress,
    copy,
    copy_infinite,
    copy_safe,
    count,
    cycle,
    distinct,
    drop,
    drop_while,
    exhaust,
    filter_false,
    first,
    flatten,
    fold,
    get,
    group,
    group_longest,
    iter_chunk,
    iter_len,
    iter_slice,
    iterate,
    last,
    list_chunk,
    partition,
    partition_infinite,
    partition_safe,
    prepend,
    product,
    reduce,
    repeat,
    side_effect,
    star_map,
    step_by,
    sum,
    take,
    take_while,
    tuple_chunk,
    with_iter,
    zip,
    zip_longest,
)

__all__ = (
    # iterator class
    "Iter",
    # convenient functions to get an iterator
    "iter",
    "reversed",
    # since we are shadowing standard functions, export them as <std>
    "std_iter",
    "std_reversed",
    # decorator to wrap return value of the function into an iterator
    "return_iter",
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

std_iter = iter
std_reversed = reversed


class Iter(Iterator[T]):
    _iterator: Iterator[T]

    @overload
    def __init__(self, iterator: Iterator[T]) -> None:
        ...

    @overload
    def __init__(self, iterable: Iterable[T]) -> None:
        ...

    @overload
    def __init__(self, function: Callable[[], T], sentinel: T) -> None:
        ...

    @no_type_check
    def __init__(
        self,
        something: Union[Iterator[T], Iterable[T], Callable[[], T]],
        sentinel: MarkerOr[T] = marker,
    ) -> None:
        if sentinel is marker:
            self._iterator = std_iter(something)

        else:
            self._iterator = std_iter(something, sentinel)

    def __iter__(self) -> "Iter[T]":
        return self

    def __next__(self) -> T:
        return next(self._iterator)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}<T> at 0x{id(self):016x}>"

    @overload
    def __getitem__(self, item: int) -> T:
        ...

    @overload
    def __getitem__(self, item: slice) -> "Iter[T]":
        ...

    @no_type_check
    def __getitem__(self, item: Union[int, slice]) -> Union[T, "Iter[T]"]:
        if isinstance(item, int):
            return self.at(item)

        elif isinstance(item, slice):
            return self.slice(item.start, item.stop, item.stop)

        else:
            raise ValueError(f"Expected integer or slice, got type {type(item).__name__!r}.")

    def unwrap(self) -> Iterator[T]:
        return self._iterator

    @classmethod
    def count(cls, start: N = 0, step: N = 1) -> "Iter[N]":
        return cls(count(start, step))

    @classmethod
    def repeat(cls, to_repeat: V, times: Optional[int] = None) -> "Iter[V]":
        if times is None:
            return cls(repeat(to_repeat))

        else:
            return cls(repeat(to_repeat, times))

    @classmethod
    def reversed(cls, iterable: Reversible[T]) -> "Iter[T]":
        return cls(std_reversed(iterable))

    @classmethod
    def with_iter(cls, context_manager: ContextManager[Iterable[T]]) -> "Iter[T]":
        return cls(with_iter(context_manager))

    @classmethod
    def iterate(cls, function: Callable[[V], V], value: V) -> "Iter[V]":
        return cls(iterate(function, value))

    def iter(self) -> "Iter[T]":
        return self

    def next(self) -> T:
        return next(self._iterator)

    def next_or(self, default: U) -> Union[T, U]:
        return next(self._iterator, default)

    def next_or_null(self) -> Optional[T]:
        return self.next_or(None)

    def all(self) -> bool:
        return all(self._iterator)

    def any(self) -> bool:
        return any(self._iterator)

    def append(self: "Iter[V]", item: V) -> "Iter[V]":
        return self.__class__(append(self._iterator, item))

    def prepend(self: "Iter[V]", item: V) -> "Iter[V]":
        return self.__class__(prepend(self._iterator, item))

    def chain(self, *iterables: Iterable[T]) -> "Iter[T]":
        return self.__class__(chain(self._iterator, *iterables))

    def chain_with(self, iterables: Iterable[Iterable[T]]) -> "Iter[T]":
        return self.__class__(chain(self._iterator, chain_from_iterable(iterables)))

    def collapse(
        self, base_type: Optional[Type[Any]] = None, levels: Optional[int] = None
    ) -> "Iter[T]":
        return self.__class__(collapse(self._iterator, base_type=base_type, levels=levels))

    def reverse(self) -> "Iter[T]":
        return self.__class__(std_reversed(tuple(self._iterator)))

    def slice(self, *slice_args: int) -> "Iter[T]":
        return self.__class__(iter_slice(self._iterator, *slice_args))

    def exhaust(self) -> None:
        exhaust(self._iterator)

    def for_each(self, function: Callable[[T], None]) -> None:
        self.map(function).exhaust()

    def join(self: "Iter[str]", delim: str) -> str:
        return delim.join(self._iterator)

    def collect(self, function: Callable[[Iterator[T]], Iterable[T]]) -> Iterable[T]:
        return function(self._iterator)

    def distinct(self, key: Optional[Callable[[T], U]] = None) -> "Iter[T]":
        return self.__class__(distinct(self._iterator, key))

    unique = distinct

    def dict(self: "Iter[Tuple[KT, VT]]") -> Dict[KT, VT]:
        return dict(self._iterator)

    def list(self) -> List[T]:
        return list(self._iterator)

    def set(self) -> Set[T]:
        return set(self._iterator)

    def tuple(self) -> Tuple[T, ...]:
        return tuple(self._iterator)

    def compress(self, selectors: Iterable[U]) -> "Iter[T]":
        return self.__class__(compress(self._iterator, selectors))

    def copy(self) -> "Iter[T]":
        for_self, to_return = copy(self._iterator)

        self._iterator = for_self

        return self.__class__(to_return)

    def copy_infinite(self) -> "Iter[T]":
        for_self, to_return = copy_infinite(self._iterator)

        self._iterator = for_self

        return self.__class__(to_return)

    def copy_safe(self) -> "Iter[T]":
        for_self, to_return = copy_safe(self._iterator)

        self._iterator = for_self

        return self.__class__(to_return)

    def cycle(self) -> "Iter[T]":
        return self.__class__(cycle(self._iterator))

    def drop(self, n: int) -> "Iter[T]":
        return self.__class__(drop(self._iterator, n))

    skip = drop

    def drop_while(self, predicate: Callable[[T], Any]) -> "Iter[T]":
        return self.__class__(drop_while(predicate, self._iterator))

    skip_while = drop_while

    def take(self, n: int) -> "Iter[T]":
        return self.__class__(take(self._iterator, n))

    def take_while(self, predicate: Callable[[T], Any]) -> "Iter[T]":
        return self.__class__(take_while(predicate, self._iterator))

    def step_by(self, step: int) -> "Iter[T]":
        return self.__class__(step_by(self._iterator, step))

    def enumerate(self, start: int = 0) -> "Iter[Tuple[int, T]]":
        return self.__class__(enumerate(self._iterator, start))

    def filter(self, predicate: Callable[[T], Any]) -> "Iter[T]":
        return self.__class__(filter(predicate, self._iterator))

    def filter_false(self, predicate: Callable[[T], Any]) -> "Iter[T]":
        return self.__class__(filter_false(predicate, self._iterator))

    def find_all(self, predicate: Callable[[T], Any]) -> "Iter[T]":
        return self.filter(predicate)

    def find(self, predicate: Callable[[T], Any]) -> T:
        return self.find_all(predicate).next()

    def find_or(self, predicate: Callable[[T], Any], default: U) -> Union[T, U]:
        return self.find_all(predicate).next_or(default)

    def find_or_null(self, predicate: Callable[[T], Any]) -> Optional[T]:
        return self.find_all(predicate).next_or_null()

    def first(self) -> T:
        return first(self._iterator)

    def first_or(self, default: U) -> Union[T, U]:
        return first(self._iterator, default)

    def first_or_null(self) -> Optional[T]:
        return self.first_or(None)

    def fold(self, function: Callable[[U, T], U], initial: U) -> U:
        return fold(self._iterator, function, initial)

    def reduce(self, function: Callable[[T, T], T]) -> T:
        return reduce(function, self._iterator)

    @overload
    def max(self: "Iter[OrderT]") -> OrderT:
        ...

    @overload
    def max(self: "Iter[T]", *, key: Callable[[T], OrderT]) -> T:
        ...

    def max(self: "Iter[Any]", *, key: Optional[Callable[[Any], Any]] = None) -> Any:
        if key is None:
            return max(self._iterator)

        return max(self._iterator, key=key)

    @overload
    def min(self: "Iter[OrderT]") -> OrderT:
        ...

    @overload
    def min(self: "Iter[T]", *, key: Callable[[T], OrderT]) -> T:
        ...

    def min(self: "Iter[Any]", *, key: Optional[Callable[[Any], Any]] = None) -> Any:
        if key is None:
            return min(self._iterator)

        return min(self._iterator, key=key)

    @overload
    def max_or(self: "Iter[OrderT]", default: U) -> Union[OrderT, U]:
        ...

    @overload
    def max_or(
        self: "Iter[T]", default: U, *, key: Callable[[T], OrderT]
    ) -> Union[T, U]:
        ...

    def max_or(
        self: "Iter[Any]", default: Any, *, key: Optional[Callable[[Any], Any]] = None
    ) -> Optional[Any]:
        if key is None:
            return max(self._iterator, default=default)

        return max(self._iterator, key=key, default=default)

    @overload
    def max_or_null(self: "Iter[OrderT]") -> Optional[OrderT]:
        ...

    @overload
    def max_or_null(
        self: "Iter[T]", *, key: Callable[[T], OrderT]
    ) -> Optional[T]:
        ...

    def max_or_null(
        self: "Iter[Any]", key: Optional[Callable[[Any], Any]] = None
    ) -> Optional[Any]:
        if key is None:
            return self.max_or(None)

        return self.max_or(None, key=key)

    @overload
    def min_or(self: "Iter[OrderT]", default: U) -> Union[OrderT, U]:
        ...

    @overload
    def min_or(
        self: "Iter[T]", default: U, *, key: Callable[[T], OrderT]
    ) -> Union[T, U]:
        ...

    def min_or(
        self: "Iter[Any]",
        default: Any,
        *,
        key: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        if key is None:
            return min(self._iterator, default=default)

        return min(self._iterator, key=key, default=default)

    @overload
    def min_or_null(self: "Iter[OrderT]") -> Optional[OrderT]:
        ...

    @overload
    def min_or_null(
        self: "Iter[T]", *, key: Callable[[T], OrderT]
    ) -> Optional[T]:
        ...

    def min_or_null(
        self: "Iter[Any]", key: Optional[Callable[[Any], Any]] = None
    ) -> Optional[Any]:
        if key is None:
            return self.min_or(None)

        return self.min_or(None, key=key)

    def sum(self, start: MarkerOr[T] = marker) -> T:
        return sum(self._iterator, start)

    def product(self, start: MarkerOr[T] = marker) -> T:
        return product(self._iterator, start)

    def get_all(self, **attrs: Any) -> "Iter[T]":
        return self.__class__(get(self._iterator, **attrs))

    def get(self, **attrs: Any) -> T:
        return self.get_all(**attrs).next()

    def get_or(self, *, default: U, **attrs: Any) -> Union[T, U]:
        return self.get_all(**attrs).next_or(default)

    def get_or_null(self, **attrs: Any) -> Optional[T]:
        return self.get_all(**attrs).next_or_null()

    def last(self) -> T:
        return last(self._iterator)

    def last_or(self, default: U) -> Union[T, U]:
        return last(self._iterator, default)

    def last_or_null(self) -> Optional[T]:
        return self.last_or(None)

    def flatten(self: "Iter[Iterable[V]]") -> "Iter[V]":
        return self.__class__(flatten(self._iterator))  # type: ignore

    def group(self, n: int) -> "Iter[Tuple[T, ...]]":
        return self.__class__(group(self._iterator, n))

    def group_longest(self, n: int, fill: Optional[T] = None) -> "Iter[Tuple[Optional[T], ...]]":
        return self.__class__(group_longest(self._iterator, n, fill))

    def iter_chunk(self, n: int) -> "Iter[Iter[T]]":
        return self.__class__(self.__class__(chunk) for chunk in iter_chunk(self._iterator, n))

    def list_chunk(self, n: int) -> "Iter[List[T]]":
        return self.__class__(list_chunk(self._iterator, n))

    def tuple_chunk(self, n: int) -> "Iter[Tuple[T, ...]]":
        return self.__class__(tuple_chunk(self._iterator, n))

    def at(self, n: int) -> T:
        return at(self._iterator, n)

    def at_or(self, n: int, default: U) -> Union[T, U]:
        return at(self._iterator, n, default)

    def at_or_null(self, n: int) -> Optional[T]:
        return self.at_or(n, None)

    def at_or_last(self, n: int) -> T:
        return at_or_last(self._iterator, n)

    def length(self) -> int:
        return iter_len(self._iterator)

    def map(self, function: Callable[[T], U]) -> "Iter[U]":
        return self.__class__(map(function, self._iterator))

    def star_map(self: "Iter[Iterable[Any]]", function: Callable[..., V]) -> "Iter[V]":
        return self.__class__(star_map(function, self._iterator))  # type: ignore

    def partition(self, predicate: Callable[[T], Any]) -> "Tuple[Iter[T], Iter[T]]":
        with_true, with_false = partition(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_infinite(self, predicate: Callable[[T], Any]) -> "Tuple[Iter[T], Iter[T]]":
        with_true, with_false = partition_infinite(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_safe(self, predicate: Callable[[T], Any]) -> "Tuple[Iter[T], Iter[T]]":
        with_true, with_false = partition_safe(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    @overload
    def zip(self, __iterable_1: Iterable[T1]) -> "Iter[Tuple[T, T1]]":
        ...

    @overload
    def zip(
        self, __iterable_1: Iterable[T1], __iterable_2: Iterable[T2]
    ) -> "Iter[Tuple[T, T1, T2]]":
        ...

    @overload
    def zip(
        self, __iterable_1: Iterable[T1], __iterable_2: Iterable[T2], __iterable_3: Iterable[T3]
    ) -> "Iter[Tuple[T, T1, T2, T3]]":
        ...

    @overload
    def zip(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
        __iterable_4: Iterable[T4],
    ) -> "Iter[Tuple[T, T1, T2, T3, T4]]":
        ...

    @overload
    def zip(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
        __iterable_4: Iterable[T4],
        __iterable_5: Iterable[T5],
    ) -> "Iter[Tuple[T, T1, T2, T3, T4, T5]]":
        ...

    @overload
    def zip(
        self,
        __iterable_1: Iterable[Any],
        __iterable_2: Iterable[Any],
        __iterable_3: Iterable[Any],
        __iterable_4: Iterable[Any],
        __iterable_5: Iterable[Any],
        __iterable_6: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> "Iter[Tuple[Any, ...]]":
        ...

    @no_type_check
    def zip(self, *iterables: Iterable[Any]) -> "Iter[Tuple[Any, ...]]":
        return self.__class__(zip(self._iterator, *iterables))

    @overload
    def zip_longest(
        self, __iterable_1: Iterable[T1]
    ) -> "Iter[Tuple[Optional[T], Optional[T1]]]":
        ...

    @overload
    def zip_longest(
        self, __iterable_1: Iterable[T1], __iterable_2: Iterable[T2]
    ) -> "Iter[Tuple[Optional[T], Optional[T1], Optional[T2]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
    ) -> "Iter[Tuple[Optional[T], Optional[T1], Optional[T2], Optional[T3]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
        __iterable_4: Iterable[T4],
    ) -> "Iter[Tuple[Optional[T], Optional[T1], Optional[T2], Optional[T3], Optional[T4]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
        __iterable_4: Iterable[T4],
        __iterable_5: Iterable[T5],
    ) -> "Iter[Tuple[Optional[T], Optional[T1], Optional[T2], Optional[T3], Optional[T4], Optional[T5]]]":  # noqa
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[Any],
        __iterable_2: Iterable[Any],
        __iterable_3: Iterable[Any],
        __iterable_4: Iterable[Any],
        __iterable_5: Iterable[Any],
        __iterable_6: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> "Iter[Tuple[Optional[Any], ...]]":
        ...

    @overload
    def zip_longest(
        self, __iterable_1: Iterable[T1], *, fill: U
    ) -> "Iter[Tuple[Union[T, U], Union[T1, U]]]":
        ...

    @overload
    def zip_longest(
        self, __iterable_1: Iterable[T1], __iterable_2: Iterable[T2], *, fill: U
    ) -> "Iter[Tuple[Union[T, U], Union[T1, U], Union[T2, U]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
        *,
        fill: U
    ) -> "Iter[Tuple[Union[T, U], Union[T1, U], Union[T2, U], Union[T3, U]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
        __iterable_4: Iterable[T4],
        *,
        fill: U
    ) -> "Iter[Tuple[Union[T, U], Union[T1, U], Union[T2, U], Union[T3, U], Union[T4, U]]]":
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[T1],
        __iterable_2: Iterable[T2],
        __iterable_3: Iterable[T3],
        __iterable_4: Iterable[T4],
        __iterable_5: Iterable[T5],
        *,
        fill: U
    ) -> "Iter[Tuple[Union[T, U], Union[T1, U], Union[T2, U], Union[T3, U], Union[T4, U], Union[T5, U]]]":  # noqa
        ...

    @overload
    def zip_longest(
        self,
        __iterable_1: Iterable[Any],
        __iterable_2: Iterable[Any],
        __iterable_3: Iterable[Any],
        __iterable_4: Iterable[Any],
        __iterable_5: Iterable[Any],
        __iterable_6: Iterable[Any],
        *iterables: Iterable[Any],
        fill: U,
    ) -> "Iter[Tuple[Union[Any, U], ...]]":
        ...

    @no_type_check
    def zip_longest(
        self, *iterables: Iterable[Any], fill: Optional[U] = None
    ) -> "Iter[Tuple[Union[Optional[Any], U], ...]]":
        return self.__class__(zip_longest(self._iterator, *iterables, fill=fill))

    def side_effect(
        self,
        function: Callable[[T], None],
        before: Optional[Callable[[], None]] = None,
        after: Optional[Callable[[], None]] = None,
    ) -> "Iter[T]":
        return self.__class__(side_effect(self._iterator, function, before, after))

    def inspect(self, string: str) -> "Iter[T]":
        def print_and_string(item: Any) -> None:
            print(string, item)

        return self.side_effect(print_and_string)

    def inspect_format(self, format_string: str) -> "Iter[T]":
        def print_format(item: Any) -> None:
            print(format_string.format(item))

        return self.side_effect(print_format)


iter = Iter
reversed = iter.reversed


def return_iter(function: Callable[..., Iterable[T]]) -> Callable[..., Iter[T]]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Iter[T]:
        return iter(function(*args, **kwargs))

    return wrapper
