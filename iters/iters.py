from functools import wraps
from typing import (
    Any,
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

from iters.utils import (
    MarkerOr,
    append,
    chain,
    chain_from_iterable,
    collapse,
    compress,
    copy,
    copy_infinite,
    copy_safe,
    count,
    cycle,
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
    marker,
    nth,
    nth_or_last,
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
    zip_longest,
)

__all__ = (
    "Iter",
    "iter",
    "reversed",
    "return_iter",
    "std_iter",
    "std_reversed",
)

KT = TypeVar("KT")
VT = TypeVar("VT")

T = TypeVar("T")
U = TypeVar("U")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

Or = Union[T, Optional[U]]

std_iter = iter
std_reversed = reversed


class Iter(Generic[T]):
    @overload
    def __init__(self, iterator: Iterator[T]) -> None:  # noqa
        ...

    @overload
    def __init__(self, iterable: Iterable[T]) -> None:  # noqa
        ...

    @overload
    def __init__(self, function: Callable[[], T], sentinel: T) -> None:  # noqa
        ...

    @no_type_check
    def __init__(  # noqa
        self,
        something: Union[Iterator[T], Iterable[T], Callable[[], T]],
        sentinel: MarkerOr[T] = marker,
    ) -> None:
        if sentinel is marker:
            self._iterator: Iterator[T] = std_iter(something)

        else:
            self._iterator: Iterator[T] = std_iter(something, sentinel)

    def __iter__(self) -> "Iter[T]":
        return self

    def __next__(self) -> T:
        return next(self._iterator)  # type: ignore

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[T] at 0x{id(self):016x}>"

    @overload
    def __getitem__(self, item: int) -> T:  # noqa
        ...

    @overload
    def __getitem__(self, item: slice) -> "Iter[T]":  # noqa
        ...

    @no_type_check
    def __getitem__(self, item: Union[int, slice]) -> Union[T, "Iter[T]"]:  # noqa
        if isinstance(item, int):
            return self.nth(item)

        elif isinstance(item, slice):
            return self.slice(item.start, item.stop, item.stop)

        else:
            raise ValueError(f"Expected integer or slice, got type {type(item).__name__!r}.")

    def unwrap(self) -> Iterator[T]:
        return self._iterator  # type: ignore

    @classmethod
    def count(cls, start: T = 0, step: T = 1) -> "Iter[T]":  # type: ignore
        return cls(cast(Iterator[T], count(start, step)))

    @classmethod
    def repeat(cls, to_repeat: T, times: Optional[int] = None) -> "Iter[T]":
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
    def iterate(cls, function: Callable[[T], T], value: T) -> "Iter[T]":
        return cls(iterate(function, value))

    def iter(self) -> "Iter[T]":
        return self

    def next(self) -> T:
        return next(self._iterator)  # type: ignore

    def next_or(self, default: Optional[T]) -> Optional[T]:
        return next(self._iterator, default)  # type: ignore

    def all(self) -> bool:
        return all(self._iterator)  # type: ignore

    def any(self) -> bool:
        return any(self._iterator)  # type: ignore

    def append(self, item: T) -> "Iter[T]":
        return self.__class__(append(self._iterator, item))  # type: ignore

    def prepend(self, item: T) -> "Iter[T]":
        return self.__class__(prepend(self._iterator, item))  # type: ignore

    def chain(self, *iterables: Iterable[T]) -> "Iter[T]":
        return self.__class__(chain(self._iterator, *iterables))  # type: ignore

    def chain_with(self, iterables: Iterable[Iterable[T]]) -> "Iter[T]":
        return self.__class__(
            chain(self._iterator, chain_from_iterable(iterables))  # type: ignore
        )

    def collapse(
        self, base_type: Optional[Type[Any]] = None, levels: Optional[int] = None
    ) -> "Iter[T]":
        return self.__class__(
            collapse(self._iterator, base_type=base_type, levels=levels)  # type: ignore
        )

    def reverse(self) -> "Iter[T]":
        return self.__class__(std_reversed(tuple(self._iterator)))  # type: ignore

    def slice(self, *slice_args) -> "Iter[T]":
        return self.__class__(iter_slice(self._iterator, *slice_args))  # type: ignore

    def exhaust(self) -> None:
        exhaust(self._iterator)  # type: ignore

    def for_each(self, function: Callable[[T], None]) -> None:
        self.map(function).exhaust()

    def collect(self, function: Callable[[Iterator[T]], Iterable[T]]) -> Iterable[T]:
        return function(self._iterator)  # type: ignore

    def dict(self) -> Dict[KT, VT]:
        return dict(self._iterator)  # type: ignore

    def list(self) -> List[T]:
        return list(self._iterator)  # type: ignore

    def set(self) -> Set[T]:
        return set(self._iterator)  # type: ignore

    def tuple(self) -> Tuple[T, ...]:
        return tuple(self._iterator)  # type: ignore

    def compress(self, selectors: Iterable[U]) -> "Iter[T]":
        return self.__class__(compress(self._iterator, selectors))  # type: ignore

    def copy(self) -> "Iter[T]":
        for_self, to_return = copy(self._iterator)  # type: ignore

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

    def drop_while(self, predicate: Callable[[T], bool]) -> "Iter[T]":
        return self.__class__(drop_while(predicate, self._iterator))

    skip_while = drop_while

    def take(self, n: int) -> "Iter[T]":
        return self.__class__(take(self._iterator, n))

    def take_while(self, predicate: Callable[[T], bool]) -> "Iter[T]":
        return self.__class__(take_while(predicate, self._iterator))

    def step_by(self, step: int) -> "Iter[T]":
        return self.__class__(step_by(self._iterator, step))

    def enumerate(self, start: int = 0) -> "Iter[Tuple[int, T]]":
        return self.__class__(enumerate(self._iterator, start))

    def filter(self, predicate: Callable[[T], bool]) -> "Iter[T]":
        return self.__class__(filter(predicate, self._iterator))

    def filter_false(self, predicate: Callable[[T], bool]) -> "Iter[T]":
        return self.__class__(filter_false(predicate, self._iterator))

    def find_all(self, predicate: Callable[[T], bool]) -> "Iter[T]":
        return self.filter(predicate)

    def find(self, predicate: Callable[[T], bool], default: Optional[T] = None) -> Optional[T]:
        return self.find_all(predicate).next_or(default)

    def first(self) -> T:
        return first(self._iterator)

    def first_or(self, default: Optional[T]) -> Optional[T]:
        return first(self._iterator, default)

    def fold(self, function: Callable[[U, Union[T, U]], U], initial: U) -> U:
        return fold(self._iterator, function, initial)

    def reduce(self, function: Callable[[U, Union[T, U]], U]) -> U:
        return reduce(function, self._iterator)

    def max(self, *, key: Optional[Callable[[T], U]] = None) -> T:
        if key is None:
            return max(self._iterator)

        return max(self._iterator, key=key)

    def min(self, *, key: Optional[Callable[[T], U]] = None) -> T:
        if key is None:
            return min(self._iterator)

        return min(self._iterator, key=key)

    def max_or(
        self, default: Optional[T], *, key: Optional[Callable[[Any], Any]] = None
    ) -> Optional[T]:
        if key is None:
            return max(self._iterator, default=default)

        return max(self._iterator, key=key, default=default)

    def min_or(
        self, default: Optional[T], *, key: Optional[Callable[[Any], Any]] = None
    ) -> Optional[T]:
        if key is None:
            return min(self._iterator, default=default)

        return min(self._iterator, key=key, default=default)

    def sum(self, start: MarkerOr[T] = marker) -> U:
        return sum(self._iterator, start)

    def product(self, start: MarkerOr[T] = marker) -> U:
        return product(self._iterator, start)

    def get_all(self, **attrs) -> "Iter[T]":
        return self.__class__(get(self._iterator, **attrs))

    def get(self, *, default: Optional[T] = None, **attrs) -> Optional[T]:
        return self.get_all(**attrs).next_or(default)

    def last(self) -> T:
        return last(self._iterator)

    def last_or(self, default: Optional[T]) -> Optional[T]:
        return last(self._iterator, default)

    def flatten(self) -> "Iter[T]":
        return self.__class__(flatten(self._iterator))

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

    def nth(self, n: int) -> T:
        return nth(self._iterator, n)

    def nth_or(self, n: int, default: T) -> T:
        return nth(self._iterator, n, default)

    def nth_or_last(self, n: int) -> T:
        return nth_or_last(self._iterator, n)

    def length(self) -> int:
        return iter_len(self._iterator)

    def map(self, function: Callable[[T], U]) -> "Iter[U]":
        return self.__class__(map(function, self._iterator))

    def star_map(self, function: Callable[..., U]) -> "Iter[U]":
        return self.__class__(star_map(function, self._iterator))

    def partition(self, predicate: Callable[[T], bool]) -> "Tuple[Iter[T], Iter[T]]":
        with_true, with_false = partition(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_infinite(self, predicate: Callable[[T], bool]) -> "Tuple[Iter[T], Iter[T]]":
        with_true, with_false = partition_infinite(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    def partition_safe(self, predicate: Callable[[T], bool]) -> "Tuple[Iter[T], Iter[T]]":
        with_true, with_false = partition_safe(self._iterator, predicate)

        return self.__class__(with_true), self.__class__(with_false)

    @overload
    def zip(self, __iter_1: Iterable[T1]) -> "Iter[Tuple[T, T1]]":  # noqa
        ...

    @overload
    def zip(  # noqa
        self, __iter_1: Iterable[T1], __iter_2: Iterable[T2]
    ) -> "Iter[Tuple[T, T1, T2]]":
        ...

    @overload
    def zip(  # noqa
        self, __iter_1: Iterable[T1], __iter_2: Iterable[T2], __iter_3: Iterable[T3]
    ) -> "Iter[Tuple[T, T1, T2, T3]]":
        ...

    @overload
    def zip(  # noqa
        self,
        __iter_1: Iterable[T1],
        __iter_2: Iterable[T2],
        __iter_3: Iterable[T3],
        __iter_4: Iterable[T4],
    ) -> "Iter[Tuple[T, T1, T2, T3, T4]]":
        ...

    @overload
    def zip(  # noqa
        self,
        __iter_1: Iterable[T1],
        __iter_2: Iterable[T2],
        __iter_3: Iterable[T3],
        __iter_4: Iterable[T4],
        __iter_5: Iterable[T5],
    ) -> "Iter[Tuple[T, T1, T2, T3, T4, T5]]":
        ...

    @overload
    def zip(  # noqa
        self,
        __iter_1: Iterable[Any],
        __iter_2: Iterable[Any],
        __iter_3: Iterable[Any],
        __iter_4: Iterable[Any],
        __iter_5: Iterable[Any],
        __iter_6: Iterable[Any],
        *iterables: Iterable[Any],
    ) -> "Iter[Tuple[Any, ...]]":
        ...

    @no_type_check
    def zip(self, *iterables: Iterable[Any]) -> "Iter[Tuple[Any, ...]]":  # noqa
        return self.__class__(zip(self._iterator, *iterables))

    @overload
    def zip_longest(  # noqa
        self, __iter_1: Iterable[T1], *, fill: Optional[U] = None
    ) -> "Iter[Tuple[Or[T, U], Or[T1, U]]]":
        ...

    @overload
    def zip_longest(  # noqa
        self, __iter_1: Iterable[T1], __iter_2: Iterable[T2], *, fill: Optional[U] = None
    ) -> "Iter[Tuple[Or[T, U], Or[T1, U], Or[T2, U]]]":
        ...

    @overload
    def zip_longest(  # noqa
        self,
        __iter_1: Iterable[T1],
        __iter_2: Iterable[T2],
        __iter_3: Iterable[T3],
        *,
        fill: Optional[U] = None,
    ) -> "Iter[Tuple[Or[T, U], Or[T1, U], Or[T2, U], Or[T3, U]]]":
        ...

    @overload
    def zip_longest(  # noqa
        self,
        __iter_1: Iterable[T1],
        __iter_2: Iterable[T2],
        __iter_3: Iterable[T3],
        __iter_4: Iterable[T4],
        *,
        fill: Optional[U] = None,
    ) -> "Iter[Tuple[Or[T, U], Or[T1, U], Or[T2, U], Or[T3, U], Or[T4, U]]]":
        ...

    @overload
    def zip_longest(  # noqa
        self,
        __iter_1: Iterable[T1],
        __iter_2: Iterable[T2],
        __iter_3: Iterable[T3],
        __iter_4: Iterable[T4],
        __iter_5: Iterable[T5],
        *,
        fill: Optional[U] = None,
    ) -> "Iter[Tuple[Or[T, U], Or[T1, U], Or[T2, U], Or[T3, U], Or[T4, U], Or[T5, U]]]":
        ...

    @overload
    def zip_longest(  # noqa
        self,
        __iter_1: Iterable[Any],
        __iter_2: Iterable[Any],
        __iter_3: Iterable[Any],
        __iter_4: Iterable[Any],
        __iter_5: Iterable[Any],
        __iter_6: Iterable[Any],
        *iterables: Iterable[Any],
        fill: Optional[U] = None,
    ) -> "Iter[Tuple[Or[Any, U], ...]]":
        ...

    @no_type_check
    def zip_longest(  # noqa
        self, *iterables: Iterable[Any], fill: Optional[U] = None
    ) -> "Iter[Tuple[Or[Any, U], ...]]":
        return self.__class__(zip_longest(self._iterator, *iterables, fillvalue=fill))

    def side_effect(
        self,
        function: Callable[[T], None],
        before: Optional[Callable[[], None]] = None,
        after: Optional[Callable[[], None]] = None,
    ) -> "Iter[T]":
        return self.__class__(side_effect(self._iterator, function, before, after))

    def inspect(self, string: str) -> "Iter[T]":
        def print_and_string(item: T) -> None:
            print(string, item)

        return self.side_effect(print_and_string)

    def inspect_format(self, format_string: str) -> "Iter[T]":
        def print_format(item: T) -> None:
            print(format_string.format(item))

        return self.side_effect(print_format)


iter = Iter
reversed = iter.reversed


def return_iter(function: Callable[..., Iterable[T]]) -> Callable[..., Iter[T]]:
    @wraps(function)
    def wrapper(*args, **kwargs) -> Iter[T]:
        return iter(function(*args, **kwargs))

    return wrapper
