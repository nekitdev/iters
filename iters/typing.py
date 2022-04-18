from abc import abstractmethod
from builtins import isinstance as is_instance
from inspect import isawaitable as standard_is_awaitable
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Protocol, Self, TypeAlias, TypeGuard, runtime_checkable

__all__ = (
    # exceptions
    "AnyException",
    # tuples
    "EmptyTuple",
    "Tuple1",
    "Tuple2",
    "Tuple3",
    "Tuple4",
    "Tuple5",
    "Tuple6",
    "Tuple7",
    "Tuple8",
    # functions
    "Nullary",
    "Unary",
    "Binary",
    "Ternary",
    "Quaternary",
    # dynamic size
    "DynamicCallable",
    "DynamicTuple",
    # ordering
    "Less",
    "Greater",
    "LessOrEqual",
    "GreaterOrEqual",
    # combined
    "FullyOrdered",
    "StrictOrdered",
    "LenientOrdered",
    # either required
    "EitherFullyOrdered",
    "EitherStrictOrdered",
    "EitherLenientOrdered",
    # sum / product
    "Sum",
    "Product",
    # decorators and predicates
    "Decorator",
    "Predicate",
    "AsyncPredicate",
    # comparing
    "Compare",
    "AsyncCompare",
    # selectors
    "Selectors",
    "AsyncSelectors",
    "AnySelectors",
    # recursive (?)
    "RecursiveIterable",
    "RecursiveAsyncIterable",
    "RecursiveAnyIterable",
    # unions
    "AnyIterable",
    "AnyIterator",
    # "MaybeAnyIterable",
    # "MaybeAnyIterator",
    "MaybeAwaitable",
    # checks
    "is_awaitable",
    "is_async_iterable",
    "is_async_iterator",
    "is_bytes",
    "is_iterable",
    "is_iterator",
    "is_string",
)

AnyException: TypeAlias = BaseException  # any exception type

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
R = TypeVar("R")

F = TypeVar("F", bound=Callable)  # type: ignore
G = TypeVar("G", bound=Callable)  # type: ignore

Nullary = Callable[[], R]
Unary = Callable[[T], R]
Binary = Callable[[T, U], R]
Ternary = Callable[[T, U, V], R]
Quaternary = Callable[[T, U, V, W], R]

Decorator = Unary[F, G]  # D(F) = G (by definition)

MaybeBool = Union[bool, Any]

Predicate = Unary[T, MaybeBool]
Selectors = Iterable[MaybeBool]

AsyncPredicate = Unary[T, Awaitable[MaybeBool]]
AsyncSelectors = AsyncIterable[MaybeBool]

AnySelectors = Union[AsyncSelectors, Selectors]

Compare = Binary[T, U, MaybeBool]
AsyncCompare = Binary[T, U, Awaitable[MaybeBool]]

EmptyTuple = Tuple[()]

Tuple1 = Tuple[T]
Tuple2 = Tuple[T, T]
Tuple3 = Tuple[T, T, T]
Tuple4 = Tuple[T, T, T, T]
Tuple5 = Tuple[T, T, T, T, T]
Tuple6 = Tuple[T, T, T, T, T, T]
Tuple7 = Tuple[T, T, T, T, T, T, T]
Tuple8 = Tuple[T, T, T, T, T, T, T, T]

DynamicTuple = Tuple[T, ...]

DynamicCallable = Callable[..., T]

AnyIterable = Union[AsyncIterable[T], Iterable[T]]
AnyIterator = Union[AsyncIterator[T], Iterator[T]]

# MaybeAnyIterable = Union[AnyIterable[T], T]
# MaybeAnyIterator = Union[AnyIterator[T], T]
MaybeAwaitable = Union[Awaitable[T], T]


def is_async_iterable(iterable: AnyIterable[T]) -> TypeGuard[AsyncIterable[T]]:
    return is_instance(iterable, AsyncIterable)


def is_iterable(iterable: AnyIterable[T]) -> TypeGuard[Iterable[T]]:
    return is_instance(iterable, Iterable)


def is_async_iterator(iterator: AnyIterator[T]) -> TypeGuard[AsyncIterator[T]]:
    return is_instance(iterator, AsyncIterator)


def is_iterator(iterator: AnyIterator[T]) -> TypeGuard[Iterator[T]]:
    return is_instance(iterator, Iterator)


def is_awaitable(maybe_awaitable: MaybeAwaitable[T]) -> TypeGuard[Awaitable[T]]:
    return standard_is_awaitable(maybe_awaitable)


def is_string(item: Any) -> TypeGuard[str]:
    return isinstance(item, str)


def is_bytes(item: Any) -> TypeGuard[bytes]:
    return isinstance(item, bytes)


# XXX: perhaps even `Any` could work here...

RecursiveIterable: TypeAlias = Union[T, Iterable["RecursiveIterable[T]"]]
RecursiveAsyncIterable: TypeAlias = Union[T, AsyncIterable["RecursiveAsyncIterable[T]"]]
RecursiveAnyIterable: TypeAlias = Union[T, AnyIterable["RecursiveAnyIterable[T]"]]

# we could define the following protocols to be generic,
# but we are ultimately using them for T: Ordered[T] bounds,
# and there is no such thing as T: Ordered[Self] as of now,
# which would be required to put the bound on T


@runtime_checkable
class Less(Protocol):
    @abstractmethod
    def __lt__(self, __other: Self) -> MaybeBool:
        raise NotImplementedError


@runtime_checkable
class Greater(Protocol):
    @abstractmethod
    def __gt__(self, __other: Self) -> MaybeBool:
        raise NotImplementedError


@runtime_checkable
class StrictOrdered(Less, Greater, Protocol):
    pass


EitherStrictOrdered = Union[Less, Greater]


@runtime_checkable
class LessOrEqual(Protocol):
    @abstractmethod
    def __le__(self, __other: Self) -> MaybeBool:
        raise NotImplementedError


@runtime_checkable
class GreaterOrEqual(Protocol):
    @abstractmethod
    def __ge__(self, __other: Self) -> MaybeBool:
        raise NotImplementedError


@runtime_checkable
class LenientOrdered(LessOrEqual, GreaterOrEqual, Protocol):
    pass


EitherLenientOrdered = Union[LessOrEqual, GreaterOrEqual]


@runtime_checkable
class FullyOrdered(StrictOrdered, LenientOrdered, Protocol):
    pass


EitherFullyOrdered = Union[EitherStrictOrdered, EitherLenientOrdered]


@runtime_checkable
class Sum(Protocol):
    @abstractmethod
    def __add__(self, __other: Self) -> Self:
        raise NotImplementedError


@runtime_checkable
class Product(Protocol):
    @abstractmethod
    def __mul__(self, __other: Self) -> Self:
        raise NotImplementedError
