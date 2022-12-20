from abc import abstractmethod
from builtins import isinstance as is_instance
from builtins import issubclass as is_subclass
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import Protocol, TypeAlias, TypeGuard, runtime_checkable

__all__ = (
    # exceptions
    "AnyException",
    "AnyExceptionType",
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
    # async functions
    "AsyncNullary",
    "AsyncUnary",
    "AsyncBinary",
    "AsyncTernary",
    "AsyncQuaternary",
    # dynamic size
    "DynamicCallable",
    "DynamicTuple",
    "AsyncDynamicCallable",
    # sum / product
    "Sum",
    "Product",
    # predicates
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
    # checks
    "is_async_iterable",
    "is_async_iterator",
    "is_bytes",
    "is_iterable",
    "is_iterator",
    "is_string",
    "is_instance",
    "is_subclass",
)

AnyException: TypeAlias = BaseException  # any exception type
AnyExceptionType = Type[AnyException]

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
R = TypeVar("R")

DynamicCallable = Callable[..., T]
AnyCallable = DynamicCallable[Any]

AsyncDynamicCallable = Callable[..., Awaitable[T]]

F = TypeVar("F", bound=AnyCallable)
G = TypeVar("G", bound=AnyCallable)

Nullary = Callable[[], R]
Unary = Callable[[T], R]
Binary = Callable[[T, U], R]
Ternary = Callable[[T, U, V], R]
Quaternary = Callable[[T, U, V, W], R]

AsyncNullary = Nullary[Awaitable[R]]
AsyncUnary = Unary[T, Awaitable[R]]
AsyncBinary = Binary[T, U, Awaitable[R]]
AsyncTernary = Ternary[T, U, V, Awaitable[R]]
AsyncQuaternary = Quaternary[T, U, V, W, Awaitable[R]]

Predicate = Unary[T, bool]
Selectors = Iterable[bool]

AsyncPredicate = AsyncUnary[T, bool]
AsyncSelectors = AsyncIterable[bool]

AnySelectors = Union[AsyncSelectors, Selectors]

Compare = Binary[T, U, bool]
AsyncCompare = AsyncBinary[T, U, bool]

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

AnyIterable = Union[AsyncIterable[T], Iterable[T]]
AnyIterator = Union[AsyncIterator[T], Iterator[T]]


def is_async_iterable(iterable: AnyIterable[T]) -> TypeGuard[AsyncIterable[T]]:
    return is_instance(iterable, AsyncIterable)


def is_iterable(iterable: AnyIterable[T]) -> TypeGuard[Iterable[T]]:
    return is_instance(iterable, Iterable)


def is_async_iterator(iterator: AnyIterator[T]) -> TypeGuard[AsyncIterator[T]]:
    return is_instance(iterator, AsyncIterator)


def is_iterator(iterator: AnyIterator[T]) -> TypeGuard[Iterator[T]]:
    return is_instance(iterator, Iterator)


def is_string(item: Any) -> TypeGuard[str]:
    return is_instance(item, str)


def is_bytes(item: Any) -> TypeGuard[bytes]:
    return is_instance(item, bytes)


def is_error(item: Any) -> TypeGuard[AnyException]:
    return is_instance(item, AnyException)


RecursiveIterable: TypeAlias = Union[T, Iterable[Any]]
RecursiveAsyncIterable: TypeAlias = Union[T, AsyncIterable[Any]]
RecursiveAnyIterable: TypeAlias = Union[T, AnyIterable[Any]]


S = TypeVar("S", bound="Sum")


@runtime_checkable
class Sum(Protocol):
    @abstractmethod
    def __add__(self: S, __other: S) -> S:
        raise NotImplementedError


P = TypeVar("P", bound="Product")


@runtime_checkable
class Product(Protocol):
    @abstractmethod
    def __mul__(self: P, __other: P) -> P:
        raise NotImplementedError
