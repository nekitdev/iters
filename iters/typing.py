from __future__ import annotations

from abc import abstractmethod as required
from builtins import isinstance as is_instance
from builtins import issubclass as is_subclass
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Iterable,
    Iterator,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

from funcs.typing import Unary
from typing_extensions import Protocol, TypeGuard, runtime_checkable

__all__ = (
    # for each
    "ForEach",
    # validators
    "Validate",
    # pairs
    "Pair",
    # sum / product
    "Sum",
    "Product",
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
    "is_iterable",
    "is_async_iterator",
    "is_iterator",
    "is_bytes",
    "is_string",
    "is_slice",
    "is_sized",
    "is_instance",
    "is_subclass",
)

T = TypeVar("T")

ForEach = Unary[T, None]
Validate = Unary[T, None]

Pair = Tuple[T, T]

AnyIterable = Union[AsyncIterable[T], Iterable[T]]
AnyIterator = Union[AsyncIterator[T], Iterator[T]]

Selectors = Iterable[bool]
AsyncSelectors = AsyncIterable[bool]
AnySelectors = AnyIterable[bool]


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


def is_slice(item: Any) -> TypeGuard[slice]:
    return is_instance(item, slice)


# XXX: can not define recursive types yet

RecursiveIterable = Union[T, Iterable[Any]]
RecursiveAsyncIterable = Union[T, AsyncIterable[Any]]
RecursiveAnyIterable = Union[T, AnyIterable[Any]]


S = TypeVar("S", bound="Sum")


@runtime_checkable
class Sum(Protocol):
    @required
    def __add__(self: S, __other: S) -> S:
        raise NotImplementedError


P = TypeVar("P", bound="Product")


@runtime_checkable
class Product(Protocol):
    @required
    def __mul__(self: P, __other: P) -> P:
        raise NotImplementedError


def is_sized(item: Any) -> TypeGuard[Sized]:
    return is_instance(item, Sized)
