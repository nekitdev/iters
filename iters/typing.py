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

from funcs.typing import AsyncUnary, Unary
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
"""Represents for-each functions `(T)`."""

AsyncForEach = AsyncUnary[T, None]
"""Represents async for-each functions `async (T)`."""

Validate = Unary[T, None]
"""Represents validation functions `(T)`."""

AsyncValidate = AsyncUnary[T, None]
"""Represents async validation functions `async (T)`."""

Pair = Tuple[T, T]
"""Represents pairs `(T, T)`."""

AnyIterable = Union[AsyncIterable[T], Iterable[T]]
"""Represents any iterable, async or not."""
AnyIterator = Union[AsyncIterator[T], Iterator[T]]
"""Represents any iterator, async or not."""

Selectors = Iterable[bool]
"""Represents selectors."""
AsyncSelectors = AsyncIterable[bool]
"""Represents async selectors."""
AnySelectors = AnyIterable[bool]
"""Represents any selectors, async or not."""


def is_async_iterable(iterable: AnyIterable[T]) -> TypeGuard[AsyncIterable[T]]:
    """Checks if an [`AnyIterable[T]`][iters.typing.AnyIterable] is
    [`AsyncIterable[T]`][typing.AsyncIterable].

    Arguments:
        iterable: The iterable to check.

    Returns:
        Whether the iterable is [`AsyncIterable[T]`][typing.AsyncIterable].
    """
    return is_instance(iterable, AsyncIterable)


def is_iterable(iterable: AnyIterable[T]) -> TypeGuard[Iterable[T]]:
    """Checks if an [`AnyIterable[T]`][iters.typing.AnyIterable] is
    [`Iterable[T]`][typing.Iterable].

    Arguments:
        iterable: The iterable to check.

    Returns:
        Whether the iterable is [`Iterable[T]`][typing.Iterable].
    """
    return is_instance(iterable, Iterable)


def is_async_iterator(iterator: AnyIterator[T]) -> TypeGuard[AsyncIterator[T]]:
    """Checks if an [`AnyIterator[T]`][iters.typing.AnyIterator] is
    [`AsyncIterator[T]`][typing.AsyncIterator].

    Arguments:
        iterator: The iterator to check.

    Returns:
        Whether the iterator is [`AsyncIterator[T]`][typing.AsyncIterator].
    """
    return is_instance(iterator, AsyncIterator)


def is_iterator(iterator: AnyIterator[T]) -> TypeGuard[Iterator[T]]:
    """Checks if an [`AnyIterator[T]`][iters.typing.AnyIterator] is
    [`Iterator[T]`][typing.Iterator].

    Arguments:
        iterator: The iterator to check.

    Returns:
        Whether the iterator is [`Iterator[T]`][typing.Iterator].
    """
    return is_instance(iterator, Iterator)


def is_string(item: Any) -> TypeGuard[str]:
    """Checks if an `item` is a string (type [`str`][str]).

    Arguments:
        item: The item to check.

    Returns:
        Whether the item is of type [`str`][str].
    """
    return is_instance(item, str)


def is_bytes(item: Any) -> TypeGuard[bytes]:
    """Checks if an `item` is a byte string (type [`bytes`][bytes]).

    Arguments:
        item: The item to check.

    Returns:
        Whether the item is of type [`bytes`][bytes].
    """
    return is_instance(item, bytes)


def is_slice(item: Any) -> TypeGuard[slice]:
    """Checks if an `item` is a slice (type [`slice`][slice]).

    Arguments:
        item: The item to check.

    Returns:
        Whether the item is of type [`slice`][slice].
    """
    return is_instance(item, slice)


# XXX: can not define recursive types yet

RecursiveIterable = Union[T, Iterable[Any]]
"""Represents recursive iterables."""
RecursiveAsyncIterable = Union[T, AsyncIterable[Any]]
"""Represents recursive async iterables."""
RecursiveAnyIterable = Union[T, AnyIterable[Any]]
"""Represents any recursive iterables, async or not."""


S = TypeVar("S", bound="Sum")


@runtime_checkable
class Sum(Protocol):
    """Represents types for which adding `self: S` to `other: S` returns `S`."""

    @required
    def __add__(self: S, __other: S) -> S:
        raise NotImplementedError


P = TypeVar("P", bound="Product")


@runtime_checkable
class Product(Protocol):
    """Represents types for which multiplying `self: P` with `other: P` returns `P`."""

    @required
    def __mul__(self: P, __other: P) -> P:
        raise NotImplementedError


def is_sized(item: Any) -> TypeGuard[Sized]:
    """Checks if an item is sized (type [`Sized`][typing.Sized]).

    Arguments:
        item: The item to check.

    Returns:
        Whether the item is of type [`Sized`][typing.Sized].
    """
    return is_instance(item, Sized)
