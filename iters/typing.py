from __future__ import annotations

from typing import Optional, Protocol, TypeVar, runtime_checkable

from typing_aliases import Predicate, required
from typing_extensions import Self

__all__ = ("OptionalPredicate", "Sum", "Product")

T = TypeVar("T")

OptionalPredicate = Optional[Predicate[T]]
"""Represents optional predicates.

Passing [`None`][None] is equivalent to passing [`bool`][bool], though most functions
are optimized to reduce the overhead of calling [`bool`][bool].
"""


@runtime_checkable
class Sum(Protocol):
    """Represents types for which adding `self: S` to `other: S` returns `S`."""

    @required
    def __add__(self, __other: Self) -> Self:
        raise NotImplementedError


@runtime_checkable
class Product(Protocol):
    """Represents types for which multiplying `self: P` with `other: P` returns `P`."""

    @required
    def __mul__(self, __other: Self) -> Self:
        raise NotImplementedError
