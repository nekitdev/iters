from __future__ import annotations

from abc import abstractmethod as required
from typing import TypeVar

from typing_extensions import Protocol, runtime_checkable

__all__ = ("Sum", "Product")

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
