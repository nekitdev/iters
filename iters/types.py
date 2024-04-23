from __future__ import annotations

from typing import Any, TypeVar, Union

from solus import Singleton
from typing_extensions import TypeIs
from wraps.primitives.option import NULL, Option, Some

T = TypeVar("T")


class NoDefault(Singleton):
    pass


no_default = NoDefault()


NoDefaultOr = Union[NoDefault, T]


def is_no_default(item: Any) -> TypeIs[NoDefault]:
    return item is no_default


class Marker(Singleton):
    pass


marker = Marker()

MarkerOr = Union[Marker, T]


def is_marker(item: Any) -> TypeIs[Marker]:
    return item is marker


def wrap_marked(item: MarkerOr[T]) -> Option[T]:
    return NULL if is_marker(item) else Some(item)
