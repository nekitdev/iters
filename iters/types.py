from __future__ import annotations

from enum import Enum
from threading import Lock
from typing import Any, TypeVar

__all__ = (
    # ordering
    "Ordering",
    # markers
    "Marker",
    "marker",
    # no default
    "NoDefault",
    "no_default",
    # singletons
    "Singleton",
    "SingletonMeta",
    # simple utils
    "format_type",
    "type_name",
)


class Ordering(Enum):
    LESS = -1
    EQUAL = 0
    GREATER = 1

    def is_less(self) -> bool:
        return self is type(self).LESS

    def is_equal(self) -> bool:
        return self is type(self).EQUAL

    def is_greater(self) -> bool:
        return self is type(self).GREATER

    def is_less_or_equal(self) -> bool:
        return self.is_less() or self.is_equal()

    def is_not_equal(self) -> bool:
        return not self.is_equal()

    def is_greater_or_equal(self) -> bool:
        return self.is_greater() or self.is_equal()


T = TypeVar("T")

TYPE_FORMAT = "<{}>"

format_type = TYPE_FORMAT.format


def type_name(item: T) -> str:
    return type(item).__name__


class SingletonMeta(type):
    _INSTANCES = {}  # type: ignore
    _LOCK = Lock()  # single lock is enough here

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # slightly too magical
        instances = cls._INSTANCES
        lock = cls._LOCK

        # use double-checked locking

        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = super().__call__(*args, **kwargs)

        return instances[cls]


class Singleton(metaclass=SingletonMeta):
    def __repr__(self) -> str:
        return format_type(type_name(self))


class NoDefault(Singleton):
    pass


no_default = NoDefault()


class Marker(Singleton):
    pass


marker = Marker()
