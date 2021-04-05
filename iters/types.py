from typing import Any, Type, TypeVar, Union, cast

from typing_extensions import Protocol

__all__ = ("Marker", "MarkerOr", "Order", "Singleton", "marker")

S = TypeVar("S", bound="Singleton")
T = TypeVar("T")


class Singleton:
    INSTANCE = None

    def __new__(cls: Type[S], *args: Any, **kwargs: Any) -> S:
        if cls.INSTANCE is None:
            cls.INSTANCE = cast(S, super().__new__(cls))

        return cls.INSTANCE

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class Marker(Singleton):
    pass


marker = Marker()

MarkerOr = Union[Marker, T]


class Order(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...

    def __gt__(self, __other: Any) -> bool:
        ...
