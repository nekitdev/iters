from threading import Lock
from typing import Any, Dict, Type, TypeVar, Union, cast

from typing_extensions import Protocol

__all__ = ("Marker", "MarkerOr", "Order", "Singleton", "marker")

S = TypeVar("S", bound="Singleton")
T = TypeVar("T")

LOCK = Lock()


class Singleton:
    INSTANCES: Dict[Type[Any], Any] = {}

    def __new__(cls: Type[S], *args: Any, **kwargs: Any) -> S:
        # use double-checked locking optimization
        if cls not in cls.INSTANCES:  # check
            with LOCK:  # lock
                if cls not in cls.INSTANCES:  # check
                    cls.INSTANCES[cls] = super().__new__(cls)  # instantiate

        return cast(S, cls.INSTANCES[cls])

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
