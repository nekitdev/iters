from typing import Generic, TypeVar

from attrs import define
from typing_extensions import final

__all__ = ("State", "stateful")

T = TypeVar("T", covariant=True)


@final
@define()
class State(Generic[T]):
    """Represents the mutable state of iteration."""

    value: T
    """The wrapped value."""

    def get(self) -> T:
        return self.value

    def set(self, value: T) -> T:  # type: ignore
        self.value = value


def stateful(value: T) -> State[T]:
    """Wraps the given value into a [`State[T]`][wraps.state.State].

    Arguments:
        value: The value to wrap.

    Returns:
        The state wrapping the value.
    """
    return State(value)
