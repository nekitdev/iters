from __future__ import annotations

from typing import Sequence, TypeVar, Union, final, overload

from attrs import frozen
from typing_aliases import is_slice
from wraps.wraps import wrap_option

__all__ = ("SequenceView", "sequence_view")

T = TypeVar("T")
U = TypeVar("U")


@final
@frozen()
class SequenceView(Sequence[T]):
    """Represents views over sequences."""

    sequence: Sequence[T]
    """The sequence to view into."""

    @classmethod
    def create(cls, sequence: Sequence[U]) -> SequenceView[U]:
        return cls(sequence)  # type: ignore

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> SequenceView[T]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, SequenceView[T]]:
        if is_slice(index):
            return self.create(self.sequence[index])

        return self.sequence[index]  # type: ignore

    def __len__(self) -> int:
        return len(self.sequence)

    @wrap_option
    def get(self, index: int) -> T:
        return self.sequence[index]


sequence_view = SequenceView
