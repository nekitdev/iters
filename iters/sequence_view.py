from __future__ import annotations

from typing import Sequence, TypeVar, Union, final, overload

from attrs import frozen
from typing_aliases import is_slice
from wraps.wraps import WrapOption

__all__ = ("SequenceView", "sequence_view")

T = TypeVar("T")

wrap_index_error = WrapOption(IndexError)


@final
@frozen()
class SequenceView(Sequence[T]):
    """Represents views over sequences."""

    sequence: Sequence[T]
    """The sequence to view."""

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> SequenceView[T]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, SequenceView[T]]:
        if is_slice(index):
            return type(self)(self.sequence[index])

        return self.sequence[index]

    def __len__(self) -> int:
        return len(self.sequence)

    @wrap_index_error
    def get(self, index: int) -> T:
        return self.sequence[index]


def sequence_view(sequence: Sequence[T]) -> SequenceView[T]:
    """Returns the [`SequenceView[T]`][iters.views.sequences.SequenceView] over the given sequence.

    Arguments:
        sequence: The sequence to view into.

    Returns:
        The view over the sequence.
    """
    return SequenceView(sequence)
