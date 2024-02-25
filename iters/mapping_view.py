from typing import Any, Iterator, Mapping, TypeVar, final

from attrs import frozen
from typing_extensions import Self
from wraps.wraps import WrapOption

__all__ = ("MappingView", "mapping_view")

K = TypeVar("K")
V = TypeVar("V")

wrap_key_error = WrapOption(KeyError)


@final
@frozen()
class MappingView(Mapping[K, V]):
    """Represents view over mappings."""

    mapping: Mapping[K, V]
    """The mapping to view."""

    def __iter__(self) -> Iterator[K]:
        yield from self.mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.mapping

    def __len__(self) -> int:
        return len(self.mapping)

    @wrap_key_error
    def get_option(self, key: K) -> V:
        return self[key]

    def copy(self) -> Self:
        return type(self)(self)


mapping_view = MappingView
