from typing import Any, Dict, Hashable, Mapping, TypeVar, overload

from typing_aliases import StringDict, StringMapping

__all__ = ("merge",)

Q = TypeVar("Q", bound=Hashable)
T = TypeVar("T")


@overload
def merge(*mappings: Mapping[Q, T]) -> Dict[Q, T]:
    ...


@overload
def merge(*mappings: StringMapping[T], **keywords: T) -> StringDict[T]:
    ...


def merge(*mappings: Mapping[Any, Any], **keywords: Any) -> Dict[Any, Any]:
    """Merges multiple `mappings` and `keywords` into one dictionary.

    Arguments:
        *mappings: Mappings to merge.
        **keywords: Keywords to add to the merged dictionary.

    Returns:
        The merged dictionary.
    """
    merged: Dict[Any, Any] = {}

    for mapping in mappings:
        merged.update(mapping)

    merged.update(keywords)

    return merged
