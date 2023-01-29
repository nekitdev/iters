from typing import Any

from solus import Singleton
from typing_extensions import TypeGuard

__all__ = (
    # markers
    "Marker",
    "marker",
    # no default
    "NoDefault",
    "no_default",
)


class NoDefault(Singleton):
    """Represents the absence of default values."""


no_default = NoDefault()
"""The instance of [`NoDefault`][iters.types.NoDefault]."""


def is_no_default(item: Any) -> TypeGuard[NoDefault]:
    """Checks if the `item` is [`NoDefault`][iters.types.NoDefault].

    Returns:
        Whether the `item` is [`NoDefault`][iters.types.NoDefault].
    """
    return item is no_default


class Marker(Singleton):
    """Represents markers used for various checks."""


marker = Marker()
"""The instance of [`Marker`][iters.types.Marker]."""


def is_marker(item: Any) -> TypeGuard[Marker]:
    """Checks if the `item` is [`Marker`][iters.types.Marker].

    Returns:
        Whether the `item` is [`Marker`][iters.types.Marker].
    """
    return item is marker
