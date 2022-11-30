from solus import Singleton

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


class Marker(Singleton):
    """Represents markers used for various checks."""


marker = Marker()
"""The instance of [`Marker`][iters.types.Marker]."""
