from typing import AsyncIterator, TypeVar

from typing_aliases import AnyIterable, AsyncBinary, Binary, Pair
from wraps.result import Result

__all__ = ("async_coalesce", "async_coalesce_await")

T = TypeVar("T")


async def async_coalesce(
    function: Binary[T, T, Result[T, Pair[T]]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    ...


async def async_coalesce_await(
    function: AsyncBinary[T, T, Result[T, Pair[T]]], iterable: AnyIterable[T]
) -> AsyncIterator[T]:
    ...
