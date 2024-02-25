from typing import AsyncIterator, TypeVar

from typing_aliases import AnyIterable, AsyncBinary, AsyncUnary, Binary, Unary
from wraps.option import Option

from iters.async_utils import async_iter

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")


async def async_scan(
    state: S, function: Binary[S, T, Option[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        option = function(state, item)

        if option.is_some():
            yield option.unwrap()

        else:
            break


async def async_scan_await(
    state: S, function: AsyncBinary[S, T, Option[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        option = await function(state, item)

        if option.is_some():
            yield option.unwrap()

        else:
            break


async def async_filter_map_option(
    function: Unary[T, Option[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        option = function(item)

        if option.is_some():
            yield option.unwrap()


async def async_filter_map_option_await(
    function: AsyncUnary[T, Option[U]], iterable: AnyIterable[T]
) -> AsyncIterator[U]:
    async for item in async_iter(iterable):
        option = await function(item)

        if option.is_some():
            yield option.unwrap()
