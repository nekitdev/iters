from typing import AsyncIterator, TypeVar

from typing_aliases import AnyIterable, AsyncBinary, AsyncUnary, Binary, Unary
from wraps.primitives.option import NULL, Option, Some
from wraps.primitives.result import Error, Ok, Result

from iters.async_utils import async_chain, async_iter, async_next_unchecked
from iters.types import is_marker, marker

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


async def async_exactly_one(iterable: AnyIterable[T]) -> Result[T, Option[AsyncIterator[T]]]:
    iterator = async_iter(iterable)

    first = await async_next_unchecked(iterator, marker)

    if is_marker(first):
        return Error(NULL)

    second = await async_next_unchecked(iterator, marker)

    if not is_marker(second):
        return Error(Some(async_chain((first, second), iterator)))

    return Ok(first)


async def async_at_most_one(iterable: AnyIterable[T]) -> Result[Option[T], AsyncIterator[T]]:
    iterator = async_iter(iterable)

    first = await async_next_unchecked(iterator, marker)

    if is_marker(first):
        return Ok(NULL)

    second = await async_next_unchecked(iterator, marker)

    if not is_marker(second):
        return Error(async_chain((first, second), iterator))

    return Ok(Some(first))
