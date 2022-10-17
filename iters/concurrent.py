from functools import partial
from typing import Any, Awaitable, Callable, Iterable, List, Tuple, TypeVar, Union, overload

from typing_extensions import ParamSpec

try:
    from anyio import create_task_group
    from anyio.to_thread import run_sync as standard_run_blocking

except ImportError:
    CONCURRENT = False

else:
    CONCURRENT = True

from iters.typing import AnyException, DynamicTuple, EmptyTuple, is_error

__all__ = (
    "CONCURRENT",
    "collect",
    "collect_with_errors",
    "collect_iterable",
    "collect_iterable_with_errors",
    "run_blocking",
)

P = ParamSpec("P")

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")

T = TypeVar("T")
ET = TypeVar("ET", bound=AnyException)

Result = Union[T, ET]

AnyResult = Result[T, AnyException]

TaggedResult = Tuple[int, Result[T, ET]]
AnyTaggedResult = TaggedResult[T, AnyException]


def _sort_tagged(tagged_result: AnyTaggedResult[T]) -> int:
    tag, _ = tagged_result

    return tag


def _take_result(tagged_result: AnyTaggedResult[T]) -> AnyResult[T]:
    _, result = tagged_result

    return result


async def _append_tagged_result(
    awaitable: Awaitable[T],
    tag: int,
    array: List[AnyTaggedResult[T]],
) -> None:
    result: AnyResult[T]

    try:
        result = await awaitable

    except AnyException as error:
        result = error

    array.append((tag, result))


def unwrap_result(result: AnyResult[T]) -> T:
    if is_error(result):
        raise result

    return result  # type: ignore


if CONCURRENT:

    @overload
    async def collect_with_errors() -> EmptyTuple:
        ...

    @overload
    async def collect_with_errors(__awaitable_a: Awaitable[A]) -> Tuple[AnyResult[A]]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B]
    ) -> Tuple[AnyResult[A], AnyResult[B]]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B], __awaitable_c: Awaitable[C]
    ) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C]]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
    ) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C], AnyResult[D]]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
    ) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C], AnyResult[D], AnyResult[E]]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
        __awaitable_f: Awaitable[F],
    ) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C], AnyResult[D], AnyResult[E], AnyResult[F]]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
        __awaitable_f: Awaitable[F],
        __awaitable_g: Awaitable[G],
    ) -> Tuple[
        AnyResult[A],
        AnyResult[B],
        AnyResult[C],
        AnyResult[D],
        AnyResult[E],
        AnyResult[F],
        AnyResult[G],
    ]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
        __awaitable_f: Awaitable[F],
        __awaitable_g: Awaitable[G],
        __awaitable_h: Awaitable[H],
    ) -> Tuple[
        AnyResult[A],
        AnyResult[B],
        AnyResult[C],
        AnyResult[D],
        AnyResult[E],
        AnyResult[F],
        AnyResult[G],
        AnyResult[H],
    ]:
        ...

    @overload
    async def collect_with_errors(
        __awaitable_a: Awaitable[Any],
        __awaitable_b: Awaitable[Any],
        __awaitable_c: Awaitable[Any],
        __awaitable_d: Awaitable[Any],
        __awaitable_e: Awaitable[Any],
        __awaitable_f: Awaitable[Any],
        __awaitable_g: Awaitable[Any],
        __awaitable_h: Awaitable[Any],
        *awaitables: Awaitable[Any],
    ) -> DynamicTuple[AnyResult[Any]]:
        ...

    async def collect_with_errors(*awaitables: Awaitable[Any]) -> DynamicTuple[AnyResult[Any]]:
        return tuple(await collect_iterable_with_errors(awaitables))

    async def collect_iterable_with_errors(iterable: Iterable[Awaitable[T]]) -> List[AnyResult[T]]:
        result: List[AnyTaggedResult[T]] = []

        async with create_task_group() as task_group:
            for tag, awaitable in enumerate(iterable):
                task_group.start_soon(_append_tagged_result, awaitable, tag, result)

        result.sort(key=_sort_tagged)

        return list(map(_take_result, result))

    @overload
    async def collect() -> EmptyTuple:
        ...

    @overload
    async def collect(__awaitable_a: Awaitable[A]) -> Tuple[A]:
        ...

    @overload
    async def collect(__awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B]) -> Tuple[A, B]:
        ...

    @overload
    async def collect(
        __awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B], __awaitable_c: Awaitable[C]
    ) -> Tuple[A, B, C]:
        ...

    @overload
    async def collect(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
    ) -> Tuple[A, B, C, D]:
        ...

    @overload
    async def collect(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
    ) -> Tuple[A, B, C, D, E]:
        ...

    @overload
    async def collect(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
        __awaitable_f: Awaitable[F],
    ) -> Tuple[A, B, C, D, E, F]:
        ...

    @overload
    async def collect(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
        __awaitable_f: Awaitable[F],
        __awaitable_g: Awaitable[G],
    ) -> Tuple[A, B, C, D, E, F, G]:
        ...

    @overload
    async def collect(
        __awaitable_a: Awaitable[A],
        __awaitable_b: Awaitable[B],
        __awaitable_c: Awaitable[C],
        __awaitable_d: Awaitable[D],
        __awaitable_e: Awaitable[E],
        __awaitable_f: Awaitable[F],
        __awaitable_g: Awaitable[G],
        __awaitable_h: Awaitable[H],
    ) -> Tuple[A, B, C, D, E, F, G, H]:
        ...

    @overload
    async def collect(
        __awaitable_a: Awaitable[Any],
        __awaitable_b: Awaitable[Any],
        __awaitable_c: Awaitable[Any],
        __awaitable_d: Awaitable[Any],
        __awaitable_e: Awaitable[Any],
        __awaitable_f: Awaitable[Any],
        __awaitable_g: Awaitable[Any],
        __awaitable_h: Awaitable[Any],
        __awaitable_next: Awaitable[Any],
        *awaitables: Awaitable[Any],
    ) -> DynamicTuple[Any]:
        ...

    async def collect(*awaitables: Awaitable[Any]) -> DynamicTuple[Any]:
        return tuple(await collect_iterable(awaitables))

    async def collect_iterable(iterable: Iterable[Awaitable[T]]) -> List[T]:
        return list(map(unwrap_result, await collect_iterable_with_errors(iterable)))

    async def run_blocking(function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        return await standard_run_blocking(partial(function, *args, **kwargs))
