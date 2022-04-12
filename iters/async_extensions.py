# TODO: make this module standalone
from abc import abstractmethod
from sys import version_info
from threading import Lock as ThreadLock
from types import TracebackType as Traceback
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from anyio import (  # XXX: TCP, TLS, UDP
    AsyncFile,
    CancelScope,
    CapacityLimiter,
    Condition,
    Event,
    Lock,
    Path,
    Semaphore,
)
from anyio import create_memory_object_stream as create_memory_channel
from anyio import (  # XXX: TCP, TLS, UDP
    create_task_group,
    current_time,
    fail_after,
    from_thread,
    get_all_backends,
    move_on_after,
    open_process,
    open_signal_receiver,
    run,
    run_process,
    sleep,
    sleep_forever,
    sleep_until,
    to_process,
    to_thread,
)
from anyio.from_thread import BlockingPortal, start_blocking_portal
from anyio.lowlevel import cancel_shielded_checkpoint, checkpoint, checkpoint_if_cancelled
from anyio.streams.memory import MemoryObjectReceiveStream as MemoryReceiveChannel
from anyio.streams.memory import MemoryObjectSendStream as MemorySendChannel
from sniffio import AsyncLibraryNotFoundError, current_async_library
from typing_extensions import Protocol, Self, TypeAlias

__all__ = (
    # current async library
    "AsyncLibraryNotFoundError",
    "current_async_library",
    # backends
    "get_all_backends",
    # event loops
    "current_time",
    "run",
    "sleep",
    "sleep_forever",
    "sleep_until",
    # tasks
    "create_task_group",
    # cancellation
    "CancelScope",
    # timeouts
    "fail_after",
    "move_on_after",
    # checkpoints
    "checkpoint",
    "checkpoint_if_cancelled",
    "cancel_shielded_checkpoint",
    # communication
    "MemoryChannel",
    "MemoryChannelFactory",
    "MemoryReceiveChannel",
    "MemorySendChannel",
    "create_memory_channel",
    # synchronization
    "CapacityLimiter",
    "Condition",
    "Event",
    "Lock",
    "Semaphore",
    # running code in workers (processes and threads)
    "to_process",
    "to_thread",
    # running asynchronous code from external threads
    "BlockingPortal",
    "from_thread",
    "start_blocking_portal",
    # subprocesses
    "open_process",
    "run_process",
    # file I/O
    "AsyncFile",
    "Path",
    # signal handling
    "open_signal_receiver",
    # closing resources (asynchronously)
    "AsyncCloseable",
    "AsyncClosing",
    "aclosing",
    "aclose_forcefully",
    # asynchronous iteration
    "AsyncIterable",
    "AsyncIterator",
    "aiter",
    "anext",
    # results
    "Result",
    "AnyResult",
    "unwrap_result",
    # cancellation (extension)
    "create_cancel_scope",
    "shield",
    # timeouts (extension)
    "wait_for",
    "wait_for_or_else",
    # communication (extension)
    "Queue",
    # other extensions
    "collect",
    "collect_iterable",
    "collect_with_errors",
    "collect_iterable_with_errors",
)

PYTHON_3_10 = version_info >= (3, 10, 0)

INFINITY = "infinity"
infinity = float(INFINITY)

T = TypeVar("T")
U = TypeVar("U")

TYPE_FORMAT = "<{}>"

format_type = TYPE_FORMAT.format


def type_name(item: T) -> str:
    return type(item).__name__


class SingletonMeta(type):
    _INSTANCES = {}  # type: ignore
    _LOCK = ThreadLock()  # single lock is enough here

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # slightly too magical
        instances = cls._INSTANCES
        lock = cls._LOCK

        # use double-checked locking

        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = super().__call__(*args, **kwargs)

        return instances[cls]


class Singleton(metaclass=SingletonMeta):
    def __repr__(self) -> str:
        return format_type(type_name(self))


class NoDefault(Singleton):
    pass


no_default = NoDefault()

NOT_ASYNC_ITERABLE = "{!r} object is not an async iterable"
NOT_ASYNC_ITERATOR = "{!r} object is not an async iterator"


def not_async_iterable(item: T) -> TypeError:
    return TypeError(NOT_ASYNC_ITERABLE.format(type_name(item)))


def not_async_iterator(item: T) -> TypeError:
    return TypeError(NOT_ASYNC_ITERATOR.format(type_name(item)))


def aiter(async_iterable: AsyncIterable[T]) -> AsyncIterator[T]:
    if isinstance(async_iterable, AsyncIterable):
        return async_iterable.__aiter__()

    raise not_async_iterable(async_iterable)


@overload
async def anext(async_iterator: AsyncIterator[T]) -> T:
    ...


@overload
async def anext(async_iterator: AsyncIterator[T], default: U) -> Union[T, U]:
    ...


async def anext(async_iterator: AsyncIterator[Any], default: Any = no_default) -> Any:
    if isinstance(async_iterator, AsyncIterator):
        try:
            return await async_iterator.__anext__()

        except StopAsyncIteration:
            if default is no_default:
                raise

            return default

    raise not_async_iterator(async_iterator)


@runtime_checkable
class AsyncCloseable(Protocol):
    @abstractmethod
    def aclose(self) -> Awaitable[None]:
        """Close the resource asynchronously."""
        raise NotImplementedError


R = TypeVar("R", bound=AsyncCloseable)

AnyException: TypeAlias = BaseException

ET = TypeVar("ET", bound=AnyException)


class AsyncClosing(Generic[R]):
    def __init__(self, resource: R) -> None:
        self._resource = resource

    def unwrap(self) -> R:
        return self._resource

    async def __aenter__(self) -> R:
        return self.unwrap()

    async def __aexit__(
        self, error_type: Optional[Type[ET]], error: Optional[ET], traceback: Optional[Traceback]
    ) -> None:
        await self.unwrap().aclose()


def aclosing(resource: R) -> AsyncClosing[R]:
    return AsyncClosing(resource)


async def aclose_forcefully(resource: R) -> None:
    with create_cancel_scope() as scope:
        scope.cancel()
        await resource.aclose()


def create_cancel_scope(*, shield: bool = False) -> CancelScope:
    return CancelScope(shield=shield)


MemoryChannel = Tuple[MemorySendChannel[T], MemoryReceiveChannel[T]]


class MemoryChannelFactory(Generic[T]):
    def __init__(self, max_size: int = 0) -> None:
        self._sender, self._receiver = cast(
            MemoryChannel[T], create_memory_channel(max_size if max_size > 0 else infinity)
        )

    @property
    def sender(self) -> MemorySendChannel[T]:
        return self._sender

    @property
    def receiver(self) -> MemoryReceiveChannel[T]:
        return self._receiver

    @property
    def channel(self) -> MemoryChannel[T]:
        return (self.sender, self.receiver)


class Queue(Generic[T]):
    def __init__(self, max_size: int = 0) -> None:
        self._max_size = max_size

        self._sender, self._receiver = MemoryChannelFactory[T](max_size).channel

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def size(self) -> int:
        return self._receiver.statistics().current_buffer_used

    def is_empty(self) -> bool:
        return not self.size

    def is_full(self) -> bool:
        return self.max_size > 0 and self.size >= self.max_size

    async def put(self, item: T) -> None:
        await self._sender.send(item)

    def put_nowait(self, item: T) -> None:
        self._sender.send_nowait(item)

    async def get(self) -> T:
        return await self._receiver.receive()

    def get_nowait(self) -> T:
        return self._receiver.receive_nowait()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        return await self.get()


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
    # queue: Queue[AnyTaggedResult[T]],
) -> None:
    result: AnyResult[T]

    try:
        result = await awaitable

    except AnyException as error:
        result = error

    finally:
        # await queue.put((tag, result))
        array.append((tag, result))


DynamicTuple = Tuple[T, ...]
EmptyTuple = Tuple[()]

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
G = TypeVar("G")
H = TypeVar("H")

# XXX: is using list(...) and tuple(...) alright here?


@overload
async def collect_with_errors() -> EmptyTuple:
    ...  # pragma: overload


@overload
async def collect_with_errors(__awaitable_a: Awaitable[A]) -> Tuple[AnyResult[A]]:
    ...  # pragma: overload


@overload
async def collect_with_errors(
    __awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B]
) -> Tuple[AnyResult[A], AnyResult[B]]:
    ...  # pragma: overload


@overload
async def collect_with_errors(
    __awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B], __awaitable_c: Awaitable[C]
) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C]]:
    ...  # pragma: overload


@overload
async def collect_with_errors(
    __awaitable_a: Awaitable[A],
    __awaitable_b: Awaitable[B],
    __awaitable_c: Awaitable[C],
    __awaitable_d: Awaitable[D],
) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C], AnyResult[D]]:
    ...  # pragma: overload


@overload
async def collect_with_errors(
    __awaitable_a: Awaitable[A],
    __awaitable_b: Awaitable[B],
    __awaitable_c: Awaitable[C],
    __awaitable_d: Awaitable[D],
    __awaitable_e: Awaitable[E],
) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C], AnyResult[D], AnyResult[E]]:
    ...  # pragma: overload


@overload
async def collect_with_errors(
    __awaitable_a: Awaitable[A],
    __awaitable_b: Awaitable[B],
    __awaitable_c: Awaitable[C],
    __awaitable_d: Awaitable[D],
    __awaitable_e: Awaitable[E],
    __awaitable_f: Awaitable[F],
) -> Tuple[AnyResult[A], AnyResult[B], AnyResult[C], AnyResult[D], AnyResult[E], AnyResult[F]]:
    ...  # pragma: overload


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
    ...  # pragma: overload


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
    ...  # pragma: overload


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
    ...  # pragma: overload


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
    ...  # pragma: overload


@overload
async def collect(__awaitable_a: Awaitable[A]) -> Tuple[A]:
    ...  # pragma: overload


@overload
async def collect(__awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B]) -> Tuple[A, B]:
    ...  # pragma: overload


@overload
async def collect(
    __awaitable_a: Awaitable[A], __awaitable_b: Awaitable[B], __awaitable_c: Awaitable[C]
) -> Tuple[A, B, C]:
    ...  # pragma: overload


@overload
async def collect(
    __awaitable_a: Awaitable[A],
    __awaitable_b: Awaitable[B],
    __awaitable_c: Awaitable[C],
    __awaitable_d: Awaitable[D],
) -> Tuple[A, B, C, D]:
    ...  # pragma: overload


@overload
async def collect(
    __awaitable_a: Awaitable[A],
    __awaitable_b: Awaitable[B],
    __awaitable_c: Awaitable[C],
    __awaitable_d: Awaitable[D],
    __awaitable_e: Awaitable[E],
) -> Tuple[A, B, C, D, E]:
    ...  # pragma: overload


@overload
async def collect(
    __awaitable_a: Awaitable[A],
    __awaitable_b: Awaitable[B],
    __awaitable_c: Awaitable[C],
    __awaitable_d: Awaitable[D],
    __awaitable_e: Awaitable[E],
    __awaitable_f: Awaitable[F],
) -> Tuple[A, B, C, D, E, F]:
    ...  # pragma: overload


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
    ...  # pragma: overload


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
    ...  # pragma: overload


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
    *awaitables: Awaitable[Any],
) -> DynamicTuple[Any]:
    ...  # pragma: overload


async def collect(*awaitables: Awaitable[Any]) -> DynamicTuple[Any]:
    return tuple(await collect_iterable(awaitables))


async def collect_iterable(iterable: Iterable[Awaitable[T]]) -> List[T]:
    return list(map(unwrap_result, await collect_iterable_with_errors(iterable)))


def unwrap_result(result: AnyResult[T]) -> T:
    if isinstance(result, AnyException):
        raise result

    else:
        return result


async def shield(awaitable: Awaitable[T]) -> T:
    with create_cancel_scope(shield=True):
        return await awaitable


async def wait_for(awaitable: Awaitable[T], timeout: Optional[float], *, shield: bool = False) -> T:
    with fail_after(timeout, shield=shield):
        return await awaitable


async def wait_for_or_else(
    awaitable: Awaitable[T], default: U, timeout: Optional[float], *, shield: bool = False
) -> Union[T, U]:
    with move_on_after(timeout, shield=shield):
        return await awaitable

    return default
