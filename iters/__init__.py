"""Library that provides rich object-oriented iterators."""

__title__ = "iters"
__author__ = "nekitdev"
__copyright__ = "Copyright 2020-2021 nekitdev"
__license__ = "MIT"
__version__ = "0.10.1"

from iters.async_iters import (
    AsyncIter,
    async_iter,
    async_reversed,
    async_next,
    async_next_unchecked,
    std_async_iter,
    std_async_reversed,
    return_async_iter,
)
from iters.iters import (
    Iter,
    iter,
    reversed,
    std_iter,
    std_reversed,
    return_iter,
)

__all__ = (
    # async iterator class
    "AsyncIter",
    # convenient functions to get an async iterator
    "async_iter",
    "async_reversed",
    # next functions; checked version works on any iterator, unchecked assumes an async iterator
    "async_next",
    "async_next_unchecked",
    # since we are shadowing standard functions, export them as <std>
    "std_async_iter",
    "std_async_reversed",
    # decorator to wrap return value of the function into an async iterator
    "return_async_iter",
    # iterator class
    "Iter",
    # convenient functions to get an iterator
    "iter",
    "reversed",
    # since we are shadowing standard functions, export them as <std>
    "std_iter",
    "std_reversed",
    # decorator to wrap return value of the function into an iterator
    "return_iter",
)
