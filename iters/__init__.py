"""Composable external iteration."""

__description__ = "Composable external iteration."
__url__ = "https://github.com/nekitdev/iters.py"

__title__ = "iters"
__author__ = "nekitdev"
__license__ = "MIT"
__version__ = "1.0.0-alpha.1"

from iters.async_iters import (
    AsyncIter,
    async_iter,
    async_iter_result,
    async_next,
    async_next_unchecked,
    standard_async_iter,
    standard_async_next,
)
from iters.iters import Iter, iter, iter_result, reversed, standard_iter, standard_reversed
from iters.types import Ordering

__all__ = (
    # async iterator class
    "AsyncIter",
    # convenient function to get async iterators
    "async_iter",
    # next functions; checked version works on any iterator, unchecked assumes async iteration
    "async_next",
    "async_next_unchecked",
    # since we are shadowing standard functions
    "standard_async_iter",
    "standard_async_next",
    # wrap results of function calls into async iterators
    "async_iter_result",
    # iterator class
    "Iter",
    # convenient functions to get iterators
    "iter",
    "reversed",
    # since we are shadowing standard functions
    "standard_iter",
    "standard_reversed",
    # wrap results of function calls into iterators
    "iter_result",
    # ordering
    "Ordering",
)
