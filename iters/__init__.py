"""Composable external iteration."""

__description__ = "Composable external iteration."
__url__ = "https://github.com/nekitdev/iters"

__title__ = "iters"
__author__ = "nekitdev"
__license__ = "MIT"
__version__ = "0.1.0"

from iters.async_iters import (
    AsyncIter,
    async_iter,
    async_next,
    async_next_unchecked,
    standard_async_iter,
    standard_async_next,
    wrap_async_iter,
)
from iters.iters import Iter, iter, reversed, standard_iter, standard_reversed, wrap_iter
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
    "wrap_async_iter",
    # iterator class
    "Iter",
    # convenient functions to get iterators
    "iter",
    "reversed",
    # since we are shadowing standard functions
    "standard_iter",
    "standard_reversed",
    # wrap results of function calls into iterators
    "wrap_iter",
    # ordering
    "Ordering",
)
