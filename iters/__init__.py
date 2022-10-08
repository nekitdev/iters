"""Composable external iteration.

If you have found yourself with a *collection* of some kind, and needed to perform
an operation on the elements of said collection, you will quickly run into *iterators*.
Iterators are heavily used in idiomatic Python code, so becoming familiar with them is essential.
"""

__description__ = "Composable external iteration."
__url__ = "https://github.com/nekitdev/iters"

__title__ = "iters"
__author__ = "nekitdev"
__license__ = "MIT"
__version__ = "0.4.0"

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
    # the async iterator type
    "AsyncIter",
    # the alias of the previous type
    "async_iter",
    # next functions; checked version works on any iterator, unchecked assumes async iteration
    "async_next",
    "async_next_unchecked",
    # since we are "shadowing" standard functions
    "standard_async_iter",
    "standard_async_next",
    # wrap results of function calls into async iterators
    "wrap_async_iter",
    # the iterator type
    "Iter",
    # the alias of the previous type
    "iter",
    # the alias of `iter.reversed`
    "reversed",
    # since we are shadowing standard functions
    "standard_iter",
    "standard_reversed",
    # wrap results of function calls into iterators
    "wrap_iter",
    # ordering
    "Ordering",
)
