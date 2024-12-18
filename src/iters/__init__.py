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
__version__ = "0.18.0"

from iters import mappings, typing, utils
from iters.iters import Iter, iter, reversed, standard_iter, standard_reversed, wrap_iter
from iters.ordered_sets import OrderedSet, ordered_set, ordered_set_unchecked
from iters.states import State, state

__all__ = (
    # the iterator type
    "Iter",
    # an alias of the previous type
    "iter",
    # an alias of `iter.reversed`
    "reversed",
    # since we are shadowing standard functions
    "standard_iter",
    "standard_reversed",
    # wrap results of function calls into iterators
    "wrap_iter",
    # ordered sets
    "OrderedSet",
    "ordered_set",
    "ordered_set_unchecked",
    # state
    "State",
    "state",
    # mappings
    "mappings",
    # utils
    "utils",
    # typing
    "typing",
)
