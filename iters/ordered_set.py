from __future__ import annotations

from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    MutableSet,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from mixed_methods import mixed_method
from named import get_type_name
from typing_aliases import AnySet, is_instance, is_sized, is_slice
from typing_extensions import TypeIs
from wraps.wraps.option import wrap_option_on

__all__ = ("OrderedSet", "ordered_set", "ordered_set_unchecked")

Q = TypeVar("Q", bound=Hashable)
R = TypeVar("R", bound=Hashable)

SLICE_ALL = slice(None)
LAST = ~0
"""The last index."""

EMPTY_REPRESENTATION = "{}()"
ITEMS_REPRESENTATION = "{}({})"


ITEM_NOT_IN_ORDERED_SET = "item {!r} is not in the ordered set"
item_not_in_ordered_set = ITEM_NOT_IN_ORDERED_SET.format


T = TypeVar("T")


def is_sequence(iterable: Iterable[T]) -> TypeIs[Sequence[T]]:
    return is_instance(iterable, Sequence)


def is_any_set(iterable: Iterable[T]) -> TypeIs[AnySet[T]]:
    return is_instance(iterable, AnySet)


wrap_index_error = wrap_option_on(IndexError)


class OrderedSet(MutableSet[Q], Sequence[Q]):
    """Represents ordered sets, i.e. mutable hash sets that preserve insertion order.

    The implementation is rather simple: it uses an *array* to store the items
    and a *hash map* to store the indices of the items in the array along with ensuring uniqueness.

    The complexity of the operations assumes that *hash maps*
    have `O(1)` *insertion*, *lookup* and *deletion* as well
    as that *arrays* have `O(1)` *by-index lookup* and *length checking*.

    It is assumed that *clearing* is `O(n)`, where `n` is the number of elements.
    """

    def __init__(self, iterable: Iterable[Q] = ()) -> None:
        self._items: List[Q] = []
        self._item_to_index: Dict[Q, int] = {}

        self.update(iterable)

    @classmethod
    def create(cls, iterable: Iterable[R] = ()) -> OrderedSet[R]:
        """Creates an ordered set from an iterable.

        Complexity:
            `O(n)`, where `n` is the length of the iterable.

        Example:
            ```python
            >>> array = [0, 1, 1, 0, 1, 1, 1, 0]
            >>> order_set = ordered_set.create(array)
            >>> order_set
            OrderedSet([0, 1])
            ```

        Arguments:
            iterable: The iterable to create the ordered set from.

        Returns:
            The created ordered set.
        """
        return cls(iterable)  # type: ignore[arg-type, return-value]

    @classmethod
    def create_unchecked(cls, iterable: Iterable[R] = ()) -> OrderedSet[R]:
        """Creates an ordered set from an iterable without checking if the items are unique.

        This method is useful when constructing an ordered set from an iterable that is known to
        contain unique items only.

        Complexity:
            `O(n)`, where `n` is the length of the iterable.

        Example:
            ```python
            >>> array = [1, 2, 3]  # we know that the items are unique
            >>> order_set = ordered_set.create_unchecked(array)
            >>> order_set
            OrderedSet([1, 2, 3])
            ```

        Arguments:
            iterable: The iterable to create the ordered set from.

        Returns:
            The created ordered set.
        """
        self: OrderedSet[R] = cls.create()

        items = self._items
        item_to_index = self._item_to_index

        items.extend(iterable)

        for index, item in enumerate(items):
            item_to_index[item] = index

        return self

    @classmethod
    def create_union(cls, *iterables: Iterable[R]) -> OrderedSet[R]:
        """Creates an ordered set that is the union of given iterables.

        Arguments:
            *iterables: The iterables to create the ordered set union from.

        Returns:
            The ordered set union.
        """
        return cls.create(chain(*iterables))

    @classmethod
    def create_intersection(cls, *iterables: Iterable[R]) -> OrderedSet[R]:
        """Creates an ordered set that is the intersection of given iterables.

        The order is determined by the first iterable.

        Arguments:
            *iterables: The iterables to create the ordered set intersection from.

        Returns:
            The ordered set intersection.
        """
        if iterables:
            head, *tail = iterables

            return cls.create(head).apply_intersection(*tail)

        return cls.create()

    @classmethod
    def create_difference(cls, *iterables: Iterable[R]) -> OrderedSet[R]:
        """Creates an ordered set that is the difference of given iterables.

        The order is determined by the first iterable.

        Arguments:
            *iterables: The iterables to create the orderd set difference from.

        Returns:
            The ordered set difference.
        """
        if iterables:
            head, *tail = iterables

            return cls.create(head).apply_difference(*tail)

        return cls.create()

    @classmethod
    def create_symmetric_difference(cls, *iterables: Iterable[R]) -> OrderedSet[R]:
        """Creates an ordered set that is the symmetric difference of given iterables.

        The order is determined by the first iterable.

        Arguments:
            *iterables: The iterables to create the ordered set symmetric difference from.

        Returns:
            The ordered set symmetric difference.
        """
        if iterables:
            head, *tail = iterables

            return cls.create(head).apply_symmetric_difference(*tail)

        return cls.create()

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> Q: ...

    @overload
    def __getitem__(self, index: slice) -> OrderedSet[Q]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Q, OrderedSet[Q]]:
        if is_slice(index):
            if index == SLICE_ALL:
                return self.copy()

            return self.create_unchecked(self._items[index])

        return self._items[index]

    def copy(self) -> OrderedSet[Q]:
        """Copies the ordered set.

        This is equivalent to:

        ```python
        order_set.create_unchecked(order_set)
        ```

        Complexity:
            `O(n)`, where `n` is the length of the ordered set.

        Example:
            ```python
            >>> order_set = ordered_set([1, 2, 3])
            >>> order_set
            OrderedSet([1, 2, 3])
            >>> order_set.copy()
            OrderedSet([1, 2, 3])
            ```

        Returns:
            The copied ordered set.
        """
        return self.create_unchecked(self)

    def __contains__(self, item: Any) -> bool:
        return item in self._item_to_index

    def add(self, item: Q) -> None:
        """Adds an item to the ordered set.

        Complexity:
            `O(1)`.

        Example:
            ```python
            >>> order_set = ordered_set()
            >>> order_set
            OrderedSet()
            >>> order_set.add(0)
            >>> order_set.add(1)
            >>> order_set.add(0)
            >>> order_set
            OrderedSet([0, 1])
            ```

        Arguments:
            item: The item to add.
        """
        item_to_index = self._item_to_index

        if item not in item_to_index:
            items = self._items

            item_to_index[item] = len(items)

            items.append(item)

    append = add
    """An alias of [`add`][iters.ordered_set.OrderedSet.add]."""

    def update(self, iterable: Iterable[Q]) -> None:
        """Updates the ordered set with the items from an iterable.

        This is equivalent to:

        ```python
        for item in iterable:
            ordered_set.add(item)
        ```

        Complexity:
            `O(n)`, where `n` is the length of the iterable.

        Example:
            ```python
            >>> order_set = ordered_set()
            >>> order_set.update([0, 1])
            >>> order_set.update([1, 2, 3])
            >>> order_set
            OrderedSet([0, 1, 2, 3])
            ```

        Arguments:
            iterable: The iterable to update the ordered set with.
        """
        for item in iterable:
            self.add(item)

    extend = update
    """An alias of [`update`][iters.ordered_set.OrderedSet.update]."""

    def index(self, item: Q, start: Optional[int] = None, stop: Optional[int] = None) -> int:
        """Gets the index of an item in the ordered set.

        Complexity:
            `O(1)`.

        Example:
            ```python
            >>> order_set = ordered_set([1, 2, 3])
            >>> order_set.index(1)
            0
            >>> order_set.index(5)
            Traceback (most recent call last):
              ...
            ValueError: 5 is not in the ordered set
            ```

        Arguments:
            item: The item to get the index of.
            start: The index to start searching from.
            stop: The index to stop searching at.

        Raises:
            ValueError: The item is not in the ordered set.

        Returns:
            The index of the item.
        """
        index = self._item_to_index.get(item)

        if index is None:
            raise ValueError(item_not_in_ordered_set(item))

        if start is not None:
            if index < start:
                raise ValueError(item_not_in_ordered_set(item))

        if stop is not None:
            if index >= stop:
                raise ValueError(item_not_in_ordered_set(item))

        return index

    get_index = wrap_index_error(index)
    """An alias of [`index`][iters.ordered_set.OrderedSet.index] wrapped to return
    [`Option[int]`][wraps.option.Option] instead of erroring.
    """

    def count(self, item: Q) -> int:
        """Returns `1` if an item is in the ordered set, `0` otherwise.

        Complexity:
            `O(1)`.

        Arguments:
            item: The item to count.

        Returns:
            `1` if the `item` is in the ordered set, `0` otherwise.
        """
        return int(item in self._item_to_index)

    def pop(self, index: int = LAST) -> Q:
        """Pops an item from the ordered set at `index`.

        Complexity:
            `O(n)`, see [`discard`][iters.ordered_set.OrderedSet.discard].

        Example:
            ```python
            >>> order_set = ordered_set([0, 1])
            >>> order_set.pop()
            1
            >>> order_set.pop(0)
            0
            >>> order_set.pop()
            Traceback (most recent call last):
              ...
            IndexError: list index out of range
            ```

        Arguments:
            index: The index to pop the item from.

        Raises:
            IndexError: The index is out of range.

        Returns:
            The popped item.
        """
        items = self._items

        item = items[index]

        self.discard(item)

        return item

    get_pop = wrap_index_error(pop)
    """An alias of [`pop`][iters.ordered_set.OrderedSet.pop] wrapped to return
    [`Option[Q]`][wraps.option.Option] instead of erroring.
    """

    def discard(self, item: Q) -> None:
        """Discards an item from the ordered set.

        Complexity:
            `O(n)`, where `n` is the length of the ordered set.
            This is because all indices after the removed index must be decremented.

        Example:
            ```python
            >>> order_set = ordered_set([0, 1])
            >>> order_set.discard(1)
            >>> order_set
            OrderedSet([0])
            >>> order_set.discard(1)
            >>> order_set.discard(0)
            >>> order_set
            OrderedSet()
            ```

        Arguments:
            item: The item to discard.
        """
        item_to_index = self._item_to_index

        if item in item_to_index:
            index = item_to_index[item]

            del self._items[index]

            for item_in, index_in in item_to_index.items():
                if index_in >= index:
                    item_to_index[item_in] -= 1

    def remove(self, item: Q) -> None:
        """A checked version of [`discard`][iters.ordered_set.OrderedSet.discard].

        Complexity: `O(n)`, see [`discard`][iters.ordered_set.OrderedSet.discard].

        Example:
            ```python
            >>> order_set = ordered_set([0, 1])
            >>> order_set.remove(1)
            >>> order_set
            OrderedSet([0])
            >>> order_set.remove(1)
            Traceback (most recent call last):
              ...
            ValueError: 1 is not in the ordered set
            >>> order_set.remove(0)
            >>> order_set
            OrderedSet()
            ```

        Arguments:
            item: The item to remove.

        Raises:
            ValueError: The item is not in the ordered set.
        """
        if item in self:
            self.discard(item)

        else:
            raise ValueError(item_not_in_ordered_set(item))

    def insert(self, index: int, item: Q) -> None:
        """Inserts an item into the ordered set at `index`.

        Complexity:
            `O(n)`, where `n` is the length of the ordered set.
            This is because all indices after the inserted index must be incremented.

        Example:
            ```python
            >>> order_set = ordered_set([1, 3])
            >>> order_set.insert(1, 2)
            >>> order_set
            OrderedSet([1, 2, 3])
            ```

        Arguments:
            index: The index to insert the item at.
            item: The item to insert.
        """
        item_to_index = self._item_to_index

        if item in item_to_index:
            return

        items = self._items

        if index < len(items):
            items.insert(index, item)

            for item_in, index_in in item_to_index.items():
                if index_in >= index:
                    item_to_index[item_in] += 1

            item_to_index[item] = index

        else:
            self.append(item)

    def clear(self) -> None:
        """Clears the ordered set.

        Complexity:
            `O(n)`.
        """
        self._items.clear()
        self._item_to_index.clear()

    def __iter__(self) -> Iterator[Q]:
        return iter(self._items)

    def __reversed__(self) -> Iterator[Q]:
        return reversed(self._items)

    def __repr__(self) -> str:
        name = get_type_name(self)

        items = self._items

        if not items:
            return EMPTY_REPRESENTATION.format(name)

        return ITEMS_REPRESENTATION.format(name, items)

    def __eq__(self, other: Any) -> bool:
        try:
            iterator = iter(other)

        except TypeError:
            return False

        if is_sequence(other):
            return self._items == list(iterator)

        return set(self._item_to_index) == set(iterator)

    def apply_union(self, *iterables: Iterable[Q]) -> OrderedSet[Q]:
        """Returns the union of the ordered set and `iterables`.

        Arguments:
            *iterables: The iterables to find the union with.

        Returns:
            The union of the ordered set and `iterables`.
        """
        if iterables:
            return self.create_union(self, *iterables)

        return self.copy()

    union = mixed_method(create_union, apply_union)
    """Mixes [`create_union`][iters.ordered_set.OrderedSet.create_union]
    and [`apply_union`][iters.ordered_set.OrderedSet.apply_union].
    """

    def apply_intersection(self, *iterables: Iterable[Q]) -> OrderedSet[Q]:
        """Returns the intersection of the ordered set and `iterables`.

        Arguments:
            *iterables: The iterables to find the intersection with.

        Returns:
            The intersection of the ordered set and `iterables`.
        """
        if iterables:
            intersection = set.intersection(*map(set, iterables))

            iterator = (item for item in self if item in intersection)

            return self.create_unchecked(iterator)

        return self.copy()

    intersection = mixed_method(create_intersection, apply_intersection)
    """Mixes [`create_intersection`][iters.ordered_set.OrderedSet.create_intersection]
    and [`apply_intersection`][iters.ordered_set.OrderedSet.apply_intersection].
    """

    def intersection_update(self, *iterables: Iterable[Q]) -> None:
        """Updates the ordered set to be the intersection of itself and `iterables`.

        Arguments:
            *iterables: The iterables to find the intersection with.
        """
        if iterables:
            intersection = self.intersection(*iterables)

            self.clear()

            self.update(intersection)

    def apply_difference(self, *iterables: Iterable[Q]) -> OrderedSet[Q]:
        """Returns the difference of the ordered set and `iterables`.

        Arguments:
            *iterables: The iterables to find the difference with.

        Returns:
            The difference of the ordered set and `iterables`.
        """
        if iterables:
            union = set.union(*map(set, iterables))
            iterator = (item for item in self if item not in union)

            return self.create_unchecked(iterator)

        return self.copy()

    difference = mixed_method(create_difference, apply_difference)
    """Mixes [`create_difference`][iters.ordered_set.OrderedSet.create_difference]
    and [`apply_difference`][iters.ordered_set.OrderedSet.apply_difference].
    """

    def difference_update(self, *iterables: Iterable[Q]) -> None:
        """Updates the ordered set to be the difference of itself and `iterables`.

        Arguments:
            *iterables: The iterables to find the difference with.
        """
        if iterables:
            difference = self.difference(*iterables)

            self.clear()

            self.update(difference)

    def single_symmetric_difference(self, other: Iterable[Q]) -> OrderedSet[Q]:
        ordered = self.create(other)

        return self.difference(ordered).union(ordered.difference(self))

    def apply_symmetric_difference(self, *iterables: Iterable[Q]) -> OrderedSet[Q]:
        """Returns the symmetric difference of the ordered set and `iterables`.

        Arguments:
            *iterables: The iterables to find the symmetric difference with.

        Returns:
            The symmetric difference of the ordered set and `iterables`.
        """
        if iterables:
            result = self

            for iterable in iterables:
                result = result.single_symmetric_difference(iterable)

            return result

        return self.copy()

    symmetric_difference = mixed_method(create_symmetric_difference, apply_symmetric_difference)
    """Mixes
    [`create_symmetric_difference`][iters.ordered_set.OrderedSet.create_symmetric_difference] and
    [`apply_symmetric_difference`][iters.ordered_set.OrderedSet.apply_symmetric_difference].
    """

    def symmetric_difference_update(self, *iterables: Iterable[Q]) -> None:
        """Updates the ordered set to be the symmetric difference of itself and `iterables`.

        Arguments:
            *iterables: The iterables to find the symmetric difference with.
        """
        if iterables:
            symmetric_difference = self.symmetric_difference(*iterables)

            self.clear()

            self.update(symmetric_difference)

    def is_subset(self, other: Iterable[Q]) -> bool:
        """Checks if the ordered set is a subset of `other`.

        Arguments:
            other: The iterable to check if the ordered set is a subset of.

        Returns:
            Whether the ordered set is a subset of `other`.
        """
        if is_sized(other):  # cover obvious cases
            if len(self) > len(other):
                return False

        if is_any_set(other):  # speedup for sets
            return all(item in other for item in self)

        other_set = set(other)

        return len(self) <= len(other_set) and all(item in other_set for item in self)

    def is_strict_subset(self, other: Iterable[Q]) -> bool:
        """Checks if the ordered set is a strict subset of `other`.

        Arguments:
            other: The iterable to check if the ordered set is a strict subset of.

        Returns:
            Whether the ordered set is a strict subset of `other`.
        """
        if is_sized(other):  # cover obvious cases
            if len(self) >= len(other):
                return False

        if is_any_set(other):  # speedup for sets
            return all(item in other for item in self)

        other_set = set(other)  # default case

        return len(self) < len(other_set) and all(item in other_set for item in self)

    def is_superset(self, other: Iterable[Q]) -> bool:
        """Checks if the ordered set is a superset of `other`.

        Arguments:
            other: The iterable to check if the ordered set is a superset of.

        Returns:
            Whether the ordered set is a superset of `other`.
        """
        if is_sized(other):  # speedup for sized iterables
            return len(self) >= len(other) and all(item in self for item in other)

        return all(item in self for item in other)  # default case

    def is_strict_superset(self, other: Iterable[Q]) -> bool:
        """Checks if the ordered set is a strict superset of `other`.

        Arguments:
            other: The iterable to check if the ordered set is a strict superset of.

        Returns:
            Whether the ordered set is a strict superset of `other`.
        """
        if is_sized(other):  # speedup for sized iterables
            return len(self) > len(other) and all(item in self for item in other)

        array = list(other)  # default case

        return len(self) > len(array) and all(item in self for item in array)

    def is_disjoint(self, other: Iterable[Q]) -> bool:
        """Checks if the ordered set is disjoint with `other`.

        Arguments:
            other: The iterable to check if the ordered set is disjoint with.

        Returns:
            Whether the ordered set is disjoint with `other`.
        """
        return none(item in self for item in other)

    # I honestly hate these names ~ nekit

    issubset = is_subset
    issuperset = is_superset
    isdisjoint = is_disjoint


ordered_set = OrderedSet
"""An alias of [`OrderedSet`][iters.ordered_set.OrderedSet]."""
ordered_set_unchecked = ordered_set.create_unchecked
"""An alias of [`ordered_set.create_unchecked`][iters.ordered_set.OrderedSet.create_unchecked]."""

from iters.utils import chain, none
