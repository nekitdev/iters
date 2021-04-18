iters.py
========

.. image:: https://img.shields.io/pypi/l/iters.py.svg
    :target: https://opensource.org/licenses/MIT
    :alt: Project License

.. image:: https://img.shields.io/pypi/v/iters.py.svg
    :target: https://pypi.python.org/pypi/iters.py
    :alt: Library Version

.. image:: https://img.shields.io/pypi/pyversions/iters.py.svg
    :target: https://pypi.python.org/pypi/iters.py
    :alt: Required Python Versions

.. image:: https://img.shields.io/pypi/status/iters.py.svg
    :target: https://github.com/nekitdev/iters.py
    :alt: Development Status

.. image:: https://img.shields.io/pypi/dw/iters.py.svg
    :target: https://pypi.python.org/pypi/iters.py
    :alt: Library Downloads / Week

.. image:: https://app.codacy.com/project/badge/Grade/62e846851c28459d8b59a541db8dd68b
    :target: https://www.codacy.com/gh/nekitdev/iters.py/dashboard
    :alt: Code Quality

iters.py is a module that implements rich iterators for Python.

Key Features
------------

- Iterators that allow chaining several methods
- Large support for many functions, both synchronous and asynchronous
- Library is completely typed, meaning many errors can be caught even before running programs

Installing
----------

**Python 3.6 or higher is required**

To install the library, you can just run the following command:

.. code:: sh

    # Linux / OS X
    python3 -m pip install --upgrade iters.py

    # Windows
    py -3 -m pip install --upgrade iters.py

Or to install it from source:

.. code:: sh

    $ git clone https://github.com/nekitdev/iters.py
    $ cd iters.py
    $ python3 -m pip install --upgrade .

Examples
--------

Using iterators and their methods:

.. code:: python

    from iters import iter

    print(
        iter(range(10))  # [0; 9] range
        .filter(lambda x: x % 2)  # leave odd numbers only
        .map(lambda x: x * x)  # square odd numbers
        .sum()  # compute the sum
    )

    # OUTPUT: 165

Equivalent implementation in pure Python code:

.. code:: python

    # leave odd numbers only -> square them -> compute the sum
    sum(
        map(
            lambda x: x * x, filter(
                lambda x: x % 2, range(10)
            )
        )
    )

Asynchronous iterators are not very different from normal iterators:

.. code:: python

    from iters import async_iter

    array = [[[[[1], 2], 3], 4], 5]

    collapsed = await async_iter(array).collapse().list()  # async is really similar

    print(collapsed)

    # OUTPUT: [1, 2, 3, 4, 5]

Typing and Type Inference
-------------------------

As the library is fully typed, different operations with iterators follow types quite closely.

Note that ``Iter[T]`` is covariant over ``T``, and so is ``AsyncIter[T]``
as they are derived from ``Iterator[T]`` and ``AsyncIterator[T]`` respectively.

Either way, here is one example of typing usage:

.. code:: python

    from iters import iter

    # some arbitrary sequence of items
    items = [0, 1, 2, 3, 4, 5]  # List[int]

    # create an iterator over items
    iterator = iter(items)  # Iter[int]

    # convert into groups like (0, 1), (2, 3), (4, 5)
    grouping = iterator.group(2)  # Iter[Tuple[int, int]]

    # finally, collect groups into mapping
    mapping = grouping.dict()  # Dict[int, int]

    for key, value in mapping.items():
        # key: int
        # value: int

        print(key + value)  # perfectly valid!

And another one:

.. code:: python

    from typing import TypeVar

    from iters import iter

    N = TypeVar("N", int, float, complex)  # some number

    Z = 1  # int
    C = 1  # int


    def function(z: N, c: N = C) -> N:
        return z * z + c


    # Z_1 = function(Z), Z_2 = function(Z_1), ...
    iterator = iter.iterate(function, Z)  # Iter[int]

    # take several results from the beginning
    numbers = iterator.take(5)  # Iter[int]

    # convert them to strings so they can be joined later
    strings = numbers.map(str)  # Iter[str]

    # create the string to display
    display = strings.join(" -> ")  # str

    print(display)

Authors
-------

This project is mainly developed by `nekitdev <https://github.com/nekitdev>`_.
