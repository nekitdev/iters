iters.py
========

.. image:: https://img.shields.io/pypi/l/iters.py.svg
    :target: https://opensource.org/licenses/MIT
    :alt: Project License

.. image:: https://img.shields.io/pypi/v/iters.py.svg
    :target: https://pypi.python.org/pypi/iters.py
    :alt: PyPI Library Version

.. image:: https://img.shields.io/pypi/pyversions/iters.py.svg
    :target: https://pypi.python.org/pypi/iters.py
    :alt: Required Python Versions

.. image:: https://img.shields.io/pypi/status/iters.py.svg
    :target: https://github.com/nekitdev/iters.py
    :alt: Project Development Status

.. image:: https://img.shields.io/pypi/dm/iters.py.svg
    :target: https://pypi.python.org/pypi/iters.py
    :alt: Library Downloads/Month

iters.py is a module that implements rich iterators for Python.

Key Features
------------

- Iterators that allow chaining several methods
- Large support for many functions, both synchronous and asynchronous

Installing
----------

**Python 3.6 or higher is required**

To install the library, you can just run the following command:

.. code:: sh

    # Linux/OS X
    python3 -m pip install -U iters.py

    # Windows
    py -3 -m pip install -U iters.py

Or to install it from source:

.. code:: sh

    $ git clone https://github.com/nekitdev/iters.py
    $ cd iters.py
    $ python3 -m pip install .

Examples
--------

.. code:: python

    from iters import iter

    print(
        iter(range(10))  # [0; 9] range
        .filter(lambda x: x % 2)  # leave odd numbers only
        .map(lambda x: x * x)  # square odd numbers
        .sum()  # compute the sum
    )

    # OUTPUT: 165

.. code:: python

    from iters import async_iter

    array = [[[[[1], 2], 3], 4], 5]

    collapsed = await async_iter(array).collapse().list()  # async is really similar

    print(collapsed)

    # OUTPUT: [1, 2, 3, 4, 5]

Authors
-------

This project is mainly developed by `nekitdev <https://github.com/nekitdev>`_.
