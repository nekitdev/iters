# `iters`

[![License][License Badge]][License]
[![Version][Version Badge]][Package]
[![Downloads][Downloads Badge]][Package]
[![Discord][Discord Badge]][Discord]

[![Documentation][Documentation Badge]][Documentation]
[![Check][Check Badge]][Actions]
[![Test][Test Badge]][Actions]
[![Coverage][Coverage Badge]][Coverage]

> *Composable external iteration.*

If you have found yourself with a *collection* of some kind, and needed to perform
an operation on the elements of said collection, you will quickly run into *iterators*.
Iterators are heavily used in idiomatic Python code, so becoming familiar with them is essential.

## Installing

**Python 3.7 or above is required.**

### pip

Installing the library with `pip` is quite simple:

```console
$ pip install iters
```

Alternatively, the library can be installed from source:

```console
$ git clone https://github.com/nekitdev/iters.git
$ cd iters
$ python -m pip install .
```

### poetry

You can add `iters` as a dependency with the following command:

```console
$ poetry add iters
```

Or by directly specifying it in the configuration like so:

```toml
[tool.poetry.dependencies]
iters = "^0.4.0"
```

Alternatively, you can add it directly from the source:

```toml
[tool.poetry.dependencies.iters]
git = "https://github.com/nekitdev/iters.git"
```

## Examples

### Simple

Squaring only even numbers in some sequence:

```python
from iters import iter


def is_even(value: int) -> bool:
    return not value % 2


def square(value: int) -> int:
    return value * value


numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

result = iter(numbers).filter(is_even).map(square).list()

print(result)  # [0, 4, 16, 36, 64]
```

### Asynchronous

Asynchronous iteration is fully supported by `iters`, and its API is similar to its
synchronous counterpart.

## Documentation

You can find the documentation [here][Documentation].

## Support

If you need support with the library, you can send an [email][Email]
or refer to the official [Discord server][Discord].

## Changelog

You can find the changelog [here][Changelog].

## Security Policy

You can find the Security Policy of `iters` [here][Security].

## Contributing

If you are interested in contributing to `iters`, make sure to take a look at the
[Contributing Guide][Contributing Guide], as well as the [Code of Conduct][Code of Conduct].

## License

`iters` is licensed under the MIT License terms. See [License][License] for details.

[Email]: mailto:support@nekit.dev

[Discord]: https://nekit.dev/discord

[Actions]: https://github.com/nekitdev/iters/actions

[Changelog]: https://github.com/nekitdev/iters/blob/main/CHANGELOG.md
[Code of Conduct]: https://github.com/nekitdev/iters/blob/main/CODE_OF_CONDUCT.md
[Contributing Guide]: https://github.com/nekitdev/iters/blob/main/CONTRIBUTING.md
[Security]: https://github.com/nekitdev/iters/blob/main/SECURITY.md

[License]: https://github.com/nekitdev/iters/blob/main/LICENSE

[Package]: https://pypi.org/project/iters
[Coverage]: https://codecov.io/gh/nekitdev/iters
[Documentation]: https://nekitdev.github.io/iters

[Discord Badge]: https://img.shields.io/badge/chat-discord-5865f2
[License Badge]: https://img.shields.io/pypi/l/iters
[Version Badge]: https://img.shields.io/pypi/v/iters
[Downloads Badge]: https://img.shields.io/pypi/dm/iters

[Documentation Badge]: https://github.com/nekitdev/iters/workflows/docs/badge.svg
[Check Badge]: https://github.com/nekitdev/iters/workflows/check/badge.svg
[Test Badge]: https://github.com/nekitdev/iters/workflows/test/badge.svg
[Coverage Badge]: https://codecov.io/gh/nekitdev/iters/branch/main/graph/badge.svg
