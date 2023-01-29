# Changelog

<!-- changelogging: start -->

## 0.11.0 (2023-01-29)

### Internal

- `async-extensions` is now used instead of reimplementing `collect_iterable` functionality.

## 0.10.0 (2023-01-08)

### Internal

- Marked the internals of the `OrderedSet[Q]` private.

## 0.9.0 (2023-01-07)

### Features

- Added `collect_iter` method for `AsyncIter[T]` and `Iter[T]`.

## 0.8.0 (2022-12-22)

### Features

- Added `into_iter` method for `AsyncIter[T]`.
- Added `into_async_iter` method for `Iter[T]`.

## 0.7.0 (2022-12-20)

### Features

- Added `OrderedSet[Q]` type within the `iters.ordered_set` module.
- Added `ordered_set` method to `Iter[T]` and `AsyncIter[T]`.

## 0.6.0 (2022-11-08)

### Internal

- Migrated to using [`named`](https://github.com/nekitdev/named) and
  [`solus`](https://github.com/nekitdev/solus) packages instead of
  reimplementing their functionality. ([#18](https://github.com/nekitdev/iters/pull/18))

## 0.5.0 (2022-10-11)

### Changes

- Functions taking `Predicate[T]` have been updated to accept `Optional[Predicate[T]]`.
  Passing `None` as an argument is identical to passing `bool`.

  There are three functions which do not accept `None`, though:
  - `drop_while`
  - `skip_while`
  - `take_while`

  This choice is motivated by the fact that it does not make much sense to `do_while(None)`.

## 0.4.0 (2022-10-08)

### Changes

- The following functions have been changed:
  - `async_iter` is now an alias of `AsyncIter`;
  - `iter` is now an alias of `Iter`;
  - `reversed` is now an alias of `iter.reversed`.

## 0.3.0 (2022-08-17)

### Changes

- Changed functions of various arity returning `Awaitable[T]` to async functions returning `T`.
  ([#15](https://github.com/nekitdev/iters/pull/15))

## 0.2.0 (2022-08-15)

### Changes

- Added `await async_iter`, equivalent to `await async_iter.list()`.

## 0.1.0 (2022-08-01)

Initial release.
