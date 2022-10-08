# Changelog

<!-- changelogging: start -->

## 0.4.0 (2022-10-08)

### Changes

- The following functions have been changed:
  - `async_iter` is now an alias of `AsyncIter`;
  - `iter` is now an alias of `Iter`;
  - `reversed` is now an alias of `iter.reversed`.

## 0.3.0 (2022-08-17)

### Changes

- Change functions of various arity returning `Awaitable[T]` to async functions returning `T`.
  ([#15](https://github.com/nekitdev/iters/pull/15))

## 0.2.0 (2022-08-15)

### Changes

- Add `await async_iter`, equivalent to `await async_iter.list()`.

## 0.1.0 (2022-08-01)

Initial release.
