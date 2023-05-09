# Predicates

`iters` defines all `predicate` arguments as [`Optional[Predicate[T]]`][typing.Optional]
where `T` is the item type of the iterable.

Passing [`None`][None] as the predicate argument is equivalent to passing [`bool`][bool],
though most functions are optimized to avoid the overhead of function calls to it.
