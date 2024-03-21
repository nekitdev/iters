from iters.state import stateful


def test_stateful() -> None:
    value = 13
    other = 42

    state = stateful(value)

    assert state.get() is value
    assert state.set(other).get() is other
    assert state.get() is other
