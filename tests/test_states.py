from iters.states import state


def test_states() -> None:
    value = 13
    other = 42

    test_state = state(value)

    assert test_state.get() is value
    assert test_state.set(other).get() is other
    assert test_state.get() is other
