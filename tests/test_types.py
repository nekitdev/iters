from iters.types import Marker, NoDefault


def test_identity() -> None:
    assert Marker() is Marker()
    assert NoDefault() is NoDefault()
