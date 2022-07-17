from iters.types import Marker, NoDefault, Singleton


def test_singleton_identity() -> None:
    assert Singleton() is Singleton()
    assert Marker() is Marker()
    assert NoDefault() is NoDefault()
