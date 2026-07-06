"""Unit tests for `engine.LazyEngine` — the load/unload seam behind idle-unload.

No CUDA or real model here: a fake factory stands in for `Engine`, so this
exercises the lazy build / drop / rebuild lifecycle in isolation.
"""

import numpy as np

from susurro.engine import LazyEngine


class FakeInner:
    """Stand-in for a warm `Engine`: counts transcribe calls, returns canned text."""

    def __init__(self, text="hi"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.text


def _factory_spy():
    """Returns (factory, builds) where builds collects each FakeInner produced."""
    builds: list[FakeInner] = []

    def factory():
        inner = FakeInner()
        builds.append(inner)
        return inner

    return factory, builds


AUDIO = np.zeros(4, dtype=np.float32)


def test_starts_unloaded_and_loads_on_first_transcribe():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    assert engine.loaded is False
    assert builds == []  # factory not called until needed

    assert engine.transcribe(AUDIO) == "hi"
    assert engine.loaded is True
    assert len(builds) == 1  # built exactly once


def test_reuses_inner_across_transcribes():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.transcribe(AUDIO)
    engine.transcribe(AUDIO)
    assert len(builds) == 1  # not rebuilt
    assert builds[0].calls == 2


def test_unload_drops_inner_and_next_transcribe_rebuilds():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.transcribe(AUDIO)
    assert engine.loaded is True

    engine.unload()
    assert engine.loaded is False

    engine.transcribe(AUDIO)  # pays the reload
    assert len(builds) == 2  # a fresh inner was built
    assert builds[1] is not builds[0]


def test_unload_when_already_unloaded_is_noop():
    factory, _builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.unload()  # never loaded
    assert engine.loaded is False
