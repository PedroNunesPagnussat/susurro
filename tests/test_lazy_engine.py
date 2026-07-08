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
        self.language = "en"  # matches Engine's default

    def set_language(self, code):
        self.language = code

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


# --- load (build-only preload) --------------------------------------------

def test_load_builds_inner_engine_once():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    assert engine.loaded is False

    engine.load()
    assert engine.loaded is True
    assert len(builds) == 1  # built exactly once


def test_load_when_already_loaded_is_noop():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.load()
    engine.load()  # already built — must not rebuild
    assert len(builds) == 1


def test_load_applies_remembered_language_to_fresh_engine():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory, language="pt")
    engine.load()
    assert builds[0].language == "pt"  # remembered language reached the fresh engine


def test_load_then_transcribe_does_not_rebuild():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.load()
    assert engine.transcribe(AUDIO) == "hi"  # reuses the preloaded engine
    assert len(builds) == 1  # no second build on transcribe


# --- language -------------------------------------------------------------

def test_ctor_language_is_applied_on_first_load():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory, language="pt")
    engine.transcribe(AUDIO)  # first load
    assert builds[0].language == "pt"  # ctor language reached the fresh engine


def test_set_language_applies_to_a_loaded_engine():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.transcribe(AUDIO)  # load
    engine.set_language("pt")
    assert builds[0].language == "pt"


def test_set_language_before_load_is_applied_on_first_load():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.set_language("pt")  # no engine yet — just remembered
    assert builds == []
    engine.transcribe(AUDIO)
    assert builds[0].language == "pt"


def test_language_is_reapplied_after_unload_reload():
    factory, builds = _factory_spy()
    engine = LazyEngine(factory)
    engine.transcribe(AUDIO)
    engine.set_language("pt")
    engine.unload()

    engine.transcribe(AUDIO)  # rebuilds a fresh engine
    assert len(builds) == 2
    assert builds[1].language == "pt"  # remembered language survived the reload
