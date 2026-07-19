"""Unit tests for the bench model registry — the id -> ModelSpec map the runner
and `--list-models` read.

Pure: availability is an import probe (never constructs a CUDA model), so these
run with no GPU and no model download. Prior art: fake factory in
`tests/test_lazy_engine.py`.
"""

import pytest

from susurro.bench import registry


def test_registry_lists_faster_whisper_baseline_first():
    # Order matters: the report puts the baseline (turbo) first, so the registry is
    # the ordered source of that ordering. Later backends append after the fw pair.
    assert list(registry.REGISTRY)[:2] == ["fw-large-v3-turbo", "fw-large-v3"]


def test_faster_whisper_specs_are_available_when_importable():
    # faster_whisper is a core dep, so both fw contenders report available here.
    assert registry.REGISTRY["fw-large-v3-turbo"].available() is True
    assert registry.REGISTRY["fw-large-v3"].available() is True


def test_availability_is_an_import_probe_not_a_model_build(monkeypatch):
    # available() must never construct a model (no CUDA, no download): flipping the
    # import probe to "not importable" flips availability, proving it's cheap.
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda _name: None)
    assert registry.REGISTRY["fw-large-v3-turbo"].available() is False


def test_resolve_none_returns_every_spec_in_registry_order():
    assert [s.id for s in registry.resolve(None)] == list(registry.REGISTRY)


def test_whispercpp_backend_is_registered_and_import_probed():
    # The optional whisper.cpp backend must be a registry slot whose availability is
    # a pywhispercpp import probe: absent runtime -> unavailable (skipped, not a crash).
    spec = registry.REGISTRY["whispercpp-turbo"]
    assert "whisper.cpp" in spec.label.lower() or "whispercpp" in spec.label.lower()

    import importlib.util

    real = importlib.util.find_spec

    def fake_find_spec(name):
        return None if name == "pywhispercpp" else real(name)

    # unavailable when pywhispercpp can't be imported ...
    registry.importlib.util.find_spec = fake_find_spec
    try:
        assert spec.available() is False
    finally:
        registry.importlib.util.find_spec = real
    # ... and available once the import probe would succeed.
    registry.importlib.util.find_spec = lambda _name: object()
    try:
        assert spec.available() is True
    finally:
        registry.importlib.util.find_spec = real


def test_resolve_selects_only_the_requested_ids_preserving_order():
    specs = registry.resolve(["fw-large-v3", "fw-large-v3-turbo"])
    # resolve keeps registry order, not the argument order.
    assert [s.id for s in specs] == ["fw-large-v3-turbo", "fw-large-v3"]


def test_resolve_unknown_id_raises_with_the_known_ids():
    with pytest.raises(KeyError) as exc:
        registry.resolve(["fw-large-v3-turbo", "nope"])
    assert "nope" in str(exc.value)
    assert "fw-large-v3-turbo" in str(exc.value)  # lists what's valid
