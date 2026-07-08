"""Unit tests for the no-daemon mic test's config wiring (`susurro.__main__`).

The loop itself needs a mic + model, but the CLI-over-config resolver is pure, so
we lock the precedence (flag > config > default, `--cpu` forces CPU) here.
"""

import argparse

from susurro.__main__ import _apply_cli
from susurro.config import Config, EngineConfig


def _args(**over):
    base = dict(model=None, device=None, cpu=False, lang=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_no_flags_leaves_config_untouched():
    config = Config(engine=EngineConfig(model="tiny", language="pt"))
    out = _apply_cli(config, _args())
    assert out.engine.model == "tiny"
    assert out.engine.language == "pt"


def test_flags_override_config():
    config = Config(engine=EngineConfig(model="tiny", language="en"))
    out = _apply_cli(config, _args(model="medium", lang="pt", device=2))
    assert out.engine.model == "medium"
    assert out.engine.language == "pt"
    assert out.audio.device == 2


def test_cpu_flag_forces_cpu():
    config = Config(engine=EngineConfig(device="cuda"))
    assert _apply_cli(config, _args(cpu=True)).engine.device == "cpu"
