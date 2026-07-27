"""Unit tests for the config loader (`susurro.config`).

The loader is pure but for one file read, so it exercises fully against files
written into `tmp_path` — never the real repo-local config. We lock the default values, the
defaults<file<CLI precedence surface (parse + override), the fail-loud validation
(wrong type / out of range / bad enum / malformed TOML), and the unknown-key
warnings.
"""

from pathlib import Path

import pytest

from susurro.config import (
    AudioConfig,
    Config,
    ConfigError,
    DaemonConfig,
    default_config_path,
    load_config,
)


def _write(path, text):
    path.write_text(text)
    return path


# --- defaults --------------------------------------------------------------


def test_missing_default_file_yields_all_defaults(tmp_path, monkeypatch):
    # Point the default at a path that doesn't exist so we exercise the missing-file
    # branch (not the real repo-local config.toml).
    monkeypatch.setattr("susurro.config.default_config_path", lambda: tmp_path / "config.toml")
    cfg = load_config()
    assert cfg == Config()


def test_empty_file_yields_all_defaults(tmp_path):
    cfg = load_config(_write(tmp_path / "c.toml", ""))
    assert cfg == Config()


def test_default_values_match_the_current_magic_numbers():
    cfg = Config()
    assert cfg.engine.model == "large-v3-turbo"
    assert cfg.engine.compute_type == "int8"
    assert cfg.engine.device == "cuda"
    assert cfg.engine.language == "en"
    assert cfg.engine.beam_size == 5
    assert cfg.engine.vad_filter is True
    assert cfg.audio.sample_rate == 16_000
    assert cfg.audio.channels == 1
    assert cfg.audio.device is None
    assert cfg.daemon.max_record_s == 60.0
    assert cfg.daemon.idle_timeout_s == 300.0
    assert cfg.daemon.notify is True
    assert cfg.notify.timeout_s == 5.0


# --- parse + override ------------------------------------------------------

_FULL = """
[engine]
model = "medium"
compute_type = "float16"
device = "cpu"
language = "pt"
beam_size = 1
vad_filter = false

[audio]
sample_rate = 48000
channels = 2
device = "USB mic"

[daemon]
max_record_s = 90.0
idle_timeout_s = 0
notify = false

[notify]
timeout_s = 10.0
"""


def test_full_file_maps_every_value(tmp_path):
    cfg = load_config(_write(tmp_path / "c.toml", _FULL))
    assert cfg.engine == type(cfg.engine)(
        model="medium",
        compute_type="float16",
        device="cpu",
        language="pt",
        beam_size=1,
        vad_filter=False,
    )
    assert cfg.audio.sample_rate == 48000
    assert cfg.audio.channels == 2
    assert cfg.audio.device == "USB mic"
    assert cfg.daemon.max_record_s == 90.0
    assert cfg.daemon.idle_timeout_s == 0  # <=0 is allowed (disables idle-unload)
    assert cfg.daemon.notify is False
    assert cfg.notify.timeout_s == 10.0


def test_partial_file_overrides_only_named_values(tmp_path):
    cfg = load_config(_write(tmp_path / "c.toml", '[engine]\nlanguage = "pt"\n'))
    assert cfg.engine.language == "pt"  # overridden
    assert cfg.engine.model == "large-v3-turbo"  # untouched -> default
    assert cfg.audio == AudioConfig()  # whole table absent -> defaults
    assert cfg.daemon == DaemonConfig()


def test_audio_device_accepts_integer_index(tmp_path):
    cfg = load_config(_write(tmp_path / "c.toml", "[audio]\ndevice = 4\n"))
    assert cfg.audio.device == 4


def test_float_field_accepts_bare_integer(tmp_path):
    cfg = load_config(_write(tmp_path / "c.toml", "[daemon]\nmax_record_s = 45\n"))
    assert cfg.daemon.max_record_s == 45.0
    assert isinstance(cfg.daemon.max_record_s, float)


def test_idle_timeout_accepts_zero_and_negative(tmp_path):
    # <=0 is the documented "disable idle-unload" sentinel, not an error.
    cfg = load_config(_write(tmp_path / "c.toml", "[daemon]\nidle_timeout_s = -5\n"))
    assert cfg.daemon.idle_timeout_s == -5.0


# --- fail loud -------------------------------------------------------------


def test_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_malformed_toml_is_an_error(tmp_path):
    p = _write(tmp_path / "c.toml", "this is not = = valid toml")
    with pytest.raises(ConfigError, match="c.toml"):  # path surfaced in the message
        load_config(p)


def test_wrong_type_is_an_error(tmp_path):
    p = _write(tmp_path / "c.toml", '[engine]\nbeam_size = "five"\n')
    with pytest.raises(ConfigError, match="beam_size"):
        load_config(p)


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_beam_size_must_be_a_positive_int(tmp_path, value):
    p = _write(tmp_path / "c.toml", f"[engine]\nbeam_size = {value}\n")
    with pytest.raises(ConfigError, match="beam_size"):
        load_config(p)


def test_bool_is_rejected_where_int_expected(tmp_path):
    # TOML `true` is a bool; bool is a subclass of int, so guard it explicitly.
    p = _write(tmp_path / "c.toml", "[engine]\nbeam_size = true\n")
    with pytest.raises(ConfigError, match="beam_size"):
        load_config(p)


def test_int_is_rejected_where_bool_expected(tmp_path):
    p = _write(tmp_path / "c.toml", "[engine]\nvad_filter = 1\n")
    with pytest.raises(ConfigError, match="vad_filter"):
        load_config(p)


def test_engine_device_must_be_cuda_or_cpu(tmp_path):
    p = _write(tmp_path / "c.toml", '[engine]\ndevice = "rocm"\n')
    with pytest.raises(ConfigError, match="device"):
        load_config(p)


def test_positive_float_rejects_zero(tmp_path):
    p = _write(tmp_path / "c.toml", "[daemon]\nmax_record_s = 0\n")
    with pytest.raises(ConfigError, match="max_record_s"):
        load_config(p)


# --- fail loud: non-finite floats ------------------------------------------
#
# `nan`, `inf` and `-inf` are legal TOML floats, and every `<= 0` range check is
# False for all three — so without an explicit finiteness test they'd sail through
# as ordinary values (`max_record_s = nan` silently ends every recording instantly;
# `inf` blows up the daemon's `settimeout`).


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "+inf", "1e400"])
def test_max_record_rejects_non_finite(tmp_path, value):
    p = _write(tmp_path / "c.toml", f"[daemon]\nmax_record_s = {value}\n")
    with pytest.raises(ConfigError, match="max_record_s must be a finite positive number"):
        load_config(p)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_notify_timeout_rejects_non_finite(tmp_path, value):
    p = _write(tmp_path / "c.toml", f"[notify]\ntimeout_s = {value}\n")
    with pytest.raises(ConfigError, match="timeout_s must be a finite positive number"):
        load_config(p)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_idle_timeout_rejects_non_finite(tmp_path, value):
    # `_number` accepts <=0 as the idle-unload-off sentinel, but not nan/inf.
    p = _write(tmp_path / "c.toml", f"[daemon]\nidle_timeout_s = {value}\n")
    with pytest.raises(ConfigError, match="idle_timeout_s must be a finite number"):
        load_config(p)


@pytest.mark.parametrize("value", ["0", "-1", "-0.5", "300.0", "1e6"])
def test_idle_timeout_still_accepts_every_finite_value(tmp_path, value):
    cfg = load_config(_write(tmp_path / "c.toml", f"[daemon]\nidle_timeout_s = {value}\n"))
    assert cfg.daemon.idle_timeout_s == float(value)


def test_non_finite_error_names_the_offending_value(tmp_path):
    # The section builder's table/key/source context must survive the new branch.
    p = _write(tmp_path / "c.toml", "[daemon]\nmax_record_s = inf\n")
    with pytest.raises(ConfigError, match=r"c\.toml: \[daemon\] max_record_s .* \(got inf\)"):
        load_config(p)


def test_audio_device_rejects_float(tmp_path):
    p = _write(tmp_path / "c.toml", "[audio]\ndevice = 1.5\n")
    with pytest.raises(ConfigError, match="device"):
        load_config(p)


def test_section_must_be_a_table(tmp_path):
    p = _write(tmp_path / "c.toml", "engine = 5\n")
    with pytest.raises(ConfigError, match="engine"):
        load_config(p)


# --- forward-compatible: unknown keys/tables warn, don't fail ---------------


def test_unknown_key_warns_and_is_ignored(tmp_path, capsys):
    p = _write(tmp_path / "c.toml", '[engine]\nfuture_knob = 1\nlanguage = "pt"\n')
    cfg = load_config(p)
    assert cfg.engine.language == "pt"  # the known key still applied
    assert cfg.engine.model == "large-v3-turbo"  # rest defaulted
    assert "future_knob" in capsys.readouterr().err


def test_unknown_table_warns_and_is_ignored(tmp_path, capsys):
    p = _write(tmp_path / "c.toml", "[experimental]\nx = 1\n")
    cfg = load_config(p)
    assert cfg == Config()  # nothing usable -> all defaults
    assert "experimental" in capsys.readouterr().err


# --- default path ----------------------------------------------------------


def test_default_config_path_is_repo_local():
    # Derived from the test file's own location (tests/ -> repo root), independently
    # of how the loader computes it (src/susurro/ -> repo root).
    repo_root = Path(__file__).resolve().parent.parent
    assert default_config_path() == repo_root / "config.toml"


def test_unreadable_path_is_a_clean_config_error(tmp_path):
    # A directory (or otherwise unreadable file) must fail loud as ConfigError, not
    # leak an OSError past the caller's `except ConfigError`.
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(ConfigError):
        load_config(d)
