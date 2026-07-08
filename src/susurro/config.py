"""Single source of truth for tunables: one repo-local `config.toml`.

Replaces the magic numbers + per-module CLI defaults scattered across the code.
`load_config()` reads the file once into a frozen `Config` (nested dataclasses
mirroring the TOML tables); each CLI then lets flags override the loaded values
(defaults < config file < CLI flag).

Fail-loud by design: a malformed file or an invalid value (wrong type, out of
range, bad enum) raises `ConfigError` rather than silently running on a default,
so a typo in the config is caught at startup instead of surprising you later.
Unknown keys/tables only warn (forward-compatible with older configs). A missing
default file is fine (all defaults); a missing *explicitly requested* file errors.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    """A config file that exists but can't be used: bad TOML or an invalid value."""


class _Invalid(Exception):
    """Internal: a validator's complaint (the "must be …" clause), wrapped into a
    `ConfigError` with the table/key/source context by the section builder."""


@dataclass(frozen=True)
class EngineConfig:
    model: str = "large-v3-turbo"
    compute_type: str = "int8"
    device: str = "cuda"  # "cuda" | "cpu"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    device: int | str | None = None  # input device index or name substring


@dataclass(frozen=True)
class DaemonConfig:
    max_record_s: float = 60.0
    idle_timeout_s: float = 300.0  # <=0 disables idle-unload
    notify: bool = True


@dataclass(frozen=True)
class NotifyConfig:
    timeout_s: float = 5.0  # notify-send subprocess backstop (not a latency knob)


@dataclass(frozen=True)
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)


def pick(cli: object, cfg: object) -> object:
    """CLI-over-config helper: the flag wins when it was passed (not None), else the
    config value. The shared primitive behind each CLI's flag-override step."""
    return cli if cli is not None else cfg


def default_config_path() -> Path:
    """The repo-local `config.toml` at the project root, resolved relative to this
    source file so it's found regardless of the working directory the daemon was
    launched from (Hyprland execs it from an arbitrary cwd). Repo-local for now —
    deliberately not `~/.config`."""
    # src/susurro/config.py -> parents[2] is the repo root.
    return Path(__file__).resolve().parents[2] / "config.toml"


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path`, or the default location when `path` is None.

    A missing default file yields all defaults; a missing *explicit* `path` is an
    error (you asked for a file that isn't there). Malformed TOML or an invalid
    value raises `ConfigError`; unknown keys/tables warn and are ignored.
    """
    explicit = path is not None
    path = Path(path) if explicit else default_config_path()
    if not path.exists():
        if explicit:
            raise ConfigError(f"config file not found: {path}")
        return Config()

    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:  # a directory, a permission error, a broken symlink…
        raise ConfigError(f"{path}: cannot read config: {exc}") from exc

    return _build_config(data, path)


# --- validators: value -> coerced value, or raise _Invalid("must be …") -----

def _str(v: object) -> str:
    if not isinstance(v, str):
        raise _Invalid("must be a string")
    return v


def _bool(v: object) -> bool:
    if not isinstance(v, bool):
        raise _Invalid("must be true or false")
    return v


def _pos_int(v: object) -> int:
    # bool is a subclass of int, so reject it before the int check.
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise _Invalid("must be a positive integer")
    return v


def _pos_float(v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise _Invalid("must be a positive number")
    return float(v)


def _number(v: object) -> float:
    # Any real number: used where <=0 is a meaningful sentinel (idle-unload off).
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _Invalid("must be a number")
    return float(v)


def _input_device(v: object) -> int | str:
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise _Invalid("must be an integer index or a string name")
    return v


def _compute_device(v: object) -> str:
    if v not in ("cuda", "cpu"):
        raise _Invalid('must be "cuda" or "cpu"')
    return v  # type: ignore[return-value]  # narrowed by the membership check


# table -> {field: validator}. Also the set of *known* keys (unknowns only warn).
_SCHEMA: dict[str, dict[str, Callable[[object], object]]] = {
    "engine": {
        "model": _str,
        "compute_type": _str,
        "device": _compute_device,
        "language": _str,
        "beam_size": _pos_int,
        "vad_filter": _bool,
    },
    "audio": {
        "sample_rate": _pos_int,
        "channels": _pos_int,
        "device": _input_device,
    },
    "daemon": {
        "max_record_s": _pos_float,
        "idle_timeout_s": _number,
        "notify": _bool,
    },
    "notify": {
        "timeout_s": _pos_float,
    },
}

_SECTIONS = {
    "engine": EngineConfig,
    "audio": AudioConfig,
    "daemon": DaemonConfig,
    "notify": NotifyConfig,
}


def _warn(msg: str) -> None:
    print(f"susurro: {msg}", file=sys.stderr, flush=True)


def _section(cls: type, schema: dict, raw: dict, table: str, source: Path) -> object:
    kwargs = {}
    for key, value in raw.items():
        validator = schema.get(key)
        if validator is None:
            _warn(f"{source}: unknown key [{table}] {key} — ignored")
            continue
        try:
            kwargs[key] = validator(value)
        except _Invalid as exc:
            raise ConfigError(f"{source}: [{table}] {key} {exc} (got {value!r})") from None
    return cls(**kwargs)


def _build_config(data: dict, source: Path) -> Config:
    sections = {}
    for table, cls in _SECTIONS.items():
        raw = data.get(table, {})
        if not isinstance(raw, dict):
            raise ConfigError(f"{source}: [{table}] must be a table")
        sections[table] = _section(cls, _SCHEMA[table], raw, table, source)
    for unknown in set(data) - set(_SECTIONS):
        _warn(f"{source}: unknown section [{unknown}] — ignored")
    return Config(**sections)
