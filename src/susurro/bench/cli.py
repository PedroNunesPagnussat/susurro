"""`susurro-bench` — the benchmark entrypoint: `record` (capture yourself reading
the scripts) and `run` (transcribe every recording with every model and report).

Import-light by contract: importing this module must not pull in a heavy runtime.
The subcommand bodies lazy-import their implementation (`recording` for `record`,
`runner` for `run`) only when that command actually runs, so `--list-models` and
`--help` stay instant and a missing optional runtime never blocks the CLI.

    uv run susurro-bench --list-models   # ids + availability, then exit
    uv run susurro-bench record          # read the scripts into bench/recordings/
    uv run susurro-bench record --device 4   # capture from a specific input
    uv run susurro-bench run             # transcribe + score + report

`record` reads the same `[audio]` config the daemon does, so the input you already
configured is the one it captures from.
"""

from __future__ import annotations

import argparse
import sys

from . import registry


def parse_models(value: str | None) -> list[str] | None:
    """Turn a `--models a,b,c` string into a list of ids, or None for "all".

    None/blank -> None ("no filter": the runner takes every available model). The
    ids aren't validated here — `registry.resolve` fails loud on an unknown id."""
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _add_models_flag(parser: argparse.ArgumentParser) -> None:
    """The shared model selector. Lives on the subcommands that run models."""
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model ids to run (default: all available)",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="susurro-bench", description=__doc__)
    p.add_argument(
        "--list-models",
        action="store_true",
        help="list model ids + availability and exit",
    )
    sub = p.add_subparsers(dest="command")

    rec = sub.add_parser("record", help="record yourself reading each script")
    rec.add_argument("--only", default=None, help="re-record just this script id")
    rec.add_argument(
        "--redo",
        action="store_true",
        help="re-record scripts that already have a recording",
    )
    # Same spelling and precedence as `susurro`/`susurro-daemon` (defaults < config
    # file < flag). `--device` stays a raw string here — `_cli.parse_device` turns it
    # into an index or a name substring in the command body, because importing the
    # config layer at parser-build time would break this module's import-light
    # contract for `--help` and `--list-models`.
    rec.add_argument("--config", default=None, help="path to config.toml")
    rec.add_argument(
        "--device", default=None, help="input device index or name (overrides [audio] device)"
    )

    run = sub.add_parser("run", help="transcribe every recording with every model, report")
    _add_models_flag(run)
    return p


def _list_models() -> int:
    """Print each registered model with its import-probe availability, then exit.

    Availability is a cheap probe (`ModelSpec.available`) — no model is built here."""
    for spec in registry.REGISTRY.values():
        status = "available" if spec.available() else "unavailable"
        print(f"  {spec.id:<22} {status:<12} {spec.label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        return _list_models()

    if args.command == "record":
        from .recording import record_command

        return record_command(args)
    if args.command == "run":
        from .runner import run_command

        return run_command(args)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
