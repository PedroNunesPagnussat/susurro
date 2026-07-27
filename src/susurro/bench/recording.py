"""`susurro-bench record` — walk the scripts and capture the owner reading each.

Three pieces of pure logic (unit-tested): `save_wav` (the inverse of
`audio.load_wav`, so a take round-trips), `plan_recordings` (which ids to record
given what's on disk and the flags), and `resolve_audio` (the `[audio]` settings to
capture with). `record_command` is the interactive shell around them — it drives the
real `Recorder` and prompts a person, so it's verified by hand, not in a unit test.

Capture settings come from the same `config.toml` the daemon reads, not hardcoded
defaults: an `[audio] device` the owner needed for dictation is the one needed here,
and ten silent takes from the wrong input are only discovered after the fact.
"""

from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

from .._cli import parse_device
from ..audio import SAMPLE_RATE, Recorder
from ..config import AudioConfig, ConfigError, load_config, pick
from . import paths


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono float32 [-1, 1] audio as a 16-bit PCM WAV — the exact format
    `audio.load_wav` reads back, so `load_wav(save_wav(x)) == x` within 16-bit
    quantization. Out-of-range samples are clamped (a clipped mic must not wrap
    around to the opposite sign through int16 overflow)."""
    clamped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    # load_wav divides by 32768, so scale by 32768 and clip the +1.0 edge into the
    # int16 max (32767); mid values that are multiples of 1/32768 round-trip exactly.
    pcm = np.clip(np.round(clamped * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def script_ids(scripts_dir: Path) -> list[str]:
    """The recordable script ids: sorted stems of `*.txt` in `scripts_dir`."""
    return sorted(p.stem for p in scripts_dir.glob("*.txt"))


def recorded_ids(recordings_dir: Path) -> set[str]:
    """Stems of the `*.wav` takes already captured (empty if the dir is absent)."""
    if not recordings_dir.exists():
        return set()
    return {p.stem for p in recordings_dir.glob("*.wav")}


def plan_recordings(
    ids: list[str],
    recorded: set[str],
    *,
    only: str | None = None,
    redo: bool = False,
) -> list[str]:
    """Decide which script ids to record, in script order.

    `only` re-records exactly one id (even if already captured); `redo` re-records
    every script; the default skips ids that already have a take. An `only` id that
    isn't a known script raises `KeyError` so a typo fails loud, not silent."""
    if only is not None:
        if only not in ids:
            raise KeyError(f"unknown script id {only!r}; known: {', '.join(ids)}")
        return [only]
    if redo:
        return list(ids)
    return [i for i in ids if i not in recorded]


def resolve_audio(args: argparse.Namespace) -> AudioConfig:
    """The `[audio]` settings to capture with: the config file, with `--device`
    layered on top (built-in defaults < config file < CLI flag — the same precedence
    `susurro` and `susurro-daemon` use).

    Raises `ConfigError` on a bad config, which includes the rate this harness needs:
    every backend is promised 16kHz mono and nothing resamples, and `config` refuses
    any other `[audio] sample_rate` outright — so `record` can't write ten takes that
    `run` would only reject later, without a second check here that could drift."""
    config = load_config(args.config)
    # `--device` is a raw string from argparse (see cli.py) — parse it with the same
    # rule the other CLIs use: digits are a PortAudio index, anything else a name.
    flag = parse_device(args.device) if args.device is not None else None
    return replace(config.audio, device=pick(flag, config.audio.device))


def _capture_one(recorder: Recorder, id_: str, text: str) -> np.ndarray:
    """Show the script and capture one take: Enter to start, Enter to stop."""
    print(f"\n=== {id_} ===\n{text}\n")
    input("Press Enter to START recording ...")
    recorder.start()
    input("Recording — read the script, then press Enter to STOP ...")
    return recorder.stop()


def record_command(args: argparse.Namespace) -> int:
    """Interactive: for each planned script, show it and capture a take to
    `bench/recordings/<id>.wav`. Not unit-tested (needs a mic and a person)."""
    # Resolve the capture settings first: a config error must surface before the
    # person is asked to read anything.
    try:
        audio_cfg = resolve_audio(args)
    except ConfigError as exc:
        print(f"susurro-bench: {exc}", file=sys.stderr)
        return 1

    scripts = paths.scripts_dir()
    recordings = paths.recordings_dir()
    ids = script_ids(scripts)
    if not ids:
        print(f"susurro-bench: no scripts found in {scripts}", file=sys.stderr)
        return 1

    try:
        plan = plan_recordings(ids, recorded_ids(recordings), only=args.only, redo=args.redo)
    except KeyError as exc:
        print(f"susurro-bench: {exc}", file=sys.stderr)
        return 2

    if not plan:
        print(f"all {len(ids)} scripts already recorded — use --redo or --only <id> to redo.")
        return 0

    recordings.mkdir(parents=True, exist_ok=True)
    # Headroom over the longest read so a slow reader is never cut off mid-script.
    recorder = Recorder(
        max_duration_s=120.0,
        samplerate=audio_cfg.sample_rate,
        channels=audio_cfg.channels,
        device=audio_cfg.device,
    )
    print(
        f"recording {len(plan)} script(s) into {recordings} "
        f"(input: {audio_cfg.device if audio_cfg.device is not None else 'default'}). "
        "Ctrl-C to stop."
    )
    try:
        for id_ in plan:
            text = (scripts / f"{id_}.txt").read_text().strip()
            audio = _capture_one(recorder, id_, text)
            # One rate for the whole path: what we captured at, what the WAV header
            # says, and what `run` divides by for RTF — never a second default.
            save_wav(recordings / f"{id_}.wav", audio, sample_rate=audio_cfg.sample_rate)
            print(f"saved {id_}.wav ({len(audio) / audio_cfg.sample_rate:.1f}s)")
    except KeyboardInterrupt:
        if recorder.recording:
            recorder.stop()  # release the mic; the in-progress take is discarded
        print("\nstopped.")
    return 0
