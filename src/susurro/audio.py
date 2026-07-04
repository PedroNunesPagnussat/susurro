"""Mic capture — 16kHz mono float32, callback-based (PortAudio).

Two capture modes share one assembly path (`_WindowBuffer`): `record_window`
grabs a fixed-length window (Phase 1's smoke loop), and `Recorder` does
variable-length start/stop capture for Phase 2's hold-to-talk (open the stream
on `start`, close it on `stop`, get everything spoken in between).

The block-assembly (`_WindowBuffer`) and `load_wav` are pure and hardware-free so
they can be unit-tested without a mic; only `record_window`, `Recorder`, and the
device helpers touch PortAudio, and they lazy-import `sounddevice` so importing
this module never requires an audio server.

The capture format (16kHz mono float32 numpy) feeds faster-whisper directly with
no resampling, and the callback/start-stop model is reused verbatim in Phase 2.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"


class _WindowBuffer:
    """Accumulates PortAudio callback blocks into one mono float32 window.

    Hardware-free: the audio callback just calls `add`; `result` assembles the
    final array. Kept separate from the stream so the assembly logic is testable.

    An optional `max_samples` cap makes `add` drop blocks once full, so a
    variable-length recording that never gets a stop (missed key release) can't
    grow the buffer without bound — the daemon's safety timeout is the primary
    stop, this is the memory backstop.
    """

    def __init__(self, max_samples: int | None = None) -> None:
        self._blocks: list[np.ndarray] = []
        self._n = 0  # samples accumulated so far (after mono downmix)
        self._max = max_samples  # hard cap: stop stashing past this (bounds memory)
        self.xruns = 0  # count of callbacks flagged with a non-empty status (xruns)
        self.capped = False  # True once the cap dropped at least one block

    def add(self, indata: np.ndarray, status: object = None) -> None:
        if status:
            self.xruns += 1
        if self._max is not None and self._n >= self._max:
            self.capped = True  # already full — drop the block so memory stays bounded
            return
        block = np.asarray(indata, dtype=np.float32)
        if block.ndim == 2:  # (frames, channels) -> mono
            block = block[:, 0]
        # PortAudio reuses its buffer between callbacks, so copy before stashing.
        self._blocks.append(block.copy())
        self._n += block.shape[0]

    def result(self, max_samples: int | None = None) -> np.ndarray:
        if max_samples is None:
            max_samples = self._max  # default to the construction-time cap
        if not self._blocks:
            return np.zeros(0, dtype=np.float32)
        arr = np.concatenate(self._blocks)
        if max_samples is not None:
            arr = arr[:max_samples]
        return arr


def record_window(
    duration_s: float,
    samplerate: int = SAMPLE_RATE,
    device: int | str | None = None,
) -> np.ndarray:
    """Record a fixed-length window from the mic and return it as mono float32.

    Blocks for ~`duration_s`. `device` selects the input (index or name substring);
    None uses the PortAudio default input.
    """
    import sounddevice as sd

    buf = _WindowBuffer()

    def _callback(indata, _frames, _time, status):  # noqa: ANN001 (PortAudio sig)
        buf.add(indata, status)

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device,
            callback=_callback,
        ):
            sd.sleep(int(duration_s * 1000))
    except sd.PortAudioError as exc:  # no device, bad rate, server down...
        raise RuntimeError(f"audio capture failed: {exc}") from exc

    return buf.result(max_samples=int(duration_s * samplerate))


class Recorder:
    """Variable-length mic capture for hold-to-talk: `start()` opens the stream,
    `stop()` closes it and returns everything captured as mono float32.

    Reuses the same callback/`_WindowBuffer` assembly as `record_window`, but the
    lifetime is caller-driven instead of a fixed duration. `max_duration_s` caps
    the capture (and thus memory) so a missed `stop` can't accumulate forever; the
    daemon's safety timeout is expected to call `stop()` well before this bites.

    Not thread-safe for concurrent start/stop; single-client by design (the daemon
    serializes). The PortAudio callback runs on its own thread, but `stop()` halts
    the stream before reading the buffer, so the assembled result is race-free.
    """

    def __init__(
        self,
        max_duration_s: float = 30.0,
        samplerate: int = SAMPLE_RATE,
        device: int | str | None = None,
    ) -> None:
        self.max_duration_s = max_duration_s
        self.samplerate = samplerate
        self.device = device
        self._stream = None  # active sd.InputStream while recording, else None

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Open the input stream and begin accumulating audio. Raises if already
        recording or if PortAudio can't open the device."""
        if self._stream is not None:
            raise RuntimeError("already recording")
        import sounddevice as sd

        buf = _WindowBuffer(max_samples=int(self.max_duration_s * self.samplerate))

        def _callback(indata, _frames, _time, status):  # noqa: ANN001 (PortAudio sig)
            buf.add(indata, status)  # closes over buf, immune to stop() nulling state

        try:
            stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=CHANNELS,
                dtype=DTYPE,
                device=self.device,
                callback=_callback,
            )
            stream.start()
        except sd.PortAudioError as exc:  # no device, bad rate, server down...
            raise RuntimeError(f"audio capture failed: {exc}") from exc

        self._stream = stream
        self._buf = buf

    def stop(self) -> np.ndarray:
        """Close the stream and return the captured audio (mono float32, trimmed
        to `max_duration_s`). Raises if not currently recording."""
        if self._stream is None:
            raise RuntimeError("not recording")
        stream, buf = self._stream, self._buf
        self._stream = None
        try:
            stream.stop()  # halts callbacks before we read the buffer
        finally:
            stream.close()
        return buf.result()


def list_input_devices() -> list[tuple[int, str]]:
    """Return (index, name) for every device that can capture audio."""
    import sounddevice as sd

    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def default_input_device() -> tuple[int, str] | None:
    """Return (index, name) of the default input device, or None if there is none."""
    import sounddevice as sd

    try:
        idx = sd.default.device[0]
    except Exception:
        return None
    if idx is None or idx < 0:
        return None
    return idx, sd.query_devices(idx)["name"]


def load_wav(path: str | Path) -> np.ndarray:
    """Load a 16-bit PCM WAV as mono float32 in [-1, 1] at its native rate.

    Used for offline testing/feeding the Engine without a mic (the committed
    fixture is already 16kHz mono, so no resampling is needed).
    """
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if sampwidth != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got sampwidth={sampwidth}")

    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels)[:, 0]
    return data
