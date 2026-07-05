"""Mic capture: variable-length start/stop recording into a mono float32 buffer.

`Recorder` opens a PortAudio input stream on `start()` and closes it on `stop()`,
returning everything captured in between as 16kHz mono float32 (the format
faster-whisper consumes directly, no resampling).

`_WindowBuffer` (the block-assembly) and `load_wav` are pure and hardware-free, so
they unit-test without a mic; only `Recorder` and the device helper touch
PortAudio, and they lazy-import `sounddevice` so importing this module never
requires an audio server.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"


class _WindowBuffer:
    """Accumulates PortAudio callback blocks into one mono float32 array.

    Hardware-free: the audio callback calls `add`; `result` assembles the final
    array. `max_samples` caps the buffer so a recording that never gets a stop (a
    missed key release) can't grow memory without bound — the daemon's safety
    timeout is the primary stop, this is the backstop.
    """

    def __init__(self, max_samples: int | None = None) -> None:
        self._blocks: list[np.ndarray] = []
        self._n = 0  # samples accumulated so far (after mono downmix)
        self._max = max_samples
        self.xruns = 0  # callbacks flagged with a non-empty status (overflows)
        self.capped = False  # True once the cap dropped at least one block

    def add(self, indata: np.ndarray, status: object = None) -> None:
        if status:
            self.xruns += 1
        if self._max is not None and self._n >= self._max:
            self.capped = True  # already full — drop the block, memory stays bounded
            return
        block = np.asarray(indata, dtype=np.float32)
        if block.ndim == 2:  # (frames, channels) -> mono
            block = block[:, 0]
        # PortAudio reuses its buffer between callbacks, so copy before stashing.
        self._blocks.append(block.copy())
        self._n += block.shape[0]

    def result(self) -> np.ndarray:
        if not self._blocks:
            return np.zeros(0, dtype=np.float32)
        arr = np.concatenate(self._blocks)
        if self._max is not None:
            arr = arr[: self._max]
        return arr


class Recorder:
    """Variable-length mic capture: `start()` opens the stream, `stop()` closes it
    and returns everything captured as mono float32.

    `max_duration_s` caps the capture (and thus memory) so a missed `stop` can't
    accumulate forever; the daemon's safety timeout is expected to call `stop()`
    well before this bites.

    Single-client by design (the daemon serializes). The PortAudio callback runs
    on its own thread, but `stop()` halts the stream before reading the buffer, so
    the assembled result is race-free.
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
        """Close the stream and return the captured audio (mono float32, trimmed to
        `max_duration_s`). Raises if not currently recording."""
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


def load_wav(path: str | Path) -> np.ndarray:
    """Load a 16-bit PCM WAV as mono float32 in [-1, 1] at its native rate.

    Feeds the Engine offline (without a mic) for tests and model evaluation.
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
