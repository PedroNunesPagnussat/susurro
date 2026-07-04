# susurro — Work Log

Append-only record of what's been done. Newest at the bottom.

## 2026-07-03 — Phase 1

**Step 1 (scaffold):** `uv` project, src layout, `pyproject.toml` pinning `faster-whisper`,
`sounddevice`, `numpy`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`. `uv sync` OK on Python 3.13
→ `ctranslate2==4.8.1`, `faster-whisper==1.2.1`, cuDNN 9.24, cuBLAS 12.9. System `libportaudio`
present.

**Step 0 (GPU spike):** PASS on the GTX 1060.
- `large-v3-turbo` int8 on `device="cuda"` — venv-local cuBLAS/cuDNN wheels preloaded via
  `susurro._cuda.preload_cuda_libs()` (LD_LIBRARY_PATH is unreliable in-process; ctypes preload
  instead).
- VRAM: 916 → 2035 MiB (**delta 1119 MiB**), leaves ample headroom on 6GB.
- Transcript of the JFK fixture is verbatim-correct.
- **Warm transcribe: 0.938s for 11.0s of audio** → well under the 1.5s DoD (a 3s window is
  cheaper still). Warm-model premise holds; no CPU fallback needed.
- Fixture: `tests/fixtures/jfk_16k_mono.wav` (canonical public-domain JFK clip, ffmpeg → 16kHz
  mono s16, 11s).

**Step 2 (audio capture):** `susurro.audio` — `record_window()` (sounddevice/PortAudio, 16kHz
mono float32, callback-based), `_WindowBuffer` (hardware-free block assembly), device helpers,
`load_wav()`. `sounddevice` lazy-imported so the module imports without an audio server. 7 unit
tests (buffer + WAV load), no hardware needed.

**Step 3 (engine + formatter):** `susurro.formatter` (`Formatter` protocol + `RuleBasedFormatter`:
whitespace trim/collapse, drop empty/no-speech, no filler removal) — 9 unit tests, TDD red→green.
`susurro.engine.Engine` — warm model loaded once, `transcribe(np.ndarray)->str` with the agreed
defaults (`language="en"`, `vad_filter=True`, `condition_on_previous_text=False`, `beam_size=5`,
int8/cuda), formatter applied, pluggable formatter DI. Engine seam test (2, CUDA-gated): JFK words
present, silence→"". 6.4s.

**Step 4 (harness loop):** `susurro.__main__` — argparse (`--duration/--device/--model/--cpu/
--list-devices`), warm-up pass, capture→transcribe→format→print loop, quiet-marker (`·`) on
silence, clean Ctrl-C. Non-interactive paths smoke-tested (`--help`, `--list-devices` → sees the
fifine USB mic at idx 7).

**Suite:** 18 passed (9 formatter + 7 audio + 2 engine). All modules import clean.

**Step 5 (verify DoD):** PENDING — needs a live mic run by the user (`uv run susurro`, speak).
Automated proxy already met: warm transcribe 0.938s (<1.5s target), silence→"" verified.
