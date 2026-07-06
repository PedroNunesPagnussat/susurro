4. Create a way to test and evaluate the voice-to-text models to decide which one is the best for my application. It should consider both time-to-text (transcribe time) and transcription quality.
    - Benchmark set: Claude generates the reference text, I record each script in my own voice, every model runs on my hardware. So time and accuracy are exclusive to my use case.
    - **Models to compare:**
        - **Current:** faster-whisper `large-v3-turbo` (int8, CUDA) — happy with it, this is the baseline.
        - **`large-v3`** (non-turbo) — turbo quality/speed tradeoff.
        - **whisper.cpp** `large-v3-turbo` — same model, different runtime.
        - **Parakeet** (`parakeet-tdt-0.6b-v2`) — non-Whisper architecture; English-first.
    - Metrics per model: WER (accuracy) + transcribe latency.
5. grill me in the decision to add a LLM at the end of the recording to check if the text has no clear transcription errors and to translate things like `slash` to `/` or `dot` to `.` `at` to `@` when it make sense
    - This will also require a lot of tests and evaluation, like step 4, to check if it's doing things like I want to do it.
6. I will probably want to make this public and make a LinkedIn post about it.
7. Config script / file to set the values we're currently magic-numbering across the code, instead of hardcoded defaults + CLI flags scattered per module. Values to pull in:
    - Engine (`engine.py`): `model` (`large-v3-turbo`), `compute_type` (`int8`), `device` (`cuda`), `language` (`en`), `beam_size` (5), `vad_filter`.
    - Audio (`audio.py`): `SAMPLE_RATE` (16000), `CHANNELS` (1), max window duration (30s), input `device`.
    - Daemon (`daemon.py`): `max_record` (60s), `notify` on/off.
    - Notify (`notify.py`): toast timeout (5s).
    - Idea: single `~/.config/susurro/config.toml` loaded once, CLI flags override it.
8. I need a easy way to transcribe to portuguese
