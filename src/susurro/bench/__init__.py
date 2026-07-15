"""susurro-bench: an offline harness to compare voice-to-text models on the
owner's own hardware and voice, judged on accuracy (WER/CER) and transcribe
latency (RTF).

Kept import-light on purpose: importing `susurro.bench` (or any of `registry`,
`cli`) must NOT drag in a heavy runtime (faster-whisper/CTranslate2, NeMo,
pywhispercpp). Every backend is built lazily, only when a run actually selects it,
so collection stays cheap and an uninstalled optional runtime never blocks import.
"""
