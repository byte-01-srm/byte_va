# BYTE – Robot Dog Voice Assistant

Offline, low-latency voice pipeline for a robot dog. All inference is local — no cloud, no internet required at runtime.

**Pipeline:**
1. **VAD** (Silero) — detects speech activity
2. **Keyword Spotter** (Zipformer) — listens for *"Hey Byte"* or *"Hello Byte"*
3. **Whisper** (`base.en` by default in `transcriber.py`) — transcribes the follow-up command

---

## Setup

### 1. Prerequisites
- Python 3.9 – 3.11 (64-bit)
- A working microphone

### 2. Create & activate the virtual environment
```powershell
python -m venv byte_env
.\byte_env\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download models & generate `keywords.txt`
```powershell
python download_models.py
```

This will:
- Download `silero_vad.onnx` (~2 MB)
- Download & extract the Zipformer KWS model (~13 MB)
- Write `keywords_raw.txt` and BPE-tokenize it into `keywords.txt` using `sherpa_onnx` (`text2token` in Python; no separate CLI step)

### 5. Run
```powershell
python app.py
```

---

## Usage

| Step | You say | BYTE prints |
|------|---------|-------------|
| Activation | *"Hey Byte"* or *"Hello Byte"* | e.g. `[BYTE] 'Hey Byte' detected! Listening for command …` |
| Command | *"sit"*, *"stay"*, *"fetch"* … | `[BYTE] Command: sit` |
| Idle | (silence) | `[BYTE] Listening … Say 'Hey Byte' or 'Hello Byte' to activate.` |

Press **Ctrl+C** to quit.

---

## Project Structure

```
.
├── app.py                 # Main entry point – state machine
├── audio_stream.py        # Non-blocking sounddevice ring buffer
├── wake_word_engine.py    # sherpa-onnx VAD + Keyword Spotter
├── transcriber.py         # faster-whisper transcription
├── download_models.py     # One-time model download + tokenization
├── test_kws.py            # KWS / wake-word tests (optional)
├── sherpa_mic_example.py  # Standalone sherpa mic demo (optional)
├── requirements.txt       # Python dependencies
├── keywords_raw.txt       # Wake-word definitions (generated; git-ignored)
├── keywords.txt           # BPE-tokenized wake words (generated; git-ignored)
├── byte_env/              # Python venv (git-ignored)
└── models/                # Downloaded ONNX weights (git-ignored)
    ├── silero_vad.onnx
    └── sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/
```

---

## Windows Notes

**onnxruntime DLL conflict:** On Windows, the system `onnxruntime.dll` in `System32` can shadow the one bundled with `sherpa-onnx`, causing an API version mismatch at import time. `wake_word_engine.py` automatically calls `os.add_dll_directory()` pointing at the `sherpa_onnx` package directory *before* the first import, resolving this without any manual intervention.

---

## Tuning

| Constant | File | Default | Effect |
|----------|------|---------|--------|
| `COMMAND_DURATION_SEC` | `app.py` | `3` | Recording window after wake word (seconds) |
| Silero `threshold` | `wake_word_engine.py` (`SileroVadModelConfig`) | `0.5` | VAD sensitivity (lower = more sensitive) |
| `keywords_score` / `keywords_threshold` | `wake_word_engine.py` | `2.3` / `0.002` | KWS boosting vs how easily phrases trigger |
| `beam_size` | `transcriber.py` (`transcribe`) | `3` | Whisper beam search (higher = slower, often more accurate) |
| `model_size` | `transcriber.py` (`Transcriber.__init__`) | `"base.en"` | Whisper model id (e.g. `tiny.en` for less RAM) |
| `:… #…` suffixes | `keywords_raw.txt` | — | Per-line KWS boost score & trigger threshold |
