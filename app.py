import time
import threading
from enum import Enum, auto

import numpy as np
import requests

from audio_stream import AudioStream
from transcriber import Transcriber
from wake_word_engine import WakeWordEngine

# --------------------------------------------------------------------------- #
# Tunable constants
# --------------------------------------------------------------------------- #
SAMPLE_RATE          = 16_000
VAD_CHUNK_SIZE       = 4096        # samples per VAD tick (~128 ms)
COMMAND_DURATION_SEC = 3           # seconds to record after wake word
COMMAND_SAMPLES      = int(SAMPLE_RATE * COMMAND_DURATION_SEC)

# --------------------------------------------------------------------------- #
# State server config  (same host/endpoint as keyboard_controller.py)
# --------------------------------------------------------------------------- #
SOCKET_HOST  = "10.127.205.34"
SOCKET_PORT  = "8000"
STATE_URL    = f"http://{SOCKET_HOST}:{SOCKET_PORT}/change-state"
POST_TIMEOUT = 2.0    # seconds
COMMAND_CYCLES = 3    # cycles to send with every voice command

# --------------------------------------------------------------------------- #
# Static states — sent without a cycles field (they are persistent poses)
# --------------------------------------------------------------------------- #
STATIC_STATES = {"SIT", "STAND"}

# --------------------------------------------------------------------------- #
# Simple keyword → state map  (checked after spin is resolved)
# --------------------------------------------------------------------------- #
INTENT_MAP: dict[str, str] = {
    "stand":    "STAND",
    "sit":      "SIT",
    "forward":  "FORWARD",
    "walk":     "FORWARD",
    "backward": "BACKWARD",
    "back":     "BACKWARD",
    "left":     "LEFT",
    "right":    "RIGHT",
    "jump":     "JUMP",
}

# Words that indicate counter-clockwise direction
_CCW_WORDS = {"counter", "anti", "anticlockwise", "counterclockwise"}

# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
class State(Enum):
    LISTENING    = auto()
    RECORDING    = auto()
    TRANSCRIBING = auto()


# --------------------------------------------------------------------------- #
# Intent detection
# --------------------------------------------------------------------------- #
def detect_intent(transcription: str) -> str | None:
    """
    Resolve a transcription to a state string.

    Spin / rotate commands are handled first so direction words
    ("clockwise" / "counter") take priority over the simple map.
    Everything else falls through to INTENT_MAP.
    """
    lower = transcription.lower()

    # --- Spin / rotate ---------------------------------------------------- #
    if "spin" in lower or "rotate" in lower:
        # Counter-clockwise if any CCW word is present
        if any(w in lower for w in _CCW_WORDS):
            return "CCW"
        return "CW"   # default spin direction is clockwise

    # --- Simple keyword scan ---------------------------------------------- #
    for keyword, state_name in INTENT_MAP.items():
        if keyword in lower:
            return state_name

    return None


# --------------------------------------------------------------------------- #
# HTTP dispatch  (fire-and-forget, non-blocking)
# --------------------------------------------------------------------------- #
def _post_state(state_name: str, cycles: int | None) -> None:
    """Runs in a daemon thread — never blocks the main voice loop."""
    payload: dict = {
        "new_state": state_name,
        "sender_id": "voice_assistant",
    }
    if cycles is not None:
        payload["cycles"] = cycles

    cycles_label = f"cycles={cycles}" if cycles is not None else "static"
    try:
        r = requests.post(STATE_URL, json=payload, timeout=POST_TIMEOUT)
        status = "✓" if r.status_code == 200 else f"✗ HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        status = "✗ no connection"
    except requests.exceptions.Timeout:
        status = "✗ timeout"
    except Exception as e:
        status = f"✗ {e}"

    print(f"[BYTE] POST [{state_name}] {cycles_label}  →  {status}", flush=True)


def send_command(state_name: str) -> None:
    """
    Dispatch a non-blocking POST.
    Static states (SIT, STAND) omit the cycles field entirely.
    All other states include cycles=COMMAND_CYCLES.
    """
    cycles = 1 if state_name in STATIC_STATES else COMMAND_CYCLES
    threading.Thread(target=_post_state, args=(state_name, cycles), daemon=True).start()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 60)
    print("  BYTE – Robot Dog Voice Assistant")
    print("=" * 60)

    print("[BYTE] Initialising audio stream …", flush=True)
    audio = AudioStream(sample_rate=SAMPLE_RATE)

    print("[BYTE] Loading wake-word engine …", flush=True)
    engine = WakeWordEngine()

    print("[BYTE] Loading transcriber …", flush=True)
    transcriber = Transcriber()

    audio.start()
    print()
    print("[BYTE] Listening … Say 'Hey Byte' or 'Hello Byte' to activate.")
    print(f"       Commands will POST to {STATE_URL} with cycles={COMMAND_CYCLES}")
    print("       Press Ctrl+C to quit.\n")

    app_state = State.LISTENING
    recording_start: float = 0.0

    try:
        while True:
            # ---------------------------------------------------------------- #
            # LISTENING  – feed small chunks to VAD + KWS                      #
            # ---------------------------------------------------------------- #
            if app_state is State.LISTENING:
                if audio.available() < VAD_CHUNK_SIZE:
                    time.sleep(0.01)
                    continue

                chunk = audio.drain(VAD_CHUNK_SIZE)
                chunk = chunk * 1.5
                chunk = np.clip(chunk, -1.0, 1.0)

                max_vol = np.max(np.abs(chunk))
                if max_vol > 0.1:
                    print(f"[Debug] Vol: {max_vol:.2f}", flush=True)

                if engine.process(chunk):
                    kw = engine.last_keyword
                    kw_display = kw.replace("bite", "byte").replace("Bite", "byte").title()
                    print(f"[BYTE] '{kw_display}' detected! Listening for command …",
                          flush=True)
                    audio.clear()
                    engine.reset()
                    recording_start = time.monotonic()
                    app_state = State.RECORDING

            # ---------------------------------------------------------------- #
            # RECORDING  – accumulate COMMAND_DURATION_SEC of audio            #
            # ---------------------------------------------------------------- #
            elif app_state is State.RECORDING:
                elapsed = time.monotonic() - recording_start
                if elapsed >= COMMAND_DURATION_SEC:
                    app_state = State.TRANSCRIBING
                else:
                    time.sleep(0.02)

            # ---------------------------------------------------------------- #
            # TRANSCRIBING  – run Whisper, resolve intent, POST to state server #
            # ---------------------------------------------------------------- #
            elif app_state is State.TRANSCRIBING:
                command_audio = audio.read_chunk(COMMAND_SAMPLES)

                if command_audio.size == 0:
                    print("[BYTE] No audio captured – returning to LISTENING.",
                          flush=True)
                else:
                    print("[BYTE] Transcribing …", flush=True)
                    result = transcriber.transcribe(command_audio)

                    if result:
                        print(f"\n[BYTE] Heard: \"{result}\"\n", flush=True)

                        intent = detect_intent(result)
                        if intent:
                            cycles_label = "static" if intent in STATIC_STATES else f"cycles={COMMAND_CYCLES}"
                            print(f"[BYTE] Intent → {intent}  ({cycles_label})", flush=True)
                            send_command(intent)
                        else:
                            print("[BYTE] No matching command found.\n", flush=True)
                    else:
                        print("[BYTE] (nothing transcribed)\n", flush=True)

                # Always clear and return to listening
                audio.clear()
                engine.reset()
                print("[BYTE] Listening … Say 'Hey Byte' or 'Hello Byte' to activate.")
                app_state = State.LISTENING

    except KeyboardInterrupt:
        print("\n[BYTE] Shutting down …")
    finally:
        audio.stop()
        print("[BYTE] Goodbye.")


if __name__ == "__main__":
    main()