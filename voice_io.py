"""
Gideon - Offline Voice Input and Output (Level 3.5)

Speech to text:  faster-whisper (base.en, int8) running on the CPU.
Text to speech:  kokoro-onnx (Kokoro v1.0) running on the CPU.

Nothing here talks to the internet at run time. The Whisper model is
downloaded once (see the setup commands), and the Kokoro files live in the
local `models` folder.

Speed design (Level 3.5):
  - Engines live in a process-wide registry (`cached_engine`), so they are
    loaded exactly once and survive every Streamlit rerun.
  - `start_warmup` loads them in a background thread right after launch and
    runs one tiny dummy inference, so the very first voice turn is already warm.
  - This module never imports Streamlit, so the registry is safe to use from
    background threads.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:  # reported with a friendly message where it is needed
    sf = None

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

WHISPER_MODEL = "base.en"
TARGET_SR = 16000

DEFAULT_VOICE = "bf_emma"
DEFAULT_SPEED = 1.05
VOICE_CHOICES = ["bf_emma", "af_bella", "af_heart", "bf_isabella", "af_nicole", "af_sarah"]
MAX_SPEECH_CHARS = 320


def package_available(name: str) -> bool:
    """True if a Python package can be imported (without importing it)."""
    return importlib.util.find_spec(name) is not None


def kokoro_files_ready() -> bool:
    return KOKORO_MODEL_PATH.exists() and KOKORO_VOICES_PATH.exists()


def _require_soundfile() -> None:
    if sf is None:
        raise RuntimeError("The 'soundfile' package is missing. Run: pip install soundfile")


# --------------------------------------------------------------------------
# Process-wide engine registry (load once, reuse forever)
# --------------------------------------------------------------------------

_ENGINES: Dict[str, Tuple[Any, str]] = {}
_ENGINE_LOCKS: Dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()
_WARMUP_STARTED = False


def cached_engine(name: str, loader: Callable[[], Tuple[Any, str]]) -> Tuple[Any, str]:
    """Return (engine, error) for `name`, running `loader` only the first time.

    Each name has its own lock, so a slow load of one engine never blocks the others.
    """
    with _REGISTRY_LOCK:
        lock = _ENGINE_LOCKS.setdefault(name, threading.Lock())
    with lock:
        if name not in _ENGINES:
            _ENGINES[name] = loader()
        return _ENGINES[name]


def clear_engines() -> None:
    """Forget every loaded engine (used by the Reload voice engines button)."""
    global _WARMUP_STARTED
    with _REGISTRY_LOCK:
        _ENGINES.clear()
        _WARMUP_STARTED = False


def start_warmup(loaders: Dict[str, Callable[[], Tuple[Any, str]]]) -> None:
    """Load engines one after another in a background daemon thread. Runs once per process."""
    global _WARMUP_STARTED
    with _REGISTRY_LOCK:
        if _WARMUP_STARTED:
            return
        _WARMUP_STARTED = True

    def worker() -> None:
        for name, loader in loaders.items():
            try:
                cached_engine(name, loader)
            except Exception:
                continue  # a failed warm-up must never crash the app

    threading.Thread(target=worker, name="gideon-warmup", daemon=True).start()


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------


def wav_bytes_to_mono16k(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes into a mono float32 array at 16 kHz."""
    _require_soundfile()
    data, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != TARGET_SR and mono.size > 1:
        target_length = int(round(mono.size * TARGET_SR / rate))
        positions = np.linspace(0, mono.size - 1, num=target_length)
        mono = np.interp(positions, np.arange(mono.size), mono)
    return np.ascontiguousarray(mono, dtype=np.float32)


def wav_to_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float audio array as 16-bit PCM WAV bytes."""
    _require_soundfile()
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def clean_for_speech(text: str) -> str:
    """Strip markdown and symbols, collapse whitespace, and cap the length."""
    text = re.sub(r"[*_`#>~|]", "", text)
    text = text.replace("·", " ").replace("→", " ")
    text = " ".join(text.split())
    if len(text) <= MAX_SPEECH_CHARS:
        return text
    cut = text[:MAX_SPEECH_CHARS]
    last_stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: last_stop + 1] if last_stop > 80 else cut.rstrip()


# --------------------------------------------------------------------------
# Speech to text
# --------------------------------------------------------------------------


class Transcriber:
    def __init__(self, model_name: str = WHISPER_MODEL) -> None:
        from faster_whisper import WhisperModel

        threads = max(4, min(8, (os.cpu_count() or 8) // 2))
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)

    def warm_up(self) -> None:
        """Run one dummy inference so the first real turn does not pay start-up costs."""
        silence = np.zeros(TARGET_SR, dtype=np.float32)
        segments, _info = self._model.transcribe(
            silence, language="en", beam_size=1, vad_filter=False, without_timestamps=True
        )
        for _segment in segments:
            pass

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono 16 kHz float32 array. Returns "" if no speech was found."""
        if audio.size < int(TARGET_SR * 0.3):
            return ""
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def load_transcriber() -> Tuple[Optional[Transcriber], str]:
    """Return (engine, "") on success or (None, friendly error message)."""
    if not package_available("faster_whisper"):
        return None, "faster-whisper is not installed. Run: pip install faster-whisper"
    try:
        engine = Transcriber()
    except Exception as exc:
        return None, (
            f"Speech recognition could not start ({type(exc).__name__}: {exc}). "
            f"The first run needs internet once to download the {WHISPER_MODEL} model; "
            "see the setup commands to pre-download it."
        )
    try:
        engine.warm_up()
    except Exception:
        pass  # warm-up is only an optimisation
    return engine, ""


# --------------------------------------------------------------------------
# Text to speech
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Speech:
    wav: bytes
    seconds: float


class Synthesizer:
    def __init__(self) -> None:
        if not kokoro_files_ready():
            raise FileNotFoundError(
                f"Kokoro files not found in {MODELS_DIR}. Download kokoro-v1.0.onnx and voices-v1.0.bin "
                "using the setup commands."
            )
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))

    def warm_up(self) -> None:
        """Synthesize one short phrase so the first spoken reply is fast."""
        self.speak("Ready.", DEFAULT_VOICE, DEFAULT_SPEED)

    def speak(self, text: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> Speech:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return Speech(wav=b"", seconds=0.0)
        language = "en-gb" if voice.startswith("b") else "en-us"
        samples, sample_rate = self._kokoro.create(cleaned, voice=voice, speed=speed, lang=language)
        seconds = float(len(samples)) / float(sample_rate) if sample_rate else 0.0
        return Speech(wav=wav_to_bytes(samples, int(sample_rate)), seconds=seconds)


def load_synthesizer() -> Tuple[Optional[Synthesizer], str]:
    """Return (engine, "") on success or (None, friendly error message)."""
    if not package_available("kokoro_onnx"):
        return None, "kokoro-onnx is not installed. Run: pip install kokoro-onnx soundfile"
    if not kokoro_files_ready():
        return None, (
            f"Voice files are missing from {MODELS_DIR}. Download kokoro-v1.0.onnx and "
            "voices-v1.0.bin with the setup commands, then press Reload voice engines."
        )
    try:
        engine = Synthesizer()
    except Exception as exc:
        return None, f"Voice output could not start ({type(exc).__name__}: {exc})"
    try:
        engine.warm_up()
    except Exception:
        pass  # warm-up is only an optimisation
    return engine, ""
