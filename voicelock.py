"""
Gideon - Speaker Biometric Voice Lock (Level 3)

Turns a few seconds of speech into a 192-number "vocal embedding" using the
SpeechBrain ECAPA-TDNN speaker model (runs locally on the CPU), then compares
embeddings with cosine similarity.

Enrollment: three recordings are averaged into one voiceprint. The voiceprint
is encrypted with the vault's Fernet key and saved as voiceprint.enc. The raw
recordings are never written to disk.

Verification: every incoming voice input is embedded and compared with the
stored voiceprint. Below the strictness threshold, the input is blocked.

Honest limits (put these in your README threat model):
  - This gates VOICE input only. Typed input is intentionally not gated.
  - A recording of your voice, or a good voice clone, can fool any speaker
    verification system that has no liveness detection. Treat this as a strong
    deterrent against casual or accidental use, not as a login.
  - Anyone with both gideon.key and voiceprint.enc can read the voiceprint.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from vault import BASE_DIR, Vault
from voice_io import MODELS_DIR, TARGET_SR, package_available

VOICEPRINT_PATH = BASE_DIR / "voiceprint.enc"
ECAPA_DIR = MODELS_DIR / "ecapa-voxceleb"
ECAPA_RUNTIME_DIR = MODELS_DIR / "ecapa-runtime"

ENROLL_SAMPLES = 3
MIN_ENROLL_VOICED_SECONDS = 2.0
MIN_VOICED_SECONDS = 1.5
MIN_ENROLL_CONSISTENCY = 0.40
DEFAULT_THRESHOLD = 0.45
MIN_THRESHOLD = 0.20
MAX_THRESHOLD = 0.90

ENROLL_PROMPTS = (
    "The quick brown fox jumps over the lazy dog, and then everything goes quiet again.",
    "I am setting up my personal assistant so that it only listens to me.",
    "Today I will focus on one small step at a time, starting with the easiest one.",
)


# --------------------------------------------------------------------------
# Audio preprocessing
# --------------------------------------------------------------------------


def trim_silence(wav: np.ndarray, frame_ms: int = 30) -> Tuple[np.ndarray, float]:
    """Cut leading and trailing silence. Returns (trimmed audio, seconds of voiced speech)."""
    frame = int(TARGET_SR * frame_ms / 1000)
    if wav.size < frame * 3:
        return wav[:0], 0.0

    count = wav.size // frame
    frames = wav[: count * frame].reshape(count, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))

    floor = float(np.percentile(rms, 10))
    peak = float(np.percentile(rms, 95))
    threshold = max(0.004, floor + 0.25 * (peak - floor))
    voiced = rms > threshold
    if not voiced.any():
        return wav[:0], 0.0

    indices = np.flatnonzero(voiced)
    start = max(0, int(indices[0]) - 2)
    end = min(count, int(indices[-1]) + 3)
    return wav[start * frame : end * frame], float(voiced.sum()) * frame_ms / 1000.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


# --------------------------------------------------------------------------
# Speaker encoder (SpeechBrain ECAPA-TDNN)
# --------------------------------------------------------------------------


def model_files_ready() -> bool:
    return (ECAPA_DIR / "hyperparams.yaml").exists() and (ECAPA_DIR / "embedding_model.ckpt").exists()


class SpeakerEncoder:
    def __init__(self) -> None:
        if not model_files_ready():
            raise FileNotFoundError(
                f"Speaker model files not found in {ECAPA_DIR}. Download them with the setup commands."
            )
        import torch

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older SpeechBrain releases
            from speechbrain.pretrained import EncoderClassifier

        kwargs = {
            "source": str(ECAPA_DIR),
            "savedir": str(ECAPA_RUNTIME_DIR),
            "run_opts": {"device": "cpu"},
        }
        # Copy files instead of symlinking: Windows blocks symlinks for normal users.
        if "local_strategy" in inspect.signature(EncoderClassifier.from_hparams).parameters:
            try:
                from speechbrain.utils.fetching import LocalStrategy

                kwargs["local_strategy"] = LocalStrategy.COPY
            except ImportError:
                pass

        self._torch = torch
        self._classifier = EncoderClassifier.from_hparams(**kwargs)

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        """Return an L2-normalized 192-dimensional embedding for mono 16 kHz audio."""
        torch = self._torch
        peak = float(np.max(np.abs(wav16k))) or 1.0
        signal = torch.from_numpy((wav16k / peak * 0.9).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            embedding = self._classifier.encode_batch(signal)
        vector = embedding.squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


def load_encoder() -> Tuple[Optional[SpeakerEncoder], str]:
    """Return (encoder, "") on success or (None, friendly error message)."""
    missing = [name for name in ("torch", "speechbrain") if not package_available(name)]
    if missing:
        return None, (
            f"Missing packages: {', '.join(missing)}. Run the voice lock install commands from the setup guide."
        )
    if not model_files_ready():
        return None, f"Speaker model files are missing from {ECAPA_DIR}. Download them with the setup commands."
    try:
        return SpeakerEncoder(), ""
    except Exception as exc:
        return None, f"The speaker model could not start ({type(exc).__name__}: {exc})"


# --------------------------------------------------------------------------
# Voiceprint storage (encrypted with the vault key)
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class VoicePrint:
    embedding: np.ndarray
    threshold: float
    samples: int
    created: str


def voiceprint_exists() -> bool:
    return VOICEPRINT_PATH.exists()


def save_voiceprint(vault: Vault, voiceprint: VoicePrint) -> None:
    payload = json.dumps(
        {
            "v": [float(x) for x in voiceprint.embedding],
            "th": float(voiceprint.threshold),
            "n": int(voiceprint.samples),
            "d": voiceprint.created,
        }
    ).encode("utf-8")
    token = vault.encrypt_bytes(payload)
    temp_path = VOICEPRINT_PATH.with_suffix(".tmp")
    temp_path.write_bytes(token)
    os.replace(temp_path, VOICEPRINT_PATH)  # atomic: never leaves a half-written voiceprint


def load_voiceprint(vault: Vault) -> Optional[VoicePrint]:
    """Return the decrypted voiceprint, or None if none is enrolled. Raises VaultError on a bad key."""
    if not VOICEPRINT_PATH.exists():
        return None
    data = json.loads(vault.decrypt_bytes(VOICEPRINT_PATH.read_bytes()).decode("utf-8"))
    return VoicePrint(
        embedding=np.asarray(data["v"], dtype=np.float32),
        threshold=float(data.get("th", DEFAULT_THRESHOLD)),
        samples=int(data.get("n", ENROLL_SAMPLES)),
        created=str(data.get("d", "")),
    )


def delete_voiceprint() -> bool:
    """Overwrite voiceprint.enc with zeros, then delete it. Returns True if a file was removed."""
    if not VOICEPRINT_PATH.exists():
        return False
    try:
        size = VOICEPRINT_PATH.stat().st_size
        with open(VOICEPRINT_PATH, "r+b") as handle:
            handle.write(b"\x00" * size)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass  # deletion below still happens
    VOICEPRINT_PATH.unlink()
    return True


# --------------------------------------------------------------------------
# Enrollment and verification
# --------------------------------------------------------------------------


def build_voiceprint(encoder: SpeakerEncoder, waves: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
    """Average several recordings into one voiceprint. Raises ValueError with a friendly message."""
    embeddings: List[np.ndarray] = []
    for index, wave in enumerate(waves, start=1):
        trimmed, voiced = trim_silence(wave)
        if voiced < MIN_ENROLL_VOICED_SECONDS:
            raise ValueError(
                f"Sample {index} has too little speech ({voiced:.1f} s). Speak for at least "
                f"{MIN_ENROLL_VOICED_SECONDS:.0f} seconds in a quiet room and record it again."
            )
        embeddings.append(encoder.embed(trimmed))

    scores = [
        cosine(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    if scores and min(scores) < MIN_ENROLL_CONSISTENCY:
        raise ValueError(
            f"Your samples do not match each other well (lowest similarity {min(scores):.2f}). "
            "Record all three in the same quiet spot, with the same microphone, speaking naturally."
        )

    mean = np.mean(np.stack(embeddings), axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm if norm > 0 else mean), scores


def new_voiceprint(embedding: np.ndarray, threshold: float, samples: int) -> VoicePrint:
    return VoicePrint(
        embedding=embedding,
        threshold=threshold,
        samples=samples,
        created=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


@dataclass(frozen=True)
class VerifyResult:
    allowed: bool
    score: float
    reason: str  # ok, mismatch, too_short, unavailable, no_voiceprint, error
    detail: str = ""


def verify(encoder: SpeakerEncoder, reference: np.ndarray, wav: np.ndarray, threshold: float) -> VerifyResult:
    """Compare incoming speech with the enrolled voiceprint."""
    trimmed, voiced = trim_silence(wav)
    if voiced < MIN_VOICED_SECONDS:
        return VerifyResult(False, 0.0, "too_short", f"Only {voiced:.1f} s of speech was detected.")
    score = cosine(encoder.embed(trimmed), reference)
    if score >= threshold:
        return VerifyResult(True, score, "ok")
    return VerifyResult(False, score, "mismatch", f"Similarity {score:.2f} is below the threshold {threshold:.2f}.")
