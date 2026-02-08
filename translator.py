"""
translator.py
Loads the Faster-Whisper model and processes audio buffers.
Supports translation to English and optional Japanese → romaji conversion.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Romaji support (lazy-loaded so pykakasi is optional)
# ---------------------------------------------------------------------------
_kakasi_converter = None


def _get_kakasi():
    """Lazy-init pykakasi converter."""
    global _kakasi_converter
    if _kakasi_converter is None:
        try:
            import pykakasi

            kks = pykakasi.kakasi()
            _kakasi_converter = kks
        except ImportError:
            print("[Translator] pykakasi not installed – romaji disabled.")
            _kakasi_converter = False  # sentinel: tried but unavailable
    return _kakasi_converter if _kakasi_converter else None


def to_romaji(text: str) -> str:
    """Convert Japanese text to romaji using pykakasi. Returns empty string on failure."""
    converter = _get_kakasi()
    if converter is None:
        return ""
    try:
        result = converter.convert(text)
        return " ".join(item["hepburn"] for item in result)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    """Holds one completed translation segment."""

    text: str
    romaji: str = ""
    language: str = ""
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Translator – wraps Faster-Whisper
# ---------------------------------------------------------------------------

class Translator:
    """
    Pulls audio chunks from *input_queue*, runs Faster-Whisper translate,
    and calls *on_result* with a TranslationResult for each segment.
    """

    def __init__(
        self,
        input_queue: Queue,
        on_result: Callable[[TranslationResult], None],
        model_size: str = "small",
        compute_type: str = "int8",
        device: str = "cpu",
    ):
        self.input_queue = input_queue
        self.on_result = on_result
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device

        self._model: Optional[WhisperModel] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- public API -------------------------------------------------------

    def load_model(self):
        """Download / load the Whisper model (blocking)."""
        print(
            f"[Translator] Loading faster-whisper model '{self.model_size}' "
            f"(compute_type={self.compute_type}, device={self.device}) …"
        )
        t0 = time.time()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4,
        )
        print(f"[Translator] Model loaded in {time.time() - t0:.1f}s.")

    def start(self):
        """Start the inference loop in a background thread."""
        if self._model is None:
            self.load_model()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print("[Translator] Inference thread started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[Translator] Inference thread stopped.")

    # ---- internal ---------------------------------------------------------

    def _inference_loop(self):
        while not self._stop_event.is_set():
            try:
                audio: np.ndarray = self.input_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                result = self._transcribe(audio)
                if result and result.text.strip():
                    self.on_result(result)
            except Exception as exc:
                print(f"[Translator] Inference error: {exc}")

    def _transcribe(self, audio: np.ndarray) -> Optional[TranslationResult]:
        """Run Whisper translate on a float32 16 kHz mono numpy array."""
        duration = len(audio) / 16_000
        t0 = time.time()

        segments, info = self._model.transcribe(
            audio,
            task="translate",  # always output English
            beam_size=3,
            best_of=3,
            language=None,  # auto-detect
            vad_filter=False,  # we already did VAD
            without_timestamps=True,
        )

        # Collect segment texts
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())

        full_text = " ".join(texts)
        elapsed = time.time() - t0
        detected_lang = info.language if info else "?"

        print(
            f"[Translator] {detected_lang} | {duration:.1f}s audio → "
            f"{elapsed:.2f}s inference | {full_text[:80]}"
        )

        # Generate romaji if the detected language is Japanese
        romaji = ""
        if detected_lang == "ja":
            # For romaji we want the *original* text, not the translation.
            # Re-run a quick transcription in transcribe mode to get the
            # Japanese source text.
            romaji = self._get_romaji(audio)

        return TranslationResult(
            text=full_text,
            romaji=romaji,
            language=detected_lang,
            duration=duration,
            timestamp=time.time(),
        )

    def _get_romaji(self, audio: np.ndarray) -> str:
        """
        Re-transcribe in 'transcribe' mode to get the original Japanese,
        then convert to romaji.
        """
        try:
            segments, _ = self._model.transcribe(
                audio,
                task="transcribe",
                language="ja",
                beam_size=1,
                best_of=1,
                without_timestamps=True,
            )
            jp_text = " ".join(seg.text.strip() for seg in segments)
            return to_romaji(jp_text)
        except Exception as exc:
            print(f"[Translator] Romaji generation failed: {exc}")
            return ""
