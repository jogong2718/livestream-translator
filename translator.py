"""
translator.py
Loads the Faster-Whisper model and processes audio buffers.
Supports translation to English and optional Japanese → romaji conversion.
"""

from __future__ import annotations

import re
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


def _contains_japanese(text: str) -> bool:
    """Return True if text contains Japanese characters."""
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text))


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
        cpu_threads: int = 4,
        source_language: Optional[str] = None,
    ):
        self.input_queue = input_queue
        self.on_result = on_result
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.cpu_threads = cpu_threads
        self.source_language = source_language

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
            cpu_threads=self.cpu_threads,
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
                item = self.input_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                audio = item
                result = self._transcribe(
                    audio,
                )
                if result and result.text.strip():
                    self.on_result(result)
            except Exception as exc:
                print(f"[Translator] Inference error: {exc}")

    def _transcribe(
        self,
        audio: np.ndarray,
    ) -> Optional[TranslationResult]:
        """Run Whisper translate on a float32 16 kHz mono numpy array."""
        duration = len(audio) / 16_000
        t0 = time.time()

        t_transcribe_start = time.time()
        segments, info = self._model.transcribe(
            audio,
            task="translate",  # always output English
            beam_size=3,
            best_of=3,
            language=self.source_language,  # auto-detect if None
            vad_filter=False,  # we already did VAD
            without_timestamps=True,
        )
        transcribe_elapsed = time.time() - t_transcribe_start

        # Collect segment texts
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())

        full_text = " ".join(texts)
        elapsed = time.time() - t0
        detected_lang = info.language if info else "?"

        print(
            f"[Translator] final {detected_lang} | {duration:.1f}s audio → "
            f"{transcribe_elapsed:.2f}s whisper | {elapsed:.2f}s total | {full_text[:80]}"
        )

        # Generate romaji only for Japanese to keep latency down
        romaji = ""
        if detected_lang == "ja":
            # For romaji we want the *original* text, not the translation.
            # Re-run a quick transcription in transcribe mode to get the
            # Japanese source text, then only keep romaji if it contains JP.
            t_romaji_start = time.time()
            jp_text = self._get_japanese_text(audio, is_partial=False)
            if _contains_japanese(jp_text):
                romaji = to_romaji(jp_text)
                if not romaji:
                    romaji = jp_text
                    print("[Translator] Romaji unavailable; showing Japanese text instead.")
            romaji_elapsed = time.time() - t_romaji_start
            print(f"[Translator] romaji generation: {romaji_elapsed:.2f}s")

        return TranslationResult(
            text=full_text,
            romaji=romaji,
            language=detected_lang,
            duration=duration,
            timestamp=time.time(),
        )

    def _get_japanese_text(self, audio: np.ndarray, is_partial: bool = False) -> str:
        """Re-transcribe in 'transcribe' mode to get the original Japanese."""
        try:
            beam_size = 1 if is_partial else 2
            best_of = 1 if is_partial else 2
            segments, _ = self._model.transcribe(
                audio,
                task="transcribe",
                language="ja",
                beam_size=beam_size,
                best_of=best_of,
                without_timestamps=True,
            )
            return " ".join(seg.text.strip() for seg in segments)
        except Exception as exc:
            print(f"[Translator] Japanese transcription failed: {exc}")
            return ""
