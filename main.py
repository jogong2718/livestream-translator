#!/usr/bin/env python3
"""
main.py
Orchestrates audio capture → Whisper translation → overlay display.

Usage:
    python main.py [--model small] [--device blackhole] [--list-devices]
"""

from __future__ import annotations

import argparse
import signal
import sys
import warnings
from queue import Queue

from PyQt6.QtWidgets import QApplication

from audio_stream import AudioStream, list_audio_devices
from overlay_ui import OverlayWindow
from translator import TranslationResult, Translator

warnings.filterwarnings(
    "ignore",
    message="macOS does not support loopback recording functionality",
    category=Warning,
    module=r"soundcard\.coreaudio",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time livestream translator overlay"
    )
    p.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: small)",
    )
    p.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="CTranslate2 compute type (default: int8 for CPU)",
    )
    p.add_argument(
        "--device-keyword",
        type=str,
        default=None,
        help="Keyword to match audio input device name (e.g. 'blackhole')",
    )
    p.add_argument(
        "--vad-threshold",
        type=float,
        default=0.35,
        help="Silero VAD speech probability threshold (default: 0.35)",
    )
    p.add_argument(
        "--min-chunk-sec",
        type=float,
        default=0.6,
        help="Minimum final speech chunk length in seconds (default: 0.6)",
    )
    p.add_argument(
        "--min-partial-chunk-sec",
        type=float,
        default=0.4,
        help="Minimum partial chunk length in seconds (default: 0.4)",
    )
    p.add_argument(
        "--max-chunk-sec",
        type=float,
        default=10.0,
        help="Maximum speech chunk length in seconds (default: 10.0)",
    )
    p.add_argument(
        "--partial-window-sec",
        type=float,
        default=4.0,
        help="Tail window length for partial updates in seconds (default: 4.0)",
    )
    p.add_argument(
        "--partial-update-sec",
        type=float,
        default=2.0,
        help="How often to emit partial updates in seconds (default: 2.0)",
    )
    p.add_argument(
        "--source-language",
        type=str,
        default=None,
        help="Force source language (e.g. 'ja' for Japanese). Default: auto-detect.",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # List devices mode
    # ------------------------------------------------------------------
    if args.list_devices:
        print("\nAvailable audio input devices:\n")
        for d in list_audio_devices():
            print(f"  [{d['index']}] {d['name']}")
        print()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Qt application (must be created before any widgets)
    # ------------------------------------------------------------------
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # ------------------------------------------------------------------
    # Overlay window
    # ------------------------------------------------------------------
    overlay = OverlayWindow()
    overlay.set_text("Starting up… loading model…")
    overlay.show()

    # ------------------------------------------------------------------
    # Shared queue: AudioStream → Translator
    # ------------------------------------------------------------------
    audio_queue: Queue = Queue(maxsize=20)

    # ------------------------------------------------------------------
    # Callback: Translator → Overlay  (called from translator thread)
    # ------------------------------------------------------------------
    def on_translation(result: TranslationResult):
        """Push translated text to the overlay (thread-safe via signal)."""
        display = result.text
        romaji = result.romaji
        if result.language:
            display = f"[{result.language.upper()}] {result.text}"
        overlay.set_text(display, romaji)

    # ------------------------------------------------------------------
    # Translator (inference thread)
    # ------------------------------------------------------------------
    translator = Translator(
        input_queue=audio_queue,
        on_result=on_translation,
        model_size=args.model,
        compute_type=args.compute_type,
        device="cpu",
        source_language=args.source_language,
    )
    # Load model synchronously so the user sees progress in the console
    translator.load_model()
    overlay.set_text("Model loaded. Listening…")

    # ------------------------------------------------------------------
    # Audio stream (capture thread)
    # ------------------------------------------------------------------
    audio_stream = AudioStream(
        output_queue=audio_queue,
        device_keyword=args.device_keyword,
        vad_threshold=args.vad_threshold,
        min_speech_chunk_sec=args.min_chunk_sec,
        min_partial_chunk_sec=args.min_partial_chunk_sec,
        max_speech_chunk_sec=args.max_chunk_sec,
        partial_window_sec=args.partial_window_sec,
        partial_update_sec=args.partial_update_sec,
    )

    # ------------------------------------------------------------------
    # Start background threads
    # ------------------------------------------------------------------
    translator.start()
    audio_stream.start()

    # ------------------------------------------------------------------
    # Graceful shutdown on Ctrl-C
    # ------------------------------------------------------------------
    def shutdown(*_):
        print("\n[Main] Shutting down…")
        audio_stream.stop()
        translator.stop()
        app.quit()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Allow Python signals to be processed while Qt event loop runs
    from PyQt6.QtCore import QTimer

    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)  # lets Python handle signals

    # ------------------------------------------------------------------
    # Run Qt event loop
    # ------------------------------------------------------------------
    print("[Main] Application running. Press Ctrl-C to quit.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
