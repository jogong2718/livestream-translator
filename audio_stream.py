"""
audio_stream.py
Handles device selection and captures system audio into a thread-safe buffer.
Uses SoundCard for cross-platform audio capture (requires a loopback virtual
audio device like BlackHole on macOS).
Integrates Silero VAD for voice-activity-gated chunking.
"""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Optional

import numpy as np
import soundcard as sc
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono
# Silero VAD expects 512 samples (32 ms at 16 kHz)
BLOCK_SIZE = 512  # samples per VAD frame
BLOCK_DURATION_MS = int(1000 * BLOCK_SIZE / SAMPLE_RATE)

# How many seconds of audio to accumulate before sending a chunk to Whisper
MIN_SPEECH_CHUNK_SEC = 1.5
MAX_SPEECH_CHUNK_SEC = 10.0

# Padding: keep a little audio before/after speech to avoid clipping words
SPEECH_PAD_MS = 300
SPEECH_PAD_FRAMES = int(SPEECH_PAD_MS / BLOCK_DURATION_MS)

# Silence duration (in frames) that ends a speech segment
SILENCE_FRAMES_THRESHOLD = int(600 / BLOCK_DURATION_MS)  # 600 ms silence


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict]:
    """Return a list of available input (microphone) devices."""
    devices = []
    for i, mic in enumerate(sc.all_microphones(include_loopback=True)):
        devices.append({"index": i, "name": mic.name, "device": mic})
    return devices


def find_loopback_device(keyword: Optional[str] = None) -> sc.Microphone:
    """
    Try to find a loopback / virtual audio device automatically.
    On macOS this is typically "BlackHole 2ch" or "BlackHole 16ch".
    Falls back to the default microphone if nothing is found.
    """
    all_mics = sc.all_microphones(include_loopback=True)

    # Priority keywords to search for (common virtual audio drivers)
    search_terms = ["blackhole", "loopback", "virtual", "stereo mix", "what u hear"]
    if keyword:
        search_terms.insert(0, keyword.lower())

    for term in search_terms:
        for mic in all_mics:
            if term in mic.name.lower():
                print(f"[AudioStream] Found loopback device: {mic.name}")
                return mic

    # Fallback: default mic
    default = sc.default_microphone()
    print(f"[AudioStream] No loopback device found – falling back to default mic: {default.name}")
    return default


# ---------------------------------------------------------------------------
# Silero VAD wrapper
# ---------------------------------------------------------------------------

class SileroVAD:
    """Lightweight wrapper around the Silero VAD model."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        # reset internal states
        self.model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Return True if the VAD-sized audio chunk contains speech.
        *audio_chunk* must be float32, mono, 16 kHz.
        """
        tensor = torch.from_numpy(audio_chunk).float()
        with torch.no_grad():
            prob = self.model(tensor, SAMPLE_RATE).item()
        return prob >= self.threshold

    def reset(self):
        self.model.reset_states()


# ---------------------------------------------------------------------------
# AudioStream – captures audio and produces VAD-gated chunks
# ---------------------------------------------------------------------------

class AudioStream:
    """
    Continuously records from *device*, runs Silero VAD per frame,
    and pushes complete speech segments onto *output_queue* as
    numpy float32 arrays (16 kHz mono).
    """

    def __init__(
        self,
        output_queue: Queue,
        device: Optional[sc.Microphone] = None,
        device_keyword: Optional[str] = None,
        vad_threshold: float = 0.35,
    ):
        self.output_queue = output_queue
        self.device = device or find_loopback_device(device_keyword)
        self.vad = SileroVAD(threshold=vad_threshold)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- public API -------------------------------------------------------

    def start(self):
        """Start capturing audio in a background daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[AudioStream] Capture thread started.")

    def stop(self):
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        print("[AudioStream] Capture thread stopped.")

    # ---- internal ---------------------------------------------------------

    def _capture_loop(self):
        """Main loop: read frames → VAD → accumulate → enqueue."""
        speech_buffer: list[np.ndarray] = []
        silent_frames = 0
        is_speaking = False

        try:
            with self.device.recorder(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SIZE,
            ) as recorder:
                print(f"[AudioStream] Recording from: {self.device.name}")
                while not self._stop_event.is_set():
                    # Read one VAD-sized frame
                    frame = recorder.record(numframes=BLOCK_SIZE)
                    # SoundCard returns (frames, channels) – squeeze to mono
                    mono = frame[:, 0].astype(np.float32)

                    if self.vad.is_speech(mono):
                        silent_frames = 0
                        if not is_speaking:
                            is_speaking = True
                            # Prepend padding frames already in buffer
                            # (they were kept from the ring below)
                        speech_buffer.append(mono)
                    else:
                        if is_speaking:
                            silent_frames += 1
                            speech_buffer.append(mono)  # keep trailing audio

                            if silent_frames >= SILENCE_FRAMES_THRESHOLD:
                                # End of speech segment → push to queue
                                self._flush_buffer(speech_buffer)
                                speech_buffer = []
                                silent_frames = 0
                                is_speaking = False
                                self.vad.reset()
                        else:
                            # Keep a small ring of pre-speech padding
                            speech_buffer.append(mono)
                            if len(speech_buffer) > SPEECH_PAD_FRAMES:
                                speech_buffer.pop(0)

                    # Safety: if someone talks forever, flush at MAX length
                    total_sec = len(speech_buffer) * BLOCK_DURATION_MS / 1000
                    if is_speaking and total_sec >= MAX_SPEECH_CHUNK_SEC:
                        self._flush_buffer(speech_buffer)
                        speech_buffer = []
                        silent_frames = 0
                        is_speaking = False
                        self.vad.reset()

        except Exception as exc:
            print(f"[AudioStream] Error in capture loop: {exc}")

    def _flush_buffer(self, frames: list[np.ndarray]):
        """Concatenate buffered frames and push if long enough."""
        if not frames:
            return
        audio = np.concatenate(frames)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_CHUNK_SEC:
            return  # too short – probably noise
        print(f"[AudioStream] Flushing {duration:.2f}s speech segment to queue.")
        self.output_queue.put(audio)
