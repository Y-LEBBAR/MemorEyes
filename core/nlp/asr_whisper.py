# core/nlp/asr_whisper.py
# -----------------------
# Simple ASR loop using Whisper + sounddevice.
# Listens to the default microphone and yields short text chunks.
#
# Dependencies (install in your venv):
#   pip install openai-whisper sounddevice
#
# Note: Whisper uses PyTorch under the hood; make sure torch is installed.

from __future__ import annotations

import queue
import threading
import time
from typing import List

import numpy as np
import sounddevice as sd
import whisper


SAMPLE_RATE = 16000        # Whisper expects 16 kHz
CHUNK_SECONDS = 4.0        # length of each audio chunk in seconds
MODEL_NAME = "base"        # or "small" if your laptop can handle it


def _transcribe_array(model: whisper.Whisper, audio: np.ndarray) -> str:
    """
    Transcribe a mono float32 array (16 kHz) with Whisper.
    """
    # Whisper expects 16000 Hz mono, float32 in [-1, 1]
    # Pad/trim to 30s window
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text.strip()


def start_asr_background(transcript_lines: List[str]) -> threading.Event:
    """
    Start a background thread that continuously records audio chunks from the
    microphone, runs Whisper, and appends recognized text to transcript_lines.

    Returns:
        stop_event: set this to True/stop_event.set() to stop the worker.
    """
    stop_event = threading.Event()

    def worker():
        model = whisper.load_model(MODEL_NAME)
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1

        while not stop_event.is_set():
            # Record one chunk
            frames = int(CHUNK_SECONDS * SAMPLE_RATE)
            audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()

            # Convert to 1D mono vector
            audio_vec = audio[:, 0]

            try:
                text = _transcribe_array(model, audio_vec)
            except Exception as e:
                # Keep running even if one chunk fails
                text = ""
            if text:
                timestamp = time.strftime("%H:%M:%S")
                line = f"[{timestamp}] {text}"
                transcript_lines.append(line)

            # Small sleep to avoid tight looping
            time.sleep(0.1)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return stop_event
