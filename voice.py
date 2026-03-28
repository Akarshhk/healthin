"""
Offline Hindi voice recognition using Vosk.
Provides both live-mic and file-based transcription.
"""

import json, os, wave

VOSK_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "vosk-model-hi-0.22", "vosk-model-hi-0.22"
)

_model = None

def _get_model():
    """Lazy-load the Vosk model (heavy, ~50 MB)."""
    global _model
    if _model is None:
        from vosk import Model
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise FileNotFoundError(
                f"Vosk Hindi model not found at {VOSK_MODEL_PATH}. "
                "Download from https://alphacephei.com/vosk/models"
            )
        _model = Model(VOSK_MODEL_PATH)
    return _model


def transcribe_audio_file(filepath: str) -> str:
    """
    Transcribe a WAV audio file to text using Vosk offline ASR.
    Accepts the file path that Gradio's gr.Audio returns.
    Returns the transcribed text string.
    """
    from vosk import KaldiRecognizer

    model = _get_model()

    wf = wave.open(filepath, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Audio must be mono 16-bit WAV")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    text_parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get("text"):
                text_parts.append(result["text"])

    # Get final partial result
    final = json.loads(rec.FinalResult())
    if final.get("text"):
        text_parts.append(final["text"])

    wf.close()
    return " ".join(text_parts).strip()


def listen_mic(duration: int = 5, samplerate: int = 16000) -> str:
    """
    Record from microphone for `duration` seconds and transcribe.
    Requires sounddevice.
    """
    import queue
    import sounddevice as sd
    from vosk import KaldiRecognizer

    model = _get_model()
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    rec = KaldiRecognizer(model, samplerate)
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                           dtype="int16", channels=1, callback=callback):
        for _ in range(0, int(samplerate / 8000 * duration)):
            data = q.get()
            rec.AcceptWaveform(data)

    return json.loads(rec.FinalResult())["text"]