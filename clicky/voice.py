"""Record during an explicit shortcut hold. Transcribe locally or with cloud STT."""

import json
import os
import re
import tempfile
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen

import AppKit as A
import ApplicationServices as AX
import AVFoundation as AV

from . import ax

WHISPER = "mlx-community/whisper-small.en-mlx"
MULTILINGUAL_WHISPER = "mlx-community/whisper-small-mlx"
DICTATION_LANGUAGES = {
    "default": "Provider Default", "auto": "Detect Automatically", "ar": "Arabic", "en": "English",
    "fr": "French", "de": "German", "it": "Italian", "ja": "Japanese", "pt": "Portuguese", "es": "Spanish",
}
WHISPER_NO_SPEECH_THRESHOLD = 0.6  # Whisper’s default segment decision threshold.


def custom_spellings(entries):
    words = {}
    for number, line in enumerate(entries.splitlines(), 1):
        if not line.strip():
            continue
        spoken, separator, written = line.partition("=")
        spoken = " ".join(spoken.split())
        written = written.strip() if separator else spoken
        if not spoken or not written:
            raise ValueError(f"Enter both the spoken and written forms on line {number}.")
        key = spoken.casefold()
        if key in words and words[key][1] != written:
            raise ValueError(f"Use one spelling for “{spoken}” on line {number}.")
        words[key] = (spoken, written)
    return sorted(words.values(), key=lambda pair: len(pair[0]), reverse=True)


def apply_spellings(text, entries):
    if not text or not entries.strip():
        return text
    words = custom_spellings(entries)
    if not words:
        return text
    # Match longest phrases first. Replacements are literal and applied once.
    phrases = []
    for index, (spoken, _) in enumerate(words):
        phrase = r"\s+".join(re.escape(word) for word in spoken.split())
        phrases.append(f"(?P<word{index}>{phrase})")
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(phrases) + r")(?!\w)", re.IGNORECASE)
    return pattern.sub(lambda m: words[m.lastindex - 1][1], text)


@dataclass(frozen=True)
class DictationField:
    pid: int
    window: object
    control: object
    selection: tuple
    selected_digest: bytes = field(repr=False)

    @classmethod
    def current(cls):
        if ax.session_locked():
            return None
        pid, _ = ax.frontmost()
        if not pid or not AX.AXIsProcessTrusted():
            return None
        app = AX.AXUIElementCreateApplication(pid)
        AX.AXUIElementSetMessagingTimeout(app, 0.2)
        control = ax.attr(app, "AXFocusedUIElement")
        window = ax.attr(app, "AXFocusedWindow")
        if (
            window is None
            or ax.attr(control, "AXRole") not in ("AXTextArea", "AXTextField", "AXComboBox")
            or ax.attr(control, "AXSubrole") == "AXSecureTextField"
            or ax.attr(control, "AXProtectedContent")
            or AX.AXUIElementIsAttributeSettable(control, "AXSelectedText", None) != (0, True)
        ):
            return None
        text, span = ax.text_selection(control)
        if span is None or (span[1] and not isinstance(text, str)):
            return None
        span, digest = ax.typing_anchor(text or "", span)
        return cls(pid, window, control, span, digest)

    def insert(self, text):
        if self != self.current():
            return False
        # Replace exactly the original selection using the app's text API.
        return AX.AXUIElementSetAttributeValue(self.control, "AXSelectedText", text) == 0


class Recording:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="clicky-voice-")
        self.path = Path(self.directory.name) / "speech.wav"
        self.recorder = None

    def start(self):
        media = AV.AVMediaTypeAudio
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(media)
        if status == AV.AVAuthorizationStatusNotDetermined:
            AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(media, lambda allowed: None)
            raise ValueError("Approve Microphone access, then hold the shortcut again.")
        if status != AV.AVAuthorizationStatusAuthorized:
            raise ValueError("Enable Microphone access in System Settings to use push-to-talk.")
        settings = {
            AV.AVFormatIDKey: int.from_bytes(b"lpcm"),
            AV.AVSampleRateKey: 16000.0,
            AV.AVNumberOfChannelsKey: 1,
            AV.AVLinearPCMBitDepthKey: 16,
            AV.AVLinearPCMIsFloatKey: False,
            AV.AVLinearPCMIsBigEndianKey: False,
        }
        self.recorder, error = AV.AVAudioRecorder.alloc().initWithURL_settings_error_(
            A.NSURL.fileURLWithPath_(str(self.path)), settings, None
        )
        if error or not self.recorder or not self.recorder.recordForDuration_(60):
            raise ValueError("The microphone could not start.")

    def stop(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

    def close(self):
        self.stop()
        self.directory.cleanup()


def transcribe(path, provider, language="default"):
    if language not in DICTATION_LANGUAGES:
        raise ValueError("Choose a supported dictation language in Settings.")
    key = os.getenv("OPENAI_API_KEY")
    if provider == "OpenAI" and key:
        boundary = uuid.uuid4().hex
        language_field = ""
        if language not in ("default", "auto"):
            language_field = f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{language}\r\n'
        header = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
            f"{language_field}"
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        body = header + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "multipart/form-data; boundary=" + boundary,
            },
        )
        with urlopen(req, timeout=35) as response:
            return json.load(response)["text"].strip()
    code = {"default": "en", "auto": None}.get(language, language)
    model = WHISPER if code == "en" else MULTILINGUAL_WHISPER
    try:
        import mlx_whisper
        import numpy as np
        from huggingface_hub import hf_hub_download

        checkpoint = Path(hf_hub_download(model, "config.json", local_files_only=True)).parent
        if not (checkpoint / "weights.npz").is_file() and not (checkpoint / "weights.safetensors").is_file():
            raise FileNotFoundError(model)
    except (ImportError, OSError) as exc:
        raise ValueError(f"Install the local extras and download {model} for this dictation language.") from exc
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("Whisper requires a mono 16 kHz PCM recording.")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768
    result = mlx_whisper.transcribe(
        samples, path_or_hf_repo=str(checkpoint), language=code, verbose=None,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )
    # Whisper can emit confident text even when its acoustic prediction is "no speech".
    # An explicit dictation should stay empty if every segment made that prediction.
    segments = result["segments"]
    if not segments or all(s["no_speech_prob"] > WHISPER_NO_SPEECH_THRESHOLD for s in segments):
        return ""
    return result["text"].strip()
