"""Sensor-event arbitration and Whisper input/output policies."""

import re
from collections import deque
from collections.abc import Iterable, Mapping
from enum import StrEnum


class Mode(StrEnum):
    ACTIVE = "ACTIVE"
    STEALTH = "STEALTH"
    SLEEP = "SLEEP"


class SpeechSource(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"


SPEECH_JAW_LIMIT = 0.28
JAW_CLICK_MIN = 0.2974
JAW_RELEASE_MS = 400
VISUAL_RELEASE_MS = 350


class ModeArbitrator:
    """Consume ordered event edges at times from one monotonic clock in ms.

    Send speech_started for each detected onset and speech_ended for each end.
    Call try_jaw_click for a debounced click candidate when it can be dispatched.
    A mode change clears pending speech and cooldowns.
    """

    def __init__(self, mode: Mode = Mode.ACTIVE):
        self._mode = Mode.ACTIVE
        self._last_active_mode = Mode.ACTIVE
        self._speech: set[SpeechSource] = set()
        self._speech_locks_jaw = False
        self._jaw_blocked_until_ms = -float("inf")
        self._visual_blocked_until_ms = -float("inf")
        self.set_mode(mode)

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def last_active_mode(self) -> Mode:
        return self._last_active_mode

    @property
    def gaze_enabled(self) -> bool:
        return self.mode != Mode.SLEEP

    @property
    def silent_enabled(self) -> bool:
        return self.mode != Mode.SLEEP

    @property
    def microphone_enabled(self) -> bool:
        return self.mode == Mode.ACTIVE

    @property
    def whisper_enabled(self) -> bool:
        return self.microphone_enabled

    @property
    def agent_enabled(self) -> bool:
        return self.mode != Mode.SLEEP

    def set_mode(self, mode: Mode) -> None:
        mode = Mode(mode)
        if mode == self.mode:
            return
        self._mode = mode
        if mode != Mode.SLEEP:
            self._last_active_mode = mode
        self._speech.clear()
        self._speech_locks_jaw = False
        self._jaw_blocked_until_ms = -float("inf")
        self._visual_blocked_until_ms = -float("inf")

    def wake(self) -> Mode:
        if self.mode == Mode.SLEEP:
            self.set_mode(self.last_active_mode)
        return self.mode

    def can_start_visual_speech(self, now_ms: float) -> bool:
        return (
            self.silent_enabled
            and SpeechSource.VISUAL not in self._speech
            and now_ms >= self._visual_blocked_until_ms
        )

    def speech_started(
        self, source: SpeechSource, jaw_open: float, now_ms: float
    ) -> bool:
        """Return whether the detector's start was accepted."""
        source = SpeechSource(source)
        if source == SpeechSource.VISUAL:
            if not self.can_start_visual_speech(now_ms):
                return False
        elif not self.microphone_enabled or source in self._speech:
            return False
        self._speech.add(source)
        if jaw_open < SPEECH_JAW_LIMIT:
            self._speech_locks_jaw = True
        return True

    def speech_ended(self, source: SpeechSource, now_ms: float) -> None:
        source = SpeechSource(source)
        if source not in self._speech:
            return
        self._speech.remove(source)
        if not self._speech and self._speech_locks_jaw:
            self._speech_locks_jaw = False
            self._jaw_blocked_until_ms = now_ms + JAW_RELEASE_MS

    def can_jaw_click(self, now_ms: float) -> bool:
        return (
            self.gaze_enabled
            and not self._speech_locks_jaw
            and now_ms >= self._jaw_blocked_until_ms
        )

    def try_jaw_click(self, jaw_open: float, now_ms: float) -> bool:
        """Accept a wide click candidate and start the visual onset cooldown."""
        if jaw_open >= JAW_CLICK_MIN and self.can_jaw_click(now_ms):
            self._visual_blocked_until_ms = now_ms + VISUAL_RELEASE_MS
            return True
        return False


# Each detector input is 20 ms of mono PCM16 at 16 kHz. Keep the spectral
# calculation identical when fitting training-fold thresholds and at runtime.
FRAME_SAMPLES = 320
NOISE_FRAMES = 25
ONSET_FRAMES = 3


def _speech_power(frames):
    import numpy as np
    frames = np.asarray(frames)
    if frames.shape[-1] != FRAME_SAMPLES:
        raise ValueError("Expected 320 samples of mono PCM")
    spectrum = np.fft.rfft(frames * np.hanning(FRAME_SAMPLES), axis=-1)
    # 50 Hz bins. Use 200 Hz to 4 kHz to suppress rumble and high-frequency hiss.
    return np.maximum(np.sum(abs(spectrum[..., 4:80]) ** 2, axis=-1), 1.0)


class AudioSpeechDetector:
    """Emit edges for sustained excess speech-band power over recent background."""

    def __init__(self, visual_ratio, acoustic_ratio, release_ms, noise_floor=None):
        self.visual_ratio = visual_ratio
        self.acoustic_ratio = acoustic_ratio
        self.release_ms = release_ms
        self.noise_samples = deque(maxlen=NOISE_FRAMES)
        self.noise_floor = noise_floor
        self.consecutive = 0
        self.active = False
        self.last_loud = None

    def update(self, pcm, now_ms, visual_active=False):
        import numpy as np
        power = float(_speech_power(pcm))
        if len(self.noise_samples) < NOISE_FRAMES:
            self.noise_samples.append(power)
            return None
        ratio = self.visual_ratio if visual_active else self.acoustic_ratio
        background = float(np.median(self.noise_samples))
        if self.noise_floor is not None:
            background = max(background, self.noise_floor)
        loud = power > background * ratio
        if not self.active:
            self.noise_samples.append(power)
        self.consecutive = self.consecutive + 1 if loud else 0
        if loud:
            self.last_loud = now_ms
        if not self.active and self.consecutive >= ONSET_FRAMES:
            self.active = True
            return True
        if self.active and not loud and now_ms - self.last_loud >= self.release_ms:
            self.active = False
            self.consecutive = 0
            return False
        return None


def recorded_audio(root, mode):
    """Read all saved clips of a mode and reject missing or incompatible audio."""
    import json
    import wave
    from pathlib import Path
    import numpy as np
    for path in sorted(Path(root).glob('*/*/meta.json')):
        meta = json.loads(path.read_text())
        if not meta.get('saved') or meta.get('mode') != mode:
            continue
        with wave.open(str(path.parent / 'audio_16k.wav'), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError(f"{path.parent}: expected mono PCM16 at 16 kHz")
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        yield path.parent, meta, pcm


def recorded_silent_audio(root):
    return recorded_audio(root, 'silent')


def audio_speech_ratios(root, exclude_session=None):
    """Fit onset ceilings with a between-clip variability margin on silent training data."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    visual_peaks, acoustic_peaks = [], []
    for path, meta, pcm in recorded_silent_audio(root):
        if path.parent.name == exclude_session:
            continue
        n = len(pcm) // FRAME_SAMPLES
        if n < NOISE_FRAMES + ONSET_FRAMES:
            raise ValueError(f"{path}: too short to evaluate an acoustic onset")
        power = _speech_power(pcm[:n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES))
        floors = np.median(sliding_window_view(power, NOISE_FRAMES)[:-1], axis=1)
        excess = 10 * np.log10(power[NOISE_FRAMES:] / floors)
        peaks = sliding_window_view(excess, ONSET_FRAMES).min(axis=1)
        acoustic_peaks.append(max(peaks))
        times = meta['audio_start_ms'] + (NOISE_FRAMES + ONSET_FRAMES + np.arange(len(peaks))) * 20
        visual = np.zeros(len(times), dtype=bool)
        for start, end in meta['vad_segments_ms']:
            visual |= (times >= start) & (times <= end)
        if visual.any():
            visual_peaks.append(max(peaks[visual]))
    if not acoustic_peaks:
        raise ValueError('No recorded silent clips available for acoustic calibration')

    def ceiling(peaks):
        return float(10 ** ((max(peaks) + np.std(peaks)) / 10))

    acoustic = ceiling(acoustic_peaks)
    visual = min(acoustic, ceiling(visual_peaks)) if visual_peaks else acoustic
    return visual, acoustic


class AVAudioEngineEchoCanceler:
    """Voice-processed duplex Core Audio, with 16 kHz mono PortAudio-compatible callbacks.

    Start/stop and poll run on the main thread. The native tap only converts
    captured PCM, reblocks it and forwards it with its acquisition timestamp.
    """

    def __init__(self, callback, device=None):
        self.callback = callback
        self.device = device
        self.engine = self.mixer = self.tap = self.observer = self.fallback = None
        self.route = None
        self.route_changed = False
        self.tap_error = None
        self.active = False

    @staticmethod
    def _default_route():
        import CoreAudio as ca
        import struct

        def prop(obj, selector, fmt):
            size = struct.calcsize(fmt)
            addr = ca.AudioObjectPropertyAddress(selector, ca.kAudioObjectPropertyScopeGlobal,
                                                 ca.kAudioObjectPropertyElementMain)
            status, length, data = ca.AudioObjectGetPropertyData(obj, addr, 0, b"", size, None)
            if status or length != size:
                raise RuntimeError("Could not read the default Core Audio route")
            return struct.unpack(fmt, data)[0]

        devices = [prop(ca.kAudioObjectSystemObject, selector, "=I") for selector in (
            ca.kAudioHardwarePropertyDefaultInputDevice, ca.kAudioHardwarePropertyDefaultOutputDevice)]
        if not all(devices):
            raise RuntimeError("No default duplex audio route")
        rates = [prop(dev, ca.kAudioDevicePropertyNominalSampleRate, "=d") for dev in devices]
        if min(rates) <= 0 or rates[0] != rates[1]:
            raise RuntimeError(f"Input/output sample rates differ or are unavailable ({rates})")
        if devices[0] != devices[1]:
            bluetooth = {ca.kAudioDeviceTransportTypeBluetooth, ca.kAudioDeviceTransportTypeBluetoothLE}
            if any(prop(dev, ca.kAudioDevicePropertyTransportType, "=I") in bluetooth for dev in devices):
                raise RuntimeError("Bluetooth input and output use different devices")
        return (*devices, *rates)

    def _input_tap(self):
        import AVFAudio as av
        import time
        import numpy as np
        from types import SimpleNamespace

        pending = np.empty(0, dtype=np.int16)
        previous_end = None

        def capture(buffer, when):
            nonlocal pending, previous_end
            try:
                fmt = buffer.format()
                if (fmt.sampleRate() != 16000 or fmt.channelCount() != 1 or
                        fmt.commonFormat() != av.AVAudioPCMFormatFloat32 or not when.isHostTimeValid()):
                    raise RuntimeError("Native capture format or clock changed")
                size = buffer.frameLength()
                if not size:
                    return
                raw = buffer.audioBufferList()[0].mData
                floating = np.frombuffer(raw, dtype=np.float32, count=size)
                pcm = np.rint(np.clip(floating * 32768, -32768, 32767)).astype(np.int16)
                position = when.sampleTime() if when.isSampleTimeValid() else None
                if position is not None and previous_end is not None and position != previous_end:
                    raise RuntimeError("Native capture sample clock changed")
                previous_end = None if position is None else position + size
                first = av.AVAudioTime.secondsForHostTime_(when.hostTime()) - len(pending) / 16000
                frames = np.concatenate((pending, pcm))
                consumed = len(frames) // FRAME_SAMPLES * FRAME_SAMPLES
                for i in range(0, consumed, FRAME_SAMPLES):
                    # CPython on macOS and AVAudioTime both use mach_absolute_time.
                    now = time.monotonic()
                    timing = SimpleNamespace(inputBufferAdcTime=first + i / 16000, currentTime=now)
                    self.callback(frames[i:i + FRAME_SAMPLES, None], FRAME_SAMPLES, timing, "")
                pending = frames[consumed:].copy()
            except Exception as exc:
                self.tap_error = str(exc)

        return capture

    def _start_native(self):
        import AVFAudio as av
        from Foundation import NSNotificationCenter
        import objc

        if self.device is not None:
            raise RuntimeError("An explicit PortAudio input device was selected")
        self.route = self._default_route()
        self.engine = engine = av.AVAudioEngine.alloc().init()
        source = engine.inputNode()
        # Apple's Objective-C selector is setVoiceProcessingEnabled:error:.
        if not source.setVoiceProcessingEnabled_error_(True, objc.NULL) or not source.isVoiceProcessingEnabled():
            raise RuntimeError("Core Audio could not enable voice processing")
        native = source.outputFormatForBus_(0)
        if native.sampleRate() <= 0 or not native.channelCount():
            raise RuntimeError("The voice-processed input format is unavailable")
        self.mixer = mixer = av.AVAudioMixerNode.alloc().init()
        engine.attachNode_(mixer)
        engine.connect_to_format_(source, mixer, native)
        mono = av.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(16000, 1)
        # The mixer performs continuous native sample-rate conversion. Its tap
        # runs upstream of the muted main mixer to prevent microphone loopback.
        main = engine.mainMixerNode()
        main.setOutputVolume_(0.0)
        engine.connect_to_format_(mixer, main, mono)
        capture = self._input_tap()
        mixer.installTapOnBus_bufferSize_format_block_(0, FRAME_SAMPLES, mono, capture)
        self.tap = capture
        engine.prepare()
        if not engine.startAndReturnError_(objc.NULL):
            raise RuntimeError("Core Audio could not start voice-processed input")

        def changed(notification):
            self.route_changed = True

        self.observer = NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
            av.AVAudioEngineConfigurationChangeNotification, engine, None, changed)

    def _stop_native(self):
        if self.observer is not None:
            from Foundation import NSNotificationCenter
            NSNotificationCenter.defaultCenter().removeObserver_(self.observer)
            self.observer = None
        if self.engine is not None:
            self.engine.stop()
        if self.tap is not None:
            self.mixer.removeTapOnBus_(0)
        self.tap = self.mixer = self.engine = None
        self.route_changed = False
        self.tap_error = None

    def _start_fallback(self, reason, reset_capture=None):
        self._stop_native()
        self.active = False
        if reset_capture is not None:
            reset_capture()
        import sounddevice as sd
        print(f"Microphone: using sounddevice without echo cancellation ({reason}).", flush=True)
        stream = sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=FRAME_SAMPLES,
                                device=self.device, callback=self.callback)
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self.fallback = stream
        self.active = True

    def start(self):
        if self.active:
            return
        try:
            self._start_native()
        except Exception as exc:
            self._start_fallback(exc)
        else:
            self.active = True
            print("Microphone: Core Audio echo cancellation enabled.", flush=True)

    def poll(self, reset_capture=None):
        """Recover a stopped or reconfigured native engine outside the tap thread."""
        if self.engine is None:
            return False
        reason = self.tap_error
        if not self.engine.isRunning():
            reason = "The native audio route stopped"
        elif self.route_changed:
            self.route_changed = False
            try:
                if self._default_route() != self.route:
                    reason = "The default audio route changed"
            except (RuntimeError, OSError) as exc:
                reason = str(exc)
        if reason:
            self._start_fallback(reason, reset_capture)
            return True
        return False

    def stop(self):
        self.active = False
        self._stop_native()
        if self.fallback is not None:
            self.fallback.close()
            self.fallback = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class EarconPlayer:
    """Play cached macOS system sounds asynchronously, when enabled."""
    SOUNDS = {'click': 'Tink', 'menu': 'Pop', 'ACTIVE': 'Glass',
              'STEALTH': 'Morse', 'SLEEP': 'Submarine', 'lock': 'Pop',
              'tension': 'Tink', 'release': 'Glass'}

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._sounds = {}
        self._players = {}
        self._last_tension = -float('inf')

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            for sound in (*self._sounds.values(), *self._players.values()):
                if sound is not None:
                    sound.stop()
        return self.enabled

    def play(self, cue, tension=None):
        if not self.enabled or cue not in self.SOUNDS:
            return
        try:
            import AppKit
            name = self.SOUNDS[cue]
            if tension is not None:
                # NSSound has no playback-rate property. Use the same system
                # samples with AVAudioPlayer for the variable-rate cues.
                import time
                import AVFoundation as AV
                now = time.monotonic()
                if cue == 'tension' and now - self._last_tension < .18:
                    return
                if cue == 'tension':
                    self._last_tension = now
                if name not in self._players:
                    url = AppKit.NSURL.fileURLWithPath_(f'/System/Library/Sounds/{name}.aiff')
                    player, _ = AV.AVAudioPlayer.alloc().initWithContentsOfURL_error_(url, None)
                    if player is not None:
                        player.setEnableRate_(True)
                        player.prepareToPlay()
                    self._players[name] = player
                if (player := self._players[name]) is not None:
                    player.setCurrentTime_(0.)
                    player.setVolume_(.06 if cue == 'tension' else .12)
                    player.setRate_(.85 + .45 * max(0., min(1., tension)))
                    player.play()
                    return
            if name not in self._sounds:
                sound = AppKit.NSSound.soundNamed_(name)
                if sound is not None:
                    sound.setVolume_(0.15)
                self._sounds[name] = sound
            sound = self._sounds[name]
            if sound is not None:
                if sound.isPlaying():
                    sound.stop()
                sound.play()
        except Exception:
            # Sound availability must never prevent a control action.
            return


class ContextHotwordBiaser:
    """Build a Whisper initial_prompt from visible frontmost-window AX labels."""

    ROLES = frozenset({"AXButton", "AXTab", "AXLink", "AXMenuItem"})
    MAX_LABELS = 45

    def initial_prompt(self, entries: Iterable[Mapping[str, object]]) -> str:
        """Accept window-index entries with role and name, in priority order."""
        labels = []
        seen = set()
        for entry in entries:
            if entry.get("role") not in self.ROLES or not entry.get("visible", True):
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            label = " ".join(name.split())
            key = label.casefold()
            if not label or key in seen:
                continue
            labels.append(label)
            seen.add(key)
            if len(labels) == self.MAX_LABELS:
                break
        return ", ".join(labels)


_HALLUCINATIONS = {("whisper",), ("thank", "you"), ("you",)}


def filter_whisper_transcript(text: str, duration_s: float, snr_db: float) -> str:
    """Drop known short, low-SNR outputs. Duration is the speech interval."""
    if duration_s < 0.45 and snr_db < 9.0:
        words = tuple(re.findall(r"\w+", text.casefold()))
        if words in _HALLUCINATIONS:
            return ""
    return text


if __name__ == "__main__":
    from pathlib import Path
    import unittest

    import numpy as np

    class ModeArbitratorTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            path = Path(__file__).with_name("data") / "session_3" / "top_left_00.npz"
            with np.load(path) as clip:
                column = clip["blendshape_names"].tolist().index("jawOpen")
                jaw = clip["blendshapes"][clip["face_present"].astype(bool), column]
                cls.narrow_jaw = float(jaw.min())
                cls.wide_jaw = float(jaw.max())
            assert cls.narrow_jaw < SPEECH_JAW_LIMIT
            assert cls.wide_jaw >= JAW_CLICK_MIN

        def test_speech_blocks_click_until_400_ms_after_end(self):
            for source in SpeechSource:
                with self.subTest(source=source):
                    arb = ModeArbitrator()
                    self.assertTrue(arb.speech_started(source, self.narrow_jaw, 0))
                    self.assertFalse(arb.speech_started(source, self.narrow_jaw, 1))
                    self.assertFalse(arb.try_jaw_click(self.wide_jaw, 600))
                    arb.speech_ended(source, 800)
                    self.assertFalse(arb.try_jaw_click(self.wide_jaw, 1199))
                    self.assertTrue(arb.try_jaw_click(self.wide_jaw, 1200))

        def test_overlapping_speech_releases_after_last_end(self):
            arb = ModeArbitrator()
            arb.speech_started(SpeechSource.VISUAL, self.narrow_jaw, 0)
            arb.speech_started(SpeechSource.AUDIO, self.wide_jaw, 10)
            arb.speech_ended(SpeechSource.VISUAL, 100)
            self.assertFalse(arb.can_jaw_click(500))
            arb.speech_ended(SpeechSource.AUDIO, 500)
            self.assertFalse(arb.can_jaw_click(899))
            self.assertTrue(arb.can_jaw_click(900))

        def test_wide_speech_onset_does_not_block_clicks(self):
            arb = ModeArbitrator()
            self.assertTrue(arb.speech_started(SpeechSource.AUDIO, self.wide_jaw, 0))
            self.assertTrue(arb.try_jaw_click(self.wide_jaw, 1))

        def test_jaw_click_blocks_visual_triggers_for_350_ms(self):
            arb = ModeArbitrator()
            self.assertFalse(arb.try_jaw_click(self.narrow_jaw, 0))
            self.assertTrue(arb.can_start_visual_speech(0))
            self.assertTrue(arb.try_jaw_click(self.wide_jaw, 100))
            self.assertFalse(arb.speech_started(SpeechSource.VISUAL, self.narrow_jaw, 449))
            self.assertTrue(arb.speech_started(SpeechSource.AUDIO, self.narrow_jaw, 449))
            self.assertTrue(arb.speech_started(SpeechSource.VISUAL, self.narrow_jaw, 450))

        def test_mode_gates_and_wake(self):
            for mode in Mode.ACTIVE, Mode.STEALTH:
                with self.subTest(mode=mode):
                    arb = ModeArbitrator(mode)
                    self.assertTrue(arb.gaze_enabled)
                    self.assertTrue(arb.silent_enabled)
                    self.assertTrue(arb.agent_enabled)
                    self.assertEqual(arb.microphone_enabled, mode == Mode.ACTIVE)
                    self.assertEqual(arb.whisper_enabled, mode == Mode.ACTIVE)
                    self.assertEqual(
                        arb.speech_started(SpeechSource.AUDIO, self.narrow_jaw, 0),
                        mode == Mode.ACTIVE,
                    )
                    self.assertTrue(arb.speech_started(SpeechSource.VISUAL, self.narrow_jaw, 0))
                    arb.set_mode(Mode.SLEEP)
                    arb.set_mode(Mode.SLEEP)
                    self.assertEqual(arb.last_active_mode, mode)
                    self.assertFalse(any((
                        arb.gaze_enabled, arb.silent_enabled, arb.agent_enabled,
                        arb.microphone_enabled, arb.whisper_enabled,
                    )))
                    for source in SpeechSource:
                        self.assertFalse(arb.speech_started(source, self.narrow_jaw, 1))
                    self.assertFalse(arb.try_jaw_click(self.wide_jaw, 1))
                    self.assertEqual(arb.wake(), mode)
                    self.assertTrue(arb.try_jaw_click(self.wide_jaw, 2))

    class EchoCaptureTests(unittest.TestCase):
        def test_native_tap_reblocks_recorded_whisper(self):
            try:
                import AVFAudio as av
                from Foundation import NSURL
                import objc
            except ImportError:
                self.skipTest("Requires the macOS AVFAudio PyObjC bindings")
            folder, meta, pcm = next(recorded_audio(Path(__file__).with_name("data") / "silent_speech", "whisper"))
            url = NSURL.fileURLWithPath_(str(folder / "audio_16k.wav"))
            file = av.AVAudioFile.alloc().initForReading_error_(url, objc.NULL)
            blocks = []

            def receive(samples, count, timing, status):
                self.assertEqual(count, FRAME_SAMPLES)
                self.assertEqual(status, "")
                blocks.append((timing.inputBufferAdcTime, samples[:, 0].copy()))

            backend = AVAudioEngineEchoCanceler(receive)
            tap = backend._input_tap()
            # AVFoundation returns variable tap sizes. Read an existing WAV in
            # non-aligned chunks so real samples span callback boundaries.
            buffer = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(file.processingFormat(), 500)
            cursor = 0
            while cursor < file.length():
                self.assertTrue(file.readIntoBuffer_frameCount_error_(buffer, min(500, file.length() - cursor), objc.NULL))
                first = meta["audio_start_ms"] / 1000 + cursor / 16000
                when = av.AVAudioTime.timeWithHostTime_sampleTime_atRate_(
                    av.AVAudioTime.hostTimeForSeconds_(first), cursor, 16000)
                tap(buffer, when)
                self.assertIsNone(backend.tap_error)
                cursor += buffer.frameLength()
            count = len(pcm) // FRAME_SAMPLES
            self.assertEqual(len(blocks), count)
            np.testing.assert_array_equal(np.concatenate([samples for _, samples in blocks]), pcm[:count * FRAME_SAMPLES])
            np.testing.assert_allclose([t for t, _ in blocks],
                                       meta["audio_start_ms"] / 1000 + .02 * np.arange(count), rtol=0, atol=1e-6)

    class WhisperPolicyTests(unittest.TestCase):
        def test_prompt_filters_and_caps_unique_labels(self):
            entries = [
                {"role": "AXButton", "name": "  New\tWindow "},
                {"role": "AXTab", "name": "new window"},
                {"role": "AXLink", "name": "Help"},
                {"role": "AXMenuItem", "name": "Export"},
                {"role": "AXHeading", "name": "Omit heading"},
                {"role": "AXButton", "name": "Omit hidden", "visible": False},
                {"role": "AXLink", "name": " "},
            ]
            entries.extend({"role": "AXTab", "name": f"Tab {i}"} for i in range(50))
            biaser = ContextHotwordBiaser()
            self.assertEqual(biaser.initial_prompt([]), "")
            self.assertEqual(
                biaser.initial_prompt(entries).split(", "),
                ["New Window", "Help", "Export"] + [f"Tab {i}" for i in range(42)],
            )

        def test_filters_only_known_short_low_snr_transcripts(self):
            for text in ("Whisper.", " Thank YOU! ", "you"):
                with self.subTest(text=text):
                    self.assertEqual(filter_whisper_transcript(text, 0.449, 8.99), "")
                    self.assertEqual(filter_whisper_transcript(text, 0.45, 8.99), text)
                    self.assertEqual(filter_whisper_transcript(text, 0.449, 9.0), text)
            for text in ("Stop.", "Open Safari.", "Thank you for that.", "Whisper mode."):
                with self.subTest(text=text):
                    self.assertEqual(filter_whisper_transcript(text, 0.449, 8.99), text)

    unittest.main(verbosity=2)
