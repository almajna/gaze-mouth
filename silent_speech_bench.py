"""Collect a silent/whispered phrase benchmark using the project's shadow mouth trigger."""
import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from queue import SimpleQueue
from tempfile import TemporaryDirectory
import wave

import cv2
import mediapipe as mp
import numpy as np

from capture import ensure_model, extract, make_landmarker
from eval_offline import TRIGGER_CONFIG_PATH, lip_gap
from live_app import Trigger, play


ROOT = Path(__file__).resolve().parent
DIGEST = "9b09a39ffd2d5463d62e352f035595c112762661265ee44ebaa87bd4593424e8"
FPS, RATE, SIDE = 25, 16000, 96
FRAME_MS = 1000 // FPS
ON_MS, OFF_MS = 120, 350
NOISE_MS = 1000
PRE_ROLL_MS, POST_ROLL_MS = 250, 150
INSTALL = "uv pip install --python .venv-gf/bin/python sounddevice"
RUN = ".venv-gf/bin/python silent_speech_bench.py --collect"

# Broad reference pronunciations. A plain camera recording has no phoneme alignment.
PROMPTS = [
    ("open terminal", "OW P AH N | T ER M AH N AH L"),
    ("scroll down", "S K R OW L | D AW N"),
    ("alpha bravo", "AE L F AH | B R AA V OW"),
    ("summarize this", "S AH M ER AY Z | DH IH S"),
    ("what time is it", "W AH T | T AY M | IH Z | IH T"),
    ("zoom out", "Z UW M | AW T"),
    ("copy image", "K AA P IY | IH M IH JH"),
    ("go to audio", "G OW | T UW | AO D IY OW"),
    ("move it over", "M UW V | IH T | OW V ER"),
    ("undo typing", "AH N D UW | T AY P IH NG"),
    ("open a new tab", "OW P AH N | AH | N UW | T AE B"),
    ("page up", "P EY JH | AH P"),
    ("echo india", "EH K OW | IH N D IY AH"),
    ("read it aloud", "R IY D | IH T | AH L AW D"),
    ("where am I", "W EH R | AE M | AY"),
    ("save as audio", "S EY V | AE Z | AO D IY OW"),
    ("show all files", "SH OW | AO L | F AY L Z"),
    ("lower volume", "L OW ER | V AA L Y UW M"),
    ("go home now", "G OW | HH OW M | N AW"),
    ("pause the video", "P AO Z | DH AH | V IH D IY OW"),
]


def verified_config():
    raw = TRIGGER_CONFIG_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError(f"Unexpected SHA-256 for {TRIGGER_CONFIG_PATH}")
    return json.loads(raw)


def shadow_update(trigger, cfg, t, lm, jaw):
    active = lm is not None and jaw >= cfg["jawOpen"] and lip_gap({"lm": lm[None]})[0] >= cfg["lip_gap"]
    return trigger.update(t, active)


def mouth_geometry(lm, width, height, pucker):
    """Face-relative similarity affine. Eye span sets scale, preserving lip expansion."""
    xy = lm[[33, 263, 61, 291, 13, 14], :2] * [width, height]
    eye1, eye2, left, right, upper, lower = xy
    axis = eye2 - eye1
    span = np.linalg.norm(axis)
    if not np.isfinite(xy).all() or not np.isfinite(pucker) or span < 1:
        return None, None
    u = axis / span
    v = np.array([-u[1], u[0]])
    basis = np.stack((u, v))
    center = (left + right + upper + lower) / 4
    scale = SIDE / span
    affine = np.column_stack((basis * scale, (SIDE - 1) / 2 - scale * (basis @ center)))
    features = np.array([np.linalg.norm(right - left) / span, pucker, (lower - upper) @ v / span])
    return affine, features


def reference_data():
    """Per-axis non-speech velocity p95 from recorded fixation windows and negative activities."""
    sessions = {}
    for folder in sorted((ROOT / "data").glob("session_*")):
        speeds = []
        for path in sorted(folder.glob("*.npz")):
            with np.load(path) as d:
                ids = np.flatnonzero(np.isin(d["cue"], ["fixate", "negative"]) & d["face_present"])
                if len(ids) < 2:
                    continue
                ts, lm = d["timestamps_ms"], d["landmarks"]
                pk = d["blendshapes"][:, list(d["blendshape_names"]).index("mouthPucker")]
                w, h = int(d["width"]), int(d["height"])
                prev = None
                for i in ids:
                    _, f = mouth_geometry(lm[i], w, h, pk[i])
                    if prev is not None and f is not None and i == prev + 1 and ts[i] > ts[prev]:
                        speeds.append(abs(f - last) * 1000 / (ts[i] - ts[prev]))
                    prev = i if f is not None else None
                    last = f
        if speeds:
            sessions[folder] = np.stack(speeds)
    if not sessions:
        raise ValueError("No recorded fixation windows available to estimate V-VAD thresholds")
    return sessions


class VisualVAD:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.prev_t = self.prev = self.moving = self.still = None
        self.active = False

    def update(self, t, features):
        """Return the segment's first moving frame on confirmation and its end on release."""
        start = end = None
        if features is None or self.prev is None or not 0 < t - self.prev_t <= ON_MS:
            self.moving = self.still = None
            # Missing tracking is unknown, never evidence of articulation or stillness.
            self.prev_t, self.prev = t, features
            return start, end
        moving = np.any(abs(features - self.prev) * 1000 / (t - self.prev_t) > self.thresholds)
        self.prev_t, self.prev = t, features
        if moving:
            self.still = None
            if self.moving is None:
                self.moving = t
            if not self.active and t - self.moving >= ON_MS:
                self.active, start = True, self.moving
        else:
            self.moving = None
            if self.still is None:
                self.still = t
            if self.active and t - self.still >= OFF_MS:
                self.active, end = False, t
        return start, end


def check_recordings(cfg, sessions):
    """Leave each negative session out of the V-VAD noise estimate when scoring it."""
    for folder in sessions:
        others = [v for k, v in sessions.items() if k != folder]
        if not others:
            raise ValueError("Need a second session for a held-out V-VAD check")
        thresholds = np.percentile(np.vstack(others), 95, axis=0)
        clicks = starts = clips = false_clips = 0
        for path in sorted(folder.glob("neg_*.npz")):
            vad, trigger = VisualVAD(thresholds), Trigger(cfg["debounce_ms"])
            clip_starts = 0
            with np.load(path) as d:
                names = list(d["blendshape_names"])
                jaw, pucker = d["blendshapes"][:, [names.index("jawOpen"), names.index("mouthPucker")]].T
                lm, w, h = d["landmarks"], int(d["width"]), int(d["height"])
                for i, (t, present) in enumerate(zip(d["timestamps_ms"], d["face_present"])):
                    clicks += shadow_update(trigger, cfg, int(t), lm[i] if present else None, jaw[i]) is not None
                    f = mouth_geometry(lm[i], w, h, pucker[i])[1] if present else None
                    clip_starts += vad.update(int(t), f)[0] is not None
            clips += 1
            starts += clip_starts
            false_clips += clip_starts > 0
        print(f"{folder.name}: {clicks} false shadow clicks; held-out V-VAD activated on "
              f"{false_clips}/{clips} negative clips ({starts} starts)")


def audio_window(blocks, start, n):
    """Slice continuous mic samples by PortAudio ADC time (seconds). Reject unavailable/overflowed audio."""
    parts = []
    remaining = n
    for adc, pcm, status in blocks:
        if not parts:
            offset = round((start - adc) * RATE)
            if offset >= len(pcm):
                continue
            if offset < 0:
                break
        else:
            offset = 0
        if status:
            raise ValueError(f"microphone reported {status}")
        take = min(remaining, len(pcm) - offset)
        parts.append(pcm[offset:offset + take])
        remaining -= take
        if not remaining:
            return np.concatenate(parts)
    raise ValueError("microphone buffer does not cover the requested interval")


def whisper_snr(signal, noise):
    def power(x):
        return np.var(x.astype(np.float64))
    noise_power = power(noise)
    speech_power = power(signal) - noise_power
    if not noise_power or speech_power <= 0:
        return None
    return float(10 * np.log10(speech_power / noise_power))


@dataclass
class MouthFrame:
    t: int
    crop: np.ndarray | None
    landmarks: np.ndarray
    shapes: np.ndarray
    names: list | None


def segment_bounds(segments):
    first = segments[0][0] - PRE_ROLL_MS
    end = segments[-1][1] + POST_ROLL_MS
    ticks = (end - first + FRAME_MS - 1) // FRAME_MS
    return first, first + ticks * FRAME_MS


def save_segment(directory, meta, frames, blocks, epoch):
    """Use actual captured crops nearest each 40 ms tick. Audio has exactly 640 samples per tick."""
    first, last = segment_bounds(meta["vad_segments_ms"])
    times = np.array([f.t for f in frames])
    ticks = np.arange(first, last, FRAME_MS)
    right = times.searchsorted(ticks).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    idx = np.where(abs(times[left] - ticks) <= abs(times[right] - ticks), left, right)
    if np.any(abs(times[idx] - ticks) > FRAME_MS) or any(frames[i].crop is None for i in idx):
        raise ValueError("missing mouth tracking or camera frames in the segment")
    mouth = np.stack([frames[i].crop for i in idx])
    names = frames[idx[0]].names
    if not names or len(names) != 52 or any(frames[i].names != names for i in idx):
        raise ValueError("missing or inconsistent blendshape channels")
    shapes = np.stack([frames[i].shapes for i in idx])
    if shapes.shape != (len(mouth), 52) or not np.isfinite(shapes).all():
        raise ValueError("invalid blendshape observations")
    pcm = audio_window(blocks, epoch + first / 1000, len(mouth) * (RATE // FPS))
    meta.update(video_fps=FPS, sample_rate=RATE, channels=1, audio_start_ms=first,
                blendshape_names=names, pre_roll_ms=PRE_ROLL_MS,
                post_roll_ms=last-meta["vad_segments_ms"][-1][1], source_frame_times_ms=times[idx].tolist(), whisper_snr_db=None)
    if meta["mode"] == "whisper":
        noise = audio_window(blocks, epoch + (meta["speech_start_ms"] - NOISE_MS) / 1000, RATE)
        # Exclude V-VAD release padding from the speech power estimate.
        lo = max(first, meta["speech_start_ms"])
        hi = min(first + len(mouth) * FRAME_MS, meta["speech_end_ms"], meta["vad_segments_ms"][-1][1] - OFF_MS)
        if hi > lo:
            meta["whisper_snr_db"] = whisper_snr(pcm[(lo-first) * RATE // 1000:(hi-first) * RATE // 1000], noise)
    meta["saved"] = True
    with TemporaryDirectory(dir=directory.parent) as work:
        stage = Path(work) / "clip"
        stage.mkdir()
        np.save(stage / "mouth_96x96.npy", mouth)
        np.save(stage / "blendshapes.npy", shapes)
        with wave.open(str(stage / "audio_16k.wav"), "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            wav.writeframes(pcm.astype("<i2", copy=False).tobytes())
        (stage / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        stage.rename(directory)


def report(attempts):
    n = len(attempts)
    print(f"\nCollision report for {n} marked phrase intervals (including retries)")
    for name in ("raw", "gated"):
        counts = [sum(name == "raw" or not e["vad_active"] for e in a["speech_clicks"]) for a in attempts]
        rate = f"{100 * sum(c > 0 for c in counts) / n:.1f}%" if n else "n/a"
        print(f"{name}: {sum(counts)} collisions; {sum(c > 0 for c in counts)}/{n} intervals collided ({rate})")
    print(f"V-VAD misses: {sum(not a['vad_segments_ms'] for a in attempts)}/{n}; "
          f"clips saved: {sum(a['saved'] for a in attempts)}/{len(PROMPTS)}")
    for field, label in (("capture_latency_ms", "Speech capture latency (Space to first V-VAD decision, ms)"),
                         ("confirmation_latency_ms", "First visual motion to V-VAD decision (ms)"),
                         ("whisper_snr_db", "Whisper SNR (dB)")):
        values = [a[field] for a in attempts if a.get(field) is not None]
        stats = f"mean {np.mean(values):.2f}, p95 {np.percentile(values, 95):.2f}" if values else "n/a"
        print(f"{label}: {stats} (n={len(values)})")
    whispers = [a for a in attempts if a["mode"] == "whisper" and a["saved"]]
    print(f"Whispers with unmeasurable SNR: {sum(a.get('whisper_snr_db') is None for a in whispers)}/{len(whispers)}")


def collect(cfg, thresholds, camera, mic):
    import sounddevice as sd

    ensure_model()
    directory = ROOT / "data" / "silent_speech" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    queue = SimpleQueue()

    def callback(indata, count, timing, status):
        queue.put((timing.inputBufferAdcTime, indata[:, 0].copy(), str(status)))

    attempts = []
    attempt = None
    print(f"Output: {directory}\nFirst ten: completely silent. Next ten: barely whispered.\nSPACE at speech onset; ENTER at phrase end. R retries. Q exits.")
    try:
        with ExitStack() as resources:
            landmarker = resources.enter_context(make_landmarker())
            cap = cv2.VideoCapture(camera)
            resources.callback(cap.release)
            resources.callback(cv2.destroyAllWindows)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            stream = resources.enter_context(sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                                            device=mic, callback=callback))
            events = resources.enter_context((directory / "events.jsonl").open("x", buffering=1))
            results = resources.enter_context((directory / "attempts.jsonl").open("x", buffering=1))
            epoch = stream.time
            trigger, vad = Trigger(cfg["debounce_ms"]), VisualVAD(thresholds)
            blocks = deque()  # ADC ring also retains the longer room-noise baseline.
            ready_frames = deque()
            trial, attempt, previous_t = 0, None, -1
            ready_since = 0
            saved_at = None
            while trial < len(PROMPTS):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera frame acquisition failed")
                t = max(previous_t + 1, round((stream.time - epoch) * 1000))
                previous_t = t
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
                lm, bs, names, _, present = extract(result)
                crop = features = None
                jaw = 0
                if present and names:
                    jaw = float(bs[names.index("jawOpen")])
                    affine, features = mouth_geometry(lm, frame.shape[1], frame.shape[0], bs[names.index("mouthPucker")])
                    if affine is not None:
                        crop = cv2.warpAffine(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), affine, (SIDE, SIDE),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                start, end = vad.update(t, features)
                onset = shadow_update(trigger, cfg, t, lm if present and names else None, jaw)
                phrase, phones = PROMPTS[trial]
                mode = "silent" if trial < 10 else "whisper"
                while not queue.empty():
                    blocks.append(queue.get_nowait())
                observation = MouthFrame(t, crop, lm, bs, names)
                if attempt is None:
                    ready_frames.append(observation)
                    while len(ready_frames) > 1 and ready_frames[1].t <= t - PRE_ROLL_MS:
                        ready_frames.popleft()
                    while blocks and blocks[0][0] + len(blocks[0][1]) / RATE < epoch + (t - max(NOISE_MS, PRE_ROLL_MS)) / 1000:
                        blocks.popleft()
                else:
                    frames.append(observation)
                    stop = attempt["speech_end_ms"]
                    if start is not None and (stop is None or start <= stop):
                        attempt["vad_segments_ms"].append([start, None])
                        if attempt["capture_latency_ms"] is None:
                            decision = round((stream.time - epoch) * 1000)
                            attempt["capture_latency_ms"] = decision - attempt["speech_start_ms"]
                            attempt["confirmation_latency_ms"] = decision - start
                    segments = attempt["vad_segments_ms"]
                    if end is not None and segments and segments[-1][1] is None:
                        segments[-1][1] = end
                if onset is not None:
                    event = dict(event="SHADOW_CLICK", fire_ms=t, hold_onset_ms=onset, vad_active=vad.active,
                                 phrase=phrase, reference_phonemes=phones, mode=mode, trial=trial+1,
                                 attempt_number=len(attempts)+1 if attempt is not None else None,
                                 in_speech_window=attempt is not None and attempt["speech_end_ms"] is None)
                    print(json.dumps(event), flush=True)
                    events.write(json.dumps(event) + "\n")
                    if event["in_speech_window"]:
                        attempt["speech_clicks"].append(event)

                preview = cv2.resize(frame, (800, 450))
                if crop is not None:
                    lip_points = lm[[61, 291, 0, 17, 13, 14], :2] * [800, 450]
                    x1, y1 = np.floor(lip_points.min(axis=0) - 6).astype(int)
                    x2, y2 = np.ceil(lip_points.max(axis=0) + 6).astype(int)
                    lip_color = (0, 75, 255) if attempt is not None else (40, 230, 80)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), lip_color, 2)
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (799, 149), (15, 18, 24), -1)
                cv2.addWeighted(overlay, 0.8, preview, 0.2, 0, preview)

                def hud_text(text, pos, scale, color, thickness):
                    x, y = pos
                    cv2.putText(preview, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (0, 0, 0), thickness, cv2.LINE_AA)
                    cv2.putText(preview, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                scale, color, thickness, cv2.LINE_AA)

                hud_text(f'{trial+1}/20 {mode.upper()}: "{phrase}"', (12, 36), 1.0, (0, 255, 255), 2)
                hud_text("SPACE = start | ENTER = finish | R = redo | Q = quit", (12, 72), .58, (255, 255, 255), 1)
                face_status = "LOCKED" if crop is not None else "LOST"
                vad_status = "SPEAKING" if vad.active else "IDLE"
                hud_text(f"V-VAD: {vad_status}   FACE: {face_status}", (12, 105), .62,
                         (255, 255, 0) if vad.active else (255, 255, 255), 1)
                if saved_at is not None:
                    message, color = "SAVED! Advancing to next phrase...", (35, 190, 45)
                elif attempt is None:
                    ready = crop is not None and not vad.active and t - ready_since >= NOISE_MS
                    message = "STATUS: READY - Press SPACE to start" if ready else "STATUS: WAIT - Hold still for READY"
                    color = (35, 160, 45) if ready else (70, 85, 100)
                elif attempt["speech_end_ms"] is None:
                    elapsed = max(0, t - attempt["speech_start_ms"]) / 1000
                    message, color = f"RECORDING ({elapsed:.1f}s) - Mouth phrase, then press ENTER", (0, 0, 220)
                else:
                    message, color = "ENDING - Hold still while saving | R = redo", (0, 100, 190)
                cv2.rectangle(preview, (0, 150), (799, 195), color, -1)
                hud_text(message, (12, 180), .66, (255, 255, 255), 2)
                cv2.imshow("Silent speech benchmark", preview)
                key = cv2.waitKey(1) & 255
                if key in (27, ord("q")):
                    break
                mark = round((stream.time - epoch) * 1000)
                if saved_at is not None:
                    if mark - saved_at >= 800:
                        trial += 1
                        saved_at = None
                        ready_since = mark
                    continue
                if attempt is None:
                    if key == 32 and crop is not None and not vad.active and mark - ready_since >= NOISE_MS:
                        frames = list(ready_frames)
                        ready_frames.clear()
                        vad = VisualVAD(thresholds)
                        attempt = dict(attempt_number=len(attempts)+1, trial=trial+1, phrase=phrase, reference_phonemes=phones, mode=mode,
                                       speech_start_ms=mark, speech_end_ms=None, speech_clicks=[], saved=False,
                                       capture_latency_ms=None, confirmation_latency_ms=None, vad_segments_ms=[], config_sha256=DIGEST,
                                       velocity_p95=thresholds.tolist(), run=directory.name)
                        play("/System/Library/Sounds/Tink.aiff")
                    continue
                retry = key == ord("r")
                if retry and attempt["speech_end_ms"] is None:
                    attempt["speech_end_ms"] = mark
                if key in (10, 13) and attempt["speech_end_ms"] is None:
                    attempt["speech_end_ms"] = mark
                if attempt["speech_end_ms"] is None:
                    continue
                segments = attempt["vad_segments_ms"]
                if not retry:
                    if t < attempt["speech_end_ms"] + OFF_MS or (segments and segments[-1][1] is None):
                        continue
                    if segments:
                        # The ADC callback may lag video. Wait until the last output tick has audio.
                        _, last = segment_bounds(segments)
                        if t < last:
                            continue
                        needed = epoch + last / 1000
                        available = blocks[-1][0] + len(blocks[-1][1]) / RATE if blocks else epoch
                        if available < needed and t < max(last, attempt["speech_end_ms"]) + 1000:
                            continue
                try:
                    if retry:
                        vad = VisualVAD(thresholds)
                        raise ValueError("Operator requested another take")
                    if not segments:
                        raise ValueError("V-VAD missed this phrase")
                    save_segment(directory / f"{trial+1:02}_{mode}", attempt, frames, blocks, epoch)
                except ValueError as exc:
                    attempt.update(saved=False, failure=str(exc))
                    print(f"Retry {phrase!r}: {exc}")
                else:
                    print(f"Saved {trial+1:02}_{mode}")
                    saved_at = round((stream.time - epoch) * 1000)
                    play("/System/Library/Sounds/Glass.aiff")
                attempts.append(attempt)
                results.write(json.dumps(attempt) + "\n")
                ready_since, attempt = t, None
    finally:
        if attempt is not None:
            attempt["failure"] = "Collection interrupted"
            with (directory / "attempts.jsonl").open("a") as f:
                f.write(json.dumps(attempt) + "\n")
            if attempt["speech_end_ms"] is not None:
                attempts.append(attempt)
            else:
                print("Unfinished phrase logged and excluded from marked-interval rates.")
        report(attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f"Install: {INSTALL}\nRun: {RUN}",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect", action="store_true", help="run the interactive 20-phrase pass")
    mode.add_argument("--check", action="store_true", help="check the detector on recorded negative clips")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mic", type=int, help="sounddevice input device index (default: system input)")
    args = parser.parse_args()
    cfg = verified_config()
    print(f"Verified trigger config SHA-256 {DIGEST}\nInstall: {INSTALL}\nRun: {RUN}")
    if args.check or args.collect:
        sessions = reference_data()
        if args.check:
            check_recordings(cfg, sessions)
        else:
            collect(cfg, np.percentile(np.vstack(list(sessions.values())), 95, axis=0), args.camera, args.mic)


if __name__ == "__main__":
    main()
