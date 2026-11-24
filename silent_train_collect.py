"""Record one randomized personal silent-command session with synchronized training streams."""
import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from queue import SimpleQueue
import json
import random

import cv2
import mediapipe as mp
import numpy as np

from capture import ensure_model, extract, make_landmarker
from live_app import play
from silent_speech_bench import (ROOT, PROMPTS, PRE_ROLL_MS, OFF_MS, NOISE_MS, FPS, RATE, SIDE,
                                 MouthFrame, VisualVAD, mouth_geometry, reference_data, save_segment,
                                 segment_bounds)


CLASSES = tuple(phrase for phrase, _ in PROMPTS[:10]) + (
    "page up", "page down", "new tab", "close tab", "go back", "click here", "select all",
    "delete that", "pause video", "lower volume", "__NULL__")
NULL_MOVES = ("Swallow", "Press lips", "Glance aside", "Smile slightly")


def make_schedule(takes):
    rng = random.SystemRandom()
    moves = list(NULL_MOVES)
    rng.shuffle(moves)
    schedule = [(label, moves[i % len(moves)] if label == "__NULL__" else label)
                for i in range(takes) for label in CLASSES]
    rng.shuffle(schedule)
    return schedule


def collect(schedule, thresholds, camera, mic):
    import sounddevice as sd

    ensure_model()
    directory = ROOT / "data" / "silent_train" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    queue = SimpleQueue()

    def callback(indata, count, timing, status):
        queue.put((timing.inputBufferAdcTime, indata[:, 0].copy(), str(status)))

    saved = 0
    attempt = None
    print(f"Output: {directory}\nAll takes are silent. Read the prompt before pressing SPACE.\nSPACE at speech onset; ENTER at phrase end. R retries. Q exits.")
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
            results = resources.enter_context((directory / "attempts.jsonl").open("x", buffering=1))
            epoch = stream.time
            vad = VisualVAD(thresholds)
            blocks = deque()  # ADC ring also retains the longer room-noise baseline.
            ready_frames = deque()
            trial, attempt, previous_t = 0, None, -1
            ready_since = 0
            saved_at = None
            while trial < len(schedule):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera frame acquisition failed")
                t = max(previous_t + 1, round((stream.time - epoch) * 1000))
                previous_t = t
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
                lm, bs, names, _, present = extract(result)
                crop = features = None
                if present and names:
                    affine, features = mouth_geometry(lm, frame.shape[1], frame.shape[0], bs[names.index("mouthPucker")])
                    if affine is not None:
                        crop = cv2.warpAffine(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), affine, (SIDE, SIDE),
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                start, end = vad.update(t, features)
                phrase, instruction = schedule[trial]
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

                hud_text(f'{trial+1}/{len(schedule)} SILENT: {instruction}', (12, 36), 1.0, (0, 255, 255), 2)
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
                    message, color = f"RECORDING ({elapsed:.1f}s) - Perform prompt, then press ENTER", (0, 0, 220)
                else:
                    message, color = "ENDING - Hold still while saving | R = redo", (0, 100, 190)
                cv2.rectangle(preview, (0, 150), (799, 195), color, -1)
                hud_text(message, (12, 180), .66, (255, 255, 255), 2)
                cv2.imshow("Silent command training", preview)
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
                        attempt = dict(trial=trial+1, phrase=phrase, instruction=instruction, mode="silent",
                                       speech_start_ms=mark, speech_end_ms=None, vad_segments_ms=[],
                                       capture_latency_ms=None, saved=False, run=directory.name)
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
                # Mark nulls independently of detector activation, including glances.
                if phrase == "__NULL__":
                    segments = [[attempt["speech_start_ms"], attempt["speech_end_ms"]]]
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
                    if phrase == "__NULL__":
                        attempt["detected_segments_ms"] = attempt["vad_segments_ms"]
                        attempt["vad_segments_ms"] = segments
                        attempt["segment_source"] = "manual_null"
                    save_segment(directory / f"{trial+1:03}", attempt, frames, blocks, epoch)
                except ValueError as exc:
                    attempt.update(saved=False, failure=str(exc))
                    print(f"Retry {phrase!r}: {exc}")
                else:
                    saved += 1
                    print(f"Saved {trial+1}/{len(schedule)}")
                    saved_at = round((stream.time - epoch) * 1000)
                    play("/System/Library/Sounds/Glass.aiff")
                results.write(json.dumps(attempt) + "\n")
                ready_since, attempt = t, None
    finally:
        if attempt is not None:
            attempt["failure"] = "Collection interrupted"
            with (directory / "attempts.jsonl").open("a") as f:
                f.write(json.dumps(attempt) + "\n")
        print(f"Saved {saved}/{len(schedule)} clips in {directory}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--takes-per-class", type=int, default=3)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mic", type=int)
    args = ap.parse_args()
    if args.takes_per_class < 1:
        ap.error("--takes-per-class must be positive")
    speeds = np.vstack(list(reference_data().values()))
    collect(make_schedule(args.takes_per_class), np.percentile(speeds, 95, axis=0), args.camera, args.mic)


if __name__ == "__main__":
    main()
