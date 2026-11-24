#!/usr/bin/env python3
"""Record one silent sentence per clip. Focus the OpenCV window to use its controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import textwrap
from time import monotonic_ns

import cv2
import numpy as np

from silent_avsr.features import DATA
from silent_avsr.sentence_video import (
    FACEMARK, MEAN_FACE, RATE_HZ, Landmarks68, align_mouth, sentence_crops,
)

WINDOW = "Sentence recorder"
SPLITS = {"eval_holdout_sentences": "eval_holdout", "train_sentences": "train_sentences"}
MAX_RECORDING_NS = 30_000_000_000


def save_clip(path, frames, timestamps, landmarks, reference, metadata):
    points = np.full((len(frames), 68, 2), np.nan, np.float32)
    for i, fit in enumerate(landmarks):
        if fit is not None:
            points[i] = fit
    crops, grid = sentence_crops(frames, timestamps, points, reference)
    metadata.update(video_fps=RATE_HZ, raw_frame_count=len(frames), crop_frame_count=len(crops),
                    missing_face_frames=sum(fit is None for fit in landmarks),
                    raw_pixel_format="BGR8", alignment="Auto-AVSR 68-point mean face / OpenCV LBF",
                    timestamp_origin="first captured frame", recorded_at=datetime.now(timezone.utc).isoformat())
    # Stage both outputs so a failed capture or encode leaves the previous archive intact.
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".staging-") as staging:
        staged = Path(staging) / path.name
        preview = staged.with_suffix(".mp4")
        writer = cv2.VideoWriter(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), RATE_HZ, (88, 88))
        try:
            if not writer.isOpened():
                raise OSError("Could not open the MP4 writer")
            for frame in crops:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()
        np.savez(staged, raw_frames=np.stack(frames),
                 raw_timestamps_ns=np.asarray(timestamps, dtype=np.int64) - timestamps[0],
                 landmarks_68=points, mouth_88x88=crops, crop_timestamps_ns=grid,
                 metadata_json=json.dumps(metadata))
        preview.replace(path.with_suffix(".mp4"))
        staged.replace(path)


def hud(frame, crop, index, sentences, recording, status):
    view = cv2.resize(frame, (640, 480))
    cv2.rectangle(view, (0, 0), (640, 126), (22, 22, 22), -1)
    headline = "List complete" if index == len(sentences) else f"Sentence {index + 1} / {len(sentences)}"
    if recording:
        headline += "   RECORDING"
    cv2.putText(view, headline, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (120, 220, 255), 2, cv2.LINE_AA)
    if index < len(sentences):
        for row, line in enumerate(textwrap.wrap(sentences[index], 48)[:4]):
            cv2.putText(view, line, (16, 52 + row * 20), cv2.FONT_HERSHEY_SIMPLEX, .55,
                        (255, 255, 255), 1, cv2.LINE_AA)
    if crop is not None:
        view[134:222, 544:632] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(view, (0, 424), (640, 480), (22, 22, 22), -1)
    for line, y in [(status, 446), ("Space  Start / stop     R  Redo last     Q  Quit", 469)]:
        cv2.putText(view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True, type=Path, help="eval_holdout_sentences.txt or train_sentences.txt")
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--facemark", default=FACEMARK, type=Path)
    args = parser.parse_args()
    split = SPLITS.get(args.list.stem)
    if split is None:
        parser.error("--list must be named eval_holdout_sentences.txt or train_sentences.txt")
    try:
        lines = args.list.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    sentences = [line.strip() for line in lines if line.strip()]
    if not sentences:
        parser.error("The sentence list is empty")
    root = DATA / split
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"{i:03}.npz" for i in range(1, len(sentences) + 1)]
    completed = set()
    for i, (path, sentence) in enumerate(zip(paths, sentences)):
        if path.is_file():
            with np.load(path, allow_pickle=False) as clip:
                metadata = json.loads(str(clip["metadata_json"]))
            if metadata["sentence"] != sentence:
                parser.error(f"{path} belongs to a different sentence list")
            completed.add(i)
    if any(path not in paths for path in root.glob("*.npz")):
        parser.error(f"{root} contains clip numbers outside this list")
    index = next((i for i in range(len(paths)) if i not in completed), len(paths))
    last = max(completed, default=None)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    reference = np.load(MEAN_FACE, allow_pickle=False)
    detect = Landmarks68(args.facemark)
    print("Focus the OpenCV window. Space starts/stops, R redoes the last clip, Q quits.", flush=True)
    if index < len(paths):
        print(f"Sentence {index + 1} / {len(paths)}  {sentences[index]}", flush=True)
    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        recording = False
        status = "Ready"
        while True:
            ok, frame = camera.read()
            timestamp = monotonic_ns()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fit = detect(gray)
            try:
                crop = None if fit is None else align_mouth(gray, fit, reference)
            except ValueError:
                fit = crop = None
            if recording:
                frames.append(frame)
                timestamps.append(timestamp)
                landmarks.append(fit)
            hud(frame, crop, index, sentences, recording, status if crop is not None else "Face out of view")
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if not recording and key in (ord("r"), ord("R")) and last is not None:
                index = last
                status = "Redo ready. Space starts recording."
                print(f"Redo sentence {index + 1}  {sentences[index]}", flush=True)
            if recording and (key == 32 or timestamp - timestamps[0] >= MAX_RECORDING_NS):
                recording = False
                try:
                    save_clip(paths[index], frames, timestamps, landmarks, reference,
                              dict(sentence=sentences[index], sentence_number=index + 1,
                                   split=split, session=session, kind="sentence"))
                except (OSError, ValueError) as exc:
                    print(f"Clip {index + 1} not saved: {exc}", flush=True)
                    status = "Clip not saved. Press Space to retry."
                else:
                    print(f"Saved {paths[index]}: {sentences[index]}", flush=True)
                    completed.add(index)
                    last = index
                    index = next((i for i in range(len(paths)) if i not in completed), len(paths))
                    status = "List complete. R redoes the last clip." if index == len(paths) else "Saved. Ready."
                    if index < len(paths):
                        print(f"Sentence {index + 1} / {len(paths)}  {sentences[index]}", flush=True)
                frames, timestamps, landmarks = [], [], []
            elif key == 32 and index < len(paths):
                frames, timestamps, landmarks = [frame], [timestamp], [fit]
                recording = True
                status = "Space stops recording."
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
