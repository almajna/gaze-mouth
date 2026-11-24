#!/usr/bin/env python3
"""Convert the recorded non-speech session clips to 25 Hz sentence-format archives (crops only).

Same processing as record_sentences.py: OpenCV face + 68 landmarks per recorded frame, then
silent_avsr.sentence_video.sentence_crops on the recorded timestamps. Nothing is generated.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from silent_avsr.features import DATA
from silent_avsr.sentence_video import MEAN_FACE, RATE_HZ, Landmarks68, align_mouth, sentence_crops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=Path, nargs="+", default=sorted(DATA.glob("session_[0-9]*")))
    ap.add_argument("--out", type=Path, default=DATA / "nonspeech_25hz")
    args = ap.parse_args()
    detect, reference = Landmarks68(), np.load(MEAN_FACE, allow_pickle=False)
    done = 0
    for session in args.sessions:
        for video in sorted(session.glob("*.mp4")):
            with np.load(video.with_suffix(".npz"), allow_pickle=False) as take:
                stamps = take["timestamps_ms"].astype(np.int64) * 1_000_000
                label = str(take["label"])
            capture, frames = cv2.VideoCapture(str(video)), []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            if len(frames) != len(stamps):
                raise ValueError(f"{video}: {len(frames)} frames but {len(stamps)} timestamps")
            points, missing = [], 0
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fit = detect(gray)
                if fit is not None:
                    try:
                        align_mouth(gray, fit, reference)
                    except (ValueError, cv2.error):
                        fit = None
                missing += fit is None
                points.append(np.full((68, 2), np.nan, np.float32) if fit is None else fit)
            crops, grid = sentence_crops(frames, stamps, np.stack(points), reference)
            group = "negative" if label.startswith("neg_") else "mouth_open_gesture"
            meta = dict(sentence="", split="nonspeech", session=session.name, kind="negative", group=group,
                        label=label, source=str(video.relative_to(DATA)), video_fps=RATE_HZ,
                        raw_frame_count=len(frames), crop_frame_count=len(crops), missing_face_frames=int(missing))
            target = args.out / session.name / f"{video.stem}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, mouth_88x88=crops, crop_timestamps_ns=grid, metadata_json=json.dumps(meta))
            done += 1
    print(f"{done} clips written to {args.out}")


if __name__ == "__main__":
    main()
