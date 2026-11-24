"""Guided webcam capture: synchronized .mp4 + MediaPipe FaceLandmarker .npz per clip.

See docs/plan.md. Run:  .venv-gf/bin/python -m training.capture --fullscreen
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parents[1] / "face_landmarker.task"
CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]
NEGATIVES = {
    "neg_blink": "Look at screen CENTER and blink naturally a few times",
    "neg_swallow": "Look at screen CENTER and swallow once or twice",
    "neg_look_down": "Look DOWN at your keyboard, then back up",
}
COUNTDOWN_MS = 3000
MAX_READ_FAILURES = 10
N_LM, N_BS = 478, 52
FONT = cv2.FONT_HERSHEY_SIMPLEX
CUE_SOUND = "/System/Library/Sounds/Tink.aiff"
RED, GREEN = (0, 0, 255), (0, 255, 0)  # BGR


def ensure_model():
    if MODEL_PATH.exists():
        return
    print(f"Downloading model -> {MODEL_PATH}")
    tmp = MODEL_PATH.with_suffix(".task.part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)


def make_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def build_protocol(args):
    steps = []
    for rep in range(args.reps):
        for c in CORNERS:
            steps.append(dict(label=c, rep=rep, duration_ms=int((args.fixate + args.hold) * 1000)))
    for rep in range(args.reps):
        for n in NEGATIVES:
            steps.append(dict(label=n, rep=rep, duration_ms=int(args.neg_seconds * 1000)))
    return steps


def cue_at(step, t_ms, args):
    """Cue for a frame captured t_ms after recording started (capture clock)."""
    if step["label"] in NEGATIVES:
        return "negative"
    return "fixate" if t_ms < args.fixate * 1000 else "gesture"


def instruction(step, cue, args):
    if step["label"] in NEGATIVES:
        return NEGATIVES[step["label"]]
    corner = step["label"].replace("_", " ").upper()
    if cue == "fixate":
        return f"Look at the {corner} dot"
    return f"Keep looking {corner} + {args.gesture.upper()}"


def countdown_instruction(step, args):
    # Negatives: neutral prompt so the action isn't done before recording starts.
    if step["label"] in NEGATIVES:
        return "Get ready... look at screen CENTER"
    return instruction(step, "fixate", args)


def next_free_path(out_dir, label):
    i = 0
    while (out_dir / f"{label}_{i:02d}.npz").exists() or (out_dir / f"{label}_{i:02d}.mp4").exists():
        i += 1
    return out_dir / f"{label}_{i:02d}"


def extract(result):
    """Return (landmarks(478,3), blendshapes(52,), names|None, matrix(4,4), present)."""
    if not result.face_landmarks:
        nan = np.float32(np.nan)
        return (np.full((N_LM, 3), nan, np.float32), np.full(N_BS, nan, np.float32),
                None, np.full((4, 4), nan, np.float32), False)
    lms = result.face_landmarks[0]
    lm = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
    if lm.shape[0] != N_LM:
        sys.exit(f"Expected {N_LM} landmarks, got {lm.shape[0]} — wrong model?")
    cats = result.face_blendshapes[0]
    bs = np.array([c.score for c in cats], np.float32)
    names = [c.category_name for c in cats]
    mat = (np.asarray(result.facial_transformation_matrixes[0], np.float32)
           if result.facial_transformation_matrixes else np.full((4, 4), np.nan, np.float32))
    return lm, bs, names, mat, True


class Clip:
    def __init__(self, step, out_dir, fps, gesture):
        self.step, self.gesture, self.fps = step, gesture, fps
        self.base = next_free_path(out_dir, step["label"])
        self.writer = None
        self.w = self.h = None
        self.lm, self.bs, self.mat, self.present, self.ts, self.cue = [], [], [], [], [], []
        self.names = None
        self.cue_sounded = False

    def _open_writer(self, frame):
        # Size from the real frame, not cap.get(): VideoWriter silently drops mismatched frames.
        self.h, self.w = frame.shape[:2]
        self.writer = cv2.VideoWriter(str(self.base) + ".mp4",
                                      cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.w, self.h))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open mp4 VideoWriter for {self.base}.mp4 ({self.w}x{self.h})")

    def add(self, frame, t_rel_ms, data, cue):
        if self.writer is None:
            self._open_writer(frame)
        elif frame.shape[:2] != (self.h, self.w):
            raise RuntimeError(f"Frame size changed mid-clip: {frame.shape[1]}x{frame.shape[0]} "
                               f"vs {self.w}x{self.h}")
        lm, bs, names, mat, present = data
        self.writer.write(frame)
        self.lm.append(lm); self.bs.append(bs); self.mat.append(mat)
        self.present.append(present); self.ts.append(t_rel_ms); self.cue.append(cue)
        if names is not None and self.names is None:
            self.names = names

    def discard(self):
        if self.writer is not None:
            self.writer.release()
        Path(str(self.base) + ".mp4").unlink(missing_ok=True)

    def save(self):
        if not self.ts:
            self.discard()
            print(f"skipped {self.base.name}: no frames recorded")
            return
        self.writer.release()
        n = len(self.ts)
        np.savez_compressed(
            str(self.base) + ".npz",
            landmarks=np.stack(self.lm), blendshapes=np.stack(self.bs),
            blendshape_names=np.array(self.names or [], dtype=str),
            head_matrix=np.stack(self.mat), face_present=np.array(self.present, bool),
            timestamps_ms=np.array(self.ts, np.int64), cue=np.array(self.cue, dtype=str),
            label=self.step["label"], rep=self.step["rep"], gesture=self.gesture,
            width=self.w, height=self.h, video_fps=float(self.fps),
        )
        dur = (self.ts[-1] - self.ts[0]) / 1000.0 if n > 1 else 0.0
        fps = (n - 1) / dur if dur > 0 else 0.0
        warn = "" if self.names else "  WARNING: no face in any frame (no blendshape names)"
        print(f"saved {self.base.name} (rep {self.step['rep'] + 1}): {n} frames, "
              f"{fps:.1f} fps measured, face {np.mean(self.present) * 100:.0f}%{warn}")


def put_centered(view, txt, y, scale, thick):
    w = view.shape[1]
    (tw, _), _ = cv2.getTextSize(txt, FONT, scale, thick)
    x = max(5, (w - tw) // 2)
    cv2.putText(view, txt, (x, y), FONT, scale, (0, 0, 0), thick + 2)
    cv2.putText(view, txt, (x, y), FONT, scale, (255, 255, 255), thick)


def play_cue_sound():
    """Non-blocking macOS alert at the fixate->gesture switch; silently skipped elsewhere."""
    try:
        subprocess.Popen(["afplay", CUE_SOUND], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def draw(view, data, status, prompt, sub, target, rec, target_color=(0, 0, 255)):
    h, w = view.shape[:2]
    s = w / 1280.0  # overlay scale relative to a 1280-px-wide frame
    th = max(1, round(2 * s))
    lm, bs, names, _, present = data
    if present:
        for x, y, _z in lm:  # mirror x to match the flipped preview
            cv2.circle(view, (int((1 - x) * w), int(y * h)), max(1, round(s)), (0, 255, 0), -1)
        if names:
            for i, key in enumerate(["jawOpen", "mouthPucker"]):
                v = float(bs[names.index(key)]) if key in names else 0.0
                y0 = h - int(60 * s) + i * int(25 * s)
                bar = int(200 * s)
                cv2.rectangle(view, (10, y0), (10 + int(bar * v), y0 + int(15 * s)), (0, 200, 255), -1)
                cv2.putText(view, f"{key} {v:.2f}", (20 + bar, y0 + int(13 * s)),
                            FONT, 0.5 * s, (255, 255, 255), max(1, round(s)))
    else:
        cv2.putText(view, "NO FACE", (10, h - int(70 * s)), FONT, 0.8 * s, (0, 0, 255), th)
    if target:
        m = int(30 * s)
        tx = m if "left" in target else w - m
        ty = m if "top" in target else h - m
        cv2.circle(view, (tx, ty), int(18 * s), target_color, -1)
        cv2.circle(view, (tx, ty), max(2, int(5 * s)), (255, 255, 255), -1)
    if prompt:
        put_centered(view, prompt, h // 2 - int(20 * s), 0.9 * s, th)
    if sub:
        put_centered(view, sub, h // 2 + int(20 * s), 0.7 * s, th)
    cv2.putText(view, status, (10, h - int(12 * s)), FONT, 0.5 * s, (200, 200, 200), max(1, round(s)))
    if rec:
        cv2.circle(view, (w - int(120 * s), int(60 * s)), int(10 * s), (0, 0, 255), -1)
        cv2.putText(view, "REC", (w - int(100 * s), int(67 * s)), FONT, 0.7 * s, (0, 0, 255), th)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="session_1")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fixate", type=float, default=2.0)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--neg-seconds", type=float, default=4.0)
    ap.add_argument("--gesture", default="open mouth")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    ensure_model()
    out_dir = Path(__file__).resolve().parents[1] / "data" / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera} (check macOS camera permission for your terminal).")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the newest frame, not a stale queue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = cam_fps if 5 <= cam_fps <= 120 else 30.0
    print(f"camera reports {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f" @ {cam_fps:.1f} fps (mp4 written at {video_fps:.1f}); out: {out_dir}")

    win = "capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    steps = build_protocol(args)
    idx, running, phase, phase_t0, clip = 0, False, "idle", 0, None
    last_ts, read_failures, printed_size = -1, 0, False
    landmarker = make_landmarker()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= MAX_READ_FAILURES:
                    print(f"camera read failed {MAX_READ_FAILURES}x in a row, aborting"); break
                continue
            read_failures = 0
            # Capture clock: taken right after read, used for MediaPipe, timestamps_ms AND cues.
            t_ms = int(time.monotonic() * 1000)
            if t_ms <= last_ts:
                t_ms = last_ts + 1
            last_ts = t_ms
            if not printed_size:
                print(f"actual frame size {frame.shape[1]}x{frame.shape[0]}")
                printed_size = True

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
            data = extract(result)

            step = steps[idx] if idx < len(steps) else None
            prompt = sub = target = None
            target_color = RED
            if step is None:
                prompt = "Done! Press q to quit."
            elif not running:
                prompt = "SPACE = start/resume   q = quit"
                sub = f"next: {step['label']} rep {step['rep'] + 1}/{args.reps}"
            elif phase == "countdown":
                left = COUNTDOWN_MS - (t_ms - phase_t0)
                prompt = countdown_instruction(step, args)
                sub = f"recording in {left / 1000:.1f}s"
                target = step["label"] if step["label"] in CORNERS else None
                if left <= 0:
                    phase, phase_t0 = "record", t_ms
                    clip = Clip(step, out_dir, video_fps, args.gesture)
            if running and step is not None and phase == "record":
                t_rel = t_ms - phase_t0
                if t_rel >= step["duration_ms"]:
                    clip.save(); clip = None
                    idx += 1
                    phase, phase_t0 = "countdown", t_ms
                    if idx >= len(steps):
                        running = False
                else:
                    cue = cue_at(step, t_rel, args)
                    if cue == "gesture":
                        target_color = GREEN
                        if not clip.cue_sounded:
                            play_cue_sound()
                            clip.cue_sounded = True
                    clip.add(frame, t_rel, data, cue)
                    prompt = instruction(step, cue, args)
                    sub = f"{(step['duration_ms'] - t_rel) / 1000:.1f}s left"
                    target = step["label"] if step["label"] in CORNERS else None

            view = cv2.flip(frame, 1)
            status = f"clip {min(idx + 1, len(steps))}/{len(steps)}  face={'yes' if data[4] else 'NO'}"
            draw(view, data, status, prompt, sub, target, clip is not None, target_color)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and step is not None:
                if running:  # pause: drop any partial clip
                    running = False
                    if clip:
                        clip.discard(); clip = None
                else:
                    running, phase, phase_t0 = True, "countdown", t_ms
    finally:
        if clip:
            clip.discard()
            print("discarded unfinished clip")
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
