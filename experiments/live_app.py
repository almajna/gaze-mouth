"""Live 4-target prototype: gaze picks the quadrant, a held open-mouth selects it.

Needs a validated logs/trigger_config.json (eval_offline.py must report DEPLOYABLE: YES).
See docs/plan.md.  Run:  .venv-gf/bin/python -m experiments.live_app
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from eval.eval_offline import (CORNERS, TRIGGER_CONFIG_PATH, gaze_features, lip_gap,
                               nearest_centroid_fit, nearest_centroid_predict)
from handsfree.trigger import COOLDOWN_MS, POP_SOUND, Trigger, load_trigger_config, play
from training.capture import ensure_model, extract, make_landmarker

ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "logs" / "live_metrics.csv"
CSV_FIELDS = ["run_id", "trial", "target", "selected", "correct", "response_time_ms", "accidental_triggers"]

W, H = 1280, 720
BOX_W, BOX_H, MARGIN = 380, 230, 20
SETTLE_MS, CALIB_DOT_MS, CALIB_SKIP_MS = 1000, 1500, 300
MIN_CALIB_FRAMES = 10
REST_MS, TIMEOUT_MS = 800, 8000
PRE_ONSET_MS = 300  # gaze used for selection = frames just before the mouth started opening
SMOOTH_MS = 150  # live indicator = majority vote of per-frame predictions in this window
CALIB_MIN_ACC = 0.85  # leave-one-frame-out accuracy below this -> recalibrate
CALIB_MAX_ATTEMPTS = 3  # then continue with the best attempt (warned on screen) instead of looping forever
CALIB_RESULT_MS = 2000
FONT = cv2.FONT_HERSHEY_SIMPLEX
GAZE_COLS = {"iris": [0, 1], "iris_head": [0, 1, 2, 3]}


def box(corner):
    x0 = MARGIN if "left" in corner else W - MARGIN - BOX_W
    y0 = MARGIN if "top" in corner else H - MARGIN - BOX_H
    return x0, y0, x0 + BOX_W, y0 + BOX_H


def box_center(corner):
    x0, y0, x1, y1 = box(corner)
    return (x0 + x1) // 2, (y0 + y1) // 2


def frame_features(lm, mat, cols, aspect):
    one = dict(lm=lm[None], hm=mat[None], aspect=aspect)
    return gaze_features(one)[0, cols], float(lip_gap(one)[0])


def smoothed_gaze(hist, t):
    """Majority vote over predictions from the last SMOOTH_MS; ties go to the most recent class."""
    win = [p for ts_, p in hist if ts_ > t - SMOOTH_MS]
    if not win:
        return None
    counts = Counter(win)
    top = max(counts.values())
    return next(p for p in reversed(win) if counts[p] == top)


def calib_lofo_accuracy(X, y, cols):
    """Leave-one-frame-out accuracy of the nearest-centroid calibration. Optimistic: adjacent frames are
    correlated, so this is a floor check for a broken calibration, not a live accuracy estimate."""
    hits = 0
    for i in range(len(X)):
        m = np.arange(len(X)) != i
        hits += nearest_centroid_predict(nearest_centroid_fit(X[m], y[m], cols), X[i:i + 1])[0] == y[i]
    return hits / len(X)


def put_center(img, txt, y, scale=0.8, color=(230, 230, 230), thick=2):
    (tw, _), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(img, txt, ((W - tw) // 2, y), FONT, scale, color, thick)


def render(frame, phase, target, gaze, calib_corner, lines, face_ok):
    img = np.full((H, W, 3), 25, np.uint8)
    for c in CORNERS:
        x0, y0, x1, y1 = box(c)
        if phase == "target" and c == target:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 170, 255), -1)  # active target: amber
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), (55, 55, 55), -1)
        if gaze == c and phase in ("rest", "target"):
            cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (200, 200, 120), 2)  # subtle gaze outline
    if calib_corner:
        cx, cy = box_center(calib_corner)
        cv2.circle(img, (cx, cy), 22, (0, 0, 255), -1)
        cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1)
    thumb = cv2.resize(cv2.flip(frame, 1), (256, 144))
    ty, tx = H // 2 - 72 + 40, W // 2 - 128
    img[ty:ty + 144, tx:tx + 256] = thumb
    if not face_ok:
        cv2.rectangle(img, (tx, ty), (tx + 256, ty + 144), (0, 0, 255), 2)
    for i, (txt, scale) in enumerate(lines):
        put_center(img, txt, H // 2 - 80 + i * 32, scale)
    return img


def summarize(rows, calib_accidental, extra_accidental):
    n = len(rows)
    print("\n==== live_app summary ====")
    acc_total = sum(r["accidental_triggers"] for r in rows) + calib_accidental + extra_accidental
    print(f"accidental triggers: {acc_total} (rest periods {sum(r['accidental_triggers'] for r in rows)}, "
          f"calibration {calib_accidental}, not attributed to a logged trial {extra_accidental})")
    if not n:
        print("no completed trials")
        return
    correct = sum(r["correct"] for r in rows)
    timeouts = sum(r["selected"] == "timeout" for r in rows)
    rts = [r["response_time_ms"] for r in rows if r["selected"] != "timeout"]
    print(f"accuracy: {correct}/{n} = {correct / n:.1%}  (timeouts {timeouts})")
    print(f"median response time: {np.median(rts):.0f} ms over {len(rts)} triggered trials" if rts
          else "median response time: n/a (no triggers)")
    wrong = [(r["trial"], r["target"], r["selected"]) for r in rows if not r["correct"]]
    if wrong:
        print("errors (trial, target, selected): " + ", ".join(map(str, wrong)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gaze-features", choices=list(GAZE_COLS), default="iris_head")
    ap.add_argument("--trigger-config", type=Path, default=TRIGGER_CONFIG_PATH)
    ap.add_argument("--windowed", action="store_true")
    args = ap.parse_args()
    if args.trials < 1:
        sys.exit("--trials must be >= 1")

    tcfg = load_trigger_config(args.trigger_config)
    tj, tg, deb = float(tcfg["jawOpen"]), float(tcfg["lip_gap"]), int(tcfg["debounce_ms"])
    print(f"trigger: jawOpen>={tj}, lip_gap>={tg}, hold>={deb}ms, cooldown {COOLDOWN_MS}ms + release")
    cols = GAZE_COLS[args.gaze_features]

    rng = random.Random(args.seed)
    targets = [CORNERS[i % 4] for i in range(args.trials)]
    rng.shuffle(targets)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    ensure_model()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera} (check macOS camera permission).")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "live_app"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    METRICS_CSV.parent.mkdir(exist_ok=True)
    new_file = not METRICS_CSV.exists()
    csv_f = open(METRICS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    trig = Trigger(deb)
    hist = deque(maxlen=200)  # (t_ms, predicted quadrant)
    rows, calib_X, calib_y = [], [], []
    calib_accidental = rest_accidental = 0
    model, calib_attempt, calib_best, calib_msg = None, 1, None, []
    phase, phase_t0, trial = "calib", None, 0
    last_ts, read_failures = -1, 0
    landmarker = make_landmarker()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= 10:
                    print("camera read failed 10x in a row, aborting"); break
                continue
            read_failures = 0
            t = int(time.monotonic() * 1000)
            t = max(t, last_ts + 1)
            last_ts = t
            if phase_t0 is None:
                phase_t0 = t
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm, bs, names, mat, present = extract(
                landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t))

            aspect = frame.shape[1] / frame.shape[0]
            feat, gap = (frame_features(lm, mat, cols, aspect) if present else (None, float("nan")))
            feat_ok = present and np.isfinite(feat).all()
            jaw = float(bs[names.index("jawOpen")]) if present and names else float("nan")
            mouth = bool(present and jaw >= tj and gap >= tg)
            onset = trig.update(t, mouth)
            fired = onset is not None
            if fired:
                play(POP_SOUND)

            gaze = None
            if model is not None and feat_ok:
                hist.append((t, str(nearest_centroid_predict(model, feat[None])[0])))
            if hist:
                gaze = smoothed_gaze(hist, t)

            el = t - phase_t0
            calib_corner, lines = None, []
            if phase == "calib":
                if fired:
                    calib_accidental += 1
                if el < SETTLE_MS:
                    calib_corner = CORNERS[0]
                    lines = [("Calibration: look at the red dot", 0.9), ("settling...", 0.6)]
                else:
                    k = (el - SETTLE_MS) // CALIB_DOT_MS
                    if k < 4:
                        calib_corner = CORNERS[k]
                        lines = [("Calibration: look at the red dot", 0.9), (f"{k + 1}/4", 0.6)]
                        if (el - SETTLE_MS) % CALIB_DOT_MS >= CALIB_SKIP_MS and feat_ok:
                            calib_X.append(feat); calib_y.append(CORNERS[k])
                    else:
                        y = np.array(calib_y)
                        counts = {c: int((y == c).sum()) for c in CORNERS}
                        if min(counts.values()) < MIN_CALIB_FRAMES:
                            print(f"calibration too few valid frames {counts}; repeating")
                            calib_X, calib_y, phase_t0 = [], [], t
                        else:
                            X = np.array(calib_X)
                            acc = calib_lofo_accuracy(X, y, cols)
                            cand = nearest_centroid_fit(X, y, cols)
                            print(f"calibration attempt {calib_attempt}: frames {counts}, "
                                  f"leave-one-frame-out accuracy {acc:.0%}")
                            if calib_best is None or acc > calib_best[0]:
                                calib_best = (acc, cand)
                            if acc >= CALIB_MIN_ACC:
                                model = cand
                                calib_msg = [(f"Calibration OK: {acc:.0%} (leave-one-frame-out)", 0.9),
                                             ("starting trials...", 0.6)]
                                phase = "calib_ok"
                            elif calib_attempt < CALIB_MAX_ATTEMPTS:
                                calib_msg = [(f"Calibration {acc:.0%} < {CALIB_MIN_ACC:.0%}: recalibrating", 0.9),
                                             (f"attempt {calib_attempt + 1}/{CALIB_MAX_ATTEMPTS}"
                                              " - keep head still, eyes on the dot", 0.6)]
                                phase = "calib_retry"
                            else:
                                model = calib_best[1]
                                calib_msg = [(f"WARNING: best calibration only {calib_best[0]:.0%} "
                                              f"after {CALIB_MAX_ATTEMPTS} attempts", 0.9),
                                             ("continuing; expect gaze errors", 0.6)]
                                print(calib_msg[0][0])
                                phase = "calib_ok"
                            calib_X, calib_y, phase_t0 = [], [], t
            elif phase in ("calib_ok", "calib_retry"):
                if fired:
                    calib_accidental += 1
                lines = calib_msg
                if el >= CALIB_RESULT_MS:
                    if phase == "calib_ok":
                        phase, phase_t0 = "rest", t
                    else:
                        calib_attempt += 1
                        hist.clear()
                        phase, phase_t0 = "calib", t
            elif phase == "rest":
                if fired:
                    rest_accidental += 1
                lines = [(f"Trial {trial + 1}/{args.trials}", 0.9), ("get ready...", 0.6)]
                if el >= REST_MS:
                    phase, phase_t0 = "target", t
            elif phase == "target":
                target = targets[trial]
                lines = [(f"Trial {trial + 1}/{args.trials}: look at the lit target, open mouth to select", 0.7)]
                selected = None
                if fired:
                    pre = [p for ts_, p in hist if onset - PRE_ONSET_MS <= ts_ < onset]
                    selected = Counter(pre).most_common(1)[0][0] if pre else (gaze or "none")
                    rt = t - phase_t0
                elif el >= TIMEOUT_MS:
                    selected, rt = "timeout", TIMEOUT_MS
                if selected is not None:
                    row = dict(run_id=run_id, trial=trial + 1, target=target, selected=selected,
                               correct=selected == target, response_time_ms=rt,
                               accidental_triggers=rest_accidental)
                    rows.append(row); writer.writerow(row); csv_f.flush()
                    print(f"trial {trial + 1:2d}: target {target:12s} selected {selected:12s} "
                          f"{'OK ' if row['correct'] else 'ERR'} {rt} ms, accidental {rest_accidental}")
                    rest_accidental = 0
                    trial += 1
                    phase, phase_t0 = ("rest", t) if trial < args.trials else ("done", t)
            else:  # done
                if fired:
                    rest_accidental += 1
                correct = sum(r["correct"] for r in rows)
                lines = [(f"Done: {correct}/{len(rows)} correct", 0.9), ("press q to quit", 0.6)]

            img = render(frame, phase, targets[trial] if trial < args.trials else None,
                         gaze, calib_corner, lines, present)
            cv2.imshow(win, img)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        csv_f.close()
        summarize(rows, calib_accidental, rest_accidental)
        print(f"trials logged to {METRICS_CSV.relative_to(ROOT)} (run_id {run_id})")


if __name__ == "__main__":
    main()
