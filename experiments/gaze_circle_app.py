"""Gaze bubble (EyeTrax GazeEstimator + Kalman-EMA) with magnetic circular targets and the validated
mouth-open click. Reads logs/trigger_config.json (never writes it). See docs/plan.md.

Run:  python -m experiments.gaze_circle_app [--calibration 9p|dense] [--model ridge|tiny_mlp|svr|elastic_net]
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import eyetrax
from eyetrax.calibration import run_9_point_calibration, run_dense_grid_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.models import create_model
from eyetrax.utils.screen import get_screen_size
from sklearn.exceptions import NotFittedError

from training.capture import MODEL_PATH, ensure_model, extract, make_landmarker
from eval.eval_offline import TRIGGER_CONFIG_PATH, lip_gap
from handsfree.trigger import POP_SOUND, Trigger, load_trigger_config, play

ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "logs" / "circle_metrics.csv"
CSV_FIELDS = ["run_id", "trial", "model", "calibration", "target_x", "target_y", "raw_gaze_x", "raw_gaze_y",
              "gaze_x", "gaze_y", "error_px", "raw_error_px", "magnetic_hit", "strict_hit", "response_time_ms",
              "timeout", "accidental_triggers", "gaze_window_samples"]
EYETRAX_TESTED = "0.4.0"
MODEL_CHOICES = ["ridge", "tiny_mlp", "svr", "elastic_net"]

BUBBLE_R, TARGET_R, MAGNET_PX, RING_PAD = 70, 140, 50, 14
REST_MS, TIMEOUT_MS, PRE_ONSET_MS, FEEDBACK_MS, CAL_RESULT_MS = 600, 8000, 300, 450, 1500
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN, GREEN, RED, AMBER = (255, 255, 0), (80, 220, 80), (60, 60, 230), (0, 190, 255)  # BGR


class LandmarkerTap:
    """Pass-through for EyeTrax's FaceLandmarker slot: EyeTrax runs inference on OUR blendshape-enabled
    landmarker, and we read the same result for the mouth trigger -> one inference per frame."""

    def __init__(self, landmarker):
        self.landmarker, self.last = landmarker, None

    def detect_for_video(self, image, ts_ms):
        self.last = self.landmarker.detect_for_video(image, ts_ms)
        return self.last

    def close(self):
        self.landmarker.close()


def build_estimator(model_name):
    if eyetrax.__version__ != EYETRAX_TESTED:
        print(f"WARNING: eyetrax {eyetrax.__version__} != tested {EYETRAX_TESTED}; landmarker swap may break")
    ensure_model()
    ge = GazeEstimator(model_name=model_name, face_landmarker_model=str(MODEL_PATH))
    if not hasattr(ge, "_face_landmarker"):
        sys.exit("eyetrax GazeEstimator has no _face_landmarker attribute; cannot share the landmarker")
    ge._face_landmarker.close()
    tap = LandmarkerTap(make_landmarker())  # blendshapes + transformation matrices enabled
    ge._face_landmarker = tap
    calib = {}
    orig_train = ge.train

    def train_and_keep(X, y, variable_scaling=None):  # keep EyeTrax's calibration samples for error reporting
        calib["X"], calib["y"] = np.asarray(X, float), np.asarray(y, float)
        orig_train(X, y, variable_scaling)

    ge.train = train_and_keep
    return ge, tap, calib


def calibrate(ge, calib, args):
    calib.clear()
    if args.calibration == "dense":
        run_dense_grid_calibration(ge, rows=5, cols=5, camera_index=args.camera)
    else:
        run_9_point_calibration(ge, camera_index=args.camera)
    if "X" not in calib:
        return None  # aborted (Esc) or no usable samples
    X, y = calib["X"], calib["y"]
    train_err = float(np.linalg.norm(ge.predict(X) - y, axis=1).mean())
    pts = np.unique(y, axis=0)
    lopo = []
    for p in pts:  # leave-one-calibration-point-out, same model type
        m = (y == p).all(1)
        if (~m).sum() < 2:
            continue
        mdl = create_model(args.model)
        mdl.train(X[~m], y[~m])
        lopo.append(np.linalg.norm(mdl.predict(X[m]) - y[m], axis=1))
    lopo_err = float(np.concatenate(lopo).mean()) if lopo else float("nan")
    return dict(samples=len(X), points=len(pts), train_err=train_err, lopo_err=lopo_err)


def make_smoother(ge, args):
    sm = KalmanEMASmoother(make_kalman(), ema_alpha=args.ema_alpha)
    if not args.no_tune:
        sm.tune(ge, camera_index=args.camera)  # EyeTrax's quick pass that sets measurement noise
    return sm


def make_targets(n, sw, sh, rng):
    grid = [(int(fx * sw), int(fy * sh)) for fy in (0.25, 0.5, 0.75) for fx in (0.2, 0.5, 0.8)]
    pts = (grid * (n // len(grid) + 1))[:n]
    for _ in range(1000):
        rng.shuffle(pts)
        if all(a != b for a, b in zip(pts, pts[1:])):
            break
    return pts


def open_metrics_csv():
    METRICS_CSV.parent.mkdir(exist_ok=True)
    header = ",".join(CSV_FIELDS)
    if METRICS_CSV.exists():
        with open(METRICS_CSV) as f:
            first = f.readline().strip()
        if first != header:
            stamp = datetime.fromtimestamp(METRICS_CSV.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
            legacy = METRICS_CSV.with_name(f"circle_metrics.legacy-{stamp}.csv")
            os.replace(METRICS_CSV, legacy)
            print(f"existing {METRICS_CSV.name} had a different schema -> moved to {legacy.name}")
    new = not METRICS_CSV.exists()
    f = open(METRICS_CSV, "a", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new:
        w.writeheader()
    return f, w


def open_capture(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index} (check macOS camera permission).")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ---------- drawing ----------

def put_center(img, txt, y, scale=0.8, color=(230, 230, 230), thick=2):
    (tw, _), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(img, txt, ((img.shape[1] - tw) // 2, y), FONT, scale, color, thick)


def blend_circle(img, c, r, color, alpha):
    over = img.copy()
    cv2.circle(over, c, r, color, -1)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)


def draw_feedback(img, fb, t):
    if fb is None:
        return
    c, hit, t0 = fb
    k = (t - t0) / FEEDBACK_MS
    if k >= 1:
        return
    if hit:
        cv2.circle(img, c, int(TARGET_R * (1 + 0.6 * k)), GREEN, max(2, int(14 * (1 - k))))
        blend_circle(img, c, int(TARGET_R * (1 - k)), GREEN, 0.5 * (1 - k))
    else:
        over = np.empty_like(img)
        over[:] = RED
        cv2.addWeighted(over, 0.35 * (1 - k), img, 1 - 0.35 * (1 - k), 0, img)


def summarize(rows, accidental_other, cal):
    print("\n==== gaze_circle_app summary ====")
    if cal:
        print(f"calibration: {cal['points']} points, {cal['samples']} samples; fit error {cal['train_err']:.0f} px, "
              f"leave-one-point-out {cal['lopo_err']:.0f} px")
    rest_acc = sum(r["accidental_triggers"] for r in rows)
    print(f"accidental triggers: {rest_acc + accidental_other} (rest periods {rest_acc}, other {accidental_other}; "
          "not monitored during EyeTrax calibration)")
    if not rows:
        print("no completed trials")
        return
    n = len(rows)
    mh, sh_ = sum(r["magnetic_hit"] for r in rows), sum(r["strict_hit"] for r in rows)
    to = sum(r["timeout"] for r in rows)
    rts = [r["response_time_ms"] for r in rows if not r["timeout"]]
    errs = [r["error_px"] for r in rows if r["error_px"] != ""]
    raw = [r["raw_error_px"] for r in rows if r["raw_error_px"] != ""]
    print(f"magnetic hit rate: {mh}/{n} = {mh / n:.1%}   strict (inside {TARGET_R}px): {sh_}/{n} = {sh_ / n:.1%}")
    print(f"timeouts (no mouth trigger within {TIMEOUT_MS / 1000:.0f}s): {to}/{n}")
    print(f"median response time: {np.median(rts):.0f} ms over {len(rts)} triggered trials" if rts
          else "median response time: n/a (no triggers)")
    if errs:
        print(f"gaze error at selection: median filtered {np.median(errs):.0f} px"
              + (f", median raw {np.median(raw):.0f} px" if raw else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calibration", choices=["9p", "dense"], default="9p", help="dense = 5x5 grid")
    ap.add_argument("--model", choices=MODEL_CHOICES, default="ridge",
                    help="EyeTrax gaze model; tiny_mlp is the non-linear (MLP) option")
    ap.add_argument("--ema-alpha", type=float, default=0.35, help="EyeTrax kalman_ema alpha (higher = smoother)")
    ap.add_argument("--no-tune", action="store_true", help="skip EyeTrax's Kalman fine-tuning pass")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--trigger-config", type=Path, default=TRIGGER_CONFIG_PATH)
    args = ap.parse_args()
    if args.trials < 1 or not 0.0 <= args.ema_alpha <= 1.0:
        sys.exit("--trials must be >= 1 and --ema-alpha in [0, 1]")

    tcfg = load_trigger_config(args.trigger_config)  # read-only
    tj, tg, deb = float(tcfg["jawOpen"]), float(tcfg["lip_gap"]), int(tcfg["debounce_ms"])
    print(f"trigger (validated, read-only): jawOpen>={tj}, lip_gap>={tg}, hold>={deb}ms, "
          f"cooldown 600ms + release, {Path(POP_SOUND).name}")
    print(f"gaze: eyetrax {eyetrax.__version__} model={args.model} calibration={args.calibration} "
          f"filter=kalman_ema(alpha={args.ema_alpha}){' no-tune' if args.no_tune else ''}")

    sw, sh = get_screen_size()
    rng = random.Random(args.seed)
    targets = make_targets(args.trials, sw, sh, rng)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    ge, tap, calib = build_estimator(args.model)
    cal = calibrate(ge, calib, args)
    if cal is None:
        sys.exit("calibration aborted or produced no samples")
    print(f"calibration: {cal}")
    smoother = make_smoother(ge, args)

    csv_f, writer = open_metrics_csv()
    cap = open_capture(args.camera)
    win = "gaze_circle_app"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    trig = Trigger(deb)
    hist = deque(maxlen=400)  # (t_ms, raw_xy, filtered_xy)
    rows, bubble, feedback = [], None, None
    rest_acc = other_acc = 0
    phase, phase_t0, trial = "cal_result", None, 0
    last_ts, read_failures = -1, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= 10:
                    print("camera read failed 10x in a row, aborting"); break
                continue
            read_failures = 0
            t = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = t
            if phase_t0 is None:
                phase_t0 = t

            feats, blink = ge.extract_features(frame)  # runs our landmarker once via the tap
            lm, bs, names, _mat, present = extract(tap.last)
            if present and names:
                jaw = float(bs[names.index("jawOpen")])
                gap = float(lip_gap(dict(lm=lm[None]))[0])  # validated definition (normalized coords)
                mouth = jaw >= tj and gap >= tg
            else:
                mouth = False
            onset = trig.update(t, mouth)
            fired = onset is not None
            if fired:
                play(POP_SOUND)

            if feats is not None and not blink:
                try:
                    raw = ge.predict(feats[None])[0]
                except NotFittedError:
                    sys.exit("gaze model not fitted (calibration failed)")
                raw = np.clip(raw, [0, 0], [sw - 1, sh - 1])
                fx, fy = smoother.step(int(raw[0]), int(raw[1]))
                bubble = (int(np.clip(fx, 0, sw - 1)), int(np.clip(fy, 0, sh - 1)))
                hist.append((t, raw, bubble))
            # blink / no face: bubble holds its last position (EyeTrax demo skips these frames too)

            img = np.full((sh, sw, 3), 28, np.uint8)
            el = t - phase_t0
            lines, near, tgt = [], False, None

            if phase == "cal_result":
                other_acc += fired
                lines = [(f"Calibration ({args.calibration}, {args.model}): fit {cal['train_err']:.0f} px, "
                          f"leave-one-point-out {cal['lopo_err']:.0f} px", 0.75),
                         ("press r to recalibrate, or wait to start", 0.6)]
                if el >= CAL_RESULT_MS:
                    phase, phase_t0 = "rest", t
            elif phase == "rest":
                rest_acc += fired
                lines = [(f"Trial {trial + 1}/{args.trials}", 0.7)]
                if el >= REST_MS:
                    phase, phase_t0 = "target", t
            elif phase == "target":
                tgt = targets[trial]
                if bubble is not None:
                    near = np.hypot(bubble[0] - tgt[0], bubble[1] - tgt[1]) <= TARGET_R + MAGNET_PX
                if fired or el >= TIMEOUT_MS:
                    ref = onset if fired else t
                    win_pts = [(r, f) for ts_, r, f in hist if ref - PRE_ONSET_MS <= ts_ < ref]
                    if win_pts:
                        raw_xy = np.mean([r for r, _ in win_pts], 0)
                        filt_xy = np.mean([f for _, f in win_pts], 0)
                    elif bubble is not None:
                        raw_xy = filt_xy = np.array(bubble, float)
                    else:
                        raw_xy = filt_xy = None
                    if filt_xy is not None:
                        err = float(np.hypot(*(filt_xy - tgt)))
                        raw_err = float(np.hypot(*(raw_xy - tgt)))
                    else:
                        err = raw_err = ""
                    mag = bool(fired and err != "" and err <= TARGET_R + MAGNET_PX)
                    strict = bool(fired and err != "" and err <= TARGET_R)
                    row = dict(run_id=run_id, trial=trial + 1, model=args.model, calibration=args.calibration,
                               target_x=tgt[0], target_y=tgt[1],
                               raw_gaze_x=(round(raw_xy[0]) if raw_xy is not None else ""),
                               raw_gaze_y=(round(raw_xy[1]) if raw_xy is not None else ""),
                               gaze_x=(round(filt_xy[0]) if filt_xy is not None else ""),
                               gaze_y=(round(filt_xy[1]) if filt_xy is not None else ""),
                               error_px=(round(err, 1) if err != "" else ""),
                               raw_error_px=(round(raw_err, 1) if raw_err != "" else ""),
                               magnetic_hit=mag, strict_hit=strict,
                               response_time_ms=(t - phase_t0) if fired else TIMEOUT_MS, timeout=not fired,
                               accidental_triggers=rest_acc, gaze_window_samples=len(win_pts))
                    rows.append(row); writer.writerow(row); csv_f.flush()
                    print(f"trial {trial + 1:2d}: target {tgt} gaze ({row['gaze_x']},{row['gaze_y']}) "
                          f"raw ({row['raw_gaze_x']},{row['raw_gaze_y']}) err {row['error_px']} px -> "
                          f"{'HIT ' if mag else 'MISS'}{' strict' if strict else ''}"
                          f"{' (timeout)' if not fired else ''} {row['response_time_ms']} ms, "
                          f"accidental {rest_acc}, window samples {len(win_pts)}")
                    feedback = (tgt, mag, t)
                    rest_acc = 0
                    trial += 1
                    phase, phase_t0 = ("rest", t) if trial < args.trials else ("done", t)
                    tgt = None
            else:  # done
                other_acc += fired
                mh = sum(r["magnetic_hit"] for r in rows)
                lines = [(f"Done: {mh}/{len(rows)} hits", 0.9), ("press q to quit", 0.6)]

            # --- draw ---
            if tgt is not None:
                if near:
                    blend_circle(img, tgt, TARGET_R + 26, AMBER, 0.22)  # hover glow
                cv2.circle(img, tgt, TARGET_R, (0, 150, 230) if not near else (0, 200, 255), -1)
                cv2.circle(img, tgt, 9, (255, 255, 255), -1)
            draw_feedback(img, feedback, t)
            if bubble is not None and phase in ("rest", "target", "done"):
                blend_circle(img, bubble, BUBBLE_R, CYAN, 0.25)
                cv2.circle(img, bubble, 4, CYAN, -1)
                if near and tgt is not None:  # magnetic snap: selection ring locks onto the target
                    cv2.circle(img, tgt, TARGET_R + RING_PAD, (255, 255, 255), 4)
                else:
                    cv2.circle(img, bubble, BUBBLE_R, CYAN, 2)
            if not present:
                put_center(img, "face not detected", sh - 30, 0.6, (80, 80, 255))
            for i, (txt, sc) in enumerate(lines):
                put_center(img, txt, (40 + i * 30) if phase in ("rest", "target") else sh // 2 + i * 36, sc)
            cv2.imshow(win, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and phase != "done":
                cap.release()
                cv2.destroyWindow(win)
                new_cal = calibrate(ge, calib, args)
                if new_cal is not None:
                    cal = new_cal
                    print(f"recalibration: {cal}")
                else:
                    print("recalibration aborted; keeping previous calibration")
                smoother = make_smoother(ge, args)
                hist.clear(); bubble = None
                cap = open_capture(args.camera)
                cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                phase, phase_t0 = "cal_result", None  # an interrupted trial restarts, unlogged
    finally:
        ge.close()  # closes our landmarker via the tap and clears the slot (no double close in __del__)
        cap.release()
        cv2.destroyAllWindows()
        csv_f.close()
        summarize(rows, other_acc, cal)
        print(f"trials logged to {METRICS_CSV.relative_to(ROOT)} (run_id {run_id})")


if __name__ == "__main__":
    main()
