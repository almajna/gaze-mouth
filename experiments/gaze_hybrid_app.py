"""Hybrid gaze bubble: EyeTrax landmark features + CLAHE eye-patch pixels + eyelid blendshapes -> RidgeCV frozen
after 9-point calibration + regularized 2D affine click corrector, EyeTrax Kalman + One-Euro filter, freeze on jaw drop, and the validated
mouth-open click (logs/trigger_config.json, read-only). See docs/plan.md ("v2 hybrid gaze").

Run:  python -m experiments.gaze_hybrid_app [--beta 0.005] [--no-adapt] [--corr-lambda-a 8] [--corr-lambda-b 1.5]
"""
import argparse
import csv
import math
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
from eyetrax.calibration import run_9_point_calibration
from eyetrax.filters import KalmanSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from training.capture import MODEL_PATH, ensure_model, extract, make_landmarker
from eval.eval_offline import TRIGGER_CONFIG_PATH, lip_gap
from handsfree.trigger import POP_SOUND, Trigger, load_trigger_config, play

ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "logs" / "hybrid_metrics.csv"
CSV_FIELDS = ["run_id", "trial", "adapt", "lopo_err_px", "target_x", "target_y", "gaze_x", "gaze_y", "error_px",
              "preonset_gaze_x", "preonset_gaze_y", "preonset_error_px", "raw_gaze_x", "raw_gaze_y", "raw_error_px",
              "magnetic_hit", "strict_hit", "response_time_ms", "timeout", "accidental_triggers", "freeze_ms",
              "gaze_window_samples", "corr_clicks", "corr_skipped", "corr_dx", "corr_dy", "refit_ms"]
EYETRAX_TESTED = "0.4.0"
BASELINE = dict(run="v1-milestone 20261119-030154 (EyeTrax ridge/9p)", median_error_px=116.0, strict_hit=0.60,
                magnetic_hit=0.95, median_rt_ms=2262.0, timeouts=0)
REFERENCE = dict(run="v2 reference 20261119-034127 (hybrid v2.1, full refit)", median_error_px=94.0, strict_hit=0.75,
                 magnetic_hit=0.90, median_rt_ms=1350.0, timeouts=0)

# eye patches
PATCH_W, PATCH_H, PATCH_UP = 16, 10, 4
PATCH_X0, PATCH_X1, PATCH_Y = 2.0, 14.0, 5.0  # canonical corner positions inside the 16x10 box
EYES = [(33, 133), (362, 263)]  # (image-left corner, image-right corner) per eye
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
N_LANDMARK_FEATS, N_PATCH_FEATS = 486, 2 * PATCH_W * PATCH_H
LID_BLENDSHAPES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight"]  # eyelid droop

# base model (frozen after calibration) / affine residual corrector
ALPHAS = [50.0, 150.0, 400.0, 800.0]
CORR_MAX, CORR_DECAY, CORR_GATE_PX, CORR_CLAMP_PX = 30, 0.92, 260.0, 160.0
CORR_LAMBDA_A, CORR_LAMBDA_B = 8.0, 1.5  # penalties on (A - I) and b, in half-screen units

# UI / trials
FREEZE_JAW = 0.12
BUBBLE_R, TARGET_R, MAGNET_PX, RING_PAD = 70, 140, 50, 14
REST_MS, TIMEOUT_MS, PRE_ONSET_MS, FEEDBACK_MS, CAL_RESULT_MS = 600, 8000, 300, 450, 1500
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN, GREEN, RED, AMBER = (255, 255, 0), (80, 220, 80), (60, 60, 230), (0, 190, 255)  # BGR


# ---------- features ----------
def eye_patches(gray, lm):
    """320 z-normalized CLAHE pixels: each eye similarity-aligned (roll + scale removed) to a 16x10 box."""
    h, w = gray.shape
    out = []
    s_up = PATCH_UP
    for a, b in EYES:
        p0 = np.array([lm[a, 0] * w, lm[a, 1] * h])
        p1 = np.array([lm[b, 0] * w, lm[b, 1] * h])
        v = p1 - p0
        dist = float(np.hypot(*v))
        if dist < 4:
            return None
        s = (PATCH_X1 - PATCH_X0) * s_up / dist
        c, sn = math.cos(-math.atan2(v[1], v[0])) * s, math.sin(-math.atan2(v[1], v[0])) * s
        M = np.array([[c, -sn, 0.0], [sn, c, 0.0]])
        M[:, 2] = np.array([PATCH_X0 * s_up, PATCH_Y * s_up]) - M[:, :2] @ p0
        big = cv2.warpAffine(gray, M, (PATCH_W * s_up, PATCH_H * s_up), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        patch = cv2.resize(CLAHE.apply(big), (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        sd = patch.std()
        out.append(((patch - patch.mean()) / (sd if sd > 1e-6 else 1.0)).ravel())
    return np.concatenate(out)


class LandmarkerTap:
    """Sits in EyeTrax's FaceLandmarker slot: one inference per frame feeds EyeTrax features, eye patches and
    the mouth trigger. `ts_override` lets an offline replay use recorded timestamps."""

    def __init__(self, landmarker):
        self.landmarker, self.last, self.ts_override = landmarker, None, None

    def detect_for_video(self, image, ts_ms):
        self.last = self.landmarker.detect_for_video(image, ts_ms if self.ts_override is None else self.ts_override)
        return self.last

    def close(self):
        self.landmarker.close()


class BaseRidge:
    """StandardScaler + RidgeCV, fit once on the 9-point calibration and then frozen for the session."""

    def __init__(self):
        self.scaler, self.alpha, self.W, self.b = None, None, None, None
        self.Xc = self.Yc = None

    @staticmethod
    def _solve(X, Y, w, alpha):
        sw_ = w.sum()
        mx, my = (w[:, None] * X).sum(0) / sw_, (w[:, None] * Y).sum(0) / sw_
        r = np.sqrt(w)[:, None]
        A, B = r * (X - mx), r * (Y - my)
        if A.shape[0] < A.shape[1]:  # dual form: (A A^T + aI)^-1 B
            K = A @ A.T
            K[np.diag_indices_from(K)] += alpha
            W = A.T @ np.linalg.solve(K, B)
        else:
            G = A.T @ A
            G[np.diag_indices_from(G)] += alpha
            W = np.linalg.solve(G, A.T @ B)
        return W, my - mx @ W

    def train(self, X, y, variable_scaling=None):  # EyeTrax calibration calls this
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.scaler = StandardScaler().fit(X)
        self.Xc, self.Yc = self.scaler.transform(X), y
        self.alpha = float(RidgeCV(alphas=ALPHAS).fit(self.Xc, y).alpha_)
        self.W, self.b = self._solve(self.Xc, self.Yc, np.ones(len(y)), self.alpha)

    def predict(self, X):
        return self.scaler.transform(np.asarray(X, float)) @ self.W + self.b

    def lopo_error(self):
        errs = []
        for p in np.unique(self.Yc, axis=0):
            m = (self.Yc == p).all(1)
            W, b = self._solve(self.Xc[~m], self.Yc[~m], np.ones((~m).sum()), self.alpha)
            errs.append(np.linalg.norm(self.Xc[m] @ W + b - self.Yc[m], axis=1))
        return float(np.concatenate(errs).mean())


class AffineCorrector:
    """corrected = pred + clamp(u (A - I) + b), u = pred in half-screen units. Fit by weighted ridge on click
    residuals (target - pred), penalising (A - I) with lambda_a and b with lambda_b; weights CORR_DECAY**age."""

    def __init__(self, sw, sh, lambda_a=CORR_LAMBDA_A, lambda_b=CORR_LAMBDA_B):
        self.c = self.s = np.array([sw / 2.0, sh / 2.0])
        self.lam = np.diag([lambda_a, lambda_a, lambda_b])
        self.theta = np.zeros((3, 2))  # rows: (A - I) for u_x, u_y; b
        self.buf = deque(maxlen=CORR_MAX)  # (u_pred, u_target, click_index)
        self.clicks = self.skipped = 0
        self.last_refit_ms = 0.0

    def delta(self, pred):
        u = (np.asarray(pred, float) - self.c) / self.s
        return np.clip((np.append(u, 1.0) @ self.theta) * self.s, -CORR_CLAMP_PX, CORR_CLAMP_PX)

    def correct(self, pred):
        return np.asarray(pred, float) + self.delta(pred)

    def add(self, pred, target):
        """Add one click (base prediction, target). Returns refit ms, or None if gated out as a look-away."""
        if np.hypot(*(np.asarray(target, float) - self.correct(pred))) > CORR_GATE_PX:
            self.skipped += 1
            return None
        t0 = time.perf_counter()
        self.buf.append(((np.asarray(pred, float) - self.c) / self.s, (np.asarray(target, float) - self.c) / self.s,
                         self.clicks))
        self.clicks += 1
        U = np.array([u for u, _, _ in self.buf])
        R = np.array([t for _, t, _ in self.buf]) - U
        w = CORR_DECAY ** ((self.clicks - 1) - np.array([k for _, _, k in self.buf]))
        Z = np.hstack([U, np.ones((len(U), 1))])
        self.theta = np.linalg.solve(Z.T @ (w[:, None] * Z) + self.lam, Z.T @ (w[:, None] * R))
        self.last_refit_ms = (time.perf_counter() - t0) * 1000
        return self.last_refit_ms


class HybridEstimator(GazeEstimator):
    """EyeTrax GazeEstimator whose features are [486 EyeTrax landmark features, 320 eye-patch pixels,
    4 eyelid blendshapes]; all from the one landmarker result."""

    def __init__(self):
        super().__init__(model_name="ridge", face_landmarker_model=str(MODEL_PATH))
        self._face_landmarker.close()
        self.tap = LandmarkerTap(make_landmarker())  # blendshapes + matrices enabled
        self._face_landmarker = self.tap
        self.model = BaseRidge()

    def extract_features(self, image):
        f, blink = super().extract_features(image)
        if f is None:
            return None, blink
        lm, bs, names, _mat, present = extract(self.tap.last)
        if not present or not names:
            return None, blink
        p = eye_patches(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), lm)
        if p is None:
            return None, blink
        lids = np.array([bs[names.index(k)] for k in LID_BLENDSHAPES], float)
        return np.concatenate([f, p, lids]), blink


# ---------- filter ----------
class OneEuro2D:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def step(self, xy, t_s):
        xy = np.asarray(xy, float)
        if self.x is None:
            self.x, self.dx, self.t = xy, np.zeros(2), t_s
            return xy
        dt = max(t_s - self.t, 1e-3)
        self.t = t_s
        ad = self._a(self.dc, dt)
        self.dx = ad * (xy - self.x) / dt + (1 - ad) * self.dx
        a = self._a(self.mc + self.beta * float(np.hypot(*self.dx)), dt)
        self.x = a * xy + (1 - a) * self.x
        return self.x


# ---------- helpers ----------
def build_estimator():
    if eyetrax.__version__ != EYETRAX_TESTED:
        print(f"WARNING: eyetrax {eyetrax.__version__} != tested {EYETRAX_TESTED}; landmarker swap may break")
    ensure_model()
    return HybridEstimator()


def calibrate(ge, args):
    ge.model = BaseRidge()
    run_9_point_calibration(ge, camera_index=args.camera)  # EyeTrax native routine -> ge.train -> model.train
    if ge.model.W is None:
        return None
    m = ge.model
    train_err = float(np.linalg.norm(m.Xc @ m.W + m.b - m.Yc, axis=1).mean())
    cal = dict(samples=len(m.Yc), points=len(np.unique(m.Yc, axis=0)), alpha=m.alpha, train_err=train_err,
               lopo_err=m.lopo_error())
    print(f"calibration (EyeTrax 9p, hybrid {N_LANDMARK_FEATS}+{N_PATCH_FEATS}+{len(LID_BLENDSHAPES)} feats): {cal['samples']} samples, "
          f"RidgeCV alpha {cal['alpha']}, fit {cal['train_err']:.0f} px, LOPO {cal['lopo_err']:.0f} px")
    return cal


def make_kalman_smoother(ge, args):
    ks = KalmanSmoother(make_kalman())
    if not args.no_tune:
        ks.tune(ge, camera_index=args.camera)
    return ks


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
    if METRICS_CSV.exists():
        with open(METRICS_CSV) as f:
            if f.readline().strip() != ",".join(CSV_FIELDS):
                stamp = datetime.fromtimestamp(METRICS_CSV.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
                legacy = METRICS_CSV.with_name(f"hybrid_metrics.legacy-{stamp}.csv")
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


def open_window(win):
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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


def summarize(rows, other_acc, cal, adapt):
    print("\n==== gaze_hybrid_app summary ====")
    rest_acc = sum(r["accidental_triggers"] for r in rows)
    print(f"accidental triggers: {rest_acc + other_acc} (rest periods {rest_acc}, other {other_acc}; "
          "not monitored during EyeTrax calibration/tune)")
    if cal:
        print(f"calibration: {cal['samples']} samples, alpha {cal['alpha']}, fit {cal['train_err']:.0f} px, "
              f"LOPO {cal['lopo_err']:.0f} px; affine corrector {'ON' if adapt else 'OFF (--no-adapt)'}")
    if rows and adapt:
        print(f"affine corrector: {rows[-1]['corr_clicks']} clicks used, {rows[-1]['corr_skipped']} gated out "
              f"(>{CORR_GATE_PX:.0f} px)")
    if not rows:
        print("no completed trials")
        return
    n = len(rows)
    errs = [r["error_px"] for r in rows if r["error_px"] != ""]
    rts = [r["response_time_ms"] for r in rows if not r["timeout"]]
    cur = dict(median_error_px=float(np.median(errs)) if errs else float("nan"),
               strict_hit=sum(r["strict_hit"] for r in rows) / n, magnetic_hit=sum(r["magnetic_hit"] for r in rows) / n,
               median_rt_ms=float(np.median(rts)) if rts else float("nan"), timeouts=sum(r["timeout"] for r in rows))
    print(f"\nside-by-side ({n} trials this run; single runs of n<=20 -> wide CIs)\n  v1 = {BASELINE['run']}"
          f"\n  v2 ref = {REFERENCE['run']}")
    print(f"  {'metric':24s} {'v1':>8s} {'v2 ref':>8s} {'this run':>9s} {'vs v1':>7s} {'vs ref':>7s}")
    for name, key, f, fd in [("median error (px)", "median_error_px", "{:.0f}", "{:+.0f}"),
                             ("strict hit (<=140px)", "strict_hit", "{:.0%}", "{:+.0%}"),
                             ("magnetic hit (<=190px)", "magnetic_hit", "{:.0%}", "{:+.0%}"),
                             ("median response (ms)", "median_rt_ms", "{:.0f}", "{:+.0f}"),
                             ("timeouts", "timeouts", "{:d}", "{:+d}")]:
        c, b, r = cur[key], BASELINE[key], REFERENCE[key]
        bad = isinstance(c, float) and math.isnan(c)
        print(f"  {name:24s} {f.format(b):>8s} {f.format(r):>8s} {('n/a' if bad else f.format(c)):>9s} "
              f"{('' if bad else fd.format(c - b)):>7s} {('' if bad else fd.format(c - r)):>7s}")
    pre = [r["preonset_error_px"] for r in rows if r["preonset_error_px"] != ""]
    raw = [r["raw_error_px"] for r in rows if r["raw_error_px"] != ""]
    if pre:
        print(f"  (300 ms before freeze: filtered median {np.median(pre):.0f} px, raw median {np.median(raw):.0f} px)")
    half = n // 2
    if adapt and half >= 3 and len(errs) == n:
        print(f"  adaptation trend: median error first {half} trials {np.median(errs[:half]):.0f} px, "
              f"last {n - half} {np.median(errs[half:]):.0f} px")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-adapt", action="store_true", help="disable the affine click corrector (A/B baseline)")
    ap.add_argument("--corr-lambda-a", type=float, default=CORR_LAMBDA_A, help="penalty on affine scale/shear (A - I)")
    ap.add_argument("--corr-lambda-b", type=float, default=CORR_LAMBDA_B, help="penalty on affine translation b")
    ap.add_argument("--min-cutoff", type=float, default=0.8, help="One-Euro min cutoff (Hz)")
    ap.add_argument("--beta", type=float, default=0.015, help="One-Euro speed coefficient")
    ap.add_argument("--d-cutoff", type=float, default=1.0, help="One-Euro derivative cutoff (Hz)")
    ap.add_argument("--no-tune", action="store_true", help="skip EyeTrax's Kalman fine-tuning pass")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--trigger-config", type=Path, default=TRIGGER_CONFIG_PATH)
    args = ap.parse_args()
    if args.trials < 1 or min(args.min_cutoff, args.d_cutoff) <= 0 or args.beta < 0 or \
            min(args.corr_lambda_a, args.corr_lambda_b) <= 0:
        sys.exit("--trials >= 1, cutoffs > 0, beta >= 0, corrector lambdas > 0")
    adapt = not args.no_adapt

    tcfg = load_trigger_config(args.trigger_config)  # read-only
    tj, tg, deb = float(tcfg["jawOpen"]), float(tcfg["lip_gap"]), int(tcfg["debounce_ms"])
    print(f"trigger (validated, read-only): jawOpen>={tj}, lip_gap>={tg}, hold>={deb}ms, cooldown 600ms + release, "
          f"{Path(POP_SOUND).name}; bubble freezes while jawOpen>{FREEZE_JAW}")
    print(f"gaze: eyetrax {eyetrax.__version__} 9p + {N_PATCH_FEATS} CLAHE eye-patch pixels + {len(LID_BLENDSHAPES)} "
          f"lid blendshapes -> RidgeCV{ALPHAS} (frozen after calibration); "
          f"Kalman + One-Euro(min_cutoff={args.min_cutoff}, beta={args.beta}, d_cutoff={args.d_cutoff}); "
          + (f"affine corrector ON (lambda_A {args.corr_lambda_a}, lambda_b {args.corr_lambda_b}, decay {CORR_DECAY}, "
             f"<= {CORR_MAX} clicks, gate {CORR_GATE_PX:.0f} px, clamp +/-{CORR_CLAMP_PX:.0f} px)" if adapt
             else "affine corrector OFF"))

    sw, sh = get_screen_size()
    targets = make_targets(args.trials, sw, sh, random.Random(args.seed))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    ge = build_estimator()
    cal, rows, other_acc, csv_f, cap = None, [], 0, None, None
    try:
        cal = calibrate(ge, args)
        if cal is None:
            print("calibration aborted or produced no samples")
            return
        kalman = make_kalman_smoother(ge, args)
        euro = OneEuro2D(args.min_cutoff, args.beta, args.d_cutoff)
        corr = AffineCorrector(sw, sh, args.corr_lambda_a, args.corr_lambda_b)
        csv_f, writer = open_metrics_csv()
        cap = open_capture(args.camera)
        win = "gaze_hybrid_app"
        open_window(win)
        trig = Trigger(deb)
        hist = deque(maxlen=400)  # (t_ms, base_xy, corrected_raw_xy, filtered_xy) -- unfrozen frames only
        bubble, frozen_since, feedback = None, None, None
        rest_acc, phase, phase_t0, trial = 0, "cal_result", None, 0
        last_ts, fails = -1, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= 10:
                    print("camera read failed 10x in a row, aborting")
                    break
                continue
            fails = 0
            t = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = t
            phase_t0 = t if phase_t0 is None else phase_t0

            feats, blink = ge.extract_features(frame)  # one landmarker inference via the tap
            lm, bs, names, _mat, present = extract(ge.tap.last)
            jaw = float(bs[names.index("jawOpen")]) if present and names else 0.0
            mouth = bool(present and names and jaw >= tj and float(lip_gap(dict(lm=lm[None]))[0]) >= tg)
            onset = trig.update(t, mouth)
            fired = onset is not None
            if fired:
                play(POP_SOUND)

            if present and jaw > FREEZE_JAW:
                frozen_since = frozen_since or t
            elif not present or jaw < FREEZE_JAW:
                frozen_since = None
            frozen = frozen_since is not None
            if feats is not None and not blink and not frozen:
                base = ge.predict(feats[None])[0]  # frozen base regressor
                raw = np.clip(corr.correct(base) if adapt else base, [0, 0], [sw - 1, sh - 1])
                kx, ky = kalman.step(int(raw[0]), int(raw[1]))
                f = euro.step((kx, ky), t / 1000.0)
                bubble = (int(np.clip(f[0], 0, sw - 1)), int(np.clip(f[1], 0, sh - 1)))
                hist.append((t, base, raw, np.array(bubble, float)))

            img = np.full((sh, sw, 3), 28, np.uint8)
            el = t - phase_t0
            lines, near, tgt = [], False, None
            if phase == "cal_result":
                other_acc += fired
                lines = [(f"Calibration: fit {cal['train_err']:.0f} px, LOPO {cal['lopo_err']:.0f} px", 0.75),
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
                if bubble is not None:  # frozen bubble -> frozen snap ring
                    near = np.hypot(bubble[0] - tgt[0], bubble[1] - tgt[1]) <= TARGET_R + MAGNET_PX
                if fired or el >= TIMEOUT_MS:
                    ref = (frozen_since or onset) if fired else t  # window ends where mouth movement began
                    win_pts = [(fe, rw, fl) for ts_, fe, rw, fl in hist if ref - PRE_ONSET_MS <= ts_ < ref]
                    sel = np.array(bubble, float) if bubble is not None else None
                    pre = np.mean([fl for _, _, fl in win_pts], 0) if win_pts else None
                    raw_m = np.mean([rw for _, rw, _ in win_pts], 0) if win_pts else None
                    err = lambda p: round(float(np.hypot(*(p - tgt))), 1) if p is not None else ""
                    xy = lambda p, i: round(float(p[i])) if p is not None else ""
                    e = err(sel)
                    mag = bool(fired and e != "" and e <= TARGET_R + MAGNET_PX)
                    strict = bool(fired and e != "" and e <= TARGET_R)
                    base_m = np.mean([bs_ for bs_, _, _ in win_pts], 0) if win_pts else None
                    applied = corr.delta(base_m) if (adapt and base_m is not None) else np.zeros(2)
                    refit = ""
                    if adapt and fired and base_m is not None:  # every completed click (gated for look-aways)
                        ms = corr.add(base_m, tgt)
                        refit = round(ms, 2) if ms is not None else "gated"
                    row = dict(run_id=run_id, trial=trial + 1, adapt=adapt, lopo_err_px=round(cal["lopo_err"], 1),
                               target_x=tgt[0], target_y=tgt[1], gaze_x=xy(sel, 0), gaze_y=xy(sel, 1), error_px=e,
                               preonset_gaze_x=xy(pre, 0), preonset_gaze_y=xy(pre, 1), preonset_error_px=err(pre),
                               raw_gaze_x=xy(raw_m, 0), raw_gaze_y=xy(raw_m, 1), raw_error_px=err(raw_m),
                               magnetic_hit=mag, strict_hit=strict,
                               response_time_ms=(t - phase_t0) if fired else TIMEOUT_MS, timeout=not fired,
                               accidental_triggers=rest_acc,
                               freeze_ms=(t - frozen_since) if (fired and frozen_since) else "",
                               gaze_window_samples=len(win_pts), corr_clicks=corr.clicks, corr_skipped=corr.skipped,
                               corr_dx=round(float(applied[0]), 1), corr_dy=round(float(applied[1]), 1), refit_ms=refit)
                    rows.append(row)
                    writer.writerow(row)
                    csv_f.flush()
                    print(f"trial {trial + 1:2d}: target {tgt} gaze ({row['gaze_x']},{row['gaze_y']}) err {e} px -> "
                          f"{'HIT ' if mag else 'MISS'}{' strict' if strict else ''}{' (timeout)' if not fired else ''} "
                          f"{row['response_time_ms']} ms, correction applied ({row['corr_dx']:+.0f},{row['corr_dy']:+.0f})"
                          + (f", corrector refit {refit} ms ({corr.clicks} clicks)" if isinstance(refit, float) else
                             (", click gated out as look-away" if refit == "gated" else "")))
                    feedback, rest_acc, trial, tgt = (tgt, mag, t), 0, trial + 1, None
                    phase, phase_t0 = ("rest", t) if trial < args.trials else ("done", t)
            else:
                other_acc += fired
                lines = [(f"Done: {sum(x['magnetic_hit'] for x in rows)}/{len(rows)} hits", 0.9),
                         ("press q to quit", 0.6)]

            if tgt is not None:
                if near:
                    blend_circle(img, tgt, TARGET_R + 26, AMBER, 0.22)
                cv2.circle(img, tgt, TARGET_R, (0, 200, 255) if near else (0, 150, 230), -1)
                cv2.circle(img, tgt, 9, (255, 255, 255), -1)
            draw_feedback(img, feedback, t)
            if bubble is not None and phase in ("rest", "target", "done"):
                blend_circle(img, bubble, BUBBLE_R, CYAN, 0.40 if frozen else 0.25)
                cv2.circle(img, bubble, 4, CYAN, -1)
                if near and tgt is not None:
                    cv2.circle(img, tgt, TARGET_R + RING_PAD, (255, 255, 255), 4)
                else:
                    cv2.circle(img, bubble, BUBBLE_R, (255, 255, 255) if frozen else CYAN, 2)
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
                old_model = ge.model
                new_cal = calibrate(ge, args)
                if new_cal is not None:
                    cal = new_cal
                else:
                    ge.model = old_model
                    print("recalibration aborted; keeping previous calibration")
                kalman = make_kalman_smoother(ge, args)
                euro = OneEuro2D(args.min_cutoff, args.beta, args.d_cutoff)
                corr = AffineCorrector(sw, sh, args.corr_lambda_a, args.corr_lambda_b)
                cap = open_capture(args.camera)
                open_window(win)
                hist.clear()
                bubble, frozen_since, phase, phase_t0 = None, None, "cal_result", None  # interrupted trial restarts
    finally:
        ge.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if csv_f:
            csv_f.close()
        summarize(rows, other_acc, cal, adapt)
        if rows:
            print(f"trials logged to {METRICS_CSV.relative_to(ROOT)} (run_id {run_id})")


if __name__ == "__main__":
    main()
