"""MGazeNet gaze trials with 13-dot mouth-pop calibration.

Look at the pinpoint, then pop your mouth to record each dot. Check the fitted
warp in the four-second lock verification. Press c to recenter, h to toggle
head vernier, or q / Esc to quit. Use --use-saved to reuse a calibration.
"""
import argparse
import csv
import math
import os
import pickle
import random
import sys
import threading
import time
from tempfile import NamedTemporaryFile
from collections import deque
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pygame
import cv2
import mediapipe as mp
from gazefollower import GazeFollower
from gazefollower.camera import WebCamCamera
from gazefollower.filter import OneEuroFilter
from gazefollower.misc import CameraRunningState, DefaultConfig
from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import LinearRegression, Ridge

from capture import ensure_model, extract, make_landmarker
from eval_offline import TRIGGER_CONFIG_PATH, lip_gap
from live_app import POP_SOUND, Trigger, load_trigger_config, play

ROOT = Path(__file__).resolve().parent
METRICS_CSV = ROOT / "logs" / "gazefollower_metrics.csv"
TRACE_CSV = ROOT / "logs" / "gazefollower_trace.csv"
CALIB_PATH = ROOT / "models" / "gf_calibrator.pkl"
CSV_FIELDS = ["run_id", "trial", "cal_method", "cal_model", "cal_cv_px", "cal_fit_px", "bg", "gaze_filter",
              "refine_trials", "refined", "recenters", "offset_x", "offset_y", "target_x", "target_y", "gaze_x",
              "gaze_y", "error_px", "preonset_gaze_x", "preonset_gaze_y", "preonset_error_px", "raw_gaze_x",
              "raw_gaze_y", "raw_error_px", "magnetic_hit", "strict_hit", "response_time_ms", "timeout",
              "accidental_triggers", "freeze_ms", "gaze_window_samples", "target_r", "fixlock", "head_vernier",
              "head_dx", "head_dy", "cal_trimmed", "bank_sessions"]
TRACE_FIELDS = ["run_id", "t_ms", "phase", "trial", "target_x", "target_y", "gaze_ok", "raw_x_cm", "raw_y_cm",
                "cal_x", "cal_y", "filt_x", "filt_y", "jaw_open", "frozen", "frame_lum", "cursor_x", "cursor_y",
                "locked", "head_yaw", "head_pitch", "head_roll", "head_dist", "head_dx", "head_dy", "vernier"]
BENCHMARKS = [dict(name="v1", run="v1-milestone 20261119-030154 (EyeTrax ridge/9p)", median_error_px=116.0,
                   strict_hit=0.60, magnetic_hit=0.95, median_rt_ms=2262.0, timeouts=0),
              dict(name="v2 ref", run="v2 reference 20261119-034127 (hybrid v2.1)", median_error_px=94.0,
                   strict_hit=0.75, magnetic_hit=0.90, median_rt_ms=1350.0, timeouts=0)]

CAM_W, CAM_H = 1280, 720  # resolution the mouth trigger was validated at
FREEZE_JAW = 0.12
BUBBLE_R, TARGET_R, MAGNET_PX, RING_PAD = 70, 140, 50, 14
REST_MS, TIMEOUT_MS, PRE_ONSET_MS, FEEDBACK_MS, LOOK_MS = 600, 8000, 300, 450, 4000
MIN_CUTOFF, BETA = 0.5, 0.0005  # OneEuro on pixels; chosen on the live stream of run 20261119-040422 (docs/plan.md)
GRID = [(fx, fy) for fy in (0.25, 0.5, 0.75) for fx in (0.2, 0.5, 0.8)]
FEATURE_DIM = 258  # MGazeNet features (first two = raw gaze in cm)
POSE_DIM = 4  # head yaw, pitch, roll (deg) and camera distance (cm) for head vernier and trace
CAL_POINTS = ([(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]
              + [(x, y) for y in (0.3, 0.7) for x in (0.3, 0.7)])
FOCUS_MS, WINDOW_MS, LOCK_RADIUS = 700, 450, 65
REFINE_SETTLE_MS = 300
MIN_POINT_SAMPLES, BLINK_FRAC, OUTLIER_Z, RAW_MAD_FLOOR_CM = 5, 0.6, 3.5, 0.1
# fixation lock on the OneEuro output (docs/plan.md v3.4): lock after speed <= FIX_SPEED for > FIX_DWELL; damp
# deviations <= FIX_JITTER heavily (ramp to none at FIX_RELEASE); release on the same frame beyond FIX_RELEASE
FIX_SPEED_PX_S, FIX_DWELL_MS, FIX_JITTER_PX, FIX_RELEASE_PX, FIX_DAMP = 800.0, 120, 18.0, 45.0, 0.1  # speed: replay on runs 051840+052008
# head vernier (docs/plan.md v3.4): screen offset = -gain * (pose - ref) beyond the deadzone (yaw -> x, pitch -> y;
# sign from data/: left corners have higher yaw, the bottom row lower pitch), EMA-smoothed, clamped per axis
HEAD_DEADZONE_DEG = 1.0  # still fixation in data/: |pose - median| p95 0.8 deg (yaw and pitch)
HEAD_GAIN_PX_DEG, HEAD_CLAMP_PX, HEAD_RESET_PX, HEAD_SETTLE_MS, HEAD_TAU_MS = 25.0, 110.0, 130.0, 150, 120.0
SMOOTHING = (0.0, 5.0, 25.0, 100.0, 500.0)
ALPHAS = (0.1, 1.0, 5.0, 20.0)
CANDIDATES = ([("ridge", a, None) for a in ALPHAS]
              + [("affine_tps", None, s) for s in SMOOTHING]
              + [("ridge_tps", a, s) for a in ALPHAS for s in SMOOTHING])
CLIP_MARGIN = 0.25
REFINE_W, REFINE_DECAY, REFINE_MAX, REFINE_MAX_ERR = 0.2, 0.94, 25, 190
RECENTER_MS, RECENTER_SKIP_MS, RECENTER_MIN_SAMPLES = 1000, 300, 5
CYAN, TARGET, TARGET_NEAR, AMBER, GREEN, RED, ORB = ((0, 255, 255), (230, 150, 0), (255, 200, 0), (255, 190, 0),
                                                    (80, 220, 80), (230, 60, 60), (255, 110, 20))  # RGB
# "white" is GazeFollower's calibration-UI background (kept for continuity with v3.1); "dark" = run 040422's screen.
PALETTES = {"white": dict(bg=(255, 255, 255), text=(40, 40, 40), edge=(0, 140, 165), ring=(40, 40, 40),
                          dot=(120, 120, 120), fill_alpha=120, frozen_alpha=190),
            "dark": dict(bg=(28, 28, 28), text=(230, 230, 230), edge=CYAN, ring=(255, 255, 255),
                         dot=(150, 150, 150), fill_alpha=64, frozen_alpha=100)}


def now_ms():
    return time.time_ns() // 1_000_000  # same clock as GazeFollower's frame timestamps


def head_pose(mat):
    """(yaw, pitch, roll in degrees, camera distance in cm) from FaceLandmarker's facial transformation matrix.
    The matrix has unit scale, so the distance is the face-scale term."""
    m = np.asarray(mat, float)
    fwd = m[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return (math.degrees(math.atan2(fwd[0], fwd[2])), math.degrees(math.atan2(fwd[1], math.hypot(fwd[0], fwd[2]))),
            math.degrees(math.atan2(m[1, 0], m[0, 0])), float(-m[2, 3]))


class FixationLock:
    """Post-filter on the OneEuro cursor: after the speed has stayed <= FIX_SPEED_PX_S for > FIX_DWELL_MS it locks;
    while locked the output follows deviations <= FIX_JITTER_PX with gain FIX_DAMP (ramping to 1 at FIX_RELEASE_PX)
    and jumps straight to the input (no added lag) once the input is > FIX_RELEASE_PX from the locked point."""

    def __init__(self, speed=FIX_SPEED_PX_S, dwell=FIX_DWELL_MS, jitter=FIX_JITTER_PX, release=FIX_RELEASE_PX,
                 damp=FIX_DAMP):
        self.speed, self.dwell, self.jitter, self.release, self.damp = speed, dwell, jitter, release, damp
        self.reset()

    def reset(self):
        self.prev, self.slow_since, self.point = None, None, None

    def update(self, t, p):
        p = np.asarray(p, float)
        prev, self.prev = self.prev, (t, p)
        if self.point is not None:
            d = float(np.hypot(*(p - self.point)))
            if d > self.release:
                self.point, self.slow_since = None, None
                return p
            k = self.damp if d <= self.jitter else \
                self.damp + (1 - self.damp) * (d - self.jitter) / (self.release - self.jitter)
            self.point = self.point + k * (p - self.point)
            return self.point.copy()
        if prev is not None and t > prev[0] and np.hypot(*(p - prev[1])) * 1000.0 / (t - prev[0]) <= self.speed:
            self.slow_since = prev[0] if self.slow_since is None else self.slow_since
            if t - self.slow_since > self.dwell:
                self.point = p.copy()
        else:
            self.slow_since = None
        return p


class HeadVernier:
    """Fine-aim offset from small head rotations around a reference pose. The reference is re-taken (offset zeroed)
    whenever the coarse gaze moves > HEAD_RESET_PX from where it was taken, and it follows the head for
    HEAD_SETTLE_MS afterwards (the head catches up after the eyes). Off = zero offset."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.ref = self.anchor = self.since = self.t = None
        self.off = np.zeros(2)

    def update(self, t, coarse, pose):
        if not self.enabled:
            self.reset()
            return self.off
        if pose is None or not all(np.isfinite(pose)):
            return self.off
        pose, coarse = np.asarray(pose, float), np.asarray(coarse, float)
        if self.anchor is None or np.hypot(*(coarse - self.anchor)) > HEAD_RESET_PX:
            self.reset()
            self.anchor, self.ref, self.since, self.t = coarse, pose, t, t
            return self.off
        dt, self.t = max(t - self.t, 0), t
        if t - self.since < HEAD_SETTLE_MS:
            self.ref = pose
            return self.off
        d = pose - self.ref
        dz = np.sign(d) * np.maximum(np.abs(d) - HEAD_DEADZONE_DEG, 0.0)
        goal = np.clip(-HEAD_GAIN_PX_DEG * dz, -HEAD_CLAMP_PX, HEAD_CLAMP_PX)
        self.off = self.off + (1 - math.exp(-dt / HEAD_TAU_MS)) * (goal - self.off)
        return self.off


class MouthAndGaze:
    """Shared state between GazeFollower's camera thread (writer) and the pygame main thread (reader)."""

    def __init__(self, tj, tg, debounce_ms, fixlock=True, head_vernier=False):
        self.lock = threading.Lock()
        self.fixlock = FixationLock() if fixlock else None
        self.vernier = HeadVernier(head_vernier)
        self.head = (float("nan"),) * POSE_DIM  # head_pose() of the latest frame with a face
        self.locked = False
        self.tj, self.tg = tj, tg
        self.landmarker, self.trigger = make_landmarker(), Trigger(debounce_ms)
        self.last_ts = -1
        self.present, self.jaw, self.frozen_since = False, 0.0, None
        self.onsets = deque()  # (onset_ms, fired_ms, freeze_start_ms | None) for the main thread
        self.hist = deque(maxlen=400)  # (t_ms, calibrated_xy, filtered_xy), offset applied -- unfrozen only
        self.feats = deque(maxlen=900)  # (t_ms, MGazeNet features, jawOpen, eye openness) -- frames with a face
        self.offset = np.zeros(2)  # 1-dot recenter offset (px), added to calibrated and filtered gaze
        self.bubble = None
        self.tag = ("start", 0, None)  # (phase, trial, target) written by the main thread
        self.trace = []
        self.lum_now, self.lum = float("nan"), [0.0, 0]

    def reset_landmarker(self):
        self.landmarker.close()
        self.landmarker = make_landmarker()
        self.last_ts = -1

    def on_camera(self, state, timestamp_ns, frame_rgb):
        """Every camera frame (RGB, as GazeFollower's WebCamCamera delivers them), before gaze processing."""
        if state != CameraRunningState.SAMPLING:
            return
        self.lum_now = float(frame_rgb[::8, ::8].mean())
        self.lum[0] += self.lum_now
        self.lum[1] += 1
        self.on_frame(frame_rgb, timestamp_ns // 1_000_000)

    def on_frame(self, frame_rgb, t_ms):
        """Our validated mouth trigger on the RGB camera frame (VIDEO mode, blendshapes). Runs before gaze."""
        t = max(int(t_ms), self.last_ts + 1)
        self.last_ts = t
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)), t)
        lm, bs, names, mat, present = extract(res)
        jaw = float(bs[names.index("jawOpen")]) if present and names else 0.0
        mouth = bool(present and names and jaw >= self.tj and float(lip_gap(dict(lm=lm[None]))[0]) >= self.tg)
        onset = self.trigger.update(t, mouth)
        head = head_pose(mat) if present and np.isfinite(mat).all() else (float("nan"),) * POSE_DIM
        with self.lock:
            self.present, self.jaw, self.head = present, jaw, head
            if present and jaw > FREEZE_JAW:
                self.frozen_since = self.frozen_since or t
            else:
                self.frozen_since = None
            if onset is not None:
                self.onsets.append((onset, t, self.frozen_since))
        if onset is not None:
            play(POP_SOUND)
        return onset

    def on_gaze(self, face_info, gaze_info):
        """Subscriber (camera thread, same frame as on_frame): features for calibration; gaze -> bubble."""
        t, nan2 = self.last_ts, (float("nan"), float("nan"))
        f = c = None
        if gaze_info.filtered_gaze_coordinates is not None:
            f = np.asarray(gaze_info.filtered_gaze_coordinates, float)
            c = np.asarray(gaze_info.calibrated_gaze_coordinates, float)
            if not (np.isfinite(f).all() and np.isfinite(c).all()):
                f = c = None
        raw = gaze_info.raw_gaze_coordinates
        openness = (float(face_info.left_eye_openness) + float(face_info.right_eye_openness)) / 2 \
            if face_info is not None and face_info.status else float("nan")
        with self.lock:
            if self.present and gaze_info.status and gaze_info.features is not None:
                self.feats.append((t, np.asarray(gaze_info.features, np.float32).ravel().copy(),
                                   self.jaw, openness))
            if f is not None:
                c, f = c + self.offset, f + self.offset
            frozen = self.frozen_since is not None
            out = None
            if self.present and f is not None and not frozen:  # cursor = OneEuro -> fixation lock -> + head vernier
                out = f if self.fixlock is None else self.fixlock.update(t, f)
                self.locked = self.fixlock is not None and self.fixlock.point is not None
                out = out + self.vernier.update(t, f, self.head[:2])
                self.bubble = out
                self.hist.append((t, c, out))
            phase, trial, tgt = self.tag
            self.trace.append((t, phase, trial, *(tgt if tgt is not None else ("", "")), int(f is not None),
                               *(nan2 if raw is None else (float(raw[0]), float(raw[1]))),
                               *(nan2 if c is None else c), *(nan2 if f is None else f),
                               self.jaw, int(frozen), self.lum_now, *(nan2 if out is None else out),
                               int(self.locked), *self.head, *self.vernier.off, int(self.vernier.enabled)))


class FrameTimedOneEuro(OneEuroFilter):
    """The library's OneEuroFilter, fed the frame time in seconds (its own unit auto-detection misreads ms/ns)."""

    def __init__(self, clock_ms, **kw):
        super().__init__(**kw)
        self.clock_ms = clock_ms

    def filter_values(self, values, timestamp=-1):
        return self.filter(np.asarray(values, float), timestamp=self.clock_ms() / 1000.0)


class Cam720(WebCamCamera):
    """WebCamCamera sets width/height before the device is open (a no-op). Set them after opening, as capture.py
    does, and fail instead of streaming black frames if the camera cannot open."""

    mode = None

    def open(self):
        if not self._cap.open(self.webcam_id):
            raise RuntimeError(f"camera {self.webcam_id} failed to open")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        mode = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                round(self._cap.get(cv2.CAP_PROP_FPS), 1))
        if mode != self.mode:
            print(f"camera delivers {mode[0]}x{mode[1]} @ {mode[2]} fps (frames are resized to {self.img_width}x"
                  f"{self.img_height} if different)")
        self.mode = mode
        self._create_capture_thread()


# ---------------------------------------------------------------- calibrator
def spec_name(spec):
    kind, alpha, smooth = spec
    base = "Affine" if alpha is None else f"Ridge(258-D, alpha={alpha:g})"
    return base if kind == "ridge" else f"{base} + TPS(s={smooth:g})"


def fit_model(X, Y, spec, weights=None):
    kind, alpha, smooth = spec
    raw = X[:, :2]  # MGazeNet sets raw_gaze_coordinates = features[:2], in cm
    cols = 2 if kind == "affine_tps" else min(FEATURE_DIM, X.shape[1])
    base = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    base.fit(X[:, :cols], Y, sample_weight=weights)
    spline = None
    if smooth is not None:
        residual = Y - base.predict(X[:, :cols])
        # Preserve the selected anchor smoothing while damping optional hit corrections.
        smoothing = smooth if weights is None else (smooth + 1) / weights - 1
        spline = RBFInterpolator(raw, residual, kernel="thin_plate_spline", smoothing=smoothing)
    return cols, base, spline


def predict_model(model, X):
    cols, base, spline = model
    pred = base.predict(X[:, :cols])
    if spline is not None:
        pred += spline(X[:, :2])
    return np.clip(pred, -CLIP_MARGIN, 1 + CLIP_MARGIN)


def dot_anchor(samples):
    """One vote per dot. Spatial median of raw gaze and component medians of latent features."""
    X = np.asarray(samples, dtype=float)
    anchor = np.median(X, axis=0)
    raw = X[:, :2]
    point = anchor[:2].copy()
    # Modified Weiszfeld update also handles a median coincident with a sample.
    for _ in range(200):
        delta = raw - point
        dist = np.linalg.norm(delta, axis=1)
        away = dist > 1e-10
        direction = (delta[away] / dist[away, None]).sum(axis=0)
        magnitude = np.linalg.norm(direction)
        coincident = len(raw) - away.sum()
        if magnitude <= coincident:
            break
        step = direction * (1 - coincident / magnitude) / (1 / dist[away]).sum()
        point += step
        if np.linalg.norm(step) < 1e-8:
            break
    anchor[:2] = point
    return anchor


class NonlinearCalibration:
    def __init__(self):
        self.has_calibrated = False
        self.fitted = self.original = None
        self.refine = deque(maxlen=REFINE_MAX)

    def predict(self, sample, estimated_coordinate):
        model = self.fitted
        if model is None or not np.isfinite(sample).all():
            return False, estimated_coordinate
        return True, tuple(predict_model(model, np.asarray(sample).reshape(1, -1))[0])

    def install(self, model, X, Y, spec):
        self.anchors, self.spec = (X, Y), spec
        self.refine.clear()
        self.original = self.fitted = model
        self.has_calibrated = True

    def add_refinement(self, X, Y):
        x, y = self.anchors
        trials = [*self.refine, (dot_anchor(X)[:x.shape[1]], Y)][-REFINE_MAX:]
        weights = np.r_[np.ones(len(x)), REFINE_W * REFINE_DECAY ** np.arange(len(trials))[::-1]]
        try:
            model = fit_model(np.vstack([x, *[a for a, _ in trials]]),
                              np.vstack([y, *[b for _, b in trials]]), self.spec, weights)
        except np.linalg.LinAlgError:
            return False
        self.refine.append(trials[-1])
        self.fitted = model
        return True

    def release(self):
        pass


class TappedGazeFollower(GazeFollower):
    """Single-process GazeFollower: every frame goes through our hook first; SAMPLING is handled here so that it
    also runs before any calibration exists (the library's branch raises), which our calibration flows need."""

    def __new__(cls, *a, frame_hook=None, **kw):  # base __new__ would route to the multiprocess class
        return object.__new__(cls)

    def __init__(self, *a, frame_hook=None, **kw):
        super().__init__(*a, **kw)
        self.frame_hook = frame_hook

    def process_frame(self, state, timestamp, frame):
        if self.frame_hook is not None:
            self.frame_hook(state, timestamp, frame)
        if state != CameraRunningState.SAMPLING:
            return super().process_frame(state, timestamp, frame)
        face_info = self.face_alignment.detect(timestamp, frame)
        gaze_info = self.gaze_estimator.detect(frame, face_info)
        if gaze_info.status and gaze_info.features is not None and self.calibration.has_calibrated:
            ok, xy = self.calibration.predict(gaze_info.features, gaze_info.raw_gaze_coordinates)
            if ok:
                px = self._calibration_controller.convert_to_pixel(xy)
                gaze_info.calibrated_gaze_coordinates = px
                gaze_info.filtered_gaze_coordinates = self.gaze_filter.filter_values(px)
        self.dispatch_face_gaze_info(face_info, gaze_info)


def build_follower(camera_index, live, min_cutoff=MIN_CUTOFF, beta=BETA):
    cfg = DefaultConfig()
    cfg.use_multiprocessing = False
    cam = Cam720(webcam_id=camera_index, img_width=CAM_W, img_height=CAM_H)
    filt = FrameTimedOneEuro(lambda: live.last_ts, freq=24.0, min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
    return TappedGazeFollower(camera=cam, gaze_filter=filt, calibration=NonlinearCalibration(), config=cfg,
                              use_multiprocessing=False, frame_hook=live.on_camera)


def sync_screen_size(gf, size):
    """Make GazeFollower's fraction->pixel conversion use the real drawing surface size."""
    s = np.array([int(size[0]), int(size[1])])
    gf.config.screen_size = s
    gf.screen_size = s
    gf._calibration_controller.screen_size = s


# ---------------------------------------------------------------- calibration flows
def window_frames(feats, lo_ms, hi_ms):
    """[(t, sample, eye openness)] with lo_ms <= t < hi_ms and the mouth still closed."""
    return [(t, x, op) for t, x, jw, op in feats if lo_ms <= t < hi_ms and jw <= FREEZE_JAW and np.isfinite(x).all()]


def clean_mask(W):
    """False for blinks (eye openness < BLINK_FRAC x this window's median; relative because the lids also lower when
    looking down) and raw-gaze outliers (> OUTLIER_Z robust z)."""
    op = np.array([o for _, _, o in W], float)
    keep = np.isfinite(op) & (op > 0)
    if keep.any():
        keep &= op >= BLINK_FRAC * np.median(op[keep])
    if keep.any():
        raw = np.array([x[:2] for _, x, _ in W], float)
        med = np.median(raw[keep], 0)
        mad = np.maximum(1.4826 * np.median(np.abs(raw[keep] - med), 0), RAW_MAD_FLOOR_CM)
        keep &= (np.abs(raw - med) / mad <= OUTLIER_Z).all(1)
    return keep


def window_samples(feats, dot_on_ms, freeze_ms, win_ms, settle_ms):
    """Clean samples from the win_ms before the bubble froze, starting >= settle_ms after the dot/target appeared.
    Returns (samples, n_rejected)."""
    W = window_frames(feats, max(dot_on_ms + settle_ms, freeze_ms - win_ms), freeze_ms)
    if not W:
        return [], 0
    keep = clean_mask(W)
    return [w[1] for w, k in zip(W, keep) if k], int((~keep).sum())


def fit_points(gf, per_point, points):
    """Train on one robust anchor per dot. Score every clean held-out frame, weighting dots equally."""
    frames = [np.asarray(xs, float) for xs in per_point]
    X = np.stack([dot_anchor(xs) for xs in frames])
    Y = np.asarray(points, float)
    screen = np.asarray(gf.config.screen_size)
    scores = {}
    for spec in CANDIDATES:
        errors = []
        for i in range(len(points)):
            train = np.arange(len(points)) != i
            try:
                model = fit_model(X[train], Y[train], spec)
                prediction = predict_model(model, frames[i])
                errors.append(np.linalg.norm((prediction - Y[i]) * screen, axis=1).mean())
            except np.linalg.LinAlgError:  # degenerate spline geometry in this fold
                break
        scores[spec] = float(np.mean(errors)) if len(errors) == len(points) else math.inf

    for spec in sorted(scores, key=scores.get):
        if not np.isfinite(scores[spec]):
            break
        try:
            model = fit_model(X, Y, spec)
        except np.linalg.LinAlgError:
            continue
        gf.calibration.install(model, X, Y, spec)
        fit = np.linalg.norm((predict_model(model, X) - Y) * screen, axis=1)
        return dict(method="pop13", spec=spec, cv_px=scores[spec], fit_px=float(fit.mean()),
                    all_cv=scores, per_point=[len(xs) for xs in per_point])
    raise ValueError("No calibration candidate produced a finite LOPO score")


def recenter_offset(hist, t_start, centre, offset):
    """New offset from the RECENTER window of hist [(t, calibrated+offset, filtered+offset)], or None."""
    pts = [c for t, c, _ in hist if t_start + RECENTER_SKIP_MS <= t <= t_start + RECENTER_MS]
    if len(pts) < RECENTER_MIN_SAMPLES:
        return None, len(pts)
    return np.asarray(offset, float) + (np.asarray(centre, float) - np.median(np.array(pts), axis=0)), len(pts)


def save_calibration(path, gf, info, offset):
    path.parent.mkdir(exist_ok=True)
    cal = gf.calibration
    blob = dict(format=4, window=list(gf.config.screen_size), info=info,
                versions=[version(k) for k in ("scipy", "scikit-learn", "gazefollower")],
                anchors=cal.anchors, model=cal.original, offset=np.asarray(offset))
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        try:
            pickle.dump(blob, tmp)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def load_calibration(path, gf):
    try:
        with path.open("rb") as f:
            blob = pickle.load(f)
        if blob["format"] != 4 or blob["window"] != list(gf.config.screen_size):
            return None, "incompatible saved calibration or screen size"
        if blob["versions"] != [version(k) for k in ("scipy", "scikit-learn", "gazefollower")]:
            return None, "calibration dependency versions changed"
        info = blob["info"].copy()
        offset = np.asarray(blob["offset"], float)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            return None, "invalid saved recenter offset"
        gf.calibration.install(blob["model"], *blob["anchors"], info["spec"])
        return dict(info, offset=offset, loaded=True), None
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, ValueError, AttributeError, ImportError) as exc:
        return None, str(exc)


# ---------------------------------------------------------------- UI
def make_targets(n, sw, sh, rng):
    grid = [(int(fx * sw), int(fy * sh)) for fx, fy in GRID]
    pts = (grid * (n // len(grid) + 1))[:n]
    for _ in range(1000):
        rng.shuffle(pts)
        if all(a != b for a, b in zip(pts, pts[1:])):
            break
    return pts


def open_csv(path, fields):
    """Append-mode CSV; a file with a different header is renamed to *.legacy-<mtime>.csv first."""
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        with open(path) as f:
            if f.readline().strip() != ",".join(fields):
                stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
                os.replace(path, path.with_name(f"{path.stem}.legacy-{stamp}.csv"))
    new = not path.exists()
    f = open(path, "a", newline="")
    if new:
        csv.writer(f).writerow(fields)
    return f


def write_trace(path, run_id, trace):
    cell = lambda v: ("" if math.isnan(v) else round(float(v), 3)) if isinstance(v, float) else v
    with open_csv(path, TRACE_FIELDS) as f:
        w = csv.writer(f)
        for r in trace:
            w.writerow([run_id, *(cell(v) for v in r)])


def alpha_circle(win, color, center, r, alpha, width=0):
    s = pygame.Surface((2 * r + 4, 2 * r + 4), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, alpha), (r + 2, r + 2), r, width)
    win.blit(s, (center[0] - r - 2, center[1] - r - 2))


def draw_pinpoint(win, pos, progress):
    p = tuple(map(round, pos))
    radius = round(2 + 42 * max(0, 1 - progress))
    if radius > 2:
        pygame.draw.circle(win, ORB, p, radius, 2)
    pygame.draw.circle(win, ORB, p, 2)  # four-pixel diameter


def text_center(win, font, txt, y, color):
    surf = font.render(txt, True, color)
    win.blit(surf, ((win.get_width() - surf.get_width()) // 2, y))


def draw_scene(win, pal, font, small, t, bubble=None, frozen=False, tgt=None, near=False, feedback=None, dots=(),
               lines=(), lines_y=30, present=True, orb=None, pulse=None, target_r=TARGET_R, status="", locked_dots=()):
    sw, sh = win.get_size()
    win.fill(pal["bg"])
    for i, d in enumerate(dots):
        color = (0, 240, 65) if i in locked_dots else pal["dot"]
        pygame.draw.circle(win, color, d, 12, 0 if i in locked_dots else 2)
        pygame.draw.circle(win, color, d, 2)
    if tgt is not None:
        if near:
            alpha_circle(win, AMBER, tgt, target_r + 26, 56)
        pygame.draw.circle(win, TARGET_NEAR if near else TARGET, tgt, target_r)
        pygame.draw.circle(win, (255, 255, 255), tgt, min(9, max(3, target_r // 5)))
    if feedback is not None and t - feedback[2] < FEEDBACK_MS:
        k = (t - feedback[2]) / FEEDBACK_MS
        if feedback[1]:
            pygame.draw.circle(win, GREEN, feedback[0], int(target_r * (1 + 0.6 * k)), max(2, int(14 * (1 - k))))
        else:
            over = pygame.Surface((sw, sh), pygame.SRCALPHA)
            over.fill((*RED, int(90 * (1 - k))))
            win.blit(over, (0, 0))
    if bubble is not None:
        b = (int(bubble[0]), int(bubble[1]))
        alpha_circle(win, CYAN, b, BUBBLE_R, pal["frozen_alpha"] if frozen else pal["fill_alpha"])
        pygame.draw.circle(win, pal["edge"], b, 4)
        if near and tgt is not None:  # magnetic snap ring (frozen with the bubble)
            pygame.draw.circle(win, pal["ring"], tgt, target_r + RING_PAD, 4)
        else:
            pygame.draw.circle(win, pal["ring"] if frozen else pal["edge"], b, BUBBLE_R, 3)
    if orb is not None:
        draw_pinpoint(win, orb, 1 if pulse is None else pulse)
    if not present:
        text_center(win, small, "face not detected", sh - 40, pal["text"])
    for i, txt in enumerate(lines):
        text_center(win, font, txt, lines_y + i * 34, pal["text"])
    if status:
        win.blit(small.render(status, True, pal["text"]), (12, sh - 30))


def row_of(target_y, sh):
    return "top" if target_y < sh / 3 else ("middle" if target_y < 2 * sh / 3 else "bottom")


def describe_cal(c):
    return (f"{spec_name(c['spec'])}, LOPO mean {c['cv_px']:.1f} px, "
            f"in-sample mean {c['fit_px']:.1f} px")


def describe_candidates(c):
    return "LOPO mean pixel errors: " + ", ".join(
        f"{spec_name(spec)} {error:.1f}" for spec, error in sorted(c["all_cv"].items(), key=lambda p: p[1]))


def summarize(rows, other_acc, cal_clicks, cal, recenters, refine_n, lum, sh, target_r=TARGET_R):
    print("\n==== gaze_follower_test summary ====")
    rest_acc = sum(r["accidental_triggers"] for r in rows)
    print(f"accidental triggers: {rest_acc + other_acc} (rest periods {rest_acc}, other {other_acc}; "
          f"{cal_clicks} click(s) during verification / recenter not counted)")
    if lum[1]:
        print(f"camera frame luminance (0-255): {lum[0] / lum[1]:.0f}")
    if cal:
        print(f"calibration {describe_cal(cal)}\nrecenters this run: {recenters}; refinement trials in use at end: {refine_n}")
    if not rows:
        print("no completed trials")
        return
    n = len(rows)
    print("by target row (failures first):")
    for name in ("bottom", "middle", "top"):
        rr = [r for r in rows if row_of(r["target_y"], sh) == name]
        if rr:
            dy = [r["gaze_y"] - r["target_y"] for r in rr if r["gaze_y"] != ""]
            ee = [r["error_px"] for r in rr if r["error_px"] != ""]
            print(f"  {name:6s}: {len(rr)} trials, {sum(r['timeout'] for r in rr)} timeouts, "
                  f"{sum(not r['magnetic_hit'] for r in rr)} misses, median error "
                  f"{(f'{np.median(ee):.0f}' if ee else 'n/a')} px, median dy {(f'{np.median(dy):+.0f}' if dy else 'n/a')} px")
    half = n // 2
    if half >= 3:
        hit = lambda rr: sum(r["magnetic_hit"] for r in rr) / len(rr)
        med = lambda rr: np.median([r["error_px"] for r in rr if r["error_px"] != ""] or [float("nan")])
        print(f"first vs second half (refinement effect, confounded with practice): magnetic {hit(rows[:half]):.0%} -> "
              f"{hit(rows[half:]):.0%}, median error {med(rows[:half]):.0f} -> {med(rows[half:]):.0f} px")
    errs = [r["error_px"] for r in rows if r["error_px"] != ""]
    rts = [r["response_time_ms"] for r in rows if not r["timeout"]]
    cur = dict(median_error_px=float(np.median(errs)) if errs else float("nan"),
               strict_hit=sum(r["strict_hit"] for r in rows) / n, magnetic_hit=sum(r["magnetic_hit"] for r in rows) / n,
               median_rt_ms=float(np.median(rts)) if rts else float("nan"), timeouts=sum(r["timeout"] for r in rows))
    print(f"\nside-by-side ({n} trials; single runs of n<=20 -> wide CIs)")
    if target_r != TARGET_R:
        print(f"  NOTE: --target-r {target_r}: strict/magnetic hit rates are not comparable with the benchmarks "
              f"(which used r={TARGET_R}); median error is")
    for b in BENCHMARKS:
        print(f"  {b['name']} = {b['run']}")
    print(f"  {'metric':24s} {'v1':>8s} {'v2 ref':>8s} {'MGazeNet':>9s} {'vs v1':>7s}")
    for name, key, f, fd in [("median error (px)", "median_error_px", "{:.0f}", "{:+.0f}"),
                             (f"strict hit (<={target_r}px)", "strict_hit", "{:.0%}", "{:+.0%}"),
                             (f"magnetic hit (<={target_r + MAGNET_PX}px)", "magnetic_hit", "{:.0%}", "{:+.0%}"),
                             ("median response (ms)", "median_rt_ms", "{:.0f}", "{:+.0f}"),
                             ("timeouts", "timeouts", "{:d}", "{:+d}")]:
        c = cur[key]
        bad = isinstance(c, float) and math.isnan(c)
        print(f"  {name:24s} {f.format(BENCHMARKS[0][key]):>8s} {f.format(BENCHMARKS[1][key]):>8s} "
              f"{('n/a' if bad else f.format(c)):>9s} {('' if bad else fd.format(c - BENCHMARKS[0][key])):>7s}")
    pre = [r["preonset_error_px"] for r in rows if r["preonset_error_px"] != ""]
    raw = [r["raw_error_px"] for r in rows if r["raw_error_px"] != ""]
    if pre:
        print(f"  (300 ms before freeze: filtered median {np.median(pre):.0f} px, unfiltered median {np.median(raw):.0f} px)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--recal", dest="use_saved", action="store_false", help=argparse.SUPPRESS)
    start.add_argument("--use-saved", action="store_true", help="reuse a compatible saved 13-dot calibration")
    updates = ap.add_mutually_exclusive_group()
    updates.add_argument("--no-refine", dest="refine", action="store_false", help=argparse.SUPPRESS)
    updates.add_argument("--refine", action="store_true", help="enable in-session refinement from confirmed hits")
    ap.set_defaults(use_saved=False, refine=False)
    ap.add_argument("--cal", choices=["pop", "pop13"], help=argparse.SUPPRESS)
    ap.add_argument("--bg", choices=list(PALETTES), default="white", help="background for calibration and trials")
    ap.add_argument("--min-cutoff", type=float, default=MIN_CUTOFF, help="OneEuro min cutoff (Hz)")
    ap.add_argument("--beta", type=float, default=BETA, help="OneEuro speed coefficient (per px/s)")
    ap.add_argument("--no-fixlock", action="store_true", help="disable the velocity-gated fixation lock")
    ap.add_argument("--head-vernier", action="store_true", help="start with the head-pose fine-aim offset on "
                    "(h toggles it live)")
    ap.add_argument("--target-r", type=int, default=TARGET_R, help=f"target radius px (default {TARGET_R}, as all "
                    "earlier runs; magnetic hit = r + 50)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--trigger-config", type=Path, default=TRIGGER_CONFIG_PATH)
    args = ap.parse_args()
    if args.trials < 1:
        sys.exit("--trials must be >= 1")
    if not 10 <= args.target_r <= 300:
        sys.exit("--target-r must be in [10, 300] px")
    R = args.target_r

    tcfg = load_trigger_config(args.trigger_config)  # read-only
    tj, tg, deb = float(tcfg["jawOpen"]), float(tcfg["lip_gap"]), int(tcfg["debounce_ms"])
    print(f"trigger (validated, read-only): jawOpen>={tj}, lip_gap>={tg}, hold>={deb}ms, cooldown 600ms + release, "
          f"{Path(POP_SOUND).name}; bubble freezes while jawOpen>{FREEZE_JAW}")
    ensure_model()
    live = MouthAndGaze(tj, tg, deb, fixlock=not args.no_fixlock, head_vernier=args.head_vernier)
    gf = build_follower(args.camera, live, args.min_cutoff, args.beta)
    pal, filt_name = PALETTES[args.bg], f"oneeuro-{args.min_cutoff:g}-{args.beta:g}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    pygame.init()
    cfg_size = tuple(int(v) for v in gf.config.screen_size)
    win = pygame.display.set_mode(cfg_size, pygame.FULLSCREEN)
    pygame.display.set_caption("gaze_follower_test")
    sw, sh = win.get_size()
    if (sw, sh) != cfg_size:
        print(f"WARNING: window is {sw}x{sh} but GazeFollower assumed {cfg_size[0]}x{cfg_size[1]}; using the window size")
    sync_screen_size(gf, (sw, sh))
    centre = (sw // 2, sh // 2)
    targets = make_targets(args.trials, sw, sh, random.Random(args.seed))
    dots = [(int(fx * sw), int(fy * sh)) for fx, fy in CAL_POINTS]
    font, small = pygame.font.SysFont(None, 34), pygame.font.SysFont(None, 26)

    cal, why = load_calibration(CALIB_PATH, gf) if args.use_saved else (None, "fresh by default")
    if cal is not None:
        live.offset = np.asarray(cal["offset"], float)
        print(f"loaded {CALIB_PATH.relative_to(ROOT)}: {describe_cal(cal)}; offset "
              f"({live.offset[0]:+.0f}, {live.offset[1]:+.0f}) px -- press c to recenter if you moved")
    else:
        print(f"starting 13-dot mouth-pop calibration ({why})")
    print(f"gaze: GazeFollower MGazeNet + CV-selected calibrator, refinement {'on' if args.refine else 'off'}, "
          f"{filt_name}; camera {CAM_W}x{CAM_H}; window {sw}x{sh} px; bg {args.bg}")
    print(f"cursor: fixation lock {'off' if args.no_fixlock else f'on ({FIX_SPEED_PX_S:g} px/s for >{FIX_DWELL_MS} ms, damp <={FIX_JITTER_PX:g} px, release >{FIX_RELEASE_PX:g} px)'}; "
          f"head vernier {'on' if args.head_vernier else 'off'} (h toggles; +-{HEAD_CLAMP_PX:g} px, reset on >"
          f"{HEAD_RESET_PX:g} px gaze moves); target radius {R} px (magnetic {R + MAGNET_PX})")

    rows, other_acc, cal_clicks, recenters, csv_f, writer = [], 0, 0, 0, None, None
    gf.add_subscriber(live.on_gaze)
    camera_on = False
    try:
        gf.camera.start_sampling()  # SAMPLING from the start: calibration reads features from the same stream
        camera_on = True
        clock = pygame.time.Clock()
        phase = "wait" if cal is not None else "points"
        phase_t0, trial, rest_acc, feedback, note = now_ms(), 0, 0, None, ""
        point_idx, point_feats = 0, []
        locked_dots = set()
        resume, vernier_toggles = None, 0
        while True:
            quit_ = recenter_key = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE)):
                    quit_ = True
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_c:
                    recenter_key = True
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_h:
                    with live.lock:
                        live.vernier.enabled = not live.vernier.enabled
                        live.vernier.reset()
                        vern_on = live.vernier.enabled
                    vernier_toggles += 1
                    print(f"head vernier {'ON' if vern_on else 'OFF'} (phase {phase}, trial {trial + 1})")
            if quit_:
                break
            t = now_ms()
            with live.lock:
                events = list(live.onsets)
                live.onsets.clear()
                bubble = None if live.bubble is None else live.bubble.copy()
                frozen_since, present = live.frozen_since, live.present
                hist = list(live.hist)
            frozen = frozen_since is not None
            if bubble is not None:
                bubble = np.clip(bubble, [0, 0], [sw - 1, sh - 1])
            if recenter_key and cal is not None and phase in ("wait", "look", "rest", "target", "done"):
                if phase == "target":
                    print(f"trial {trial + 1:2d}: interrupted by recenter, will be repeated")
                resume = {"target": "rest", "wait": "look"}.get(phase, phase)
                phase, phase_t0 = "recenter", t
            el = t - phase_t0
            near, tgt, lines, lines_y, show_dots, orb, show_bubble, pulse = False, None, [], 30, False, None, True, None

            if phase == "points":
                orb, show_bubble, pulse = dots[point_idx], False, el / FOCUS_MS
                lines = [note or f"Fixate the pinpoint, then pop your mouth ({point_idx + 1}/{len(dots)})"]
                if events:
                    freeze = events[0][2]
                    xs = []
                    if freeze is not None and freeze - WINDOW_MS >= phase_t0 + FOCUS_MS:
                        with live.lock:
                            xs, _ = window_samples(live.feats, phase_t0, freeze, WINDOW_MS, FOCUS_MS)
                    if len(xs) >= MIN_POINT_SAMPLES:
                        point_feats.append(xs)
                        point_idx += 1
                        if point_idx == len(dots):
                            cal = fit_points(gf, point_feats, CAL_POINTS)
                        else:
                            orb, pulse, phase_t0 = dots[point_idx], 0, t
                        note, lines = "", []
                    else:
                        note = "Hold your gaze on the pinpoint a moment longer, then pop again"
            elif phase == "wait":
                cal_clicks += len(events)
                lines, lines_y = ["Starting gaze stream..."], sh // 2 - 17
                if bubble is not None:
                    locked_dots.clear()
                    last_checked = t - 1
                    phase, phase_t0 = "look", t
            elif phase == "look":
                cal_clicks += len(events)
                show_dots = True
                for ts, _, pos in hist:
                    if last_checked < ts <= phase_t0 + LOOK_MS:
                        last_checked = ts
                        locked_dots.update(i for i, d in enumerate(dots) if math.dist(pos, d) <= LOCK_RADIUS)
                remaining = max(0, math.ceil((LOOK_MS - el) / 1000))
                lines_y = 8
                lines = [f"Look at the dots   {len(locked_dots)}/{len(dots)} locked   {remaining} s"]
                if el >= LOOK_MS:
                    print(f"Verification: {len(locked_dots)}/{len(dots)} zones locked within {LOCK_RADIUS} px")
                    phase, phase_t0 = "rest", t
            elif phase == "recenter":
                cal_clicks += len(events)
                orb, show_bubble = centre, False
                lines, lines_y = ["Recenter: look at the orb (1 s)"], sh // 2 - 90
                if el >= RECENTER_MS:
                    new, n = recenter_offset(hist, phase_t0, centre, live.offset)
                    if new is None:
                        print(f"recenter: only {n} gaze samples, offset unchanged")
                    else:
                        print(f"recenter: offset ({live.offset[0]:+.0f}, {live.offset[1]:+.0f}) -> "
                              f"({new[0]:+.0f}, {new[1]:+.0f}) px from {n} samples")
                        with live.lock:
                            live.offset = new
                            live.hist.clear()
                            if live.fixlock:
                                live.fixlock.reset()
                            live.vernier.reset()
                            live.bubble = None
                        recenters += 1
                        save_calibration(CALIB_PATH, gf, cal, new)
                    phase, phase_t0 = ("wait" if resume == "look" else resume), t
            elif phase == "rest":
                rest_acc += len(events)
                lines = [f"Trial {trial + 1}/{args.trials}"]
                if el >= REST_MS:
                    phase, phase_t0 = "target", t
            elif phase == "target":
                tgt = targets[trial]
                if bubble is not None:
                    near = math.hypot(bubble[0] - tgt[0], bubble[1] - tgt[1]) <= R + MAGNET_PX
                fired = bool(events)
                if fired or el >= TIMEOUT_MS:
                    onset, fired_at, froz = events[0] if fired else (None, t, None)
                    ref = (froz or onset) if fired else t  # window ends where mouth movement began
                    win_pts = [(c, f) for ts_, c, f in hist if ref - PRE_ONSET_MS <= ts_ < ref]
                    sel = bubble
                    pre = np.mean([f for _, f in win_pts], 0) if win_pts else None
                    raw_m = np.mean([c for c, _ in win_pts], 0) if win_pts else None
                    err = lambda p: round(float(math.hypot(*(np.asarray(p) - tgt))), 1) if p is not None else ""
                    xy = lambda p, i: round(float(p[i])) if p is not None else ""
                    e = err(sel)
                    with live.lock:
                        vern_now, head_off = live.vernier.enabled, live.vernier.off.copy()
                    mag = bool(fired and e != "" and e <= R + MAGNET_PX)
                    strict = bool(fired and e != "" and e <= R)
                    refined = False
                    if mag and args.refine and e <= REFINE_MAX_ERR:  # anchored refinement
                        with live.lock:
                            feats = list(live.feats)
                        xs, _ = window_samples(feats, phase_t0, ref, WINDOW_MS, REFINE_SETTLE_MS)
                        if len(xs) >= MIN_POINT_SAMPLES:
                            refined = gf.calibration.add_refinement(xs, (np.asarray(tgt, float) - live.offset) / (sw, sh))
                    row = dict(run_id=run_id, trial=trial + 1,
                               cal_method=("loaded-" if cal.get("loaded") else "") + cal["method"],
                               cal_model=spec_name(cal["spec"]), cal_cv_px=round(cal["cv_px"], 1),
                               cal_fit_px=round(cal["fit_px"], 1), bg=args.bg, gaze_filter=filt_name,
                               refine_trials=len(gf.calibration.refine), refined=refined,
                               recenters=recenters, offset_x=round(float(live.offset[0])),
                               offset_y=round(float(live.offset[1])),
                               target_x=tgt[0], target_y=tgt[1], gaze_x=xy(sel, 0), gaze_y=xy(sel, 1), error_px=e,
                               preonset_gaze_x=xy(pre, 0), preonset_gaze_y=xy(pre, 1), preonset_error_px=err(pre),
                               raw_gaze_x=xy(raw_m, 0), raw_gaze_y=xy(raw_m, 1), raw_error_px=err(raw_m),
                               magnetic_hit=mag, strict_hit=strict,
                               response_time_ms=(fired_at - phase_t0) if fired else TIMEOUT_MS, timeout=not fired,
                               accidental_triggers=rest_acc,
                               freeze_ms=(fired_at - froz) if (fired and froz) else "",
                               gaze_window_samples=len(win_pts), target_r=R, fixlock=not args.no_fixlock,
                               head_vernier=vern_now, head_dx=round(float(head_off[0])),
                               head_dy=round(float(head_off[1])), cal_trimmed=False,
                               bank_sessions=0)
                    rows.append(row)
                    writer.writerow(row)
                    csv_f.flush()
                    print(f"trial {trial + 1:2d}: target {tgt} gaze ({row['gaze_x']},{row['gaze_y']}) err {e} px -> "
                          f"{'HIT ' if mag else 'MISS'}{' strict' if strict else ''}{' (timeout)' if not fired else ''} "
                          f"{row['response_time_ms']} ms{'  [refined]' if refined else ''}")
                    other_acc += max(0, len(events) - 1)
                    feedback, rest_acc, trial, tgt = (tgt, mag, t), 0, trial + 1, None
                    phase, phase_t0 = ("rest", t) if trial < args.trials else ("done", t)
            else:
                other_acc += len(events)
                lines, lines_y = [f"Done: {sum(r['magnetic_hit'] for r in rows)}/{len(rows)} hits - press q to quit"], sh // 2

            if cal is not None and phase == "points":
                print(f"Calibration: {describe_cal(cal)}\n{describe_candidates(cal)}")
                save_calibration(CALIB_PATH, gf, cal, live.offset)
                anchors, labels = gf.calibration.anchors
                groups = np.repeat(np.arange(len(dots)), cal["per_point"])
                np.savez_compressed(ROOT / "logs" / f"gf_cal_{run_id}.npz", features=np.vstack(point_feats),
                                    labels=labels[groups], group=groups, anchors=anchors, per_point=cal["per_point"])
                print(f"Saved {CALIB_PATH.relative_to(ROOT)}")
                phase, phase_t0 = "wait", t
            if phase in ("rest", "target") and csv_f is None:
                csv_f = open_csv(METRICS_CSV, CSV_FIELDS)
                writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
            with live.lock:
                live.tag = (phase, trial + 1 if phase in ("rest", "target") else 0,
                            tgt if tgt is not None else (None if orb is None else (int(orb[0]), int(orb[1]))))

            status = ""
            if phase in ("look", "rest", "target", "done", "wait"):
                with live.lock:
                    status = (f"head vernier {'ON' if live.vernier.enabled else 'off'} (h)  |  fixation lock "
                              f"{'off' if live.fixlock is None else ('LOCKED' if live.locked else 'on')}")
            draw_scene(win, pal, font, small, t, bubble=bubble if show_bubble else None, frozen=frozen, tgt=tgt,
                       near=near, feedback=feedback, dots=dots if show_dots else (), lines=lines, lines_y=lines_y,
                       present=present, orb=orb, pulse=pulse, target_r=R, status=status, locked_dots=locked_dots)
            pygame.display.flip()
            clock.tick(60)
    finally:
        gf.remove_subscriber(live.on_gaze)
        if camera_on:
            gf.camera.stop_sampling()
        gf.release()
        live.landmarker.close()
        pygame.quit()
        if csv_f:
            csv_f.close()
        if live.trace:
            write_trace(TRACE_CSV, run_id, live.trace)
            print(f"per-frame trace: {len(live.trace)} rows -> {TRACE_CSV.relative_to(ROOT)} (run_id {run_id})")
        summarize(rows, other_acc, cal_clicks, cal, recenters, len(gf.calibration.refine), live.lum, sh, R)
        if rows:
            on = [r for r in rows if r["head_vernier"]]
            print(f"head vernier: on for {len(on)}/{len(rows)} committed trials, {vernier_toggles} toggles"
                  + (f"; median error vernier on {np.median([r['error_px'] for r in on if r['error_px'] != '']):.0f}"
                     f" px vs off {np.median([r['error_px'] for r in rows if not r['head_vernier'] and r['error_px'] != ''] or [float('nan')]):.0f} px"
                     if on and any(r["error_px"] != "" for r in on) else ""))
        if rows:
            print(f"trials logged to {METRICS_CSV.relative_to(ROOT)} (run_id {run_id})")


if __name__ == "__main__":
    main()
