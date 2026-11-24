"""Gaze building blocks. Coordinates are screen pixels unless explicitly normalized.

No camera or event listeners are started on import. Callers supply monotonic seconds,
real gaze observations and physical click / accessibility target events.
"""
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class OneEuroFilter:
    """Casiez et al. 1€ filter. Sample timestamps use monotonic seconds.

    Reference: github.com/casiez/OneEuroFilter/tree/main/python.
    """

    def __init__(self, mincutoff=1., beta=0., dcutoff=1.):
        if not all(math.isfinite(v) for v in (mincutoff, beta, dcutoff)) or min(mincutoff, dcutoff) <= 0 or beta < 0:
            raise ValueError("Invalid One Euro cutoff or beta")
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.reset()

    def reset(self):
        self.timestamp = self.filtered = None
        self.derivative = 0.

    @staticmethod
    def alpha(dt, cutoff):
        tau = 1 / (2 * math.pi * cutoff)
        return dt / (dt + tau)

    def update(self, value, timestamp):
        if not math.isfinite(value) or not math.isfinite(timestamp):
            raise ValueError("One Euro samples must be finite")
        if self.timestamp is None:
            self.timestamp, self.filtered = timestamp, value
        elif timestamp > self.timestamp:
            dt = timestamp - self.timestamp
            speed = (value - self.filtered) / dt
            self.derivative += self.alpha(dt, self.dcutoff) * (speed - self.derivative)
            cutoff = self.mincutoff + self.beta * abs(self.derivative)
            self.filtered += self.alpha(dt, cutoff) * (value - self.filtered)
            self.timestamp = timestamp
        return self.filtered


# MediaPipe canonical_face_model.obj vertices in centimeters (Apache 2.0).
# OpenCV uses +y down and +z away from the camera.
RIGID_FACE_IDS = (1, 33, 263, 133, 362, 6, 168, 10, 127, 356)
FACE_IDS = (*RIGID_FACE_IDS, 152, 61, 291)
FACE_POINTS = ((0., 1.126865, -7.475604), (-4.445859, -2.663991, -3.173422),
               (4.445859, -2.663991, -3.173422), (-1.856432, -2.585245, -3.757904),
               (1.856432, -2.585245, -3.757904), (0., -2.473255, -5.788627),
               (0., -3.271027, -5.236015), (0., -8.261778, -4.481535),
               (-7.743095, -2.364999, 2.005167), (7.743095, -2.364999, 2.005167),
               (0., 9.403378, -4.264492), (-2.456206, 4.342621, -4.283884),
               (2.456206, 4.342621, -4.283884))


@dataclass(frozen=True)
class HeadPose:
    angles: tuple
    origin: tuple
    direction: tuple
    intersection: tuple
    reprojection_px: float


def solve_head_pose(landmarks, width, height):
    """Intersect the facial forward ray with the display plane at the camera."""
    import cv2
    import numpy as np

    points = np.asarray(landmarks)[list(FACE_IDS), :2].astype(float) * (width, height)
    if not np.isfinite(points).all():
        return None
    model = np.asarray(FACE_POINTS)
    # Approximate intrinsics for a webcam with no camera calibration.
    intrinsics = np.array(((width, 0, width / 2), (0, width, height / 2), (0, 0, 1.)))
    rigid = len(RIGID_FACE_IDS)
    ok, rvec, tvec = cv2.solvePnP(model[:rigid], points[:rigid], intrinsics, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    projected = cv2.projectPoints(model, rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - points, axis=1)
    inliers = errors <= max(2 * np.median(errors[:rigid]), max(errors[:rigid]))
    # Mouth corners and chin refine the fit when consistent with the rigid
    # part of the face. Speaking or swallowing can move those three anchors.
    model_in, points_in = model[inliers], points[inliers]
    rvec, tvec = cv2.solvePnPRefineLM(model_in, points_in, intrinsics, None, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    origin = rotation @ model[0] + tvec.ravel()
    direction = -rotation[:, 2]
    if origin[2] <= 0 or direction[2] >= 0 or not np.isfinite(origin).all():
        return None
    x, y = (origin - origin[2] * direction / direction[2])[:2]
    if not np.isfinite((x, y)).all():
        return None
    pitch, yaw, roll = cv2.RQDecomp3x3(rotation)[0]
    reprojected = cv2.projectPoints(model_in, rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
    error = np.sqrt(np.mean(np.sum((reprojected - points_in) ** 2, axis=1)))
    return HeadPose((yaw, pitch, roll), tuple(origin), tuple(direction), (-x, y), float(error))


def iris_fraction(landmarks, aspect=1.):
    """Horizontal iris location within each eye, used only for display routing."""
    import numpy as np

    fractions = []
    for first, last, iris in ((33, 133, 468), (362, 263, 473)):
        a, b, i = np.asarray(landmarks)[[first, last, iris], :2] * (aspect, 1.)
        axis = b - a
        length = axis @ axis
        if length > 0 and np.isfinite(i).all():
            fractions.append(float((i - a) @ axis / length))
    return 1 - sum(fractions) / len(fractions) if fractions else None


def iris_vertical(landmarks, aspect=1.):
    """How far the iris centres sit below (+) or above (-) the line through the eye corners, in
    eye-corner widths, averaged over both eyes. Corner width does not change with gaze; eyelid
    opening does (the upper lid follows the eye down), so it is not used for scale.
    `landmarks` are FaceLandmarker's normalised points; `aspect` is image width / height."""
    import numpy as np

    offsets = []
    for first, last, iris in ((33, 133, 468), (362, 263, 473)):
        a, b, i = np.asarray(landmarks, float)[[first, last, iris], :2] * (aspect, 1.)
        axis = b - a
        width = float(np.hypot(*axis))
        if width > 0 and np.isfinite(i).all():
            # (-dy, dx) points down the image for these left-to-right corner pairs, at any head roll.
            offsets.append(float((i - (a + b) / 2) @ (-axis[1], axis[0])) / width ** 2)
    return sum(offsets) / len(offsets) if offsets else None


def eye_opening(landmarks, aspect=1.):
    """Gap between the upper and lower eyelid in eye-corner widths, averaged over both eyes.
    A closing eye reads as gaze at the bottom edge, to MGazeNet and to the iris landmarks alike,
    so frames below a share of the user's open-eye value are dropped as blinks."""
    import numpy as np

    gaps = []
    for first, last, upper, lower in ((33, 133, 159, 145), (362, 263, 386, 374)):
        a, b, up, low = np.asarray(landmarks, float)[[first, last, upper, lower], :2] * (aspect, 1.)
        axis = b - a
        width = float(np.hypot(*axis))
        if width > 0 and np.isfinite((up, low)).all():
            gaps.append(abs(float((low - up) @ (-axis[1], axis[0]))) / width ** 2)
    return sum(gaps) / len(gaps) if gaps else None


# A frame counts as a blink below this share of the smallest eyelid opening among the calibration
# fixations (the bottom row, where the lids are lowest), and below the fixed value for a
# calibration that did not record openings. Both come from eval_gaze_scroll.py on data/session_*.
EYE_OPEN_SHARE = .7
EYE_OPEN_DEFAULT = .22

# One Euro settings for calibrated eye gaze in screen pixels (min cutoff Hz, beta per px/s,
# derivative cutoff Hz), chosen with eval_gaze_scroll.py on the recorded fixations.
EYE_FILTER = (.2, .012, .5)


class VerticalFusion:
    """Screen height (0 top, 1 bottom) from MGazeNet's raw gaze plus the iris offset, and head
    pitch when a user's own calibration shows that it helps.

    Fitted on the calibration anchors by ridge regression. `fit` returns None unless the fused
    model beats the existing calibration's leave-one-anchor-out error, so a calibration that
    gains nothing keeps its current behaviour."""
    FLOORS = (.5, .5, .005, 2.)  # cm, cm, eye widths, degrees: spread below this is not signal
    ALPHA = 1.
    GAIN = .9                    # a candidate must cut the error it is compared with by 10 %

    def __init__(self, mean, scale, weights, intercept):
        self.mean, self.scale, self.weights, self.intercept = mean, scale, weights, float(intercept)

    @staticmethod
    def features(raw, vertical, columns):
        import numpy as np

        return np.column_stack([np.asarray(raw, float).reshape(-1, 2),
                                np.atleast_2d(np.asarray(vertical, float))[:, :2]])[:, :columns]

    @classmethod
    def solve(cls, x, y):
        import numpy as np

        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), cls.FLOORS[:x.shape[1]])
        z = (x - mean) / scale
        weights = np.linalg.solve(z.T @ z + cls.ALPHA * np.eye(x.shape[1]), z.T @ (y - y.mean()))
        return cls(mean, scale, weights, y.mean())

    @classmethod
    def held_out_error(cls, x, y):
        import numpy as np

        keep = ~np.eye(len(y), dtype=bool)
        return float(np.mean([abs(cls.solve(x[rows], y[rows]).value(x[i]) - y[i]) for i, rows in enumerate(keep)]))

    @classmethod
    def fit(cls, raw, vertical, targets_y, baseline_error=math.inf):
        """`raw`: (n, 2) MGazeNet gaze; `vertical`: (n, 2) iris offset and head pitch in degrees;
        `baseline_error`: the current calibration's leave-one-out error in screen heights."""
        import numpy as np

        y = np.asarray(targets_y, float)
        full = cls.features(raw, vertical, 4)
        if len(y) < 6 or full.shape[0] != len(y) or not np.isfinite(full).all():
            return None
        errors = {columns: cls.held_out_error(full[:, :columns], y) for columns in (3, 4)}
        columns = 4 if errors[4] <= cls.GAIN * errors[3] else 3
        if errors[columns] > cls.GAIN * baseline_error:
            return None
        model = cls.solve(full[:, :columns], y)
        model.error = errors[columns]
        return model

    def value(self, row):
        return float(self.intercept + ((row - self.mean) / self.scale) @ self.weights)

    @property
    def uses_pitch(self):
        return len(self.weights) == 4

    def predict(self, raw, vertical):
        return self.value(self.features(raw, vertical, len(self.weights))[0])


class ScrollPacer:
    """Turns gaze height inside a reading area (0 top, 1 bottom) into pixel scroll steps.

    Scrolls down while gaze is at or below DOWN after DWELL and up while it is at or above UP
    after UP_DWELL. It stops on the first sample inside QUIET. In the bands between, it
    keeps doing what it was doing, so gaze resting on a boundary does not chatter.
    `update` returns the whole pixels to post now: positive moves the content down the screen
    (toward the start of the document), negative moves it up, matching Quartz wheel deltas."""
    DOWN, UP = .75, .12
    QUIET = (.18, .72)
    FULL_DOWN, FULL_UP = .95, .02     # full speed from here to the edge
    DWELL = .38
    UP_DWELL = .22
    SLOW, FAST = 40., 900.            # pixels per second at the zone boundary and at full depth
    RAMP = .12                        # seconds: time constant of speed changes while scrolling
    RATE = 60.                        # most events per second

    def __init__(self, dwell=None):
        self.dwell = self.DWELL if dwell is None else dwell
        self.reset()

    @staticmethod
    def below_calibrated_screen(iris, iris_limit):
        return iris_limit is not None and (iris is None or not math.isfinite(iris) or iris > iris_limit)

    def reset(self):
        self.direction = 0            # +1 reading on (content moves up), -1 back toward the top
        self.candidate, self.since = 0, None
        self.speed = self.carry = 0.
        self.last = None

    def zone(self, y):
        if y >= self.DOWN:
            return 1
        if y <= self.UP:
            return -1
        return 0 if self.QUIET[0] < y < self.QUIET[1] else None

    def depth(self, y):
        if self.direction > 0:
            depth = (y - self.DOWN) / (self.FULL_DOWN - self.DOWN)
        else:
            depth = (self.UP - y) / (self.UP - self.FULL_UP)
        return max(0., min(1., depth))

    def update(self, y, now, below_screen=False):
        zone = self.zone(y)
        if below_screen and zone != -1 and (zone == 1 or self.direction == 1 or self.candidate == 1):
            self.reset()
            return 0
        if zone == 0 or (zone and self.direction == -zone):
            self.reset()
            if zone == 0:
                return 0
        if zone and self.direction != zone:
            if self.candidate != zone:
                self.candidate, self.since = zone, now
            if now - self.since < (self.dwell if zone == 1 else self.UP_DWELL):
                return 0
            self.direction, self.last = zone, now
            self.speed = self.carry = 0.
        if not self.direction:
            if zone is None:  # the confirmation has to be unbroken time in the zone
                self.candidate, self.since = 0, None
            return 0
        dt = now - self.last
        if dt < 1 / self.RATE:
            return 0
        self.last = now
        dt = min(dt, 1 / 15)
        target = self.SLOW + (self.FAST - self.SLOW) * self.depth(y) ** 2
        self.speed += (target - self.speed) * (1 - math.exp(-dt / self.RAMP))
        self.carry += self.speed * dt
        pixels = int(self.carry)
        self.carry -= pixels
        return -self.direction * pixels


class FixationLock:
    """I-VT. Release on high velocity and acquire after a low-velocity dwell."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.timestamp = self.started = self.anchor = None
        self.recent = deque(maxlen=3)
        self.fast = 0
        self.samples = []

    def update(self, raw, filtered, timestamp, speed_threshold, dwell=.15):
        if self.timestamp is not None and timestamp <= self.timestamp:
            return self.anchor
        if self.timestamp is not None and timestamp - self.timestamp > .25:
            self.reset()
        self.recent.append((timestamp, raw))
        self.timestamp = timestamp
        velocity = fixation_speed(self.recent)
        if velocity > speed_threshold:
            self.fast += 1
            if self.fast >= 2 or self.anchor is None:
                self.started, self.anchor = None, None
                self.samples.clear()
        elif self.anchor is None:
            self.fast = 0
            if self.started is None:
                self.started = timestamp
            self.samples.append(filtered)
            if timestamp - self.started >= dwell:
                self.anchor = tuple(sum(axis) / len(axis) for axis in zip(*self.samples))
                self.samples.clear()
        else:
            self.fast = 0
        return self.anchor


def fixation_speed(samples):
    if len(samples) < 3:
        return 0.
    (start, a), (_, _), (end, b) = samples
    return math.dist(a, b) / (end - start) if 0 < end - start <= .25 else 0.


def fixation_threshold(sequences):
    import numpy as np

    speeds = [fixation_speed(samples[i:i + 3]) for samples in sequences
              for i in range(len(samples) - 2)]
    if not speeds:
        raise ValueError("At least three neutral pose samples are required")
    return float(np.percentile(speeds, 95) * 2)


def hand_measurements(result, aspect=1.):
    """Palm-normalized 3D pinch and camera-relative wrist positions per hand."""
    import numpy as np

    measures = {}
    for image, world, handedness in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
        if not handedness or len(image) != 21 or len(world) != 21:
            continue
        xyz = np.array([(p.x, p.y, p.z) for p in world])
        palm = math.dist(xyz[0], xyz[9])
        if not math.isfinite(palm) or palm <= 0:
            continue
        # MediaPipe image z is relative to each wrist. Estimate camera-relative
        # range from the metric and apparent palm spans across the finger bases.
        image_xy = np.array([(p.x * aspect, p.y) for p in image])
        bases = [5, 9, 13, 17]
        apparent = np.linalg.norm(image_xy[bases] - image_xy[0], axis=1)
        metric = np.linalg.norm(xyz[bases, :2] - xyz[0, :2], axis=1)
        visible = apparent > 0
        if not visible.any():
            continue
        depth = float(np.median(metric[visible] / apparent[visible]))
        wrist = image[0]
        pinch = math.dist(xyz[4], xyz[8]) / palm
        position = (1 - wrist.x, wrist.y, depth)
        if math.isfinite(pinch) and all(math.isfinite(v) for v in position):
            measures[handedness[0].category_name] = (pinch, position)
    return measures


def make_hand_landmarker(path):
    import mediapipe as mp

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def wrist_velocity(samples, aspect):
    if len(samples) != 3:
        return (0., 0., 0.)
    (first, start), (_, _), (last, end) = samples
    if not 0 < end - start <= .25:
        return (0., 0., 0.)
    return tuple((b - a) * scale / (end - start)
                 for a, b, scale in zip(first, last, (aspect, 1., aspect)))


class HandGestures:
    """Per-hand hysteresis on real MediaPipe measurements, with one drag owner."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history, self.flicked = {}, set()
        self.owner = self.origin = self.previous_pair = None
        self.spread_open = False

    def update(self, hands, timestamp, aspect, profile):
        events = []
        pinch_on, pinch_off, flick_threshold = (profile[key] for key in ("pinch_on", "pinch_off", "flick_speed"))
        for key in self.history.keys() - hands.keys():
            del self.history[key]
            self.flicked.discard(key)

        if self.owner is not None:
            reading = hands.get(self.owner)
            if reading is None or reading[0] >= pinch_off:
                events.append(("release", None))
                self.owner = self.origin = None
            else:
                position = reading[1]
                delta = tuple(a - b for a, b in zip(position, self.origin))
                events.append(("pull", (position, delta)))

        velocities = {}
        for key, (_, position) in hands.items():
            samples = self.history.setdefault(key, deque(maxlen=3))
            if samples and timestamp - samples[-1][1] > .125:
                samples.clear()
            if not samples or timestamp > samples[-1][1]:
                samples.append((position, timestamp))
            velocities[key] = wrist_velocity(samples, aspect)
            if math.hypot(*velocities[key]) <= flick_threshold / 2:
                self.flicked.discard(key)

        pair = None
        was_spread = self.spread_open
        if len(hands) == 2:
            positions = [position for _, position in hands.values()]
            pair = math.dist(*(tuple(c * k for c, k in zip(p, (aspect, 1., aspect))) for p in positions))
        if pair is None or self.owner is not None:
            self.previous_pair, self.spread_open = None, False
            if was_spread:
                events.append(("spread_end", None))
        elif not events:
            if self.previous_pair is None:
                self.previous_pair = pair, timestamp
            initial, started = self.previous_pair
            if pair < initial or (not self.spread_open and timestamp - started > .25):
                self.previous_pair = pair, timestamp
            elif not self.spread_open and pair > initial and timestamp > started:
                if (pair - initial) / (timestamp - started) > flick_threshold / 2:
                    self.spread_open = True
            if self.spread_open:
                return [("spread", pair / initial)] if initial > 0 else []

        for key, (pinch, position) in hands.items():
            if self.owner is None and not events and pinch < pinch_on:
                self.owner, self.origin = key, position
                events.append(("pinch", position))
            elif self.owner is None and not events and key not in self.flicked:
                vx, vy, vz = velocities[key]
                if math.sqrt(vx * vx + vy * vy + vz * vz) > flick_threshold:
                    dominant = max(range(3), key=lambda i: abs((vx, vy, vz)[i]))
                    direction = ("left" if vx < 0 else "right", "up" if vy < 0 else "down", "click")[dominant]
                    self.flicked.add(key)
                    events.append(("flick", direction))
        return events


PROFILE_PATH = ROOT / "logs/spatial_profile.json"
PROFILE_FIELDS = ("neutral", "iris_neutral", "iris_margin", "reprojection_px",
                  "saccade_speed", "pinch_on", "pinch_off", "flick_speed", "noise_floor")


def load_spatial_profile(path=PROFILE_PATH):
    try:
        profile = json.loads(path.read_text())
        if profile.get("version") != 1:
            return None
        _point(profile["neutral"])
        values = [profile[key] for key in PROFILE_FIELDS[1:]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values):
            return None
        if not profile["pinch_on"] < profile["pinch_off"]:
            return None
        return profile
    except (FileNotFoundError, ValueError, TypeError, KeyError, AttributeError):
        return None


def save_spatial_profile(profile, path=PROFILE_PATH):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        try:
            json.dump(dict(profile, version=1), tmp, allow_nan=False, indent=2)
            tmp.write("\n")
            tmp.close()
            if load_spatial_profile(Path(tmp.name)) is None:
                raise ValueError("Invalid spatial calibration")
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


class SpatialOnboarding:
    """Timing and samples for a neutral hold, two hand motions, and room tone."""

    INSTRUCTIONS = (
        ("Center your head", "Face the middle of the screen and hold still."),
        ("Calibrate your hand", "Hold one hand open and still near your face."),
        ("Calibrate your hand", "Touch your thumb to your index finger and hold."),
        ("Calibrate your hand", "Open your hand again and flick it once."),
        ("Measure room noise", "Stay quiet while the microphone measures the room."),
    )
    DURATIONS = (4, 4, 4, 6, 2)

    def __init__(self, now):
        self.started = now
        self.pose, self.open, self.closed, self.flick = [], [], [], []
        self.history = {}

    def step(self, now):
        elapsed = now - self.started
        for idx, length in enumerate(self.DURATIONS):
            if elapsed < length:
                return idx, max(0., elapsed)
            elapsed -= length
        return len(self.DURATIONS), elapsed

    def sample(self, now, pose, iris, hands, aspect):
        step, elapsed = self.step(now)
        if elapsed < .75 and step != 3:
            self.history.clear()
            return
        if step == 0 and pose is not None and iris is not None:
            self.pose.append((now, pose, iris))
        if step not in (1, 2, 3) or len(hands) != 1:
            self.history.clear()
            return
        key, (pinch, position) = next(iter(hands.items()))
        if step == 3:
            if not self.open or not self.closed:
                return
            if not hasattr(self, "unpinch"):
                opened = sorted(value for value, _ in self.open)
                closed = sorted(self.closed)
                self.unpinch = (opened[len(opened) // 2] + closed[len(closed) // 2]) / 2
                self.history.clear()
            if pinch < self.unpinch:
                self.history.clear()
                return
        history = self.history.setdefault(key, deque(maxlen=3))
        history.append((position, now))
        if step == 1:
            self.open.append((pinch, math.hypot(*wrist_velocity(history, aspect))))
        elif step == 2:
            self.closed.append(pinch)
        else:
            self.flick.append(math.hypot(*wrist_velocity(history, aspect)))

    def finish(self, noise_floor):
        import numpy as np

        if min(len(self.pose), len(self.open), len(self.closed), len(self.flick)) < 10:
            raise ValueError("Keep your face and one hand in view for each calibration step.")
        times, poses, iris = zip(*self.pose)
        points = np.array([p.intersection for p in poses])
        open_pinch, rest = zip(*self.open)
        opened = np.percentile(open_pinch, 20)
        closed = np.percentile(self.closed, 80)
        rest_speed, flick_speed = np.percentile(rest, 95), max(self.flick)
        if closed >= opened or flick_speed <= rest_speed:
            raise ValueError("Keep your hand still during the open hold, then pinch and flick when prompted.")
        if noise_floor is None:
            raise ValueError("Calibration needs a half-second of microphone input.")
        iris_center = float(np.median(iris))
        return dict(neutral=np.median(points, axis=0).tolist(), iris_neutral=iris_center,
                    iris_margin=float(max(.015, 3 * np.percentile(abs(np.array(iris) - iris_center), 95))),
                    reprojection_px=float(max(p.reprojection_px for p in poses) * 2),
                    saccade_speed=fixation_threshold([list(zip(times, points))]),
                    pinch_on=float((2 * closed + opened) / 3), pinch_off=float((closed + 2 * opened) / 3),
                    flick_speed=float((rest_speed + flick_speed) / 2), noise_floor=float(noise_floor))


def _point(p):
    x, y = map(float, p)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('coordinates must be finite')
    return x, y


class GazeBackend(ABC):
    @abstractmethod
    def start(self):
        """Start capture; requires a real camera and user participation."""

    @abstractmethod
    def get_gaze(self):
        """Return latest screen pixel coordinates or None."""

    @abstractmethod
    def close(self):
        pass


class GazeFollowerBackend(GazeBackend):
    """Existing MGazeNet pipeline and saved Ridge/TPS calibration.

    trigger_config contains mouth thresholds, not the Ridge/TPS model; that model
    is loaded separately from the project's trusted local calibration pickle.
    """
    def __init__(self, screen_size, camera_index=0, config_path=ROOT / 'logs/trigger_config.json',
                 calibration_path=ROOT / 'models/gf_calibrator.pkl'):
        self.screen_size = _point(screen_size)
        if min(self.screen_size) <= 0:
            raise ValueError('screen size must be positive')
        self.camera_index, self.config_path, self.calibration_path = camera_index, Path(config_path), Path(calibration_path)
        self.follower = self.live = None

    def prepare(self, live):
        """Build the shared follower without starting a second camera stream."""
        if self.follower is not None:
            raise RuntimeError('backend already prepared')
        from handsfree.gaze_follower import build_follower, sync_screen_size
        self.follower = build_follower(self.camera_index, live)
        sync_screen_size(self.follower, self.screen_size)
        self.live = live
        return self.follower

    def start(self):
        if self.follower is not None and self.live is not None:
            self.follower.camera.start_sampling()
            return
        if self.follower is not None:
            raise RuntimeError('backend already started')
        from handsfree.gaze_follower import MouthAndGaze, build_follower, sync_screen_size, load_calibration
        cfg = json.loads(self.config_path.read_text())
        live = MouthAndGaze(cfg['jawOpen'], cfg['lip_gap'], cfg['debounce_ms'])
        gf = None
        try:
            gf = build_follower(self.camera_index, live)
            sync_screen_size(gf, self.screen_size)
            cal, reason = load_calibration(self.calibration_path, gf)
            if cal is None:
                raise RuntimeError(f'Cannot load gaze calibration: {reason}')
            live.offset = cal['offset']
            gf.add_subscriber(live.on_gaze)
            gf.camera.start_sampling()
        except Exception:
            if gf is not None:
                gf.release()
            live.landmarker.close()
            raise
        self.follower, self.live = gf, live

    def get_gaze(self):
        if self.live is None:
            return None
        if hasattr(self.live, 'latest'):
            return None if self.live.latest is None else tuple(self.live.latest[1])
        with self.live.lock:
            return None if self.live.bubble is None else tuple(map(float, self.live.bubble))

    def close(self):
        if self.follower is not None:
            try:
                self.follower.release()
            finally:
                if not hasattr(self.live, 'latest'):
                    self.live.landmarker.close()
                self.follower = self.live = None


class BlazeGazeBackend(GazeBackend):
    """Fallback backend until a compatible WebEyeTrack Python adapter is available.

    WebEyeTrack has no installed, established Python interface in this project;
    never silently claim that its model is in use.
    """
    def __init__(self, *args, **kwargs):
        self.backend = GazeFollowerBackend(*args, **kwargs)
        self.backend_name = 'GazeFollower'
        self.fallback_reason = 'WebEyeTrack adapter is not available'

    @property
    def follower(self):
        return self.backend.follower

    def prepare(self, live):
        return self.backend.prepare(live)

    def start(self):
        return self.backend.start()

    def get_gaze(self):
        return self.backend.get_gaze()

    def close(self):
        return self.backend.close()


class HeadPoseCompensator:
    """EMA of pose-delta corrections. Gains are px/degree and px/unit distance.

    Distance is -matrix[2,3] in MediaPipe's transform, not matrix scale.
    Distance gains default to zero until fitted from real user data.
    """
    def __init__(self, yaw_gain=-25., pitch_gain=-25., distance_gain=(0., 0.),
                 tau=0.12, limit=110.):
        self.yaw_gain, self.pitch_gain = float(yaw_gain), float(pitch_gain)
        self.distance_gain = _point(distance_gain)
        if tau <= 0 or limit <= 0:
            raise ValueError('tau and limit must be positive')
        self.tau, self.limit = tau, limit
        self.reference = self.last_time = None
        self.offset = (0., 0.)

    @staticmethod
    def pose_from_matrix(matrix):
        from handsfree.gaze_follower import head_pose
        yaw, pitch, _, distance = head_pose(matrix)
        if not all(math.isfinite(v) for v in (yaw, pitch, distance)) or distance <= 0:
            raise ValueError('invalid facial transformation')
        return yaw, pitch, distance

    def calibrate(self, pose, timestamp):
        pose = tuple(map(float, pose))
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose) or pose[2] <= 0 or not math.isfinite(timestamp):
            raise ValueError('invalid pose or time')
        self.reference, self.last_time, self.offset = pose, timestamp, (0., 0.)

    def update(self, gaze, pose, timestamp):
        gaze = _point(gaze)
        if self.reference is None:
            raise RuntimeError('calibrate pose first')
        pose = tuple(map(float, pose))
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose) or pose[2] <= 0 or not math.isfinite(timestamp):
            raise ValueError('invalid pose or time')
        if timestamp < self.last_time:
            raise ValueError('timestamps must be monotonic')
        d = [v - r for v, r in zip(pose, self.reference)]
        goals = (self.yaw_gain*d[0] + self.distance_gain[0]*d[2],
                 self.pitch_gain*d[1] + self.distance_gain[1]*d[2])
        a = -math.expm1(-(timestamp-self.last_time)/self.tau)
        self.offset = tuple(old + a*(max(-self.limit, min(self.limit, goal))-old)
                            for old, goal in zip(self.offset, goals))
        self.last_time = timestamp
        return tuple(v + off for v, off in zip(gaze, self.offset))


class SmoothPursuitCalibrator:
    """0.6 s stationary lead-in, followed by a 5 s normalized orb trajectory.

    Samples are timestamped externally; callers collect gaze against target_at().
    """
    points = ((.1,.1), (.9,.1), (.9,.88), (.5,.88), (.1,.88), (.1,.1))
    def __init__(self, start_time, settle=0.6, duration=5.0):
        if not math.isfinite(start_time) or settle < 0 or duration <= 0:
            raise ValueError('invalid timing')
        self.start_time, self.settle, self.duration = start_time, settle, duration
        self.samples = []

    def target_at(self, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('invalid time')
        elapsed = timestamp - self.start_time
        if elapsed < 0 or elapsed > self.settle + self.duration:
            return None
        u = max(0., min(1., (elapsed-self.settle)/self.duration)) * (len(self.points)-1)
        i = min(int(u), len(self.points)-2)
        f = u-i
        return tuple(a+(b-a)*f for a,b in zip(self.points[i], self.points[i+1]))

    def add_sample(self, timestamp, raw_gaze):
        target = self.target_at(timestamp)
        if target is None or timestamp < self.start_time+self.settle:
            return False
        import numpy as np
        sample = np.asarray(raw_gaze, dtype=float).ravel()
        if sample.size < 2 or not np.isfinite(sample).all():
            raise ValueError('invalid gaze features')
        self.samples.append((timestamp, sample.copy(), target))
        return True

    def fit(self, anchors=None):
        """Fit the existing Ridge + TPS model to timestamped MGazeNet features.

        Targets are normalized display coordinates. The returned calibration uses
        the same serializable model format as the existing follower calibration.
        A prior and a pursuit pass each receive the same total weight. Full
        calibration anchors share all 258 features. Five-dot anchors contain
        only the two raw coordinates, so the joint fit uses those coordinates.
        """
        import numpy as np
        from handsfree.gaze_follower import FEATURE_DIM, NonlinearCalibration, fit_model
        if len(self.samples) < 10:
            raise ValueError('Pursuit needs at least 10 gaze samples')
        features = np.asarray([row[1] for row in self.samples], dtype=float)
        targets = np.asarray([row[2] for row in self.samples], dtype=float)
        if features.ndim != 2 or features.shape[1] not in (2, FEATURE_DIM):
            raise ValueError('Pursuit needs raw gaze coordinates or complete MGazeNet feature vectors')
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError('Pursuit samples must be finite')
        if np.linalg.matrix_rank(targets - targets.mean(axis=0)) < 2:
            raise ValueError('Pursuit was interrupted before the target covered both axes')
        if np.linalg.matrix_rank(features[:, :2] - features[:, :2].mean(axis=0)) < 2:
            raise ValueError('Pursuit gaze samples do not cover two dimensions')
        weights = None
        if anchors is not None:
            prior, labels = (np.asarray(part, dtype=float) for part in anchors)
            if (prior.ndim != 2 or not len(prior) or prior.shape[1] not in (2, FEATURE_DIM)
                    or labels.shape != (len(prior), 2) or not np.isfinite(prior).all()
                    or not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1))):
                raise ValueError('Incompatible saved gaze anchors')
            n_current, n_prior = len(features), len(prior)
            weights = np.repeat(min(n_current, n_prior) / np.array([n_prior, n_current]),
                                [n_prior, n_current])
            width = min(prior.shape[1], features.shape[1])
            features = np.vstack((prior[:, :width], features[:, :width]))
            targets = np.vstack((labels, targets))
        spec = ('ridge_tps', 1.0, 25.0)
        model = fit_model(features, targets, spec, weights)
        cal = NonlinearCalibration()
        cal.install(model, features, targets, spec)
        return cal


class PassiveClickCalibrator:
    """Caller supplies verified physical clicks or snapped AX jaw-click targets.

    Gaze samples and targets are pixels. The caller determines whether gaze was
    settled for 150 ms; no input hooks or accessibility permissions are created.
    """
    def __init__(self, screen_size, learning_rate=0.2):
        self.width, self.height = _point(screen_size)
        if min(self.width, self.height) <= 0 or not 0 < learning_rate <= 1:
            raise ValueError('invalid screen size or learning rate')
        self.learning_rate = learning_rate
        self.grid = [[(0., 0.) for _ in range(3)] for _ in range(3)]

    def correction(self, gaze):
        x,y = _point(gaze)
        x = max(0., min(2., 2*x/self.width)); y = max(0., min(2., 2*y/self.height))
        ix, iy = min(int(x), 1), min(int(y), 1)
        fx, fy = x-ix, y-iy
        return tuple(sum(self.grid[iy+j][ix+i][k] * (fx if i else 1-fx) * (fy if j else 1-fy)
                         for j in (0,1) for i in (0,1)) for k in (0,1))

    def observe_click(self, raw_gaze, target, settled_ms, source='physical', snapped=False):
        raw_gaze, target = _point(raw_gaze), _point(target)
        if source not in ('physical', 'jaw') or (source == 'jaw' and not snapped):
            return False
        if not math.isfinite(settled_ms) or settled_ms < 150 or math.dist(raw_gaze, target) > 240:
            return False
        ix = round(max(0., min(2., 2*raw_gaze[0]/self.width)))
        iy = round(max(0., min(2., 2*raw_gaze[1]/self.height)))
        old = self.grid[iy][ix]
        residual = tuple(t-g for g,t in zip(raw_gaze,target))
        self.grid[iy][ix] = tuple(max(-120., min(120., a+self.learning_rate*(b-a))) for a,b in zip(old,residual))
        return True

    def apply(self, gaze):
        p = _point(gaze)
        return tuple(a+b for a,b in zip(p, self.correction(p)))

    def invert(self, corrected):
        """The uncorrected gaze that `apply` maps to `corrected` (corrections are small and smooth,
        so three fixed-point steps settle well below a pixel)."""
        target = guess = _point(corrected)
        for _ in range(3):
            guess = tuple(t - c for t, c in zip(target, self.correction(guess)))
        return guess


class GazeLoupe:
    """320 px square display centered on gaze, sampling a 128 px screen square."""
    def __init__(self, gaze_x, gaze_y, size=320., magnification=2.5):
        self.center = _point((gaze_x, gaze_y))
        if size <= 0 or magnification <= 0:
            raise ValueError('size and magnification must be positive')
        self.size, self.magnification = float(size), float(magnification)

    def loupe_to_screen(self, lx, ly):
        return tuple(c + (v-self.size/2)/self.magnification for c,v in zip(self.center, _point((lx,ly))))

    def screen_to_loupe(self, sx, sy):
        return tuple(self.size/2 + (v-c)*self.magnification for c,v in zip(self.center, _point((sx,sy))))

    def cell(self, sx, sy):
        lx, ly = self.screen_to_loupe(sx, sy)
        if 0 <= lx < self.size and 0 <= ly < self.size:
            return int(ly * 3 / self.size) * 3 + int(lx * 3 / self.size) + 1
        return None

    def cell_center(self, number):
        if not 1 <= number <= 9:
            raise ValueError("Loupe cells are numbered 1 to 9")
        row, col = divmod(number - 1, 3)
        return self.loupe_to_screen((col + .5) * self.size / 3, (row + .5) * self.size / 3)


def recorded_head_rays(root=ROOT / "data"):
    """Replay saved FaceLandmarker observations without opening a device."""
    import numpy as np

    for path in sorted(root.glob("session_*/*.npz")):
        with np.load(path) as clip:
            width, height = int(clip["width"]), int(clip["height"])
            label = str(clip["label"])
            frames = []
            for lm, time_ms, present in zip(clip["landmarks"], clip["timestamps_ms"], clip["face_present"]):
                pose = solve_head_pose(lm, width, height) if present else None
                frames.append((float(time_ms) / 1000, pose))
        yield path.parent.name, label, frames


def replay_fixations(frames, threshold):
    filters = [OneEuroFilter(1., .08) for _ in range(2)]
    lock = FixationLock()
    for now, pose in frames:
        if pose is None:
            for filt in filters:
                filt.reset()
            lock.reset()
            yield None
            continue
        filtered = tuple(filt.update(v, now) for filt, v in zip(filters, pose.intersection))
        yield lock.update(pose.intersection, filtered, now, threshold)


if __name__ == '__main__':
    import unittest

    class Tests(unittest.TestCase):
        def test_hand_transitions_from_recorded_hand_observations(self):
            import cv2
            import mediapipe as mp
            import numpy as np

            clip = ROOT / "data/session_3/bottom_left_00.mp4"
            model = ROOT / "models/hand_landmarker.task"
            if not model.is_file():
                model = Path.home() / ".cache/gaze-mouth-proto/hand_landmarker.task"
            video = cv2.VideoCapture(str(clip))
            self.addCleanup(video.release)
            frames = []
            with make_hand_landmarker(model) as detector, np.load(clip.with_suffix(".npz")) as recorded:
                timestamps = recorded["timestamps_ms"].astype(int)
                for timestamp in timestamps - timestamps[0]:
                    ok, frame = video.read()
                    if not ok:
                        break
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    aspect = frame.shape[1] / frame.shape[0]
                    frames.append((timestamp / 1000, hand_measurements(detector.detect_for_video(image, int(timestamp)), aspect)))
            tracks = [(t, next(iter(hands)), hands) for t, hands in frames if len(hands) == 1]
            runs = [(index, tracks[index:index + 3]) for index in range(len(tracks) - 2)]
            start, replay = next((index, rows) for index, rows in runs
                                 if rows[0][1] == rows[1][1] == rows[2][1]
                                 and rows[2][0] - rows[0][0] < .12)
            hand = replay[0][1]
            ratios = [hands[hand][0] for _, _, hands in replay]
            wrist = [(hands[hand][1], stamp) for stamp, _, hands in replay]
            speed = math.hypot(*wrist_velocity(wrist, aspect))
            self.assertGreater(speed, 0)
            low, high = min(ratios), max(ratios)
            profile = {"pinch_on": (low + high) / 2, "pinch_off": high, "flick_speed": speed * 2}
            air = HandGestures()
            events = []
            end = next((t for t, hands in frames if t > replay[-1][0] and not hands), None)
            if end is None:
                end = frames[-1][0]
            for time, hands in frames:
                if replay[0][0] <= time <= end:
                    events.extend(kind for kind, _ in air.update(hands, time, aspect, profile))
            self.assertIn("pinch", events)
            self.assertIn("pull", events)
            self.assertIn("release", events)

            # Replay the same real track at a threshold between stationary
            # and its measured 3-frame speed to exercise the flick branch.
            profile = {"pinch_on": low / 2, "pinch_off": low, "flick_speed": speed / 2}
            air.reset()
            self.assertEqual(air.update(replay[0][2], replay[0][0], aspect, profile), [])
            self.assertEqual(air.update(replay[1][2], replay[1][0], aspect, profile), [])
            self.assertEqual([kind for kind, _ in air.update(replay[2][2], replay[2][0], aspect, profile)], ["flick"])

        def test_one_euro_reference_on_recorded_motion(self):
            frames = next(frames for _, label, frames in recorded_head_rays() if label == "neg_look_down")
            valid = [(t, pose.intersection[1]) for t, pose in frames if pose is not None]
            filt = OneEuroFilter(1., .08, 1.)
            previous_time, expected = valid[0]
            derivative = 0.
            self.assertEqual(filt.update(expected, previous_time), expected)
            for t, y in valid[1:]:
                if t <= previous_time:
                    continue
                dt = t - previous_time
                weight = 1. / (1. + 1. / (2 * math.pi * dt))
                derivative = weight * ((y - expected) / dt) + (1 - weight) * derivative
                weight = 1. / (1. + 1. / (2 * math.pi * (1 + .08 * abs(derivative)) * dt))
                expected = weight * y + (1 - weight) * expected
                self.assertAlmostEqual(filt.update(y, t), expected, places=10)
                previous_time = t

        def test_solvepnp_and_fixation_on_real_clips(self):
            clips = list(recorded_head_rays())
            self.assertEqual(len(clips), 64)
            baseline = [frames for session, label, frames in clips if session != "session_3" and label == "neg_blink"]
            threshold = fixation_threshold([[(t, pose.intersection) for t, pose in frames if pose]
                                            for frames in baseline])
            holds = releases = 0
            for session, label, frames in clips:
                for _, pose in frames:
                    if pose is not None:
                        self.assertGreater(pose.origin[2], 0)
                        self.assertAlmostEqual(math.hypot(*pose.direction), 1.)
                        self.assertTrue(math.isfinite(pose.reprojection_px))
                if session != "session_3":
                    continue
                locked = list(replay_fixations(frames, threshold))
                for prior, point in zip(locked, locked[1:]):
                    if prior is not None and point is not None:
                        holds += 1
                        self.assertEqual(prior, point)
                    if prior is not None and point is None:
                        releases += 1
            self.assertGreater(holds, 0)
            self.assertGreater(releases, 0)

        def test_profile_round_trip_with_recorded_values(self):
            from tempfile import TemporaryDirectory
            from handsfree.audio_arbitration import _speech_power, recorded_silent_audio
            import numpy as np

            _, _, frames = next(recorded_head_rays())
            poses = [p for _, p in frames if p]
            _, _, pcm = next(recorded_silent_audio(ROOT / "data/silent_speech"))
            # Exercise serialization only. Use measured scalar values for
            # gesture fields until calibration has a dedicated gesture corpus.
            motion = np.abs(np.diff([p.intersection[0] for p in poses]))
            low, high = (float(v) for v in np.percentile(motion, [20, 80]))
            profile = dict(neutral=list(poses[0].intersection), iris_neutral=.5, iris_margin=low,
                           reprojection_px=poses[0].reprojection_px, saccade_speed=high,
                           pinch_on=low, pinch_off=high, flick_speed=high, noise_floor=float(_speech_power(pcm[:320])))
            with TemporaryDirectory() as directory:
                path = Path(directory) / "spatial_profile.json"
                save_spatial_profile(profile, path)
                self.assertEqual(load_spatial_profile(path), dict(profile, version=1))
                with self.assertRaises(ValueError):
                    save_spatial_profile(dict(profile, pinch_off=0), path)
                self.assertEqual(load_spatial_profile(path), dict(profile, version=1))

        def test_pose_compensation(self):
            c = HeadPoseCompensator(yaw_gain=10, pitch_gain=-10, distance_gain=(2, -2), tau=1, limit=110)
            c.calibrate((0, 0, 10), 0)
            x,y = c.update((100,100), (2,1,12), 1)
            self.assertAlmostEqual(x, 100+24*(1-math.exp(-1)))
            self.assertAlmostEqual(y, 100-14*(1-math.exp(-1)))
            with self.assertRaises(ValueError): c.update((0,0), (0,0,10), .5)
            c.calibrate((0,0,10), 2)
            self.assertEqual(c.offset, (0.,0.))

        def test_click_limits_and_rejections(self):
            c = PassiveClickCalibrator((1000,1000), 1)
            self.assertFalse(c.observe_click((500,500), (700,500), 149))
            self.assertFalse(c.observe_click((500,500), (741,500), 150))
            self.assertFalse(c.observe_click((500,500), (700,500), 150, 'jaw'))
            self.assertTrue(c.observe_click((500,500), (700,400), 150, 'jaw', True))
            self.assertEqual(c.correction((500,500)), (120.,-100.))
            self.assertTrue(c.observe_click((500,500), (300,500), 150))
            self.assertEqual(c.correction((500,500)), (-120.,0.))
            self.assertLessEqual(max(abs(v) for v in c.correction((250,250))), 120)

        def test_pursuit(self):
            c = SmoothPursuitCalibrator(10)
            self.assertIsNone(c.target_at(9))
            self.assertEqual(c.target_at(10.5), (.1,.1))
            self.assertFalse(c.add_sample(10.5, (2,3)))
            self.assertTrue(c.add_sample(12.6, (2,3)))
            self.assertAlmostEqual(c.samples[0][2][1], .88)
            self.assertAlmostEqual(c.target_at(15.6)[1], .1)
            self.assertIsNone(c.target_at(15.61))

        def test_pursuit_fits_recorded_features_with_both_anchor_formats(self):
            import numpy as np
            from tempfile import TemporaryDirectory
            from types import SimpleNamespace
            from handsfree.gaze_follower import load_calibration, save_calibration

            pursuit = ROOT / "logs/gf_cal_20261119-051840.npz"
            dots = ROOT / "logs/gf_cal_20261119-201418.npz"
            if not pursuit.is_file() or not dots.is_file():
                self.skipTest("Recorded pursuit and fixation anchors are required")
            with np.load(pursuit) as trace, np.load(dots) as fixed:
                # The recorded pass used a different orb route. Retain its
                # original target coordinates with each timestamped observation.
                samples = list(zip(trace["t_ms"] / 1000, trace["features"], trace["labels"]))
                x, labels, group = fixed["anchors"], fixed["labels"], fixed["group"]
                y = np.stack([labels[group == idx][0] for idx in np.unique(group)])
            pursuit = SmoothPursuitCalibrator(samples[0][0])
            pursuit.samples = samples[1:]
            for anchors in (None, (x, y), (x[:, :2], y)):
                with self.subTest(width=anchors[0].shape[1] if anchors else None):
                    cal = pursuit.fit(anchors)
                    self.assertIsNotNone(cal.fitted[2])
                    if anchors is not None:
                        np.testing.assert_array_equal(cal.anchors[0][:len(x)], anchors[0])
                    ok, xy = cal.predict(samples[0][1], samples[0][1][:2])
                    self.assertTrue(ok and np.isfinite(xy).all())
                    host = SimpleNamespace(calibration=cal, config=SimpleNamespace(screen_size=(1352, 878)))
                    with TemporaryDirectory() as directory:
                        path = Path(directory) / "pursuit.pkl"
                        save_calibration(path, host, {"spec": cal.spec}, (0., 0.))
                        host.calibration = type(cal)()
                        info, reason = load_calibration(path, host)
                        self.assertIsNotNone(info, reason)
                        _, reloaded = host.calibration.predict(samples[0][1], samples[0][1][:2])
                        np.testing.assert_allclose(xy, reloaded)

        def test_loupe_roundtrip(self):
            loupe = GazeLoupe(500, 400)
            self.assertEqual(loupe.loupe_to_screen(160,160), (500,400))
            self.assertEqual(loupe.loupe_to_screen(0,0), (436,336))
            for p in ((0,0),(320,320),(61.5,123.25)):
                result = loupe.screen_to_loupe(*loupe.loupe_to_screen(*p))
                for a,b in zip(p,result): self.assertAlmostEqual(a,b)

    unittest.main()
