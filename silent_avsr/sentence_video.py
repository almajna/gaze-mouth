"""68-point face alignment and timestamp-based preprocessing for sentence clips."""
from pathlib import Path

import cv2
import numpy as np

from .features import ROOT, centre_crop

RATE_HZ = 25.0
PERIOD_NS = 40_000_000
AUTO_AVSR = ROOT / "models" / "auto_avsr"
MEAN_FACE = AUTO_AVSR / "preparation/detectors/retinaface/20words_mean_face.npy"
FACEMARK = ROOT / "models/lbfmodel.yaml"
STABLE = [28, 33, 36, 39, 42, 45, 48, 54]


class Landmarks68:
    def __init__(self, weights=FACEMARK):
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        if self.detector.empty():
            raise RuntimeError("Could not load OpenCV's frontal face detector")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(weights))

    def __call__(self, gray):
        scale = min(1.0, 320 / gray.shape[1])
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        faces = self.detector.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        box = np.rint(np.array([face]) / scale).astype(np.int32)
        ok, landmarks = self.facemark.fit(gray, box)
        return landmarks[0].reshape(68, 2) if ok else None


def align_mouth(gray, landmarks, reference):
    """Use Auto-AVSR's 68-point template, stable anchors, and 96-to-88 center crop."""
    affine, _ = cv2.estimateAffinePartial2D(landmarks[STABLE], reference[STABLE], method=cv2.LMEDS)
    if affine is None:
        raise ValueError("Could not align the detected face")
    warped = cv2.warpAffine(gray, affine, (256, 256))
    lip_center = (landmarks[48:68] @ affine[:, :2].T + affine[:, 2]).mean(axis=0)
    x, y = np.rint(np.clip(lip_center, 48, 208)).astype(int)
    return centre_crop(warped[y - 48:y + 48, x - 48:x + 48])


def sentence_crops(frames, timestamps_ns, landmarks, reference):
    times = np.asarray(timestamps_ns, dtype=np.int64)
    points = np.asarray(landmarks, dtype=np.float32)
    if len(times) < 2 or len(times) != len(frames) or points.shape != (len(times), 68, 2):
        raise ValueError("Expected matching frames, timestamps and 68-point landmarks")
    times = times - times[0]
    if np.any(np.diff(times) <= 0) or times[-1] < PERIOD_NS:
        raise ValueError("Capture a longer clip with increasing frame timestamps")
    valid = np.isfinite(points).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("No face detected in the clip")
    # Interpolate dropouts by time and hold the nearest fit at either edge.
    if not valid.all():
        flat = points.reshape(len(times), -1).copy()
        for column in flat.T:
            column[~valid] = np.interp(times[~valid], times[valid], column[valid])
        points = flat.reshape(-1, 68, 2)
    aligned = []
    for i, frame in enumerate(frames):
        margin = min(6, i, len(points) - 1 - i)
        smooth = points[i - margin:i + margin + 1].mean(axis=0)
        smooth += points[i].mean(axis=0) - smooth.mean(axis=0)
        aligned.append(align_mouth(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), smooth, reference))
    grid = np.arange(times[-1] // PERIOD_NS + 1, dtype=np.int64) * PERIOD_NS
    # Select interpolation pairs on the integer grid, with no extrapolation or FPS rounding.
    right = np.searchsorted(times, grid, side="right").clip(1, len(times) - 1)
    left = right - 1
    weight = ((grid - times[left]) / (times[right] - times[left]))[:, None, None]
    aligned = np.asarray(aligned)
    return np.rint(aligned[left] * (1 - weight) + aligned[right] * weight).astype(np.uint8), grid


def normalize_contrast(crops):
    """Apply the fixed grayscale normalization used by the USR 2.0 demo."""
    if crops.dtype != np.uint8 or crops.ndim != 3 or crops.shape[1:] != (88, 88) or not len(crops):
        raise ValueError("Expected a nonempty (T, 88, 88) uint8 grayscale clip")
    return (crops.astype(np.float32) / 255.0 - 0.421) / 0.165
