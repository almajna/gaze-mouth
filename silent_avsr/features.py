"""Input contract for the two streams, and read-only loaders for the real clips in data/.

landmarks_30hz : (T, 163) float32. 45 lip/jaw landmarks (xyz) in a head-fixed face frame, in units
                 of the outer-eye-corner span, followed by 28 mouth blendshapes.
mouth_roi_30hz : (T, 1, 88, 88) uint8. Grayscale mouth crop, stabilized by a per-frame similarity
                 transform (eye line horizontal, eye span fixed, mouth centred).
"""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RATE_HZ = 30.0
ROI_SIDE = 88
SOURCE_SIDE = 96  # side of the crops the collectors store; the eye span maps to this many pixels

LIP_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LIP_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
JAW = (152, 148, 377, 172, 397)
GEOMETRY_LANDMARKS = LIP_OUTER + LIP_INNER + JAW
RIGID_ORIGIN = (33, 263, 133, 362, 168)  # eye corners and nose bridge: unaffected by the jaw
EYE_LEFT, EYE_RIGHT, FOREHEAD, NOSE_BASE = 33, 263, 10, 2
MOUTH_BLENDSHAPES = (
    "cheekPuff", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight")
D_GEOM = 3 * len(GEOMETRY_LANDMARKS) + len(MOUTH_BLENDSHAPES)  # 163


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fill_invalid(x):
    """Replace non-finite rows with the nearest earlier finite row (or the first finite row)."""
    x = np.array(x, dtype=np.float32)
    good = np.isfinite(x.reshape(len(x), -1)).all(1)
    if not good.any():
        raise ValueError("No frame has a detected face")
    index = np.where(good, np.arange(len(x)), -1)
    index = np.maximum.accumulate(index)
    index[index < 0] = np.flatnonzero(good)[0]
    return x[index]


def geometric_features(landmarks, blendshapes, blendshape_names, width, height):
    """(T, 478, 3) normalized MediaPipe landmarks + (T, 52) blendshapes -> (T, 163) float32."""
    landmarks = fill_invalid(landmarks)
    blendshapes = fill_invalid(blendshapes)
    points = landmarks.astype(np.float64) * [width, height, width]  # z shares the x scale
    eye_axis = points[:, EYE_RIGHT] - points[:, EYE_LEFT]
    span = np.linalg.norm(eye_axis, axis=-1, keepdims=True)
    ex = eye_axis / span
    down = points[:, NOSE_BASE] - points[:, FOREHEAD]
    ey = _unit(down - (down * ex).sum(-1, keepdims=True) * ex)
    ez = np.cross(ex, ey)
    frame = np.stack((ex, ey, ez), axis=1)  # rows are the face axes
    origin = points[:, RIGID_ORIGIN].mean(1, keepdims=True)
    local = (points[:, GEOMETRY_LANDMARKS] - origin) @ frame.transpose(0, 2, 1) / span[:, None]
    names = list(blendshape_names)
    mouth = blendshapes[:, [names.index(n) for n in MOUTH_BLENDSHAPES]]
    return np.concatenate((local.reshape(len(local), -1), mouth), axis=1).astype(np.float32)


def mouth_roi(gray_frames, landmarks, side=ROI_SIDE):
    """(T, H, W) uint8 frames + (T, 478, 3) landmarks -> (T, side, side) uint8 stabilized mouth crops."""
    import cv2
    landmarks = fill_invalid(landmarks)
    crops = np.empty((len(gray_frames), side, side), np.uint8)
    for i, (frame, lm) in enumerate(zip(gray_frames, landmarks)):
        height, width = frame.shape
        eye1, eye2, left, right, upper, lower = lm[[33, 263, 61, 291, 13, 14], :2] * [width, height]
        axis = eye2 - eye1
        span = np.linalg.norm(axis)
        u = axis / span
        basis = np.stack((u, [-u[1], u[0]]))
        scale = SOURCE_SIDE / span
        centre = (left + right + upper + lower) / 4
        affine = np.column_stack((basis * scale, (side - 1) / 2 - scale * (basis @ centre)))
        crops[i] = cv2.warpAffine(frame, affine, (side, side), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return crops


def resample(x, times_ms, rate_hz=RATE_HZ):
    """Linear interpolation along axis 0 from frame timestamps onto a uniform grid."""
    times_ms = np.asarray(times_ms, np.float64)
    grid = np.arange(times_ms[0], times_ms[-1] + 1e-6, 1000.0 / rate_hz)
    position = np.interp(grid, times_ms, np.arange(len(times_ms)))
    lo = np.floor(position).astype(int)
    hi = np.minimum(lo + 1, len(times_ms) - 1)
    weight = (position - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    out = (1 - weight) * x[lo] + weight * x[hi]
    return np.rint(out).astype(np.uint8) if x.dtype == np.uint8 else out.astype(np.float32)


def centre_crop(frames, side=ROI_SIDE):
    off = (frames.shape[-1] - side) // 2
    return frames[..., off:off + side, off:off + side]


def silent_takes(root=DATA / "silent_train"):
    """Sorted (folder, meta) for every saved real silent take. Reads only."""
    takes = []
    for path in sorted(Path(root).glob("*/*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("saved") and (path.parent / "mouth_96x96.npy").is_file():
            takes.append((path.parent, meta))
    return takes


def load_silent_take(folder, meta=None, side=ROI_SIDE):
    """A real silent take as 30 Hz model input. These takes have no landmarks, so the geometry
    stream is absent. `side=96` keeps the stored crop for random-crop augmentation."""
    folder = Path(folder)
    meta = meta or json.loads((folder / "meta.json").read_text())
    video = np.load(folder / "mouth_96x96.npy", allow_pickle=False)
    if video.dtype != np.uint8 or video.shape[1:] != (SOURCE_SIDE, SOURCE_SIDE) or not len(video):
        raise ValueError(f"{folder}: expected uint8 96x96 mouth crops")
    times = np.arange(len(video)) * 1000.0 / meta["video_fps"]
    roi = resample(centre_crop(video, side), times)
    return dict(landmarks_30hz=None, mouth_roi_30hz=roi[:, None], phrase=meta["phrase"],
                session=folder.parent.name, name=f"{folder.parent.name}/{folder.name}")


def session_clips(root=DATA):
    return sorted(Path(root).glob("session_*/*.npz"))


def load_session_clip(npz_path):
    """A real recorded clip (video + paired landmarks) as 30 Hz dual-stream model input.
    These clips contain gaze fixations, mouth-open clicks and negatives, not speech."""
    import cv2
    npz_path = Path(npz_path)
    with np.load(npz_path) as d:
        landmarks, blendshapes = d["landmarks"], d["blendshapes"]
        names, times = [str(n) for n in d["blendshape_names"]], d["timestamps_ms"]
        width, height = int(d["width"]), int(d["height"])
        label, gesture = str(d["label"]), str(d["gesture"])
    video = cv2.VideoCapture(str(npz_path.with_suffix(".mp4")))
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    video.release()
    n = min(len(frames), len(landmarks))
    if n < 2:
        raise ValueError(f"{npz_path}: no readable frames")
    landmarks, blendshapes, times = landmarks[:n], blendshapes[:n], times[:n]
    geometry = geometric_features(landmarks, blendshapes, names, width, height)
    roi = mouth_roi(frames[:n], landmarks)
    return dict(landmarks_30hz=resample(geometry, times), mouth_roi_30hz=resample(roi, times)[:, None],
                label=label, gesture=gesture, name=f"{npz_path.parent.name}/{npz_path.stem}")
