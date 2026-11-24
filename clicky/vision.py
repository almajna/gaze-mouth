"""Measure screenshot regions with Vision and describe them offline with FastVLM."""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from functools import cache, cached_property
from io import BytesIO
from pathlib import Path

from PIL import Image

from .ax import clipped_frame

_worker = ThreadPoolExecutor(1, thread_name_prefix="CursorMix FastVLM")
FASTVLM = "mlx-community/FastVLM-0.5B-bf16"


@cache
def _load():
    try:
        from huggingface_hub import hf_hub_download
        from mlx_vlm import load
    except ImportError as error:
        raise RuntimeError(
            "Install CursorMix's requirements-local.lock for offline vision."
        ) from error
    path = Path(hf_hub_download(FASTVLM, "config.json", local_files_only=True)).parent
    return load(path)


def _observe(question, jpeg):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = _load()
    instruction = (
        "Give a short factual description answering the question using this screenshot. "
        "Quote visible labels when relevant. " + question[:1200]
    )
    formatted = apply_chat_template(processor, model.config, instruction, num_images=1)
    with Image.open(BytesIO(jpeg)) as image:
        return generate(
            model,
            processor,
            prompt=formatted,
            image=[image.convert("RGB")],
            max_tokens=110,
            temperature=0,
            verbose=False,
        ).text.strip()


def describe_local(question, jpeg):
    try:
        return _worker.submit(_observe, question, jpeg).result()
    except (OSError, ImportError) as exc:
        raise RuntimeError(
            "Install requirements-local.lock and download the CursorMix FastVLM checkpoint for offline vision."
        ) from exc


def recognized_lines(jpeg):
    import Vision as V

    request = V.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(V.VNRequestTextRecognitionLevelAccurate)
    handler = V.VNImageRequestHandler.alloc().initWithData_options_(jpeg, {})
    ok, _ = handler.performRequests_error_([request], None)
    if not ok:
        return []
    return [text for line in request.results() for text in line.topCandidates_(1)]


def text_frame(literal, lines, display):
    if not literal.strip():
        return None
    pattern = re.compile(
        r"(?<!\w)" + re.escape(literal.strip()) + r"(?!\w)", re.IGNORECASE
    )
    matches = [
        (line, hit) for line in lines for hit in pattern.finditer(str(line.string()))
    ]
    if len(matches) != 1:
        return None
    line, hit = matches[0]
    text = str(line.string())
    # Vision takes an NSString range measured in UTF-16 code units.
    start = len(text[: hit.start()].encode("utf-16-le")) // 2
    length = len(hit.group().encode("utf-16-le")) // 2
    region, _ = line.boundingBoxForRange_error_((start, length), None)
    if region is None:
        return None
    rect = region.boundingBox()
    x, y, w, h = display
    frame = (
        x + w * rect.origin.x,
        y + h * (1 - rect.origin.y - rect.size.height),
        w * rect.size.width,
        h * rect.size.height,
    )
    return clipped_frame(frame, (display,))


class ScreenshotRegions:
    """Map locations in one captured image into the supplied display or canvas."""

    def __init__(self, jpeg, viewport):
        self.jpeg = jpeg
        self.viewport = viewport

    @cached_property
    def lines(self):
        return recognized_lines(self.jpeg)

    def uniform(self, frame):
        """Whether the proposed frame covers a single pixel color in the captured image."""
        x, y, w, h = self.viewport
        fx, fy, fw, fh = frame
        with Image.open(BytesIO(self.jpeg)) as image:
            sx, sy = image.width / w, image.height / h
            box = (
                max(0, math.floor((fx - x) * sx)),
                max(0, math.floor((fy - y) * sy)),
                min(image.width, math.ceil((fx + fw - x) * sx)),
                min(image.height, math.ceil((fy + fh - y) * sy)),
            )
            return all(lo == hi for lo, hi in image.crop(box).convert("RGB").getextrema())

    def closeup(self, frame):
        """Return the actual pixels and display bounds around a proposed target."""
        x, y, w, h = self.viewport
        fx, fy, fw, fh = frame
        width = min(w, max(w / 3, fw * 5))
        height = min(h, max(h / 3, fh * 5))
        if width == w and height == h:
            return None
        left = max(0, min(fx + fw / 2 - width / 2 - x, w - width))
        top = max(0, min(fy + fh / 2 - height / 2 - y, h - height))
        with Image.open(BytesIO(self.jpeg)) as img, BytesIO() as out:
            sx, sy = img.width / w, img.height / h
            l, t = round(left * sx), round(top * sy)
            r, b = round((left + width) * sx), round((top + height) * sy)
            if r <= l or b <= t:
                return None
            img.crop((l, t, r, b)).convert("RGB").save(out, "JPEG", quality=92)
            return ScreenshotRegions(out.getvalue(), (x + l / sx, y + t / sy, (r - l) / sx, (b - t) / sy))

    def frame(self, box, literal=""):
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(v) in (int, float) for v in box)
        ):
            return None
        left, top, right, bottom = box
        if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
            return None
        if isinstance(literal, str) and 0 < len(literal.strip()) <= 160:
            measured = text_frame(literal, self.lines, self.viewport)
            if measured:
                return measured
        x, y, w, h = self.viewport
        return (
            x + w * left / 1000,
            y + h * top / 1000,
            w * (right - left) / 1000,
            h * (bottom - top) / 1000,
        )


def region_pixels(jpeg, display, frame, scale):
    """Sample a visible rectangle at a shared resolution across display scales."""
    if not jpeg or not display or not frame:
        return None
    x, y, w, h = frame
    dx, dy, dw, dh = display
    if x < dx or y < dy or x + w > dx + dw or y + h > dy + dh:
        return None
    size = round(w * scale), round(h * scale)
    if min(size) < 1:
        return None
    with Image.open(BytesIO(jpeg)) as image:
        sx, sy = image.width / dw, image.height / dh
        box = ((x - dx) * sx, (y - dy) * sy, (x + w - dx) * sx, (y + h - dy) * sy)
        return image.convert("RGB").resize(size, Image.Resampling.BILINEAR, box)


def match_icon(icon, view):
    """Find a single copy of a screenshot detail. Repeated or missing icons have no match."""
    import numpy as np

    source = np.asarray(view.convert("L"), dtype=np.float32)
    pattern = np.asarray(icon.convert("L"), dtype=np.float32)
    h, w = pattern.shape
    height, width = source.shape
    if min(w, h) < 4 or w > width or h > height:
        return None
    pattern -= pattern.mean()
    energy = np.sum(pattern**2)
    if energy == 0:
        return None

    # Zero-mean normalized cross-correlation tolerates Finder dimming a window.
    # Only valid origins are kept, so the FFT cannot wrap a match around an edge.
    spectrum = np.fft.rfft2(source)
    kernel = np.fft.rfft2(pattern, s=source.shape, axes=(0, 1))
    covariance = np.fft.irfft2(spectrum * kernel.conj(), s=source.shape, axes=(0, 1))
    covariance = covariance[: height - h + 1, : width - w + 1]

    def sums(values):
        integral = (
            np.pad(values, ((1, 0), (1, 0)))
            .cumsum(axis=0, dtype=np.float64)
            .cumsum(axis=1)
        )
        return (
            integral[h:, w:]
            - integral[:-h, w:]
            - integral[h:, :-w]
            + integral[:-h, :-w]
        )

    variance = sums(source**2) - sums(source) ** 2 / (w * h)
    scale = np.sqrt(energy * np.maximum(variance, 0))
    scores = np.divide(
        covariance, scale, out=np.zeros_like(covariance), where=scale > 0
    )
    y, x = np.unravel_index(scores.argmax(), scores.shape)
    best = scores[y, x]
    if best < 0.85:
        return None
    # Ignore neighboring subpixel alignments of this copy, then check for a
    # second plausible copy elsewhere. Either could have been the original.
    scores[max(0, y - h // 2) : y + h // 2 + 1, max(0, x - w // 2) : x + w // 2 + 1] = 0
    if scores.max() >= best - 0.08:
        return None

    # Window controls can share a silhouette and differ only by color. Remove
    # per-pixel brightness before checking their color on the original captures.
    def chroma(patch):
        rgb = np.asarray(patch.convert("RGB"), dtype=np.float32)
        return rgb - rgb.mean(axis=2, keepdims=True)

    original = chroma(icon)
    candidate = chroma(view.crop((int(x), int(y), int(x + w), int(y + h))))
    color_error = np.sqrt(np.mean((candidate - original) ** 2))
    if color_error > max(6, np.sqrt(np.mean(original**2)) / 2):
        return None
    return int(x), int(y)
