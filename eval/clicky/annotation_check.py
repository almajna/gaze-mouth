"""Submit a multi-mark request from the Ask panel against a real Finder recording."""

import json
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

import objc
from objc import super
from PIL import Image, ImageChops
from PyObjCTools import AppHelper

from clicky import annotations, ax, capture, memory
from clicky.app import ClickyDelegate, end_run, main
from clicky.settings import preferences

from . import corpus


def recorded_finder(state_name="start"):
    recording = next((corpus.ROOT / "data/clicky/development").glob(f"finder-roundtrip-*/{state_name}.json"))
    state = json.loads(recording.read_text())
    elements = [
        ax.Element(
            e["role"], e["label"], e["frame"], focused=e.get("focused", False), selected=e.get("selected")
        )
        for e in state["elements"]
    ]
    return capture.Context(
        ax.Window(0, state["app"], state["title"], state["frame"], elements),
        state["display"],
        recording.with_suffix(".jpg").read_bytes(),
    )


def rendered(view):
    bitmap = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
    view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), bitmap)
    with Image.open(BytesIO(bytes(bitmap.TIFFRepresentation()))) as image:
        return image.convert("RGB")


class AnnotationCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.annotation-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_({"didCompleteOnboarding": True, "history": False, "provider": "Anthropic"})
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        self.context = recorded_finder()
        self.panel.field.setStringValue_(
            "Annotate this Finder window. Draw a ring around the Clicky material folder name and "
            "an arrow to the toolbar Search button. Label both with their names. Connect the folder to Search with a labeled line."
        )
        AppHelper.callAfter(self.submit_, None)

    @objc.python_method
    def finished(self, request, prompt, error):
        super().finished(request, prompt, error)
        try:
            if error:
                raise RuntimeError(error)
            drawing = self.annotation
            assert drawing and drawing.window.isVisible() and not drawing.live
            view = drawing.view
            marks = list(view.marks)
            assert len(marks) == 2 and self.panel.output.string().strip()
            for name, shape in (("Clicky material", "ring"), ("Search", "arrow")):
                mark = next(m for m in marks if m.shape == shape and name.lower() in m.label.lower())
                referenced = self.context.window.elements[mark.target.element_id]
                assert referenced.label == name and referenced.frame == mark.target.frame
            assert len(view.links) == 1
            connection = view.links[0]
            assert marks[connection["start"]].shape == "ring" and marks[connection["end"]].shape == "arrow"
            for width, height in ((700, 440), (460, 530)):
                drawing.window.setContentSize_((width, height))
                decorated = rendered(view)
                links = view.links
                view.links = []
                connection_ink = ImageChops.difference(decorated, rendered(view))
                view.marks = []
                plain = rendered(view)
                view.marks = marks
                view.links = links
                diff = ImageChops.difference(decorated, plain)
                # A mark must actually render at each reference, including after letterboxing.
                dx, dy, dw, dh = view.viewport
                pw, ph = decorated.size
                scale = min(pw / dw, ph / dh)
                centers = []
                for mark in marks:
                    x, y, w, h = mark.target.frame
                    left = round((pw - dw * scale) / 2 + (x - dx) * scale)
                    top = round((ph - dh * scale) / 2 + (y - dy) * scale)
                    centers.append((left + w * scale / 2, top + h * scale / 2))
                    assert diff.crop((left, top, left + round(w * scale), top + round(h * scale))).getbbox()
                # Check the shaft away from either target and from the midpoint caption.
                px, py = (round(0.75 * a + 0.25 * b) for a, b in zip(*centers))
                assert connection_ink.crop((px - 3, py - 3, px + 4, py + 4)).getbbox(), (
                    "Missing connector shaft"
                )
            drawing.window.performClose_(None)
            assert not drawing.window.isVisible() and view.snapshot is None
            self.dismiss_(None)
            assert self.annotation is None
            outcomes.append(
                "Two labeled Finder targets rendered in their recorded frames at both window sizes."
            )
        except Exception as exc:  # noqa: BLE001 — surface failure from the AppKit callback
            outcomes.append(exc)
        finally:
            self.store.removePersistentDomainForName_(self.suite)
            AppHelper.callAfter(end_run)


if __name__ == "__main__":
    outcomes = []
    with tempfile.TemporaryDirectory() as temp:
        memory.DATABASE = Path(temp) / "history.sqlite3"
        main(delegate_type=AnnotationCheck, argv=[])
    if not outcomes:
        raise RuntimeError("The annotation check ended without a result.")
    if isinstance(outcomes[0], Exception):
        raise outcomes[0]
    print(outcomes[0])
    info = corpus.context("public-screens/qa-0-2.json")
    assert not info.window.elements and info.display is None
    _, marks, _, viewport = annotations.compose(
        "Box the blue Spotlight Comments section heading and label it.", "Anthropic", info, ()
    )
    assert len(marks) == 1 and marks[0].shape == "box" and marks[0].target.element_id is None
    # Reference in the original 265 × 552 public Finder Info screenshot, in image pixels.
    assert viewport == (0, 0, 265, 552)
    x, y, w, h = marks[0].target.frame
    assert 3 <= x + w / 2 < 262 and 71 <= y + h / 2 < 93
    print("The visual-only box centered on the recorded Spotlight Comments heading.")
    metadata = next((corpus.DATA / "native-calculator").glob("*.json"))
    calculator = corpus.context(metadata.relative_to(corpus.DATA))
    _, marks, _, viewport = annotations.compose(
        "Draw a box around the Equals button and a ring around All Clear.",
        "Anthropic",
        calculator,
        (),
    )
    assert viewport[0] < 0 and len(marks) == 2
    for name, shape in (("Equals", "box"), ("All Clear", "ring")):
        target = next(mark.target for mark in marks if mark.shape == shape)
        assert target.label == name and target.frame == calculator.window.elements[target.element_id].frame
        assert viewport[0] <= target.frame[0] < viewport[0] + viewport[2]
    print("Both Calculator marks bound to control frames on the recorded secondary display.")
    try:
        annotations.compose(
            "Circle the video timeline playhead and label it.", "Anthropic", recorded_finder(), ()
        )
    except ValueError as exc:
        assert "playhead" in str(exc).lower() or "timeline" in str(exc).lower(), exc
    else:
        raise AssertionError("Clicky drew a video timeline mark on the Finder recording.")
    print("The request for a missing video playhead produced no mark.")
