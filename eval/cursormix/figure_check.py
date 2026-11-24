"""Ask Clicky to illustrate a recorded Finder window and exercise the rendered 3D view."""

import tempfile
import uuid
from io import BytesIO
from pathlib import Path

import objc
from objc import super
from PIL import Image, ImageChops
from PyObjCTools import AppHelper

from cursormix import memory
from cursormix.app import ClickyDelegate, end_run, main
from cursormix.settings import preferences

from . import corpus


def rendered(view):
    with Image.open(BytesIO(bytes(view.snapshot().TIFFRepresentation()))) as image:
        return image.convert("RGB")


class FigureCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.figure-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_({"didCompleteOnboarding": True, "history": False, "provider": "Anthropic"})
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        self.context = corpus.context(corpus.rows("qa")[0]["context"])
        self.panel.field.setStringValue_(
            "Show me a 3D drawing of this Finder window. Label its folder with the actual name on screen "
            "and use different shapes to explain how its contents relate to it."
        )
        AppHelper.callAfter(self.submit_, None)

    @objc.python_method
    def finished(self, request, prompt, error):
        super().finished(request, prompt, error)
        try:
            if error:
                raise RuntimeError(error)
            window = self.figure_window
            assert window is not None and window.isVisible()
            assert window.canvas.allowsCameraControl() and len(window.nodes) >= 2
            text = " ".join(p["label"] + " " + p["detail"] for p in window.diagram["parts"])
            assert "sites" in (text + window.diagram["title"] + window.diagram["summary"]).lower()
            before = rendered(window.canvas)
            assert any(hi - lo > 30 for lo, hi in before.getextrema())
            controller = window.canvas.defaultCameraController()
            controller.rotateByX_Y_(30, 15)
            rotated = rendered(window.canvas)
            assert ImageChops.difference(before, rotated).getbbox()
            controller.dollyToTarget_(window.radius * 0.35)
            assert ImageChops.difference(rotated, rendered(window.canvas)).getbbox()
            for node in window.nodes:
                projected = window.canvas.projectPoint_(node.position())
                window.inspect_point((projected.x, projected.y))
                if window.picker.indexOfSelectedItem() > 0:
                    break
            assert window.picker.indexOfSelectedItem() > 0
            window.picker.selectItemAtIndex_(1)
            window.picker.sendAction_to_(window.picker.action(), window)
            assert window.detail.stringValue() == window.diagram["parts"][0]["detail"]
            assert window.nodes[1].opacity() < window.nodes[0].opacity()
            window.resetView_(None)
            assert window.picker.indexOfSelectedItem() == 0 and all(n.opacity() == 1 for n in window.nodes)
            assert window.detail.stringValue() == self.panel.output.string()
            self.dismiss_(None)
            assert not window.isVisible() and self.figure_window is None and not window.canvas.scene()
            outcomes.append("Native 3D drawing verified using the recorded Finder window.")
        except Exception as exc:  # noqa: BLE001 — relay AppKit callback failures to the process
            outcomes.append(exc)
        finally:
            self.store.removePersistentDomainForName_(self.suite)
            AppHelper.callAfter(end_run)


if __name__ == "__main__":
    outcomes = []
    with tempfile.TemporaryDirectory() as temp:
        memory.DATABASE = Path(temp) / "history.sqlite3"
        main(delegate_type=FigureCheck, argv=[])
    if not outcomes:
        raise RuntimeError("The 3D check ended without a result.")
    if isinstance(outcomes[0], Exception):
        raise outcomes[0]
    print(outcomes[0])
