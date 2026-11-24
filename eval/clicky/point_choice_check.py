"""Point to repeated controls in a real Finder recording through the native Ask panel."""

import tempfile
import uuid
from pathlib import Path

import objc
from objc import super
from PyObjCTools import AppHelper

from clicky import memory
from clicky.app import ClickyDelegate, end_run, main
from clicky.overlay import locate
from clicky.settings import preferences

from .annotation_check import recorded_finder


class PointChoiceCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.point-choice-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_(
            {
                "didCompleteOnboarding": True,
                "provider": "Anthropic",
                "suggestFollowups": False,
                "history": False,
                "shortcut": "Command Shift Space",
            }
        )
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        self.context = recorded_finder()
        self.rows = [self.context.window.elements[i] for i in (94, 96, 98)]
        assert all(e.label == "Eject eject" for e in self.rows)
        self.stage = "choose"
        self.panel.show()
        self.panel.field.setStringValue_(
            "Point to an Eject button in the Finder sidebar. Let me choose which visible row."
        )
        AppHelper.callLater(60, self.timeout)
        AppHelper.callAfter(self.submit_, None)

    @objc.python_method
    def offer_point_choices(self, request, prompt, targets):
        super().offer_point_choices(request, prompt, targets)
        try:
            assert [t.frame for t in targets] == [e.frame for e in self.rows]
            menu = self.panel.target_menu
            assert not menu.isHidden() and menu.numberOfItems() == 4
            if self.stage == "dismiss":
                self.dismiss_(None)
                assert self.point_choices is None and menu.isHidden() and self.overlay.target is None
                self.stop_check(None)
                return
            menu.selectItemAtIndex_(3)
            menu.sendAction_to_(menu.action(), self)
        except Exception as exc:  # noqa: BLE001 — relay native callback failures
            self.stop_check(exc)

    @objc.python_method
    def finished(self, request, prompt, error):
        super().finished(request, prompt, error)
        try:
            assert not error and self.stage == "choose", error
            assert self.point_choices is None and self.panel.target_menu.isHidden()
            assert self.overlay.target.frame == self.rows[-1].frame and self.overlay.window.isVisible()
            self.stage = "dismiss"
            self.panel.field.setStringValue_(prompt)
            AppHelper.callAfter(self.submit_, None)
        except Exception as exc:  # noqa: BLE001
            self.stop_check(exc)

    @objc.python_method
    def stop_check(self, error):
        if self.stage != "done":
            self.stage = "done"
            PointChoiceCheck.failure = error
            self.store.removePersistentDomainForName_(self.suite)
            AppHelper.callAfter(end_run)

    @objc.python_method
    def timeout(self):
        self.stop_check(TimeoutError("Timed out at the Finder control picker."))


def main_check():
    ctx = recorded_finder()
    # Finder exposes additional sidebar entries below the window, including unique labels.
    outside = ctx.window.elements[99]
    assert outside.frame[1] >= ctx.window.frame[1] + ctx.window.frame[3]
    assert sum(e.label == outside.label for e in ctx.window.elements) == 1
    try:
        locate(outside.label, ctx, "Anthropic")
    except ValueError as error:
        assert "outside the visible area" in str(error)
    else:
        raise AssertionError("The pointer accepted a control below the recorded window.")
    bottom = locate("Point to the bottom visible Eject button in the Finder sidebar.", ctx, "Anthropic")
    assert bottom.frame == ctx.window.elements[98].frame
    selected = locate("Point to the 1 KB size value on the selected file’s row.", ctx, "Anthropic")
    assert selected.element_id == 48
    assert locate("File", ctx, "Anthropic").element_id == 4
    with tempfile.TemporaryDirectory() as tmp:
        memory.DATABASE = Path(tmp) / "history.sqlite3"
        main(delegate_type=PointChoiceCheck, argv=[])
    if PointChoiceCheck.failure:
        raise PointChoiceCheck.failure
    print(
        "Off-window target rejected. Correct visible bottom, selected-row and menu controls resolved. "
        "The native picker placed the highlight and dismissal cleared a second choice."
    )


if __name__ == "__main__":
    main_check()
