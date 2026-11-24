"""Exercise the first-run native dialogs in an isolated preferences suite."""

import os
import tempfile
import uuid
from pathlib import Path

import AppKit as A
import objc
from objc import super
from PyObjCTools import AppHelper

from clicky import memory
from clicky.app import ClickyDelegate, end_run, main
from clicky.settings import preferences


def buttons(view):
    if isinstance(view, A.NSButton):
        yield view
    for child in view.subviews():
        yield from buttons(child)


class FirstLaunch(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.onboarding-eval." + uuid.uuid4().hex
        return preferences(self.suite)

    def applicationDidFinishLaunching_(self, note):
        self.dialogs = []
        timer = A.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            0.12, self, "readDialog:", None, True
        )
        A.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, A.NSModalPanelRunLoopMode)
        try:
            super().applicationDidFinishLaunching_(note)
        finally:
            timer.invalidate()
        assert self.store.boolForKey_("didCompleteOnboarding")
        assert self.dialogs.count("Skip") == 2 and "OK" in self.dialogs, self.dialogs
        self.store.removePersistentDomainForName_(self.suite)
        print("Isolated first launch displayed the shortcut and both provider-key dialogs.", flush=True)
        AppHelper.callAfter(end_run)

    def readDialog_(self, timer):
        window = A.NSApp.modalWindow()
        if not window:
            return
        choices = {button.title(): button for button in buttons(window.contentView())}
        for title in ("Skip", "OK", "Continue"):
            if title in choices:
                self.dialogs.append(title)
                choices[title].performClick_(None)
                return


if __name__ == "__main__":
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    with tempfile.TemporaryDirectory() as tmp:
        memory.DATABASE = Path(tmp) / "history.sqlite3"
        main(delegate_type=FirstLaunch, argv=[])
