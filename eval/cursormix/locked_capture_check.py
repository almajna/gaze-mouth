"""Check queued, cancelled and already-completed captures against an actual macOS session lock."""

import os
import sys
import threading
import uuid
from concurrent.futures import Future

import objc
from objc import super

from cursormix import ax, capture
from cursormix.app import ClickyDelegate, main
from cursormix.settings import preferences

from .annotation_check import recorded_finder


def observe_worker(cancel, app_only):
    accesses = []
    operations = {f.__code__ for f in (ax.read_window, capture.display_for, capture.screen_kit, capture.mss.MSS.grab)}

    def observe(frame, event, arg):
        if event == "call" and frame.f_code in operations:
            accesses.append(frame.f_code.co_name)

    sys.setprofile(observe)
    try:
        result = capture.on_summon(os.getpid(), "Clicky", cancel, app_only=app_only)
    finally:
        sys.setprofile(None)
    assert not accesses, accesses
    return result


def assert_empty(context):
    assert context and not (context.jpeg or context.display or context.window.app or context.window.elements)
    assert context.window.pid == 0 and context.issue


class LockedQueueCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        store = preferences("local.clicky.locked-capture-check." + uuid.uuid4().hex)
        store.registerDefaults_({"shortcut": "Command Shift Space"})
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        try:
            assert ax.session_locked(), "Run this check while the Mac is locked."
            for app_only in (False, True):
                assert_empty(self.capture_queue.submit(observe_worker, threading.Event(), app_only).result(5))
            release, cancel = threading.Event(), threading.Event()
            self.capture_queue.submit(release.wait, 5)
            queued = self.capture_queue.submit(observe_worker, cancel, False)
            cancel.set()
            release.set()
            cancelled = queued.result(5)
            assert_empty(cancelled)
            assert "cancel" in cancelled.issue.casefold()
            completed = Future()
            completed.set_result(recorded_finder("start"))
            assert completed.result().jpeg and completed.result().window.elements
            self.show_context(self.summon_id, completed, os.getpid(), "Clicky", None)
            assert_empty(self.context)
            assert_empty(self.original_context)
            assert not self.panel.mark.isEnabled()
            self.dismiss_(None)
            assert self.context is None and self.original_context is None
            assert ax.session_locked(), "The Mac unlocked during the check."
            print("Locked full-display and app-only requests and queued cancellation read no window or pixels. "
                  "The panel discarded a completed Finder recording delivered during the lock.")
        except Exception as error:  # noqa: BLE001 — carry callback failures into the native process exit code
            self.exit_code = 1
            print(f"Locked capture check failed: {error!r}", file=sys.stderr)


if __name__ == "__main__":
    main(delegate_type=LockedQueueCheck, argv=["--check-startup"])
