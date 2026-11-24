"""Resume a real recorded Finder round trip through History across three launches."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import AppKit as A
import objc
from objc import super
from PyObjCTools import AppHelper

from cursormix import memory, teach
from cursormix.app import ClickyDelegate, end_run, main
from cursormix.settings import preferences

from . import corpus
from .annotation_check import recorded_finder
from .onboarding_check import buttons

GOAL = (
    "Walk me through opening the Clicky material folder, then returning to the original folder "
    "with the same file selected."
)


class ResumeCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.walkthrough-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_(
            {
                "history": self.recorded_state != "private",
                "didCompleteOnboarding": True,
                "suggestFollowups": False,
                "screenContext": False,
                "provider": "Anthropic",
                "shortcut": "Command Shift Space",
            }
        )
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        AppHelper.callLater(100, self.timeout)
        AppHelper.callAfter(self.open_recording)

    @objc.python_method
    def open_recording(self):
        try:
            if self.recorded_state in ("start", "private"):
                self.ask_(None)
                return
            key, _, _ = memory.conversations(30)[0]
            state = memory.walkthrough(key, 30)
            assert state["goal"] == GOAL and state["pending"], "History lost the unfinished step."
            if self.recorded_state == "inside":
                moved = recorded_finder("moved")
                assert teach.reconcile_step(state["pending"], moved, "Anthropic") is None, (
                    "Moving the Finder window advanced the saved step."
                )
            self.store.setBool_forKey_(True, "screenContext")
            timer = A.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                0.15, self, "selectGuide:", None, True
            )
            A.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, A.NSModalPanelRunLoopMode)
            try:
                self.history_(self)
            finally:
                timer.invalidate()
            assert self.pending_guide, "History did not schedule a fresh summon."
        except Exception as error:  # noqa: BLE001
            self.stop_check(error)

    def selectGuide_(self, timer):
        window = A.NSApp.modalWindow()
        if window is None:
            return
        button = next((b for b in buttons(window.contentView()) if b.title() == "Resume Walkthrough"), None)
        if button:
            button.performClick_(None)
        else:
            A.NSApp.abortModal()

    def ask_(self, sender):
        # Replay a recorded summon without capturing the user's current desktop.
        self.store.setBool_forKey_(False, "screenContext")
        super().ask_(sender)
        self.store.setBool_forKey_(True, "screenContext")

    @objc.python_method
    def show_context(self, ticket, future, pid, name, shortcut):
        if self.recorded_state == "wrong-app":
            other = next((corpus.DATA / "native-calculator").glob("*.json"))
            observation = corpus.context(other.relative_to(corpus.DATA))
        else:
            state = {"private": "start", "clear": "returned"}.get(self.recorded_state, self.recorded_state)
            observation = recorded_finder(state)
        self.present_context(ticket, observation, shortcut)
        if self.recorded_state in ("start", "private"):
            self.panel.field.setStringValue_(GOAL)
            self.submit_(None)

        elif self.recorded_state == "clear":
            # The final verification is in flight. Use the normal dialog and its confirmation.
            timer = A.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                0.12, self, "clearSaved:", None, True
            )
            A.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, A.NSModalPanelRunLoopMode)
            try:
                self.history_(self)
            finally:
                timer.invalidate()

    def clearSaved_(self, timer):
        window = A.NSApp.modalWindow()
        if window:
            choices = {b.title(): b for b in buttons(window.contentView())}
            for title in ("Clear", "Clear History…"):
                if title in choices:
                    choices[title].performClick_(None)
                    break

    @objc.python_method
    def show_instruction(self, request, message, target, help_requests, explanation):
        super().show_instruction(request, message, target, help_requests, explanation)
        try:
            if self.recorded_state == "private":
                assert self.active_guide_key is None and not memory.conversations(30)
                self.stop_check(None)
                return
            state = memory.walkthrough(self.active_guide_key, 30)
            assert state["pending"]["instruction"] in message and state["goal"] == GOAL
            assert "Clicky-source.txt" in state["starting"]["selection"]
            if self.recorded_state == "start":
                assert not state["verified"] and message.startswith("Step 1.")
                # Terminate this disposable launch before AppKit can run normal shutdown.
                self.store.removePersistentDomainForName_(self.suite)
                os._exit(42)
            elif self.recorded_state == "inside":
                assert len(state["verified"]) == 1 and message.startswith("Step 2."), (
                    "Entering the folder while Clicky was closed did not restore Step 2."
                )
                assert state["verified"][0]["resulting_window"] == self.context.window.title
                assert state["pending"]["before"]["window"] == self.context.window.title
            else:
                raise AssertionError("A completed round trip started another step.")
            self.stop_check(None)
        except Exception as error:  # noqa: BLE001
            self.stop_check(error)

    @objc.python_method
    def finished(self, request, prompt, error):
        super().finished(request, prompt, error)
        try:
            if self.recorded_state == "wrong-app":
                assert error and "Finder" in error and self.panel.output.string() == error
                guide = memory.walkthrough(self.active_guide_key, 30)
                assert guide and not guide["verified"] and guide["pending"]["before"]["app"] == "Finder"
                self.stop_check(None)
                return
            if self.recorded_state == "clear":
                assert self.history_epoch > 0, "The confirmation did not clear History."
                assert not error and not memory.conversations(30), "A late result repopulated History."
                self.stop_check(None)
                return
            assert self.recorded_state == "returned" and not error, error
            entries = memory.conversations(30)
            assert len(entries) == 1 and not memory.walkthrough(entries[0][0], 30)
            turns = memory.turns(entries[0][0], 30)
            assert len(turns) == 1 and turns[0]["request"] == GOAL and turns[0]["reply"]
            self.stop_check(None)
        except Exception as exc:  # noqa: BLE001
            self.stop_check(exc)

    @objc.python_method
    def stop_check(self, error):
        self.cancel.set()
        ResumeCheck.failure = error
        self.store.removePersistentDomainForName_(self.suite)
        AppHelper.callAfter(end_run)

    @objc.python_method
    def timeout(self):
        self.stop_check(TimeoutError("The saved walkthrough did not reopen."))


def check_saved_progress(path):
    # Work on a copy of the committed first step from the native launch.
    private_copy = path.with_name("retention.sqlite3")
    shutil.copyfile(path, private_copy)
    memory.DATABASE = private_copy
    key, _, _ = memory.conversations(30)[0]
    state = memory.walkthrough(key, 30)
    for _ in range(2):
        memory.save_walkthrough(key, state, 30, create=True)
    assert len(memory.turns(key, 30)) == 1
    with memory.connect() as db:
        db.execute("UPDATE history SET created=0 WHERE conversation=?", (key,))
    assert memory.walkthrough(key, 30) is None
    memory.save_walkthrough(key, state, 30)
    assert not memory.conversations(30)
    memory.save_walkthrough(key, state, 30, create=True)
    memory.clear()
    memory.save_walkthrough(key, state, 30)
    memory.remember(GOAL, "Finished", completed_walkthrough=key)
    assert not memory.conversations(30), "A late callback restored cleared progress."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("private", "start", "wrong-app", "inside", "clear", "returned"))
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    if args.state:
        if not args.database:
            parser.error("--state requires a temporary --database")
        memory.DATABASE = args.database
        ResumeCheck.recorded_state = args.state
        ResumeCheck.failure = None
        main(delegate_type=ResumeCheck, argv=[])
        if ResumeCheck.failure:
            raise ResumeCheck.failure
    else:
        with tempfile.TemporaryDirectory() as temp:
            for state in ("private", "start", "wrong-app", "inside", "clear", "returned"):
                path = Path(temp) / "test.sqlite3"
                if state == "clear":
                    copy = path.with_name("clear.sqlite3")
                    shutil.copyfile(path, copy)
                    path = copy
                launch = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "eval.cursormix.guide_resume_check",
                        "--state",
                        state,
                        "--database",
                        str(path),
                    ],
                    check=False,
                    timeout=115,
                )
                if state == "start":
                    if launch.returncode != 42:
                        raise RuntimeError(f"The interrupted launch exited with {launch.returncode}.")
                    check_saved_progress(path)
                else:
                    launch.check_returncode()
        print(
            "An abrupt exit retained the step. Calculator and a Finder window move left it pending. The return finished it."
        )
