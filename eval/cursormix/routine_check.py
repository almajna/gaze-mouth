"""Exercise real scheduled briefs, the native editor, and recovery across process exits."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import AppKit as A
import objc
from objc import super
from PyObjCTools import AppHelper

from cursormix import ax, capture, memory, routines
from cursormix.app import ClickyDelegate, end_run, main
from cursormix.settings import preferences

from .corpus import ROOT
from .onboarding_check import buttons

SOURCE = ROOT / "data/clicky/task-workspace/Clicky-source.txt"
PROMPT = "Give the license name and copyright year from the selected file in one short sentence."


def invoke(item):
    assert item is not None and A.NSApp.sendAction_to_from_(
        item.action(), item.target(), item
    )


class RoutineCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.routine-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_(
            {
                "didCompleteOnboarding": True,
                "shortcut": "Command Shift Space",
                "provider": "Anthropic",
            }
        )
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        self.error = None
        self.baseline = self.generation
        AppHelper.callAfter(self.exercise)
        AppHelper.callLater(120, self.timeout)

    @objc.python_method
    def exercise(self):
        try:
            manager = self.routine_menu
            self.menu_ref = self.menu.itemWithTitle_("Routines").submenu()
            manager.menuNeedsUpdate_(self.menu_ref)
            timer = A.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                0.15, self, "fillEditor:", None, True
            )
            A.NSRunLoop.currentRunLoop().addTimer_forMode_(
                timer, A.NSModalPanelRunLoopMode
            )
            try:
                invoke(self.menu_ref.itemWithTitle_("New Routine…"))
            finally:
                timer.invalidate()
            if self.error:
                raise self.error
            self.desktop_key = routines.save(
                "Review the selected file",
                "Open the selected file in Finder.",
                "Desktop task",
                "",
                "Once",
                SOURCE.stat().st_mtime,
            )
            first_due = min(SOURCE.stat().st_mtime, time.time() - 60)
            briefs = (
                ("Copyright year", "What copyright year is printed in the file?"),
                ("Copyright holders", "Name the copyright holders exactly as printed."),
                ("License name", "What license does the first line name?"),
            )
            for index, (name, question) in enumerate(briefs, 1):
                routines.save(
                    name, question, "Background brief", str(self.input_file),
                    "Once", first_due + index, provider="Anthropic",
                )
            manager.start()
            # Keep the next timer tick out of this observation. A free slot must drain the queue itself.
            manager.timer.setFireDate_(A.NSDate.dateWithTimeIntervalSinceNow_(180))
            assert len(manager.jobs) == 3, [(r["name"], r["status"]) for r in routines.all_tasks()]
            assert manager.menu_item.title() == "Routines (3 running, 1 queued, 1 due)"
            if self.home.windows:
                view = next(iter(self.home.windows.values())).contentView()
                self.home.open(view.display_id, pinned=True)
                assert view.routine_button.title() == "Start" and "3 running, 1 queued" in view.status.stringValue()
                assert "3 running" in view.tab.accessibilityLabel()
                routines.pause(self.desktop_key, True)
                manager.refresh_featured()
                view.refresh_(None)
                assert view.routine_button.title() == "Jobs" and not view.routine_button.isHidden()
                center = A.NSNotificationCenter.defaultCenter()
                self.job_titles = []
                center.addObserver_selector_name_object_(
                    self, "inspectJobs:", A.NSMenuDidBeginTrackingNotification, None
                )
                try:
                    view.routine_button.performClick_(None)
                finally:
                    center.removeObserver_name_object_(self, A.NSMenuDidBeginTrackingNotification, None)
                assert "License name · Queued" in self.job_titles
                self.home.collapse()
                routines.pause(self.desktop_key, False)
                manager.refresh_featured()
            manager.menuNeedsUpdate_(self.menu_ref)
            assert self.menu_ref.itemWithTitle_("License name · Queued")
            invoke(self.menu_ref.itemWithTitle_("Copyright year · Running").submenu().itemWithTitle_("Cancel Run"))
            AppHelper.callLater(0.2, self.wait_result)
        except Exception as error:  # noqa: BLE001
            self.finish(error)

    def inspectJobs_(self, note):
        menu = note.object()
        if menu and menu.title() == "Routines":
            self.job_titles = [str(item.title()) for item in menu.itemArray()]
            menu.performSelector_withObject_afterDelay_inModes_(
                "cancelTrackingWithoutAnimation", None, 0.02, [A.NSEventTrackingRunLoopMode]
            )

    def fillEditor_(self, timer):
        window = A.NSApp.modalWindow()
        if not window:
            return
        try:
            controls = {}
            # NSAlert hosts its accessory in a nested view on recent macOS versions.
            queue = list(window.contentView().subviews())
            while queue:
                v = queue.pop()
                if v.accessibilityLabel():
                    controls[str(v.accessibilityLabel())] = v
                queue.extend(v.subviews())
            controls["Name"].setStringValue_("License brief")
            controls["Instructions"].setStringValue_(PROMPT)
            controls["Text file"].setStringValue_(str(self.input_file))
            controls["Repeat"].selectItemWithTitle_("Daily")
            controls["First due"].setDateValue_(
                A.NSDate.dateWithTimeIntervalSince1970_(SOURCE.stat().st_mtime)
            )
            next(
                b for b in buttons(window.contentView()) if b.title() == "Save"
            ).performClick_(None)
        except Exception as error:  # noqa: BLE001
            self.error = error
            A.NSApp.abortModal()

    @objc.python_method
    def wait_result(self):
        try:
            tasks = {t["name"]: t for t in routines.all_tasks()}
            brief = tasks["License brief"]
            if any(t["status"] in ("Saved", "Running") for t in tasks.values()):
                AppHelper.callLater(0.2, self.wait_result)
                return
            assert tasks["Copyright year"]["status"] == "Cancelled" and not tasks["Copyright year"]["result"]
            for name, expected in (("Copyright holders", "almajna"), ("License name", "mit")):
                task = tasks[name]
                assert task["status"] == "Complete" and expected in task["result"].casefold(), task["issue"]
                assert task["next_at"] is None
            assert max(brief["started"], tasks["Copyright holders"]["started"]) <= min(
                brief["result_at"], tasks["Copyright holders"]["result_at"]
            ), "The two completed briefs did not overlap."
            assert brief["status"] == "Complete", (brief["status"], brief["issue"])
            text = SOURCE.read_text()
            assert text.splitlines()[0] in brief["result"] and "2026" in brief["result"]
            assert brief["next_at"] > time.time()
            assert brief["provider"] == "Anthropic"
            assert tasks["Review the selected file"]["status"] == "Ready"
            assert not self.capture_calls and self.generation == self.baseline, (
                "A timer summoned or captured the desktop."
            )
            manager = self.routine_menu
            manager.menuNeedsUpdate_(self.menu_ref)
            sub = self.menu_ref.itemWithTitle_("License brief · Complete").submenu()
            assert sub.itemWithTitle_("View Result…")
            invoke(sub.itemWithTitle_("Pause Schedule"))
            assert next(t for t in routines.all_tasks() if t["id"] == brief["id"])[
                "paused"
            ]
            self.completed = brief
            self.input_file.unlink()
            manager.menuNeedsUpdate_(self.menu_ref)
            invoke(
                self.menu_ref.itemWithTitle_("License brief · Paused")
                .submenu()
                .itemWithTitle_("Run Now")
            )
            AppHelper.callLater(0.2, self.after_missing_file)
        except Exception as error:  # noqa: BLE001
            self.finish(error)

    @objc.python_method
    def after_missing_file(self):
        try:
            brief = next(
                t for t in routines.all_tasks() if t["id"] == self.completed["id"]
            )
            if brief["status"] == "Running":
                AppHelper.callLater(0.2, self.after_missing_file)
                return
            assert (
                brief["status"] == "Failed" and self.input_file.name in brief["issue"]
            )
            assert (
                brief["result"] == self.completed["result"]
                and brief["result_at"] == self.completed["result_at"]
            )
            assert brief["next_at"] == self.completed["next_at"]
            manager = self.routine_menu
            shutil.copy2(SOURCE, self.input_file)
            manager.menuNeedsUpdate_(self.menu_ref)
            invoke(
                self.menu_ref.itemWithTitle_("License brief · Paused")
                .submenu()
                .itemWithTitle_("Run Now")
            )
            manager.menuNeedsUpdate_(self.menu_ref)
            invoke(
                self.menu_ref.itemWithTitle_("License brief · Running")
                .submenu()
                .itemWithTitle_("Cancel Run")
            )
            AppHelper.callLater(0.2, self.after_cancel)
        except Exception as error:  # noqa: BLE001
            self.finish(error)

    @objc.python_method
    def after_cancel(self):
        try:
            brief = next(
                t for t in routines.all_tasks() if t["id"] == self.completed["id"]
            )
            if brief["status"] == "Running":
                AppHelper.callLater(0.2, self.after_cancel)
                return
            assert (
                brief["status"] == "Cancelled"
                and brief["result_at"] == self.completed["result_at"]
            )
            assert self.input_file.read_bytes() == SOURCE.read_bytes()
            manager = self.routine_menu
            manager.menuNeedsUpdate_(self.menu_ref)
            invoke(
                self.menu_ref.itemWithTitle_("License brief · Paused")
                .submenu()
                .itemWithTitle_("Resume Schedule")
            )
            manager.tick_(None)
            assert manager.featured["id"] == self.desktop_key
            if self.home.windows:
                view = next(iter(self.home.windows.values())).contentView()
                self.home.open(view.display_id, pinned=True)
                view.refresh_(None)
                assert (
                    not view.routine_button.isHidden()
                    and view.routine_button.title() == "Start"
                )
                assert "Review the selected file" in view.status.stringValue()
                self.home.collapse()
            if ax.session_locked():
                invoke(
                    self.menu_ref.itemWithTitle_("Review the selected file · Ready")
                    .submenu()
                    .itemWithTitle_("Run on Current Screen")
                )
                AppHelper.callLater(0.2, self.wait_desktop)
            else:
                self.finish(None)
        except Exception as error:  # noqa: BLE001
            self.finish(error)

    @objc.python_method
    def wait_desktop(self):
        try:
            task = next(t for t in routines.all_tasks() if t["id"] == self.desktop_key)
            if task["status"] == "Running":
                AppHelper.callLater(0.2, self.wait_desktop)
                return
            assert (
                task["status"] in ("Cancelled", "Interrupted", "Failed")
                and "unlock" in task["issue"].lower()
            ), task["issue"]
            assert not self.capture_calls and self.generation > self.baseline
            self.finish(None)
        except Exception as error:  # noqa: BLE001
            self.finish(error)

    @objc.python_method
    def timeout(self):
        self.finish(TimeoutError("The native routine trial timed out."))

    @objc.python_method
    def finish(self, error):
        RoutineCheck.failure = error
        self.routine_menu.stop()
        self.store.removePersistentDomainForName_(self.suite)
        AppHelper.callAfter(end_run)


def crash_claim(key):
    brief = next(t for t in routines.all_tasks() if t["name"] == "License brief")
    assert (
        brief["status"] == "Cancelled"
        and SOURCE.read_text().splitlines()[0] in brief["result"]
    )
    run = routines.claim(key)
    assert run is not None
    print("claimed", flush=True)
    sys.stdin.readline()
    os._exit(42)


def check_recovery():
    # Use the real file timestamp for a missed scheduled occurrence.
    key = routines.save(
        "Interrupted visit",
        "Open the selected file in Finder.",
        "Desktop task",
        "",
        "Once",
        SOURCE.stat().st_mtime,
    )
    routines.reconcile()
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "eval.cursormix.routine_check",
            "--database",
            str(memory.DATABASE),
            "--claim",
            key,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as worker:
        try:
            assert worker.stdout.readline().strip() == "claimed"
            routines.reconcile()
            assert routines.claim(key) is None, "Two processes acquired the same run."
            assert (
                next(t for t in routines.all_tasks() if t["id"] == key)["status"]
                == "Running"
            )
        finally:
            worker.communicate("\n", timeout=15)
        assert worker.returncode == 42
    routines.reconcile()
    task = next(t for t in routines.all_tasks() if t["id"] == key)
    assert task["status"] == "Interrupted" and task["next_at"] is None
    assert routines.claim() is None, (
        "An interrupted desktop action was automatically retried."
    )
    assert routines.claim(key, revision=task["revision"] - 1) is None, (
        "A stale menu choice restarted the task."
    )
    weekly = routines.save(
        "Weekly license",
        PROMPT,
        "Background brief",
        str(SOURCE),
        "Weekly",
        SOURCE.stat().st_mtime,
    )
    slot = routines.claim()
    assert slot and slot.task["id"] == weekly
    next_run = next(t["next_at"] for t in routines.all_tasks() if t["id"] == weekly)
    first = datetime.fromtimestamp(SOURCE.stat().st_mtime)
    following = datetime.fromtimestamp(next_run)
    assert next_run > time.time()
    assert (first.weekday(), first.hour, first.minute) == (
        following.weekday(),
        following.hour,
        following.minute,
    )
    slot.finish("Cancelled")
    assert routines.claim() is None
    routines.remove(weekly)
    routines.remove(key)
    print(
        "A concurrent process could not reclaim the live run. An abrupt exit left the consumed occurrence interrupted."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--claim")
    args = parser.parse_args()
    if args.claim:
        memory.DATABASE = args.database
        crash_claim(args.claim)
    with tempfile.TemporaryDirectory(prefix="clicky-routines-") as tmp:
        memory.DATABASE = Path(tmp) / "history.sqlite3"
        RoutineCheck.input_file = Path(tmp) / SOURCE.name
        shutil.copy2(SOURCE, RoutineCheck.input_file)
        RoutineCheck.failure = None
        RoutineCheck.capture_calls = []

        def observe(frame, event, arg):
            if event == "call" and frame.f_code is capture.on_summon.__code__:
                RoutineCheck.capture_calls.append(time.time())

        sys.setprofile(observe)
        threading.setprofile(observe)
        try:
            main(delegate_type=RoutineCheck, argv=[])
        finally:
            sys.setprofile(None)
            threading.setprofile(None)
        if RoutineCheck.failure:
            raise RoutineCheck.failure
        check_recovery()
    print(
        "Three briefs started together. Cancelling one preserved its siblings, and the queued brief ran before the next timer tick. The due desktop task made no capture."
    )
