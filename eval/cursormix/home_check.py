"""Open a recorded Finder question through the native display Home controls."""

import tempfile
import threading
import uuid
from pathlib import Path

import AppKit as A
import objc
from objc import super
from PyObjCTools import AppHelper

from cursormix import ax, memory
from cursormix.app import ClickyDelegate, end_run, main
from cursormix.home import display_id, top_center
from cursormix.settings import preferences

from .annotation_check import recorded_finder
from .privacy import Observation


class HomeCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.home-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_(
            {
                "didCompleteOnboarding": True,
                "displayHome": True,
                "shortcut": "Command Shift Space",
                "screenContext": True,
                "floatingPointer": False,
                "suggestFollowups": False,
                "provider": "Anthropic",
            }
        )
        return store

    def applicationDidFinishLaunching_(self, notification):
        super().applicationDidFinishLaunching_(notification)
        self.stage = "open"
        AppHelper.callAfter(self.exercise)
        AppHelper.callLater(70, self.timeout)

    @objc.python_method
    def exercise(self):
        try:
            screens = A.NSScreen.screens()
            assert self.home.windows.keys() == {display_id(s) for s in screens}
            front = ax.frontmost()
            for screen in screens:
                window = self.home.windows[display_id(screen)]
                view = window.contentView()
                assert window.isVisible() and not window.canBecomeKeyWindow()
                view.mouseEntered_(None)
                assert view.expanded and not view.open_button.isHidden()
                assert self.summon_id == 0 and self.context is None and ax.frontmost() == front
                cx, top = top_center(screen)
                assert abs(A.NSMidX(window.frame()) - cx) <= 1 and abs(A.NSMaxY(window.frame()) - top) <= 2
                view.close_button.performClick_(None)
                assert not view.expanded
            self.openHome_(None)
            self.view = next(w.contentView() for w in self.home.windows.values() if w.contentView().expanded)
            assert self.view.pinned and self.view.cancel_button.isHidden()
            AppHelper.callLater(0.6, self.ask_from_tab)
        except Exception as error:  # noqa: BLE001 — relay AppKit callback failures to the command
            self.stop_check(error)

    @objc.python_method
    def ask_from_tab(self):
        try:
            assert not self.observation.calls, "Opening Home captured screen context."
            self.store.setBool_forKey_(False, "screenContext")
            self.view.open_button.performClick_(None)
        except Exception as exc:  # noqa: BLE001
            self.stop_check(exc)

    @objc.python_method
    def show_context(self, ticket, future, pid, name, shortcut):
        super().show_context(ticket, future, pid, name, shortcut)
        if self.stage != "open":
            return
        try:
            assert self.panel.home_display == self.view.display_id and self.panel.window.isVisible()
            screen = self.panel.window.screen()
            cx, top = top_center(screen)
            assert abs(A.NSMidX(self.panel.window.frame()) - cx) <= 1
            assert abs(A.NSMaxY(self.panel.window.frame()) - (top - 24)) <= 1
            self.store.setBool_forKey_(True, "screenContext")
            self.context = recorded_finder()
            self.title = self.context.window.title
            self.stage = "answer"
            self.panel.field.setStringValue_(
                "What is the name of the open Finder folder? Give only the name."
            )
            self.panel.field.sendAction_to_(self.panel.field.action(), self)
            self.home.open(self.view.display_id, pinned=True)
            assert self.view.open_button.title() == "Show Request" and not self.view.cancel_button.isHidden()
            assert "Working" in self.view.tab.accessibilityLabel()
            request = self.generation
            self.view.open_button.performClick_(None)
            assert request == self.generation and not self.cancel.is_set()
        except Exception as exc:  # noqa: BLE001
            self.stop_check(exc)

    @objc.python_method
    def finished(self, request, prompt, error):
        super().finished(request, prompt, error)
        if self.stage == "guide":
            self.stop_check(RuntimeError(error or "The walkthrough ended before displaying an instruction."))
            return
        if self.stage != "answer":
            return
        try:
            assert not error and self.title.casefold() in self.panel.output.string().casefold()
            self.home.open(self.view.display_id, pinned=True)
            assert (
                self.view.cancel_button.isHidden()
                and self.title.casefold() in self.view.status.stringValue().casefold()
            )
            self.view.open_button.performClick_(None)
            self.stage = "guide"
            self.panel.field.setStringValue_(
                "Walk me through opening the Clicky material folder and returning to the starting folder."
            )
            self.panel.field.sendAction_to_(self.panel.field.action(), self)
        except Exception as exc:  # noqa: BLE001
            self.stop_check(exc)

    @objc.python_method
    def show_instruction(self, request, message, target, help_requests, explanation):
        super().show_instruction(request, message, target, help_requests, explanation)
        if self.stage != "guide":
            return
        try:
            assert (
                self.panel.step_question.action() == "askAboutStep:"
                and not self.panel.step_question.isHidden()
            )
            assert self.panel.field.isHidden() and self.step_help is help_requests
            self.home.open(self.view.display_id, pinned=True)
            assert (
                self.view.open_button.title() == "Show Walkthrough"
                and message in self.view.status.stringValue()
            )
            self.view.open_button.performClick_(None)
            assert self.generation == request and self.step_help is help_requests
            self.stage = "cancel"
            ticket = self.generation
            self.home.open(self.view.display_id, pinned=True)
            self.view.cancel_button.performClick_(None)
            assert self.cancel.is_set() and self.generation > ticket and not self.panel.window.isVisible()
            self.flag_(self.flags["displayHome"])
            assert not self.home.windows and self.home_item.isHidden()
            self.flag_(self.flags["displayHome"])
            assert self.home.windows and not self.home_item.isHidden()
            assert not self.observation.calls, "Home requested a capture outside the explicit Ask."
            self.stop_check(None)
        except Exception as exc:  # noqa: BLE001
            self.stop_check(exc)

    @objc.python_method
    def timeout(self):
        self.stop_check(TimeoutError("Home check timed out in " + self.stage))

    @objc.python_method
    def stop_check(self, error):
        if self.stage != "done":
            self.stage = "done"
            HomeCheck.failure = error
            self.home.sync(False)
            self.store.removePersistentDomainForName_(self.suite)
            AppHelper.callAfter(end_run)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        memory.DATABASE = Path(tmp) / "history.sqlite3"
        HomeCheck.observation = Observation()
        threading.setprofile_all_threads(HomeCheck.observation.profile)
        try:
            main(delegate_type=HomeCheck, argv=[])
        finally:
            threading.setprofile_all_threads(None)
        if HomeCheck.failure:
            raise HomeCheck.failure
    print(
        "Home preserved focus, answered the Finder question, reopened the same walkthrough and cancelled it."
    )
