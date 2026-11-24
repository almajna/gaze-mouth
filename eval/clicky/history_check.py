"""Resume real recorded excerpts from an older Clicky database through the native History menu."""

import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import ClassVar

import AppKit as A
import objc
from objc import super
from PyObjCTools import AppHelper

from clicky import memory
from clicky.app import ClickyDelegate, end_run, main
from clicky.settings import preferences

from . import corpus
from .agent_trial import descendants
from .annotation_check import recorded_finder


class HistoryCheck(ClickyDelegate):
    failures: ClassVar[list[Exception]] = []

    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.history-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_(
            {"didCompleteOnboarding": True, "screenContext": False, "floatingPointer": False}
        )
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        AppHelper.callAfter(self.exercise)

    @objc.python_method
    def exercise(self):
        timer = A.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            0.12, self, "selectSaved:", None, True
        )
        A.NSRunLoop.mainRunLoop().addTimer_forMode_(timer, A.NSModalPanelRunLoopMode)
        try:
            saved = memory.conversations(30)
            assert len(saved) == 2
            self.chosen = saved[-1][0]
            window = recorded_finder().window
            self.search_term = window.title
            filename = next(e.label for e in window.elements if e.selected and e.role == "AXTextField")
            memory.remember(
                "Which file is selected?",
                filename,
                conversation=self.chosen,
                screen=memory.screen_excerpt(window),
            )
            self.turns = memory.turns(self.chosen, 30)
            assert len(self.turns) == 2 and len(memory.turns(saved[0][0], 30)) == 1
            for row in corpus.rows("qa"):
                memory.remember(row["request"], row["reference"])
            assert self.chosen not in {entry[0] for entry in memory.conversations(30)}
            for term in (filename, window.app, "Which file is selected?"):
                assert self.chosen in {entry[0] for entry in memory.conversations(30, term)}
            self.deadline = time.monotonic() + 8
            self.history_(self)
            assert self.conversation_id == self.chosen and self.conversation == self.turns
            assert self.panel.output.string().endswith(self.turns[-1]["reply"])
            assert self.panel.field.stringValue() == ""
            self.ask_(self)
            assert self.conversation_id is None and not self.conversation and not self.panel.output.string()
            memory.clear()
            assert not memory.conversations(30)
        except Exception as exc:  # noqa: BLE001 — relay native callback failures to the command
            self.failures.append(exc)
        finally:
            timer.invalidate()
            self.store.removePersistentDomainForName_(self.suite)
            AppHelper.callAfter(end_run)

    def selectSaved_(self, timer):
        try:
            self.search_and_resume()
        except Exception as exc:  # noqa: BLE001 — return failures from the native modal to the test process
            self.failures.append(exc)
            A.NSApp.abortModal()

    @objc.python_method
    def search_and_resume(self):
        if time.monotonic() > self.deadline:
            A.NSApp.abortModal()
            return
        window = A.NSApp.modalWindow()
        if not window:
            return
        controls = list(descendants(window.contentView()))
        picker = next((v for v in controls if isinstance(v, A.NSPopUpButton)), None)
        if not picker:
            return
        search = next(v for v in controls if isinstance(v, A.NSSearchField))
        search.setStringValue_("%_")
        search.sendAction_to_(search.action(), self)
        assert not picker.numberOfItems() and not self.history_resume.isEnabled()
        search.setStringValue_(self.search_term.upper())
        search.sendAction_to_(search.action(), self)
        assert picker.numberOfItems() == 1 and self.history_resume.isEnabled()
        search.setStringValue_("")
        search.sendAction_to_(search.action(), self)
        assert picker.numberOfItems() == 50
        search.setStringValue_(self.search_term)
        search.sendAction_to_(search.action(), self)
        for i, item in enumerate(picker.itemArray()):
            if item.representedObject() == self.chosen:
                picker.selectItemAtIndex_(i)
                picker.sendAction_to_(picker.action(), self)
                break
        if self.history_preview.string().startswith(self.turns[0]["request"]):
            next(v for v in controls if isinstance(v, A.NSButton) and v.title() == "Resume").performClick_(
                None
            )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        memory.DATABASE = Path(temp) / "history.sqlite3"
        rows = corpus.rows("qa")[:2]
        with sqlite3.connect(memory.DATABASE) as db:
            db.execute("CREATE TABLE history(id INTEGER PRIMARY KEY, request TEXT, answer TEXT)")
            db.executemany(
                "INSERT INTO history(request, answer) VALUES (?,?)",
                [(r["request"], r["reference"]) for r in rows],
            )
        main(delegate_type=HistoryCheck, argv=[])
        if HistoryCheck.failures:
            raise HistoryCheck.failures[0]
        assert not memory.conversations(30)
        print("Native History found and resumed a conversation older than the latest fifty, then cleared it.")
