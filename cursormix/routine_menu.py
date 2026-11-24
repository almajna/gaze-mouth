"""Native menu, scheduler and result viewer for saved CursorMix routines."""

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import AppKit as A
import objc
from Foundation import NSDate, NSObject
from PyObjCTools import AppHelper

from . import routines
from .models import PROVIDERS, selected_provider

SCHEDULED_BRIEF_LIMIT = 3


def command(menu, title, target, selector, key=None):
    button = menu.addItemWithTitle_action_keyEquivalent_(title, selector, "")
    button.setTarget_(target)
    if key:
        button.setRepresentedObject_(key)
    return button


def display_time(value):
    return datetime.fromtimestamp(value).strftime("%a %b %d, %H:%M")


def waiting_brief(task, now):
    return (
        task["kind"] == "Background brief" and not task["paused"] and task["status"] != "Running"
        and task["next_at"] is not None and task["next_at"] <= now
    )


class RoutineMenu(NSObject):
    @objc.python_method
    def attach(self, app, menu):
        self.app = app
        self.menu = menu
        self.menu_item = app.menu.itemAtIndex_(app.menu.indexOfItemWithSubmenu_(menu))
        self.timer = None
        self.jobs = {}
        self.pending = None
        self.desktop = self.desktop_cancel = None
        self.featured = None
        self.job_summary = ""
        self.file_field = None
        self.menu.setDelegate_(self)

    @objc.python_method
    def start(self):
        self.timer = (
            A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                15, self, "tick:", None, True
            )
        )
        self.tick_(None)

    @objc.python_method
    def stop(self):
        if self.timer:
            self.timer.invalidate()
            self.timer = None
        self.cancel_pending()
        if self.desktop_cancel:
            self.desktop_cancel.set()
        for _, cancel in self.jobs.values():
            cancel.set()

    @objc.python_method
    def cancel_pending(self):
        if self.pending:
            self.pending.finish("Cancelled")
            self.pending = None

    def tick_(self, timer):
        try:
            routines.reconcile()
            while len(self.jobs) < SCHEDULED_BRIEF_LIMIT:
                run = routines.claim()
                if run is None:
                    break
                self.start_brief(run)
            self.refresh_featured()
        except (sqlite3.Error, OSError) as error:
            if self.menu.numberOfItems():
                self.menu.itemAtIndex_(0).setToolTip_(str(error))

    @objc.python_method
    def refresh_featured(self):
        tasks = routines.all_tasks()
        ready = [r for r in tasks if r["status"] == "Ready" and not r["paused"]]
        completed = [r for r in tasks if r["result_at"] or r["issue"]]
        self.featured = (
            ready[0]
            if ready
            else max(completed, key=lambda r: r["started"] or 0, default=None)
        )
        now = time.time()
        counts = (
            (sum(r["status"] == "Running" for r in tasks), "running"),
            (sum(waiting_brief(r, now) for r in tasks), "queued"),
        )
        jobs = ", ".join(f"{count} {label}" for count, label in counts if count)
        if jobs != self.job_summary:
            self.job_summary = jobs
            self.app.home.activity("speaking" if self.app.speaker else self.app.activity)
        summary = ", ".join(filter(None, (jobs, f"{len(ready)} due" if ready else "")))
        self.menu_item.setTitle_(f"Routines ({summary})" if summary else "Routines")

    def menuNeedsUpdate_(self, menu):
        menu.removeAllItems()
        command(menu, "New Routine…", self, "newRoutine:")
        try:
            tasks = routines.all_tasks()
        except (sqlite3.Error, OSError) as error:
            command(menu, str(error), None, None)
            return
        if tasks:
            menu.addItem_(A.NSMenuItem.separatorItem())
        now = time.time()
        for task in tasks:
            key = task["id"]
            state = task["status"]
            if task["paused"] and state != "Running":
                state = "Paused"
            elif waiting_brief(task, now):
                state = "Queued"
            parent = command(menu, f"{task['name']} · {state}", None, None)
            sub = A.NSMenu.alloc().initWithTitle_(task["name"])
            parent.setSubmenu_(sub)
            command(sub, task["kind"], None, None)
            if task["next_at"] and not task["paused"]:
                command(
                    sub,
                    f"{task['cadence']} · {display_time(task['next_at'])}",
                    None,
                    None,
                )
            elif task["status"] == "Ready":
                command(sub, "Due on your Mac", None, None)
            run_title = (
                "Run on Current Screen" if task["kind"] == "Desktop task" else "Run Now"
            )
            if task["status"] != "Running":
                command(sub, run_title, self, "runRoutine:", key).setTag_(
                    task["revision"]
                )
            elif (
                key in self.jobs
                or (self.desktop and self.desktop.task["id"] == key)
                or (self.pending and self.pending.task["id"] == key)
            ):
                command(sub, "Cancel Run", self, "cancelRoutine:", key)
            if task["result"] or task["issue"]:
                command(sub, "View Result…", self, "viewResult:", key)
            sub.addItem_(A.NSMenuItem.separatorItem())
            if task["status"] != "Running":
                command(sub, "Edit…", self, "editRoutine:", key)
            if task["next_at"]:
                command(
                    sub,
                    "Resume Schedule" if task["paused"] else "Pause Schedule",
                    self,
                    "pauseRoutine:",
                    key,
                )
            if task["status"] != "Running":
                command(sub, "Remove…", self, "removeRoutine:", key)

    @objc.python_method
    def lookup(self, sender):
        return next(
            (r for r in routines.all_tasks() if r["id"] == sender.representedObject()),
            None,
        )

    @objc.python_method
    def message(self, title, text, buttons=("OK",)):
        dialog = A.NSAlert.alloc().init()
        dialog.setMessageText_(title)
        dialog.setInformativeText_(text)
        for label in buttons:
            dialog.addButtonWithTitle_(label)
        A.NSApp.activateIgnoringOtherApps_(True)
        return dialog.runModal()

    def runRoutine_(self, sender):
        try:
            task = self.lookup(sender)
            if not task:
                return
            if task["kind"] == "Desktop task":
                if (
                    self.pending
                    or self.desktop
                    or not self.app.cancel_item.isHidden()
                    and self.app.activity != "idle"
                ):
                    self.message(
                        "A request is already running",
                        "Finish or cancel it before starting this routine.",
                    )
                    return
                if not all(
                    self.app.store.boolForKey_(key)
                    for key in ("screenContext", "agentTasks")
                ):
                    self.message(
                        "Desktop routine unavailable",
                        "Enable Use Screen Context and Desktop Actions in Settings.",
                    )
                    return
            run = routines.claim(task["id"], revision=sender.tag())
            if not run:
                self.refresh_featured()
                return
            if task["kind"] == "Desktop task":
                try:
                    self.app.ask_(self)
                    self.pending = run
                except Exception:
                    run.finish("Interrupted", "The desktop request could not start.")
                    raise
            else:
                self.start_brief(run)
            self.refresh_featured()
        except (sqlite3.Error, OSError, ValueError) as error:
            self.message("Could not start routine", str(error))

    @objc.python_method
    def submit_pending(self):
        if not self.pending:
            return False
        run = self.pending
        self.pending = None
        self.desktop = run
        self.app.panel.field.setStringValue_(run.task["prompt"])
        self.app.submit_request(routine=run)
        self.desktop_cancel = self.app.cancel
        return True

    @objc.python_method
    def end_desktop(self, run, status, text):
        try:
            run.finish(status, text)
        finally:
            if self.desktop is run:
                self.desktop = self.desktop_cancel = None
        self.refresh_featured()

    @objc.python_method
    def start_brief(self, run):
        cancel = threading.Event()
        key = run.task["id"]
        self.jobs[key] = run, cancel
        provider = selected_provider(run.task["provider"])

        def work():
            try:
                text = routines.brief(run.task, provider, cancel)
                status = "Complete"
            except Exception as error:  # noqa: BLE001 — retain a readable run outcome
                status, text = "Failed", str(error)
            AppHelper.callAfter(self.end_brief, key, status, text)

        threading.Thread(target=work, name="CursorMix routine", daemon=True).start()

    @objc.python_method
    def end_brief(self, key, status, text):
        run, cancel = self.jobs.pop(key)
        run.finish(
            "Cancelled" if cancel.is_set() else status, "" if cancel.is_set() else text
        )

        if self.timer is not None:
            self.tick_(None)
        else:
            self.refresh_featured()

    def cancelRoutine_(self, sender):
        key = sender.representedObject()
        if key in self.jobs:
            self.jobs[key][1].set()
        elif self.pending and self.pending.task["id"] == key:
            self.app.dismiss_(self)
        elif self.desktop and self.desktop.task["id"] == key:
            if self.desktop_cancel is self.app.cancel:
                self.app.cancelRequest_(self)
            elif self.desktop_cancel:
                self.desktop_cancel.set()

    def pauseRoutine_(self, sender):
        if task := self.lookup(sender):
            routines.pause(task["id"], not task["paused"])
            self.refresh_featured()

    def removeRoutine_(self, sender):
        task = self.lookup(sender)
        if not task:
            return
        if (
            self.message(
                "Remove routine?",
                f"Remove {task['name']}, its schedule and its saved result?",
                ("Keep", "Remove"),
            )
            == A.NSAlertSecondButtonReturn
        ):
            try:
                routines.remove(task["id"])
                self.refresh_featured()
            except ValueError as error:
                self.message("Could not remove routine", str(error))

    def viewResult_(self, sender):
        if task := self.lookup(sender):
            notes = [task["issue"]] if task["issue"] else []
            if task["result_at"]:
                notes.extend(
                    [
                        "Last completed · " + display_time(task["result_at"]),
                        task["result"],
                    ]
                )
            self.message(task["name"], "\n\n".join(notes))

    def newRoutine_(self, sender):
        self.edit({})

    def editRoutine_(self, sender):
        if task := self.lookup(sender):
            self.edit(task)

    def chooseFile_(self, sender):
        picker = A.NSOpenPanel.openPanel()
        picker.setCanChooseDirectories_(False)
        picker.setAllowsMultipleSelection_(False)
        if picker.runModal() == A.NSModalResponseOK:
            self.file_field.setStringValue_(picker.URL().path())

    @objc.python_method
    def edit(self, task):
        dialog = A.NSAlert.alloc().init()
        dialog.setMessageText_("Edit Routine" if task else "New Routine")
        dialog.setInformativeText_(
            "The saved provider receives the instructions and optional UTF-8 file (up to 32 KB) on each run. "
            "Scheduled desktop tasks wait for you to open them on the current screen."
        )
        dialog.addButtonWithTitle_("Save")
        dialog.addButtonWithTitle_("Cancel")
        view = A.NSView.alloc().initWithFrame_(((0, 0), (420, 237)))

        def field(label, y, control, width=312):
            title = A.NSTextField.labelWithString_(label)
            title.setFrame_(((0, y + 2), (97, 19)))
            control.setFrame_(((99, y), (width, 24)))
            control.setAccessibilityLabel_(label)
            view.addSubview_(title)
            view.addSubview_(control)
            return control

        def text(label, y, value, width=312):
            return field(label, y, A.NSTextField.textFieldWithString_(value), width)

        def choices(label, y, values, selected):
            popup = A.NSPopUpButton.alloc().initWithFrame_pullsDown_(
                ((0, 0), (312, 24)), False
            )
            popup.addItemsWithTitles_(values)
            popup.selectItemWithTitle_(selected)
            return field(label, y, popup)

        name = text("Name", 206, task.get("name", ""))
        instructions = text("Instructions", 173, task.get("prompt", ""))
        kind = choices(
            "Task type", 140, routines.KINDS, task.get("kind", routines.KINDS[0])
        )
        provider = choices(
            "Provider",
            107,
            PROVIDERS,
            task.get("provider")
            or selected_provider(self.app.store.stringForKey_("provider")),
        )
        source = text("Text file", 74, task.get("source", ""), 245)
        self.file_field = source
        choose = A.NSButton.buttonWithTitle_target_action_(
            "Choose…", self, "chooseFile:"
        )
        choose.setFrame_(((344, 72), (72, 26)))
        view.addSubview_(choose)
        cadence = choices(
            "Repeat", 41, routines.CADENCES, task.get("cadence", routines.CADENCES[0])
        )
        start = A.NSDatePicker.alloc().init()
        start.setDatePickerElements_(
            A.NSYearMonthDayDatePickerElementFlag | A.NSHourMinuteDatePickerElementFlag
        )
        start.setDateValue_(
            NSDate.dateWithTimeIntervalSince1970_(
                task.get("next_at") or time.time() + 3600
            )
        )
        field("First due", 8, start)
        dialog.setAccessoryView_(view)
        dialog.window().setInitialFirstResponder_(name)
        A.NSApp.activateIgnoringOtherApps_(True)
        try:
            while dialog.runModal() == A.NSAlertFirstButtonReturn:
                try:
                    path = source.stringValue().strip()
                    routines.save(
                        str(name.stringValue()),
                        str(instructions.stringValue()),
                        str(kind.titleOfSelectedItem()),
                        str(Path(path).expanduser()) if path else "",
                        str(cadence.titleOfSelectedItem()),
                        start.dateValue().timeIntervalSince1970(),
                        task.get("id"),
                        provider=str(provider.titleOfSelectedItem()),
                    )
                    self.tick_(None)
                    break
                except (OSError, ValueError, sqlite3.Error) as error:
                    dialog.setInformativeText_(str(error))
        finally:
            self.file_field = None
