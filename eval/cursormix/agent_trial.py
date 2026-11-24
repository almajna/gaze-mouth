"""Observe a real fixed-start desktop task through the running Clicky controller."""

import argparse
import hashlib
import json
import shlex
import sys
import time
import uuid
from pathlib import Path

import AppKit as A
import ApplicationServices as AX
import objc
from objc import super
from PyObjCTools import AppHelper

from cursormix import ax
from cursormix.app import ClickyDelegate, end_run, main
from cursormix.settings import preferences

from . import corpus

OBS = corpus.ROOT / "data/clicky/observations"
DEFAULT_WORKSPACE = corpus.ROOT / "data/clicky/task-workspace"


def matches(check, window, workspace=DEFAULT_WORKSPACE):
    # Every goal names an app and at least one separately observable condition.
    if not check.get("app") or window.app != check["app"]:
        return False
    tests = []
    if "has_window" in check:
        tests.append(bool(window.frame) == check["has_window"])
    if "label_contains" in check:
        tests.append(any(check["label_contains"] in e.label for e in window.elements))
    if "title_contains" in check:
        tests.append(bool(check["title_contains"]) and check["title_contains"] in window.title)
    if "label" in check:
        tests.append(any(e.label == check["label"] for e in window.elements))
    root = workspace.resolve()

    def local(name):
        path = (root / check[name]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("File predicate escapes the task workspace.")
        return path

    if "directory" in check:
        tests.append(local("directory").is_dir())
    if "absent_file" in check:
        tests.append(not local("absent_file").exists())
    if "file" in check:
        f = local("file")
        tests.append(f.is_file() == check.get("exists", True))
        if "contents" in check:
            tests.append(f.is_file() and f.read_text() == check["contents"])
        if "same_as" in check:
            ref = local("same_as")
            tests.append(f.is_file() and ref.is_file() and f.read_bytes() == ref.read_bytes())
    return bool(tests) and all(tests)


class Trial:
    def __init__(self, item, provider, mode, workspace_path=DEFAULT_WORKSPACE):
        self.workspace_path = workspace_path
        self.item, self.provider, self.mode = item, provider, mode
        self.start_ok = self.end_ok = self.cancelled = False
        self.saw_reply = self.completed = False
        self.events = []
        self.recorded = False
        self.end_at_stop = None
        self.reported_done = False
        self.stop_exception = None

    def summoned(self, shortcut, opened):
        pass

    def submitted(self, provider):
        self.provider = provider

    def dismissed(self, hidden):
        return False

    def agent_event(self, event, details):
        if event == "done":
            self.reported_done = True
            self.end_ok = matches(self.item["end"], details["window"], self.workspace_path)
        elif event == "stopped":
            self.cancelled = details["cancelled"]
            # The worker emits stopped before the result panel takes focus. Check
            # the real end condition even if the planner never announced completion.
            error = sys.exception()
            self.stop_exception = type(error).__name__ if error else None
            if not ax.session_locked():
                pid, app = ax.frontmost()
                if app != self.item["end"]["app"]:
                    self.end_at_stop = False
                else:
                    window = ax.read_window(pid, app)
                    if ax.frontmost()[0] == pid and not ax.session_locked():
                        try:
                            self.end_at_stop = matches(self.item["end"], window, self.workspace_path)
                        except OSError:
                            pass  # A disappearing fixture supplies no terminal observation.
            AppHelper.callAfter(self.finish)
        else:
            self.events.append(
                {
                    "event": event,
                    "number": details["number"],
                    "allowed": details.get("allowed"),
                    "action": hashlib.sha256(details["description"].encode()).hexdigest(),
                }
            )
            if self.mode == "cancel" and event == "proposal":
                AppHelper.callAfter(self.delegate.cancelRequest_, None)

    def finish(self):
        if self.recorded:
            return
        self.recorded = True
        if self.summon and not self.start_ok:
            print("No trial recorded. The captured window did not provide a valid start.", flush=True)
            AppHelper.callAfter(end_run)
            return
        drivers = []
        if self.navigation:
            drivers.append("navigation")
        if self.workspace:
            drivers.append("workspace approval")
        record = {
            "id": self.item["id"],
            "mode": self.mode,
            "approval_driver": "scripted " + " and ".join(drivers) if drivers else "manual",
            "provider": self.provider,
            "revision": corpus.source_revision(),
            "seal": corpus.seal("agent").read_text().strip(),
            "start": self.start_ok,
            "end": self.end_ok,
            "reported_done": self.reported_done,
            "end_at_stop": self.end_at_stop,
            "stop_exception": self.stop_exception,
            "cancelled": self.cancelled,
            "events": self.events,
            "time": time.time(),
        }
        corpus.write_once(OBS / (uuid.uuid4().hex + ".json"), json.dumps(record).encode())
        print("Saved live observation for", self.item["id"], self.mode, flush=True)
        AppHelper.callAfter(end_run)


def navigation_proposals(task, workspace=DEFAULT_WORKSPACE):
    choices = {
        "calculator": {"Open app · Calculator"},
        "calendar": {"Open app · Calendar"},
        "open-file": {f"Open file · {workspace / 'Clicky-source.txt'}", "Open app · TextEdit"},
        "finder-subfolder": {f"Open file · {workspace / 'Clicky material'}"},
        "finder-parent": {f"Open file · {workspace}", "Menu · Go > Enclosing Folder"},
        "safari-url": {
            "Open app · Safari",
            "Open url · https://example.org",
            "Open url · https://example.org/",
        },
        "chrome-switch": {"Shortcut · control+tab", "Switch tab · Example Domain"},
        "safari-switch": {"Shortcut · control+tab", "Switch tab · Example Domain"},
        "finder-info": {"Menu · File > Get Info"},
    }
    pane = {"settings-keyboard": "Keyboard", "settings-sound": "Sound", "settings-display": "Displays"}.get(
        task
    )
    if pane:
        return {
            "Open app · System Settings",
            f"Open settings · {pane}",
            f"Click · AXButton {pane}",
            f"Click · AXRow {pane}",
            f"Click · AXStaticText {pane}",
            f"Menu · View > {pane}",
        }
    return choices.get(task, set())


def fixture_command(task, description, workspace=DEFAULT_WORKSPACE):
    workspace = workspace.resolve()
    commands = {
        "terminal": ("touch", "Clicky-terminal.txt"),
        "make-folder": ("mkdir", "Clicky exports"),
        "duplicate": ("cp", "Clicky-copy.txt", "Clicky-backup.txt"),
        "rename": ("mv", "Clicky-rename.txt", "Clicky-reviewed.txt"),
    }
    if task not in commands:
        return False
    program, *names = commands[task]
    paths = [workspace / name for name in names]
    if any(path.is_symlink() for path in paths) or paths[-1].exists():
        return False
    if len(paths) == 2:
        if not paths[0].is_file():
            return False
        operation = "Copy file" if program == "cp" else "Move file"
        source, destination = (json.dumps(str(path), ensure_ascii=False) for path in paths)
        if description == f"{operation} · From {source}\nTo {destination}":
            return True
    if not description.startswith("Command · "):
        return False
    try:
        proposed = shlex.split(description.removeprefix("Command · "))
    except ValueError:
        return False
    return proposed == [program, *map(str, paths)]


def fixture_text_edit(pid, description, workspace=DEFAULT_WORKSPACE):
    if description != "Type · Ready for review\n":
        return False
    path = workspace / "Clicky-edit.txt"
    app = A.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if path.is_symlink() or not path.is_file() or not app or app.bundleIdentifier() != "com.apple.TextEdit":
        return False
    root = AX.AXUIElementCreateApplication(pid)
    window = ax.attr(root, "AXFocusedWindow")
    doc = A.NSURL.URLWithString_(str(ax.attr(window, "AXDocument") or ""))
    focus = ax.attr(root, "AXFocusedUIElement")
    if not doc or not doc.isFileURL() or doc.path() != str(path) or ax.attr(focus, "AXRole") != "AXTextArea":
        return False
    selection = ax.attr(focus, "AXSelectedTextRange")
    return bool(selection and AX.AXValueGetValue(selection, AX.kAXValueCFRangeType, None) == (True, (0, 0)))


def fixture_draft(description):
    prefix = "Compose mail · "
    if not description.startswith(prefix):
        return False
    try:
        fields = json.loads(description[len(prefix) :])
    except ValueError:
        return False
    mail = A.NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(A.NSURL.URLWithString_("mailto:"))
    return (
        isinstance(fields, dict)
        and not fields.keys() - {"to", "subject", "body"}
        and fields.get("subject") == "Clicky review request"
        and fields.get("to", "") == fields.get("body", "") == ""
        and mail is not None
        and A.NSBundle.bundleWithURL_(mail).bundleIdentifier() == "com.apple.mail"
    )


def descendants(view):
    yield view
    for child in view.subviews():
        yield from descendants(child)


class TaskDelegate(ClickyDelegate):
    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        if self.trial.summon:
            AppHelper.callAfter(self.summon_fixed_start)

    @objc.python_method
    def summon_fixed_start(self):
        if ax.session_locked():
            print("The Mac is locked. No window was inspected.", flush=True)
            self.trial.finish()
            return
        window = ax.read_window(*ax.frontmost())
        if matches(self.trial.item["start"], window, self.trial.workspace_path):
            self.ask_(self)
        else:
            print("The frontmost window does not match the fixed start.", flush=True)
            self.trial.finish()

    @objc.python_method
    def confirm_action(self, request, pid, app, description, decision):
        if not (self.trial.navigation or self.trial.workspace):
            return super().confirm_action(request, pid, app, description, decision)
        # Drive the real NSAlert only for exact, predeclared navigation on the
        # fixed task. Everything else goes through its actual Cancel button.
        self.expected_prompt = (f"Allow action in {app}?", description)
        task = self.trial.item["id"]
        permitted = self.trial.navigation and description in navigation_proposals(task, self.trial.workspace_path)
        if self.trial.workspace:
            if task == "draft":
                permitted = permitted or fixture_draft(description)
            elif task == "textedit":
                permitted = permitted or fixture_text_edit(pid, description, self.trial.workspace_path)
            else:
                permitted = permitted or fixture_command(task, description, self.trial.workspace_path)
        self.expected_button = "Allow Once" if permitted else "Cancel"
        timer = A.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "readConsent:", None, True
        )
        A.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, A.NSModalPanelRunLoopMode)
        try:
            super().confirm_action(request, pid, app, description, decision)
        finally:
            timer.invalidate()

    def readConsent_(self, timer):
        modal = A.NSApp.modalWindow()
        if not modal:
            return
        controls = list(descendants(modal.contentView()))
        labels = {str(v.stringValue()) for v in controls if isinstance(v, A.NSTextField)}
        if not all(value in labels for value in self.expected_prompt):
            return
        button = next(
            (v for v in controls if isinstance(v, A.NSButton) and v.title() == self.expected_button), None
        )
        if button:
            button.performClick_(None)

    @objc.python_method
    def settings_store(self):
        store = preferences("local.clicky.task-eval." + uuid.uuid4().hex)
        store.registerDefaults_(
            {"provider": self.trial.provider, "history": False, "passive": False, "spokenReplies": False}
        )
        self.trial.delegate = self
        return store

    @objc.python_method
    def show_context(self, ticket, future, pid, name, shortcut):
        super().show_context(ticket, future, pid, name, shortcut)
        if self.context is None or not matches(self.trial.item["start"], self.context.window, self.trial.workspace_path):
            if self.trial.summon:
                self.trial.finish()
                return
            self.panel.output.setString_(
                "Start state does not match. Restore the specified state and summon again."
            )
            return
        if matches(self.trial.item["end"], self.context.window, self.trial.workspace_path):
            if self.trial.summon:
                self.trial.finish()
                return
            self.panel.output.setString_(
                "The end condition already holds. Restore an unfinished start state."
            )
            return
        self.trial.start_ok = True
        workspace = self.trial.workspace_path
        self.panel.field.setStringValue_(self.trial.item["request"].replace("{workspace}", str(workspace)))
        self.submit_(None)

    @objc.python_method
    def finished(self, request, prompt, error):
        self.trial.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE, help="directory containing the disposable fixture files")
    parser.add_argument("provider", choices=("Anthropic", "OpenAI", "Local"))
    parser.add_argument("mode", choices=("run", "cancel"))
    parser.add_argument(
        "--summon",
        action="store_true",
        help="summon automatically if the frontmost window matches the fixed start",
    )
    parser.add_argument(
        "--verify-navigation",
        action="store_true",
        help="exercise native Allow Once only for exact, predeclared read-only navigation, cancelling other proposals",
    )
    parser.add_argument(
        "--verify-workspace",
        action="store_true",
        help="approve a verified local fixture command, text insertion, or empty unaddressed Mail draft",
    )
    args = parser.parse_args()
    corpus.verify("agent")
    row = next(row for row in corpus.rows("agent") if row["id"] == args.id)
    print(row["setup"], flush=True)
    if args.summon:
        print("Checking the fixed start before summon.", flush=True)
    else:
        print("Restore the start state, then press your Clicky shortcut (default ⌃⌥Space).", json.dumps(row["start"]), flush=True)
    trial = Trial(row, args.provider, args.mode, args.workspace.resolve())
    trial.summon = args.summon
    trial.navigation = args.verify_navigation
    trial.workspace = args.verify_workspace
    main(trial, delegate_type=TaskDelegate, argv=[])
    if args.summon and not trial.start_ok:
        raise SystemExit("The fixed start was unavailable.")
