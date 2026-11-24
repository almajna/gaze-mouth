"""Execute one validated action against its observed application."""

import ctypes
import json
import os
import shlex
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import AppKit as A
import ApplicationServices as AX
import Quartz as Q

from . import ax
from .integrations import SERVICES
from .settings import installed_panes

KEYS = {
    "cmd+n": (45, Q.kCGEventFlagMaskCommand),
    "cmd+t": (17, Q.kCGEventFlagMaskCommand),
    "cmd+f": (3, Q.kCGEventFlagMaskCommand),
    "cmd+s": (1, Q.kCGEventFlagMaskCommand),
    "cmd+o": (31, Q.kCGEventFlagMaskCommand),
    "cmd+w": (13, Q.kCGEventFlagMaskCommand),
    "cmd+c": (8, Q.kCGEventFlagMaskCommand),
    "cmd+v": (9, Q.kCGEventFlagMaskCommand),
    "cmd+a": (0, Q.kCGEventFlagMaskCommand),
    "cmd+z": (6, Q.kCGEventFlagMaskCommand),
    "control+tab": (48, Q.kCGEventFlagMaskControl),
    "tab": (48, 0),
    "return": (36, 0),
    "escape": (53, 0),
}
KINDS = (
    "click",
    "switch_tab",
    "type",
    "menu",
    "shortcut",
    "open_app",
    "open_settings",
    "open_url",
    "open_file",
    "copy_file",
    "move_file",
    "command",
    *SERVICES,
    "done",
)

COMMANDS = {
    "pwd": "/bin/pwd",
    "date": "/bin/date",
    "ls": "/bin/ls",
    "wc": "/usr/bin/wc",
    "du": "/usr/bin/du",
    "file": "/usr/bin/file",
    "mkdir": "/bin/mkdir",
    "mv": "/bin/mv",
    "cp": "/bin/cp",
    "touch": "/usr/bin/touch",
}


class ActionUnavailable(ValueError):
    """The requested operation could not be dispatched against its observed target."""


def activate(pid, cancel):
    if cancel.is_set():
        raise InterruptedError("Cancelled.")
    if ax.frontmost()[0] == pid:
        return
    app = A.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if not app:
        raise ActionUnavailable("The target application closed.")
    app.activateWithOptions_(A.NSApplicationActivateIgnoringOtherApps)
    for _ in range(30):
        if cancel.wait(0.04):
            raise InterruptedError("Cancelled.")
        if ax.frontmost()[0] == pid:
            return
    raise ActionUnavailable("The target application could not be focused.")


def keyboard(pid, key, flags, text=None):
    for down in (True, False):
        event = Q.CGEventCreateKeyboardEvent(None, key, down)
        Q.CGEventSetFlags(event, flags)
        if text is not None:
            Q.CGEventKeyboardSetUnicodeString(event, len(text.encode("utf-16-le")) // 2, text)
        Q.CGEventPostToPid(pid, event)


def press(ref):
    if ref is None or AX.AXUIElementPerformAction(ref, "AXPress") != 0:
        raise ValueError("This control could not be pressed.")


def menu_path(pid, value, cancel):
    menu = ax.attr(AX.AXUIElementCreateApplication(pid), "AXMenuBar")
    names = [part.strip() for part in value.split(">")]
    if len(names) < 2 or not all(names):
        raise ActionUnavailable("Specify a menu and a command.")
    opened = None
    try:
        for i, name in enumerate(names):
            if cancel.is_set() or ax.frontmost()[0] != pid:
                raise InterruptedError("Menu selection cancelled.")
            match = next((c for c in ax.menu_children(menu) if ax.attr(c, "AXTitle") == name), None)
            enabled = ax.attr(match, "AXEnabled")
            if match is None or (enabled is not None and not enabled):
                raise ActionUnavailable("The menu command is missing or disabled: " + name)
            if 0 < i < len(names) - 1 and not any(
                ax.attr(child, "AXRole") == "AXMenu" for child in (ax.attr(match, "AXChildren") or ())
            ):
                raise ActionUnavailable("This item does not expose a submenu: " + name)
            if cancel.is_set() or ax.frontmost()[0] != pid:
                raise InterruptedError("Menu selection cancelled.")
            press(match)
            if i == 0:
                opened = match
            menu = match
            if i + 1 < len(names):
                cancel.wait(0.1)
        opened = None
    finally:
        if opened is not None:
            for child in ax.attr(opened, "AXChildren") or ():
                if ax.attr(child, "AXRole") == "AXMenu":
                    AX.AXUIElementPerformAction(child, "AXCancel")


def file_operands(argv):
    # Inspect only forms whose switches do not take their own path arguments.
    switches = {"cp": "finpv", "mv": "finv", "mkdir": "pv", "touch": "acm"}
    if argv[0] not in switches:
        return []
    words = argv[1:]
    while words and words[0].startswith("-"):
        option, words = words[0], words[1:]
        if option == "--":
            break
        if len(option) < 2 or not set(option[1:]) <= set(switches[argv[0]]):
            return []
    return [Path.home() / word for word in words if word]


def regular_contents_match(source, target, cancel):
    # Nonblocking opens also let us reject a path replaced by a FIFO before verification.
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW

    def open_file(path):
        return os.fdopen(os.open(path, flags), "rb")

    with open_file(source) as left, open_file(target) as right:
        first, second = os.fstat(left.fileno()), os.fstat(right.fileno())
        if not all(stat.S_ISREG(s.st_mode) for s in (first, second)):
            return None
        if first.st_size != second.st_size:
            return False
        remaining = first.st_size
        while remaining:
            if cancel.is_set():
                raise InterruptedError("Verification cancelled.")
            chunk = left.read(min(1024 * 1024, remaining))
            if not chunk or chunk != right.read(len(chunk)):
                return False
            remaining -= len(chunk)
        for path, before in ((source, first), (target, second)):
            now = path.lstat()
            if not os.path.samestat(now, before) or (now.st_size, now.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                return None
        return True


def same_file_version(before, after):
    return (os.path.samestat(before, after) and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns)


@dataclass(frozen=True)
class FileTransfer:
    source: Path
    destination: Path
    original: os.stat_result
    parent: os.stat_result

    @classmethod
    def prepare(cls, value):
        paths = json.loads(value)
        if not isinstance(paths, dict) or paths.keys() != {"source", "destination"}:
            raise ValueError("Specify the source and destination filenames.")
        checked = []
        for key in ("source", "destination"):
            name = paths[key]
            if not isinstance(name, str) or not Path(name).is_absolute():
                raise ValueError("Both file paths must be absolute.")
            path = Path(name)
            if not path.name or path.name in (".", ".."):
                raise ValueError("Choose a file with a name.")
            checked.append(path.parent.resolve(strict=True) / path.name)
        source, destination = checked
        original = source.lstat()
        if not stat.S_ISREG(original.st_mode):
            raise ValueError("Choose a regular source file.")
        if os.path.lexists(destination):
            raise ValueError("The destination already exists. Choose a new filename.")
        parent = destination.parent.stat()
        if not stat.S_ISDIR(parent.st_mode):
            raise ValueError("Choose an existing destination folder.")
        return cls(source, destination, original, parent)

    def preview(self):
        # Escape line breaks and quotes inside names so neither path can add another field.
        return (f"From {json.dumps(str(self.source), ensure_ascii=False)}\n"
                f"To {json.dumps(str(self.destination), ensure_ascii=False)}")

    def check_paths(self, cancel):
        current = self.source.lstat()
        if (
            not same_file_version(self.original, current)
            or not os.path.samestat(self.parent, self.destination.parent.stat())
        ):
            raise ActionUnavailable("The source file or destination folder changed after the preview.")
        if os.path.lexists(self.destination):
            raise ActionUnavailable("The destination is occupied. Its contents were preserved.")
        if cancel.is_set():
            raise InterruptedError("File operation cancelled.")

    def execute(self, kind, cancel):
        self.check_paths(cancel)
        move = kind == "move_file"
        if not move and kind != "copy_file":
            raise ValueError("Unknown file operation.")
        if move and self.original.st_dev == self.parent.st_dev:
            rename_exclusive(self.source, self.destination)
            if not same_file_version(self.original, self.destination.lstat()):
                raise ValueError("The file changed during the move. Check the destination.")
        else:
            # Keep partial copies private. Publish only the fully verified
            # file, with one filesystem operation that refuses occupied names.
            with tempfile.TemporaryDirectory(dir=self.destination.parent, prefix=".clicky-copy-") as directory:
                staged = Path(directory) / "copy"
                manager = A.NSFileManager.defaultManager()
                ok, error = manager.copyItemAtURL_toURL_error_(
                    A.NSURL.fileURLWithPath_(str(self.source)), A.NSURL.fileURLWithPath_(str(staged)), None
                )
                if not ok:
                    raise ValueError(str(error.localizedDescription()))
                if regular_contents_match(self.source, staged, cancel) is not True:
                    raise ValueError("The source changed or the copy could not be verified.")
                published = staged.lstat()
                self.check_paths(cancel)
                rename_exclusive(staged, self.destination)
            if not move:
                return f"Copied {self.source} to {self.destination}. The file contents match ({published.st_size} bytes)."
            if cancel.is_set():
                raise InterruptedError("The copy finished before cancellation. The original was retained.")
            # The two volumes cannot share a rename. First take ownership of
            # the actual source entry, then verify both identities before unlink.
            held = self.source.with_name(".clicky-move-" + uuid4().hex)
            rename_exclusive(self.source, held)
            try:
                originals = ((held, self.original), (self.destination, published))
                for path, expected in originals:
                    info = path.lstat()
                    if not same_file_version(expected, info):
                        raise ValueError("A file changed during the move. Both copies were retained.")
                if cancel.is_set():
                    raise InterruptedError("The copy finished before cancellation. The original was retained.")
                held.unlink()
            except (OSError, ValueError):
                self.restore_source(held)
                raise
        if os.path.lexists(self.source):
            raise ValueError("The destination was moved into place but the original path is occupied.")
        return f"Moved {self.source} to {self.destination} ({self.original.st_size} bytes). The original path is absent."

    def restore_source(self, held):
        try:
            rename_exclusive(held, self.source)
        except OSError as exc:
            raise ValueError(f"The source could not be restored at its old name. The file is at {held}.") from exc


def rename_exclusive(source, destination):
    # Darwin RENAME_EXCL. FileManager's existence check alone can race
    # another simultaneous move to the same initially vacant name.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def terminal(value, cancel):
    # Run a confirmed local utility directly. A shell is unnecessary for these desktop tasks.
    argv = shlex.split(value)
    if not argv or argv[0] not in COMMANDS:
        raise ValueError("Supported terminal utilities: " + ", ".join(COMMANDS))
    operands = file_operands(argv)
    original = {}
    for path in operands[:8]:
        try:
            original[path] = path.lstat()
        except FileNotFoundError:
            original[path] = None
        except OSError:
            pass
    destinations = []
    if argv[0] in ("cp", "mv") and len(operands) >= 2:
        destination = operands[-1]
        into_folder = destination.is_dir()
        for source in operands[:-1][:8]:
            info = original.get(source)
            if info and (stat.S_ISREG(info.st_mode) or (argv[0] == "mv" and stat.S_ISDIR(info.st_mode))):
                destinations.append((source, destination / source.name if into_folder else destination))
    with tempfile.TemporaryFile() as out:
        process = subprocess.Popen(
            [COMMANDS[argv[0]], *argv[1:]],
            cwd=Path.home(),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                if cancel.wait(0.1):
                    raise InterruptedError("Command cancelled.")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        out.seek(0)
        result = out.read(1800).decode(errors="replace")
        if process.returncode:
            raise ValueError("Command failed. " + result)
        if argv[0] in ("mkdir", "mv", "cp", "touch"):
            facts = []
            for operand in operands[:8]:
                try:
                    info = operand.lstat()
                except FileNotFoundError:
                    facts.append(f"{operand}: absent")
                except OSError:
                    continue
                else:
                    if stat.S_ISREG(info.st_mode):
                        facts.append(f"{operand}: regular file, {info.st_size} bytes")
                    elif stat.S_ISDIR(info.st_mode):
                        facts.append(f"{operand}: directory")
                    if operand in original and argv[0] in ("touch", "mkdir"):
                        if original[operand] is None:
                            facts.append(f"{operand} was absent before the command and now exists.")
                        else:
                            facts.append(f"{operand} already existed before the command.")
            for source, target in destinations:
                try:
                    info = target.lstat()
                    if stat.S_ISREG(info.st_mode):
                        facts.append(f"Destination file {target}: {info.st_size} bytes")
                        if argv[0] == "cp":
                            match = regular_contents_match(source, target, cancel)
                            if match is not None:
                                relation = "match" if match else "differ from"
                                facts.append(f"Destination contents {relation} {source}.")
                    if argv[0] == "mv" and not os.path.lexists(source):
                        facts.append(f"Original path {source} is absent.")
                        if os.path.samestat(original[source], info):
                            facts.append(f"Destination {target} has the same filesystem identity as the original {source}.")
                except InterruptedError:
                    raise
                except OSError:
                    continue
            return "\n".join(filter(None, ["Command exited with status 0.", result, *facts]))
        return result or "Command exited with status 0."


def perform(kind, value, index, window, cancel):
    if cancel.is_set():
        raise InterruptedError("Cancelled.")
    if kind in ("copy_file", "move_file"):
        return value.execute(kind, cancel)
    if kind == "command":
        return terminal(value, cancel)
    if kind in SERVICES:
        draft = SERVICES[kind].prepare(json.loads(value))
        if not A.NSWorkspace.sharedWorkspace().openURL_(A.NSURL.URLWithString_(draft)):
            raise ValueError("macOS could not open the draft in a mail application.")
        return "Opened an email draft for review."
    if kind == "open_app":
        if not value or value.startswith("-") or "/" in value:
            raise ValueError("Expected an installed application name.")
        subprocess.run(["/usr/bin/open", "-a", value], check=True, timeout=8)
    elif kind == "open_settings":
        identifier = installed_panes().get(value)
        if not identifier:
            raise ActionUnavailable("Choose a pane from the installed System Settings list.")
        if cancel.is_set() or ax.session_locked():
            raise InterruptedError("Settings navigation cancelled.")
        try:
            subprocess.run(["/usr/bin/open", "-b", "com.apple.systempreferences",
                            "x-apple.systempreferences:" + identifier], check=True, timeout=8)
        except (OSError, subprocess.SubprocessError) as error:
            raise ActionUnavailable("macOS could not open the requested settings pane.") from error
        return "Requested the settings pane. Check which pane is visible."
    elif kind == "open_file":
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.exists():
            raise ValueError("Expected an existing absolute file path.")
        if not A.NSWorkspace.sharedWorkspace().openURL_(A.NSURL.fileURLWithPath_(str(path))):
            raise ValueError("macOS could not open that file.")
    elif kind == "open_url":
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError("Expected an HTTP or HTTPS web address.")
        if not A.NSWorkspace.sharedWorkspace().openURL_(A.NSURL.URLWithString_(value)):
            raise ValueError("macOS could not open that address.")
    else:
        activate(window.pid, cancel)
        current = ax.read_window(window.pid, window.app)
        if current.ref != window.ref or current.title != window.title or cancel.is_set():
            raise ActionUnavailable("The active window changed before the action.")
        if kind == "switch_tab":
            observed = window.elements[index]
            match = next(
                (
                    i
                    for i, tab in ax.browser_tabs(current).items()
                    if current.elements[i].ref == observed.ref and tab["title"] == value
                ),
                None,
            )
            if match is None:
                raise ActionUnavailable("That browser tab changed before I could select it.")
            if cancel.is_set() or ax.frontmost()[0] != window.pid:
                raise InterruptedError("Tab selection cancelled.")
            press(current.elements[match].ref)
        elif kind == "click":
            element = window.elements[index]
            live = next(
                (e for e in current.elements if e.ref == element.ref and e.label == element.label), None
            )
            if live is None or live.row != element.row:
                raise ActionUnavailable("The selected control changed before the click.")
            if (
                live.role in ("AXRow", "AXCell", "AXStaticText", "AXUnknown")
                and live.row is not None
                and AX.AXUIElementIsAttributeSettable(live.row, "AXSelected", None) == (0, True)
            ):
                if AX.AXUIElementSetAttributeValue(live.row, "AXSelected", True) != 0:
                    raise ValueError("The row could not be selected.")
            elif live.role in ax.TEXT_ROLES:
                # Clicking in a field puts the insertion point there.
                if AX.AXUIElementSetAttributeValue(live.ref, "AXFocused", True) != 0:
                    press(live.ref)
            else:
                press(live.ref)
        elif kind == "type":
            root = AX.AXUIElementCreateApplication(window.pid)
            focus = ax.attr(root, "AXFocusedUIElement")
            if not focus or ax.attr(focus, "AXSubrole") == "AXSecureTextField" or ax.attr(focus, "AXProtectedContent"):
                raise ActionUnavailable("Select an editable, non-password field first.")
            old_focus = next((e.ref for e in window.elements if e.focused), None)
            if old_focus is None or focus != old_focus:
                raise ActionUnavailable("The focused field changed before typing.")
            if ax.typing_anchor(*ax.text_selection(focus)) != window.typing_anchor:
                raise ActionUnavailable("The insertion point or highlighted text changed before typing.")
            if cancel.is_set() or ax.session_locked():
                raise InterruptedError("Typing cancelled.")
            keyboard(window.pid, 0, 0, value)
        elif kind == "shortcut":
            if value.lower() not in KEYS:
                raise ValueError("That keyboard shortcut is unavailable.")
            keyboard(window.pid, *KEYS[value.lower()])
        elif kind == "menu":
            menu_path(window.pid, value, cancel)
        else:
            raise ValueError("Unknown desktop action.")
    return "Action performed."
