"""Focused-window accessibility in global Quartz points."""

import time
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import AppKit as A
import ApplicationServices as AX
import Quartz as Q


def session_locked():
    return bool((Q.CGSessionCopyCurrentDictionary() or {}).get("CGSSessionScreenIsLocked"))


def frontmost():
    app = A.NSWorkspace.sharedWorkspace().frontmostApplication()
    return (int(app.processIdentifier()), str(app.localizedName())) if app else (0, "")


def attr(node, key):
    if node is None:
        return None
    status, value = AX.AXUIElementCopyAttributeValue(node, key, None)
    return value if status == 0 else None


def clipped_frame(frame, clips):
    x, y, w, h = frame
    right, bottom = x + w, y + h
    for clip in clips:
        if not clip:
            return None
        left, top, cw, ch = clip
        x, y = max(x, left), max(y, top)
        right, bottom = min(right, left + cw), min(bottom, top + ch)
        if x >= right or y >= bottom:
            return None
    return x, y, right - x, bottom - y


def bounds(node):
    result = []
    for name, kind in (("AXPosition", AX.kAXValueCGPointType), ("AXSize", AX.kAXValueCGSizeType)):
        value = attr(node, name)
        if value is None:
            return None
        ok, pair = AX.AXValueGetValue(value, kind, None)
        if not ok:
            return None
        result.extend(map(float, pair))
    return tuple(result) if min(result[2:]) > 0 else None


@dataclass
class Element:
    role: str
    label: str
    frame: tuple | None
    ref: object = field(default=None, repr=False)
    focused: bool = False
    selected: bool | None = None
    row: object = field(default=None, repr=False)

    def description(self):
        item = {"role": self.role, "label": self.label, "focused": self.focused}
        if self.selected is not None:
            item["selected"] = self.selected
        return item


@dataclass
class Window:
    pid: int
    app: str
    title: str = ""
    frame: tuple | None = None
    elements: list = field(default_factory=list)
    ref: object = field(default=None, repr=False)
    selected_text: str | None = None
    typing_anchor: tuple | None = field(default=None, repr=False)

    def inventory(self):
        return [{"id": i, **e.description()} for i, e in enumerate(self.elements)]

    def state(self, *, control_frames=True):
        return (
            self.pid,
            self.ref,
            self.title,
            self.frame,
            self.selected_text,
            self.typing_anchor,
            tuple((e.role, e.label, e.focused, e.selected, e.frame if control_frames else None) for e in self.elements),
        )


def finder_locations(window, request):
    """Resolve Finder's local file references for the current desktop task."""
    if window.app != "Finder":
        return {}
    parents = set()
    named = set()
    query = request.casefold()
    for element in window.elements:
        if not element.ref or not element.frame or element.role not in ("AXImage", "AXTextField"):
            continue
        name = attr(element.ref, "AXFilename")
        if not isinstance(name, str) or not name:
            continue
        url = attr(element.ref, "AXURL")
        if not isinstance(url, A.NSURL) or not url.isFileURL():
            continue
        resolved = url.filePathURL()
        if not resolved or not resolved.path():
            continue
        path = Path(resolved.path())
        parents.add(path.parent)
        if name.casefold() in query and path.name == name:
            named.add(str(path))
    locations = {"matching_paths": sorted(named)} if named else {}
    if len(parents) == 1:
        directory = parents.pop()
        if directory.name == window.title:
            locations["directory_containing_displayed_files"] = str(directory)
    return locations


def browser_tabs(window):
    """Identify tabs in the browser's own window chrome and keep their window-local AX ids."""
    if not window.ref:
        return {}
    app = A.NSRunningApplication.runningApplicationWithProcessIdentifier_(window.pid)
    if not app or app.bundleIdentifier() not in ("com.google.Chrome", "com.apple.Safari"):
        return {}
    tabs = {}
    for index, element in enumerate(window.elements):
        node = element.ref
        if element.role != "AXRadioButton" or attr(node, "AXSubrole") != "AXTabButton":
            continue
        parent = attr(node, "AXParent")
        in_group = False
        seen = set()
        while parent and parent != window.ref and parent not in seen and len(seen) < 64:
            seen.add(parent)
            role = attr(parent, "AXRole")
            if role in ("AXWebArea", "AXDocument", "AXWindow", "AXApplication"):
                break
            in_group |= role == "AXTabGroup" or (
                role == "AXOpaqueProviderGroup" and attr(parent, "AXSubrole") == "AXOpaqueProviderList"
            )
            parent = attr(parent, "AXParent")
        if in_group and parent == window.ref:
            title = attr(node, "AXTitle") or attr(node, "AXDescription")
            if isinstance(title, str) and title.strip():
                tabs[index] = {"title": title, "selected": bool(attr(node, "AXValue"))}
    return tabs


def menu_children(parent):
    children = attr(parent, "AXChildren") or ()
    submenu = next((child for child in children if attr(child, "AXRole") == "AXMenu"), None)
    return (attr(submenu, "AXChildren") or ()) if submenu is not None else children


def read_menu(window, path, cancel):
    """Read the commands exposed by the application's menu without activating it."""
    root = AX.AXUIElementCreateApplication(window.pid)
    AX.AXUIElementSetMessagingTimeout(root, 0.2)
    if (attr(root, "AXFocusedWindow") or attr(root, "AXMainWindow")) != window.ref:
        return {"issue": "The window changed. Inspect its current controls."}
    node = attr(root, "AXMenuBar")
    for name in filter(None, (part.strip() for part in path.split(">"))):
        if cancel.is_set():
            raise InterruptedError("Task cancelled.")
        node = next((child for child in menu_children(node) if attr(child, "AXTitle") == name), None)
        if node is None:
            return {"issue": "This menu path is unavailable."}
    items = []
    for child in menu_children(node):
        if cancel.is_set():
            raise InterruptedError("Task cancelled.")
        name = attr(child, "AXTitle")
        if name:
            items.append({"name": str(name)[:160], "enabled": attr(child, "AXEnabled")})
        if len(items) >= 80:
            break
    return {"menu": path, "items": items}


def selection_excerpt(text):
    """An empty highlight differs from a control that does not expose its selection."""
    if not isinstance(text, str):
        return None
    return text if len(text) <= 1600 else text[:1600] + "…"


def text_selection(control):
    """Read the full highlighted passage and its UTF-16 range from the same control."""
    text = attr(control, "AXSelectedText")
    value = attr(control, "AXSelectedTextRange")
    if isinstance(value, AX.AXValueRef):
        ok, span = AX.AXValueGetValue(value, AX.kAXValueCFRangeType, None)
        if ok:
            return text, tuple(span)
    return text, None


def typing_anchor(text, span):
    if not isinstance(text, str) and span is None:
        return None
    digest = sha256(text.encode("utf-8", "surrogatepass")).digest() if isinstance(text, str) else None
    return span, digest


TEXT_ROLES = ("AXTextField", "AXTextArea", "AXComboBox", "AXSearchField")


def read_window(pid, app):
    window = Window(pid, app)
    if session_locked():
        return Window(0, "")
    if not pid or not AX.AXIsProcessTrusted():
        return window
    root = AX.AXUIElementCreateApplication(pid)
    AX.AXUIElementSetMessagingTimeout(root, 0.15)
    node = attr(root, "AXFocusedWindow") or attr(root, "AXMainWindow")
    if node is None:
        return window
    window.ref = node
    window.title = str(attr(node, "AXTitle") or "")
    window.frame = bounds(node)
    for key, label in (
        ("AXCloseButton", "Close this window"),
        ("AXMinimizeButton", "Minimize this window"),
        ("AXFullScreenButton", "Full screen this window"),
    ):
        button = attr(node, key)
        if button is not None and bounds(button):
            window.elements.append(Element("AXButton", label, bounds(button), button))
    focus = attr(root, "AXFocusedUIElement")
    menu = attr(root, "AXMenuBar")
    # Closed menus can expose hundreds of off-screen items and exhaust the
    # window budget before an editor's active text field is reached.
    headings = list(attr(menu, "AXChildren") or ())
    queue = deque((entry, None, None) for entry in (node, *headings))
    if focus:
        queue.appendleft((focus, None, None))
    seen = set()
    deadline = time.monotonic() + 1.3
    while queue and len(seen) < 640 and time.monotonic() < deadline:
        child, row, selected = queue.popleft()
        if child in seen:
            if row is not None and child == focus and window.elements and window.elements[0].ref == focus:
                window.elements[0].row, window.elements[0].selected = row, selected
            continue
        seen.add(child)
        role = str(attr(child, "AXRole") or "")
        if attr(child, "AXSubrole") == "AXSecureTextField" or attr(child, "AXProtectedContent"):
            continue
        if child == focus:
            text, span = text_selection(child)
            window.selected_text = selection_excerpt(text)
            window.typing_anchor = typing_anchor(text, span)
        if role == "AXRow":
            row = child
            value = attr(child, "AXSelected")
            selected = bool(value) if value is not None else None
        parts = [attr(child, "AXTitle"), attr(child, "AXDescription"), attr(child, "AXValue")]
        label = " ".join(dict.fromkeys(str(x) for x in parts if isinstance(x, (str, int, float)) and str(x)))
        if not label and role in TEXT_ROLES:
            # An empty field has no words of its own. Keep it so a request can click and type in it.
            label = str(attr(child, "AXPlaceholderValue") or "") or "empty text field"
        if label or child == focus:
            window.elements.append(
                Element(role, label[:280], bounds(child), child, child == focus, selected, row)
            )
        if child not in headings:
            queue.extend(
                (descendant, row, selected) for descendant in (attr(child, "AXChildren") or ())[:256]
            )
    return Window(0, "") if session_locked() else window
