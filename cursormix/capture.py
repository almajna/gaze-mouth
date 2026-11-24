"""Take a single in-memory snapshot of the display containing the focused window."""

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from threading import Event

import AppKit as A
import ApplicationServices as AX
import Quartz as Q
from PIL import Image

from . import ax


@dataclass
class Context:
    window: ax.Window
    display: tuple | None = None
    jpeg: bytes | None = None
    issue: str = ""
    marked: bool = False

    def prompt(self, request, conversation=()):
        payload = {"request": request, "screen": json.loads(self.description(request))}
        if conversation:
            payload["prior_turns"] = conversation
        return json.dumps(payload, ensure_ascii=False)

    def control_frame(self, index):
        node = self.window.elements[index]
        if not node.frame or min(node.frame[2:]) <= 0:
            return None
        clips = [self.display] if self.display else []
        if self.window.frame and node.role not in ("AXMenuBar", "AXMenuBarItem", "AXMenu", "AXMenuItem"):
            clips.append(self.window.frame)
        frame = ax.clipped_frame(node.frame, clips)
        return node.frame if frame == tuple(node.frame) else frame

    def description(self, request="", *, positions=False, targets_only=False):
        words = set(re.findall(r"\w+", request.casefold()))
        controls = self.window.inventory()
        viewport = self.display or self.window.frame
        if positions or targets_only:
            eligible = []
            for control in controls:
                frame = self.control_frame(control["id"])
                if targets_only and not frame:
                    continue
                if positions and frame and viewport:
                    x, y, w, h = frame
                    left, top, width, height = viewport
                    cx, cy = x + w / 2 - left, y + h / 2 - top
                    if 0 <= cx < width and 0 <= cy < height:
                        control["center_percent"] = [round(100 * cx / width, 1), round(100 * cy / height, 1)]
                eligible.append(control)
            controls = eligible

        # Keep the focused control and matching labels even in a very large browser window.
        def relevance(control):
            tokens = set(re.findall(r"\w+", control["label"].casefold()))
            return (control["focused"], len(words & tokens))

        controls.sort(key=relevance, reverse=True)
        chosen = []
        remaining = 10000 - len(self.window.selected_text or "")
        for control in controls:
            remaining -= len(control["label"]) + 55
            if remaining < 0:
                break
            chosen.append(control)
        screen = {
            "app": self.window.app,
            "window": self.window.title,
            **({"selected_text": self.window.selected_text} if self.window.selected_text is not None else {}),
            "controls": chosen,
            "capture_issue": self.issue,
            **(
                {"user_marks": "Amber marks on the image identify the subject of the request."}
                if self.marked
                else {}
            ),
        }
        if positions and viewport:
            screen["positions_relative_to"] = "display" if self.display else "window"
        return json.dumps(screen, ensure_ascii=False)


def display_for(window):
    screens = A.NSScreen.screens()
    mouse = Q.CGEventGetLocation(Q.CGEventCreate(None))
    x, y, w, h = window.frame or (mouse.x, mouse.y, 1, 1)
    best = None
    for screen in screens:
        identifier = int(screen.deviceDescription()["NSScreenNumber"])
        rect = Q.CGDisplayBounds(identifier)
        sx, sy = rect.origin
        sw, sh = rect.size
        area = max(0, min(x + w, sx + sw) - max(x, sx)) * max(0, min(y + h, sy + sh) - max(y, sy))
        if best is None or area > best[0]:
            best = area, identifier, (sx, sy, sw, sh), float(screen.backingScaleFactor())
    if best is None:
        raise RuntimeError("macOS reports no connected display.")
    return best[1:]


def sc_result(start):
    finished = Event()
    result = []

    def done(value, error):
        result.extend((value, error))
        finished.set()

    start(done)
    if not finished.wait(1.5):
        # A stalled system capture service stays stalled. Skip it for a while so summons stay quick.
        global kit_unavailable_until
        kit_unavailable_until = time.monotonic() + 300
        raise RuntimeError("ScreenCaptureKit did not answer.")
    if result[0] is None:
        raise RuntimeError("ScreenCaptureKit did not return an image or display.")
    return result[0]


kit_unavailable_until = 0.0


def system_screenshot(frame, window=None):
    """Capture with the built-in screencapture tool. It does not depend on ScreenCaptureKit's service."""
    x, y, w, h = frame
    target = None
    if window is not None and window.frame:
        # Show only the app's own window on an empty display, like an app-only capture.
        for info in Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly, 0) or ():
            b = info["kCGWindowBounds"]
            bounds = (b["X"], b["Y"], b["Width"], b["Height"])
            if info.get("kCGWindowOwnerPID") == window.pid and all(
                abs(p - q) < 2 for p, q in zip(bounds, window.frame)
            ):
                target = int(info["kCGWindowNumber"])
                break
    with tempfile.TemporaryDirectory(prefix="clicky-shot-") as directory:
        path = os.path.join(directory, "screen.jpg")
        argv = ["/usr/sbin/screencapture", "-x", "-t", "jpg"]
        argv += ["-o", "-l", str(target)] if target else ["-R", f"{x:.0f},{y:.0f},{w:.0f},{h:.0f}"]
        subprocess.run([*argv, path], check=True, timeout=6, capture_output=True)
        with Image.open(path) as shot:
            shot = shot.convert("RGB")
    ratio = min(1.0, 1600 / max(w, h))
    size = (round(w * ratio), round(h * ratio))
    if target:
        wx, wy, ww, wh = window.frame
        image = Image.new("RGB", size)
        part = shot.resize((max(1, round(ww * ratio)), max(1, round(wh * ratio))), Image.Resampling.LANCZOS)
        image.paste(part, (round((wx - x) * ratio), round((wy - y) * ratio)))
    else:
        image = shot.resize(size, Image.Resampling.LANCZOS)
    with BytesIO() as stream:
        image.save(stream, "JPEG", quality=82)
        return stream.getvalue()


class CaptureInterrupted(Exception):
    pass


def check_capture_allowed(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise CaptureInterrupted("Summon cancelled.")
    if ax.session_locked():
        raise CaptureInterrupted("Unlock your Mac to use screen context.")


def screen_kit(identifier, frame, scale, app_pid=None, *, cancelled=None):
    import ScreenCaptureKit as S

    check_capture_allowed(cancelled)
    content = sc_result(
        lambda done: (
            S.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                False, True, done
            )
        )
    )
    display = next(d for d in content.displays() if d.displayID() == identifier)
    if app_pid is None:
        own = [app for app in content.applications() if app.processID() == os.getpid()]
        selected = S.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
            display, own, []
        )
    else:
        app = next(app for app in content.applications() if app.processID() == app_pid)
        selected = S.SCContentFilter.alloc().initWithDisplay_includingApplications_exceptingWindows_(
            display, [app], []
        )
        if selected.respondsToSelector_("setIncludeMenuBar:"):
            selected.setIncludeMenuBar_(True)
    config = S.SCStreamConfiguration.alloc().init()
    ratio = min(scale, 1600 / max(frame[2:]))
    config.setWidth_(round(frame[2] * ratio))
    config.setHeight_(round(frame[3] * ratio))
    config.setShowsCursor_(False)
    check_capture_allowed(cancelled)
    image = sc_result(
        lambda done: S.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            selected, config, done
        )
    )
    check_capture_allowed(cancelled)
    rep = A.NSBitmapImageRep.alloc().initWithCGImage_(image)
    return bytes(
        rep.representationUsingType_properties_(
            A.NSBitmapImageFileTypeJPEG, {A.NSImageCompressionFactor: 0.82}
        )
    )


def on_summon(pid, name, cancelled=None, *, fallback=True, app_only=False):
    try:
        check_capture_allowed(cancelled)
        context = _capture_window(pid, name, cancelled, fallback, app_only)
        check_capture_allowed(cancelled)
        return context
    except CaptureInterrupted as error:
        return Context(ax.Window(0, ""), issue=str(error))


def _capture_window(pid, name, cancelled, fallback, app_only):
    context = Context(ax.read_window(pid, name))
    check_capture_allowed(cancelled)
    trusted = AX.AXIsProcessTrusted()
    if not Q.CGPreflightScreenCaptureAccess():
        context.issue = "Screen Recording is off, so there is no screenshot." + (
            "" if trusted else " Accessibility is off too, so window contents are missing."
        )
        return context
    if not trusted:
        context.issue = "Accessibility is off, so control names and positions are missing."
    identifier, frame, scale = display_for(context.window)
    context.display = frame
    try:
        if time.monotonic() < kit_unavailable_until:
            raise RuntimeError("ScreenCaptureKit is stalled.")
        context.jpeg = screen_kit(identifier, frame, scale, pid if app_only else None, cancelled=cancelled)
    except CaptureInterrupted:
        raise
    except Exception:  # noqa: BLE001 — bridge failures fall back to the independent capture backend
        check_capture_allowed(cancelled)
        if not fallback:
            context.issue = "ScreenCaptureKit is unavailable."
            return context
        try:
            context.jpeg = system_screenshot(frame, context.window if app_only else None)
        except (OSError, ValueError, subprocess.SubprocessError):
            context.issue = "The display could not be captured."
    return context
