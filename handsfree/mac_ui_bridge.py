"""Small macOS accessibility, coordinate, and gaze-menu helpers.

Coordinates are logical screen points, not backing pixels. AX boxes and gaze
positions use the Quartz top-left global coordinate system. Accessibility
permission must be granted to the host process for window indexing.
"""

import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import AppKit
import ApplicationServices as AX
import Quartz as Q
import objc


def _request_capture_access(av, media):
    answered = threading.Event()

    def completed(granted):
        answered.set()

    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(media, completed)
    deadline = time.monotonic() + 60
    while not answered.is_set() and time.monotonic() < deadline:
        AppKit.NSRunLoop.currentRunLoop().runUntilDate_(AppKit.NSDate.dateWithTimeIntervalSinceNow_(.05))
    return av.AVCaptureDevice.authorizationStatusForMediaType_(media)


def microphone_authorized(request=False):
    import AVFoundation as av

    media = av.AVMediaTypeAudio
    status = av.AVCaptureDevice.authorizationStatusForMediaType_(media)
    if status == av.AVAuthorizationStatusNotDetermined and request:
        status = _request_capture_access(av, media)
    return status == av.AVAuthorizationStatusAuthorized


def check_macos_permissions(request=False, microphone=True):
    """Check all required grants without opening a camera or microphone."""
    settings = "System Settings → Privacy & Security"
    issues = []
    if not AX.AXIsProcessTrusted():
        issues.append(f"Accessibility: enable your launching terminal in {settings} → Accessibility.")
    try:
        import AVFoundation as av
    except ImportError:
        issues.append("Camera and Microphone: install pyobjc-framework-AVFoundation in this Python environment.")
    else:
        for label, media in (("Camera", av.AVMediaTypeVideo), ("Microphone", av.AVMediaTypeAudio)):
            if label == "Microphone" and not microphone:
                continue
            status = av.AVCaptureDevice.authorizationStatusForMediaType_(media)
            if status == av.AVAuthorizationStatusNotDetermined and request:
                print(f"{label}: choose Allow in the macOS permission dialog.", file=sys.stderr, flush=True)
                status = _request_capture_access(av, media)
            if status == av.AVAuthorizationStatusRestricted:
                issues.append(f"{label}: a system restriction blocks access. Ask your Mac administrator to allow it.")
            elif status == av.AVAuthorizationStatusNotDetermined:
                issues.append(f"{label}: launch the agent from your terminal and approve the macOS permission dialog.")
            elif status != av.AVAuthorizationStatusAuthorized:
                issues.append(f"{label}: enable your launching terminal in {settings} → {label}, then reopen it.")
    for issue in issues:
        print(issue, file=sys.stderr)
    return not issues


def permission_status():
    import AVFoundation as av
    return {
        "Accessibility": bool(AX.AXIsProcessTrusted()),
        "Camera": av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeVideo) == av.AVAuthorizationStatusAuthorized,
        "Microphone": av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeAudio) == av.AVAuthorizationStatusAuthorized,
        "Screen Recording": bool(Q.CGPreflightScreenCaptureAccess()),
    }


def unwrap_ax(result, success=0):
    """Return the value of a successful PyObjC (status, value) result."""
    if isinstance(result, tuple) and len(result) == 2 and result[0] == success:
        return result[1]
    return None


def primary_height():
    screens = AppKit.NSScreen.screens()
    if not screens:
        raise RuntimeError("No screens are available")
    return screens[0].frame().size.height


def cocoa_to_quartz(x, y, height=None):
    """Convert a global Cocoa point to a global Quartz point, in points."""
    h = primary_height() if height is None else height
    return x, h - y


def quartz_to_cocoa(x, y, height=None):
    return cocoa_to_quartz(x, y, height)


@dataclass(frozen=True)
class IndexedElement:
    element: object
    role: str
    bounds: tuple  # x, y, width, height in global AX points
    title: str = ""

    @property
    def center(self):
        x, y, w, h = self.bounds
        return x + w / 2, y + h / 2


class WindowIndexer:
    """Index visible interactive descendants of the focused window in a worker.

    Call refresh after focus or window content changes. PIDs are cached for
    this instance; call forget_pid when observing an application terminate to
    avoid retaining a reused PID. Accessibility failures produce an empty or
    partial index, rather than an exception.
    """
    ROLES = frozenset(("AXButton", "AXLink", "AXTextField", "AXTextArea", "AXSearchField",
                       "AXTab", "AXMenuItem", "AXCheckBox", "AXComboBox", "AXRadioButton",
                       "AXPopUpButton", "AXSwitch", "AXSlider"))
    CHROMIUM_NAMES = ("chrome", "chromium", "electron", "edge", "brave",
                      "opera", "vivaldi", "arc")

    def __init__(self):
        self._enabled_pids = set()
        self.elements = []
        self.snapshot = (None, "", "", (), ())
        self.generation = 0
        self.serial = 0
        self.pending = None
        self.last_refresh = -math.inf
        self.focus_pending = None
        self.last_focus_check = self.focus_at = -math.inf
        self.observers = []
        self.executor = None

    def invalidate(self, *_):
        self.generation += 1
        self.serial += 1
        self.snapshot = (None, "", "", (), ())
        self.elements = []
        self.last_refresh = self.last_focus_check = self.focus_at = -math.inf

    def start(self):
        self.executor = ThreadPoolExecutor(2, thread_name_prefix="AX window")
        self.notifications = AppKit.NSWorkspace.sharedWorkspace().notificationCenter()
        self.observers.append(self.notifications.addObserverForName_object_queue_usingBlock_(
            AppKit.NSWorkspaceDidActivateApplicationNotification, None,
            AppKit.NSOperationQueue.mainQueue(), self.invalidate))
        self.poll()

    def close(self):
        for observer in self.observers:
            self.notifications.removeObserver_(observer)
        self.observers.clear()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if self.focus_pending is not None and self.focus_pending.done():
            pending, self.focus_pending = self.focus_pending, None
            generation, captured, focus = pending.result()
            if generation == self.generation:
                if (focus[0], focus[4]) != (self.snapshot[0], self.snapshot[4]):
                    self.invalidate()
                    self.publish(self.generation, focus)
                self.focus_at = captured
        if self.pending is not None and self.pending.done():
            pending, self.pending = self.pending, None
            self.publish(*pending.result())
        if self.executor is not None and self.pending is None and now - self.last_refresh >= .5:
            self.last_refresh = now
            self.pending = self.executor.submit(self.scan, self.generation)
        if self.executor is not None and self.focus_pending is None and now - self.last_focus_check >= .1:
            self.last_focus_check = now
            self.focus_pending = self.executor.submit(self.read_focus, self.generation)

    def read_focus(self, generation):
        with objc.autorelease_pool():
            app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
            if app is None:
                return generation, time.monotonic(), (None, "", "", (), ())
            pid = app.processIdentifier()
            root = AX.AXUIElementCreateApplication(pid)
            AX.AXUIElementSetMessagingTimeout(root, .06)
            window = self._attr(root, "AXFocusedWindow")
            title = str(self._attr(window, "AXTitle") or "") if window is not None else ""
            identity = (window, self._bounds(window)) if window is not None else ()
            return generation, time.monotonic(), (pid, str(app.localizedName() or ""), title, (), identity)

    def publish(self, generation, snapshot):
        if generation != self.generation:
            return
        self.snapshot = snapshot
        self.elements = snapshot[3]
        self.serial += 1

    def forget_pid(self, pid):
        self._enabled_pids.discard(pid)

    def enable_chromium(self, pid, app_element, app_name):
        """Attempt manual accessibility once per matching PID, including failures."""
        if pid in self._enabled_pids or not any(
            name in (app_name or "").lower() for name in self.CHROMIUM_NAMES
        ):
            return
        self._enabled_pids.add(pid)
        try:
            # Unsupported is an ordinary AX error status, not an exception.
            AX.AXUIElementSetAttributeValue(app_element, "AXManualAccessibility", True)
        except Exception:
            pass

    @staticmethod
    def _attr(element, name):
        try:
            return unwrap_ax(AX.AXUIElementCopyAttributeValue(element, name, None))
        except Exception:
            return None

    @classmethod
    def _bounds(cls, element):
        pairs = []
        for name, kind in (("AXPosition", AX.kAXValueCGPointType),
                           ("AXSize", AX.kAXValueCGSizeType)):
            value = cls._attr(element, name)
            if value is None:
                return None
            try:
                pair = unwrap_ax(AX.AXValueGetValue(value, kind, None), success=True)
                a, b = pair
                pairs.extend((float(a), float(b)))
            except (TypeError, ValueError, OverflowError):
                return None
            except Exception:
                return None
        if not all(math.isfinite(v) for v in pairs) or min(pairs[2:]) <= 0:
            return None
        return tuple(pairs)

    def scan(self, generation):
        with objc.autorelease_pool():
            return self.scan_window(generation)

    def scan_window(self, generation):
        elements = []
        pid, name, window_title = None, "", ""
        window_key = ()
        try:
            app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
            if app is None:
                return generation, (pid, name, window_title, (), window_key)
            pid = app.processIdentifier()
            name = str(app.localizedName() or "")
            root = AX.AXUIElementCreateApplication(pid)
            AX.AXUIElementSetMessagingTimeout(root, 0.1)
            self._enabled_pids.intersection_update({pid})
            self.enable_chromium(pid, root, f"{name} {app.bundleIdentifier() or ''}")
            window = self._attr(root, "AXFocusedWindow")
            if window is None:
                return generation, (pid, name, window_title, (), window_key)
            window_title = str(self._attr(window, "AXTitle") or "")
            window_key = (window, self._bounds(window))
            clip = window_key[1]
            # Cap visited nodes as well as results, keeping large trees bounded.
            stack = [(window, 0)]
            visited = 0
            deadline = time.monotonic() + .25
            while stack and visited < 1200 and len(elements) < 800 and time.monotonic() < deadline:
                node, depth = stack.pop()
                visited += 1
                role = self._attr(node, "AXRole")
                if role in self.ROLES and self._attr(node, "AXEnabled") is not False:
                    bounds = self._bounds(node)
                    if bounds:
                        element = IndexedElement(node, role, bounds,
                                                 str(self._attr(node, "AXTitle") or self._attr(node, "AXDescription") or ""))
                        x, y = element.center
                        if clip is None or (clip[0] <= x <= clip[0] + clip[2] and clip[1] <= y <= clip[1] + clip[3]):
                            elements.append(element)
                if depth < 18:
                    children = self._attr(node, "AXChildren")
                    if children:
                        stack.extend((child, depth + 1) for child in reversed(tuple(children)[:1200]))
        except Exception:
            pass
        return generation, (pid, name, window_title, tuple(elements), window_key)

    def refresh(self):
        self.publish(*self.scan(self.generation))
        return self.elements

    def lookup_at_gaze(self, x, y, radius=70):
        """Return the nearest indexed box within radius points, or None."""
        if radius < 0:
            raise ValueError("radius must be nonnegative")
        def distance(item):
            bx, by, w, h = item.bounds
            return math.hypot(max(bx - x, 0, x - bx - w),
                              max(by - y, 0, y - by - h))
        nearest = min(self.elements, key=distance, default=None)
        return nearest if nearest is not None and distance(nearest) <= radius else None

    def snap(self, point, radius=220):
        nearest = min(self.elements, key=lambda item: math.dist(point, item.center), default=None)
        return nearest if nearest is not None and math.dist(point, nearest.center) <= radius else None

    def loupe_targets(self, loupe):
        cells = {}
        for item in self.elements:
            cell = loupe.cell(*item.center)
            if cell is None:
                continue
            center = loupe.cell_center(cell)
            if cell not in cells or math.dist(center, item.center) < math.dist(center, cells[cell].center):
                cells[cell] = item
        return cells


class TrackpadHandoffTracker:
    """Track physical cursor samples; ignore samples caused by your own warps.

    After the last physical movement, alpha remains zero for 2.5 seconds,
    then rises linearly over recovery_seconds. Feed actual event positions via
    update; polling alone cannot distinguish physical and programmatic moves.
    """
    def __init__(self, recovery_seconds=0.5, clock=time.monotonic):
        if recovery_seconds <= 0:
            raise ValueError("recovery_seconds must be positive")
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        self._position = None
        self._last_movement = None

    def update(self, x, y, *, physical=True, now=None):
        now = self.clock() if now is None else now
        position = (x, y)
        if physical and self._position is not None and position != self._position:
            self._last_movement = now
        self._position = position
        return self.alpha(now)

    def alpha(self, now=None):
        now = self.clock() if now is None else now
        if self._last_movement is None:
            return 1.0
        return max(0.0, min(1.0, (now - self._last_movement - 2.5) / self.recovery_seconds))

    @property
    def hud_alpha(self):
        return self.alpha()


class RadialRingMenu:
    """Six clockwise slices, with Copy centered on the rightward direction.

    Angles use screen coordinates (positive y down). The exact center has no
    direction and returns None. No radial limit is imposed on hit testing.
    """
    ITEMS = ("Copy", "Paste", "Summarize", "Back", "New Tab", "Close Tab")

    def __init__(self, center_x, center_y):
        self.center_x, self.center_y = center_x, center_y
        self.scale = 1.

    def hit_test(self, gaze_x, gaze_y):
        dx, dy = gaze_x - self.center_x, gaze_y - self.center_y
        if dx == 0 and dy == 0:
            return None
        angle = (math.degrees(math.atan2(dy, dx)) + 30) % 360
        return self.ITEMS[int(angle // 60)]


class ElasticTether:
    """Layer-backed Bézier and short-lived sparks, in Quartz screen points."""

    def __init__(self, root, height, reduce_motion=False, origin=(0, 0)):
        self.height, self.reduce_motion = height, reduce_motion
        self.origin = origin
        self.color = AppKit.NSColor.systemTealColor().CGColor()
        self.shape = Q.CAShapeLayer.layer()
        self.shape.setFrame_(root.bounds())
        self.shape.setFillColor_(None)
        self.shape.setStrokeColor_(self.color)
        self.shape.setLineCap_("round")
        self.shape.setShadowColor_(self.color)
        self.shape.setShadowRadius_(7.)
        self.shape.setShadowOpacity_(.85)
        root.addSublayer_(self.shape)
        self.emitter = Q.CAEmitterLayer.layer()
        self.emitter.setRenderMode_("additive")
        self.emitter.setBirthRate_(0.)
        stamp = Q.CGBitmapContextCreate(None, 8, 8, 8, 32, Q.CGColorSpaceCreateDeviceRGB(),
                                       Q.kCGImageAlphaPremultipliedLast)
        Q.CGContextSetRGBFillColor(stamp, .45, .95, 1., 1.)
        Q.CGContextFillEllipseInRect(stamp, Q.CGRectMake(0, 0, 8, 8))
        cell = Q.CAEmitterCell.emitterCell()
        cell.setContents_(Q.CGBitmapContextCreateImage(stamp))
        cell.setBirthRate_(190.)
        cell.setLifetime_(.20)
        cell.setVelocity_(90.)
        cell.setVelocityRange_(35.)
        cell.setEmissionRange_(math.tau)
        cell.setAlphaSpeed_(-5.)
        self.emitter.setEmitterCells_([cell])
        root.addSublayer_(self.emitter)

    def local(self, point):
        return tuple(v - o for v, o in zip(quartz_to_cocoa(*point, height=self.height), self.origin))

    def draw(self, start, end, tension=0.):
        Q.CATransaction.begin()
        Q.CATransaction.setDisableActions_(True)
        try:
            if start is None or end is None:
                self.shape.removeAnimationForKey_("tetherPath")
                self.shape.setPath_(None)
                return
            x, y = self.local(start)
            ex, ey = self.local(end)
            dx, dy = ex - x, ey - y
            length = math.hypot(dx, dy)
            tension = max(0., min(1., tension))
            bow = 0. if self.reduce_motion else min(40., length / 4) * (1 - tension)
            bx, by = (-dy * bow / length, dx * bow / length) if length else (0., 0.)
            path = Q.CGPathCreateMutable()
            Q.CGPathMoveToPoint(path, None, x, y)
            Q.CGPathAddCurveToPoint(path, None, x + dx / 3 + bx, y + dy / 3 + by,
                                    ex - dx / 3 + bx, ey - dy / 3 + by, ex, ey)
            # Continue from what the compositor is showing when a new camera
            # sample arrives, including when it interrupts a previous update.
            previous = None
            if not self.reduce_motion and self.shape.path() is not None:
                presentation = self.shape.presentationLayer()
                previous = presentation.path() if presentation is not None else self.shape.path()
            self.shape.setLineWidth_(1.5 + tension)
            self.shape.setPath_(path)
            if previous is not None:
                follow = Q.CABasicAnimation.animationWithKeyPath_("path")
                follow.setFromValue_(previous)
                follow.setToValue_(path)
                follow.setDuration_(1 / 30)
                self.shape.addAnimation_forKey_(follow, "tetherPath")
        finally:
            Q.CATransaction.commit()

    def burst(self, point):
        Q.CATransaction.begin()
        Q.CATransaction.setDisableActions_(True)
        try:
            self.emitter.setEmitterPosition_(self.local(point))
            if not self.reduce_motion:
                burst = Q.CABasicAnimation.animationWithKeyPath_("birthRate")
                burst.setFromValue_(1.)
                burst.setToValue_(0.)
                burst.setDuration_(.13)
                self.emitter.addAnimation_forKey_(burst, "sparks")
            pulse = Q.CABasicAnimation.animationWithKeyPath_("opacity")
            pulse.setFromValue_(.4)
            pulse.setToValue_(1.)
            pulse.setDuration_(.13)
            self.shape.addAnimation_forKey_(pulse, "pulse")
        finally:
            Q.CATransaction.commit()


class MenuBar(AppKit.NSObject):
    @objc.python_method
    def configure(self, callbacks, permissions):
        self.callbacks = callbacks
        self.item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        menu = AppKit.NSMenu.alloc().init()
        self.status = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
        for label, granted in permissions.items():
            menu.addItemWithTitle_action_keyEquivalent_(f"{label} · {'Allowed' if granted else 'Needs access'}", None, "")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self.controls = {}
        for index, label in enumerate(callbacks):
            entry = menu.addItemWithTitle_action_keyEquivalent_(label, "selectControl:", "")
            entry.setTarget_(self)
            entry.setTag_(index)
            self.controls[label] = entry
        self.item.setMenu_(menu)
        self.update("ACTIVE", True)
        return self

    @objc.IBAction
    def selectControl_(self, sender):
        list(self.callbacks.values())[sender.tag()]()

    @objc.python_method
    def update(self, mode, sound):
        title = f"◎ {mode}"
        if self.item.button().title() != title:
            self.item.button().setTitle_(title)
            self.status.setTitle_(mode)
        self.controls["Toggle Sound"].setState_(AppKit.NSControlStateValueOn if sound else AppKit.NSControlStateValueOff)

    @objc.python_method
    def close(self):
        AppKit.NSStatusBar.systemStatusBar().removeStatusItem_(self.item)
        self.callbacks = {}


if __name__ == "__main__":
    import unittest
    from unittest.mock import patch

    class BridgeTests(unittest.TestCase):
        def test_offscreen_tether_render(self):
            root = Q.CALayer.layer()
            root.setFrame_(Q.CGRectMake(0, 0, 160, 120))
            tether = ElasticTether(root, 120)
            tether.shape.setFrame_(root.bounds())
            tether.draw((15, 20), (135, 100), .7)
            self.assertEqual(tuple(Q.CGPathGetCurrentPoint(tether.shape.path())), (135, 20))
            self.assertEqual(len(tether.emitter.emitterCells()), 1)
            tether.burst((135, 100))
            self.assertIsNotNone(tether.emitter.animationForKey_("sparks"))
            bitmap = Q.CGBitmapContextCreate(None, 160, 120, 8, 160 * 4, Q.CGColorSpaceCreateDeviceRGB(),
                                             Q.kCGImageAlphaPremultipliedLast)
            root.renderInContext_(bitmap)
            rendered = Q.CGBitmapContextCreateImage(bitmap)
            pixels = Q.CGDataProviderCopyData(Q.CGImageGetDataProvider(rendered))
            self.assertTrue(any(bytes(pixels)[3::4]))
            tether.draw((19, 20), (135, 100), .9)
            self.assertIsNotNone(tether.shape.animationForKey_("tetherPath"))
            tether.draw(None, None)
            self.assertIsNone(tether.shape.path())
            self.assertIsNone(tether.shape.animationForKey_("tetherPath"))

        def test_cache_invalidation(self):
            index = WindowIndexer()
            item = IndexedElement(None, "AXButton", (20, 40, 40, 20), "Open")
            old = index.generation
            snapshot = (1, "Test", "Window", (item,), ())
            index.publish(old, snapshot)
            self.assertIs(index.snap(item.center), item)
            index.invalidate()
            index.publish(old, snapshot)
            self.assertIsNone(index.snap(item.center))
            index.publish(index.generation, snapshot)
            self.assertIs(index.snap(item.center), item)

        def test_center_snap_and_loupe_selection(self):
            from handsfree.gaze_engine import GazeLoupe
            index = WindowIndexer()
            item = IndexedElement(None, "AXLink", (99, 99, 2, 2))
            index.elements = [item]
            self.assertIs(index.snap((320, 100)), item)
            self.assertIsNone(index.snap((320.01, 100)))
            loupe = GazeLoupe(500, 400)
            for n in range(1, 10):
                self.assertEqual(loupe.cell(*loupe.cell_center(n)), n)
            x, y = loupe.cell_center(9)
            link = IndexedElement(None, "AXLink", (x - 3, y - 3, 6, 6))
            offset = IndexedElement(None, "AXButton", (x + 7, y, 6, 6))
            index.elements = [item, offset, link]
            self.assertEqual(index.loupe_targets(loupe), {9: link})

        def test_coordinates(self):
            for point in ((0, 0), (-1920, 200), (100, 1400), (100, -800)):
                flipped = cocoa_to_quartz(*point, height=1000)
                self.assertEqual(flipped, (point[0], 1000 - point[1]))
                self.assertEqual(quartz_to_cocoa(*flipped, height=1000), point)

        def test_unwrap(self):
            self.assertEqual(unwrap_ax((0, "ok")), "ok")
            self.assertIsNone(unwrap_ax((-25205, "bad")))
            self.assertIsNone(unwrap_ax(None))
            self.assertIsNone(unwrap_ax((0,)))
            self.assertEqual(unwrap_ax((True, (2, 3)), True), (2, 3))

        def test_pid_cache(self):
            index = WindowIndexer()
            with patch.object(AX, "AXUIElementSetAttributeValue", return_value=-25205) as setter:
                index.enable_chromium(1, object(), "Google Chrome")
                index.enable_chromium(1, object(), "Google Chrome")
                index.enable_chromium(2, object(), "Electron")
                index.enable_chromium(3, object(), "Safari")
                self.assertEqual(setter.call_count, 2)
                self.assertEqual(setter.call_args_list[0].args[1:], ("AXManualAccessibility", True))
                index.forget_pid(1)
                index.enable_chromium(1, object(), "Chrome")
                self.assertEqual(setter.call_count, 3)

        def test_handoff(self):
            tracker = TrackpadHandoffTracker()
            self.assertEqual(tracker.update(0, 0, now=0), 1)
            self.assertEqual(tracker.update(1, 0, now=1), 0)
            self.assertEqual(tracker.alpha(3.5), 0)
            self.assertAlmostEqual(tracker.alpha(3.75), .5)
            self.assertEqual(tracker.alpha(4), 1)
            self.assertEqual(tracker.update(8, 0, physical=False, now=4), 1)
            self.assertEqual(tracker.update(9, 0, now=4.1), 0)

        def test_ring(self):
            ring = RadialRingMenu(-100, 50)
            self.assertIsNone(ring.hit_test(-100, 50))
            for i, item in enumerate(ring.ITEMS):
                angle = math.radians(i * 60)
                self.assertEqual(ring.hit_test(-100 + 10 * math.cos(angle),
                                               50 + 10 * math.sin(angle)), item)
            self.assertEqual(ring.hit_test(-90, 49), "Copy")
            self.assertEqual(ring.hit_test(-90, 56), "Paste")

        def test_lookup(self):
            index = WindowIndexer()
            item = IndexedElement(None, "AXButton", (-100, 10, 20, 30))
            index.elements = [item]
            self.assertIs(index.lookup_at_gaze(-80, 20, 0), item)
            self.assertIsNone(index.lookup_at_gaze(0, 20, 70))

    unittest.main()
