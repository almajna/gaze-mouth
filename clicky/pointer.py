"""A click-through companion drawn in its own small, nonactivating Cocoa window."""

import math
import time

import AppKit as A

COLORS = {
    "Amber": (1.0, 0.64, 0.19),
    "Mint": (0.36, 0.85, 0.66),
    "Sky": (0.42, 0.73, 1.0),
    "Lilac": (0.79, 0.64, 0.96),
}


class PointerGlyph(A.NSView):
    def drawRect_(self, dirty):
        body = A.NSBezierPath.bezierPath()
        body.moveToPoint_((3, 29))
        for point in ((27, 20), (16, 16), (12, 5)):
            body.lineToPoint_(point)
        body.closePath()
        body.setLineJoinStyle_(A.NSRoundLineJoinStyle)
        A.NSColor.whiteColor().setStroke()
        body.setLineWidth_(3.5)
        body.stroke()
        A.NSColor.colorWithSRGBRed_green_blue_alpha_(*COLORS[self.color], 1).setFill()
        body.fill()
        A.NSColor.colorWithSRGBRed_green_blue_alpha_(0.13, 0.11, 0.09, 1).setStroke()
        body.setLineWidth_(1.3)
        body.stroke()
        fold = A.NSBezierPath.bezierPath()
        fold.moveToPoint_((8, 25))
        fold.lineToPoint_((16, 16))
        fold.setLineWidth_(1.4)
        fold.stroke()
        if self.activity == "idle":
            return
        badge = A.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((25, 2), (22, 17)), 8, 8)
        A.NSColor.colorWithWhite_alpha_(0.12, 0.96).setFill()
        badge.fill()
        A.NSColor.whiteColor().setStroke()
        badge.setLineWidth_(1)
        badge.stroke()
        A.NSColor.whiteColor().setFill()
        if self.activity == "thinking":
            for x in (30, 35, 40):
                A.NSBezierPath.bezierPathWithOvalInRect_(((x, 9), (2.5, 2.5))).fill()
        elif self.activity == "listening":
            for x, height in ((30, 4), (33, 8), (36, 11), (39, 6)):
                A.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ((x, 10.5 - height / 2), (2, height)), 1, 1
                ).fill()
        else:
            speaker = A.NSBezierPath.bezierPath()
            speaker.moveToPoint_((30, 9))
            for point in ((30, 12), (32, 12), (36, 15), (36, 6), (32, 9)):
                speaker.lineToPoint_(point)
            speaker.closePath()
            speaker.fill()
            wave = A.NSBezierPath.bezierPath()
            wave.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_((36, 10.5), 5, -52, 52)
            wave.stroke()


class Pointer:
    def __init__(self):
        self.window = A.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (48, 32)), A.NSWindowStyleMaskBorderless, A.NSBackingStoreBuffered, False
        )
        self.glyph = PointerGlyph.alloc().initWithFrame_(((0, 0), (48, 32)))
        self.glyph.activity = "idle"
        self.glyph.color = "Amber"
        self.window.setContentView_(self.glyph)
        self.window.setLevel_(A.NSScreenSaverWindowLevel + 1)
        self.window.setOpaque_(False)
        self.window.setHasShadow_(False)
        self.window.setBackgroundColor_(A.NSColor.clearColor())
        self.window.setIgnoresMouseEvents_(True)
        self.window.setSharingType_(A.NSWindowSharingNone)
        self.window.setCollectionBehavior_(
            A.NSWindowCollectionBehaviorCanJoinAllSpaces
            | A.NSWindowCollectionBehaviorFullScreenAuxiliary
            | A.NSWindowCollectionBehaviorIgnoresCycle
        )
        self.enabled = False
        self.timer = None
        self.destination = None
        self.last_mouse = None

    def appearance(self, *, color=None, activity=None):
        if color is not None:
            self.glyph.color = color if color in COLORS else "Amber"
        if activity is not None:
            self.glyph.activity = activity
        self.glyph.setNeedsDisplay_(True)

    def stop_timer(self):
        if self.timer:
            self.timer.invalidate()
            self.timer = None

    def suspend(self):
        self.stop_timer()
        self.last_mouse = self.destination = None
        self.window.orderOut_(None)

    def start_timer(self):
        if self.timer is None:
            self.timer = A.NSTimer.timerWithTimeInterval_repeats_block_(1 / 30, True, self.tick)
            A.NSRunLoop.mainRunLoop().addTimer_forMode_(self.timer, A.NSRunLoopCommonModes)

    def follow(self):
        self.destination = self.last_mouse = None
        if self.enabled:
            self.tick(None)
            self.window.orderFrontRegardless()
            self.start_timer()
        else:
            self.suspend()

    def point(self, tip, *, animate=True):
        if not self.enabled:
            return
        if not self.window.isVisible():
            self.follow()
        self.origin = tuple(self.window.frame().origin)
        self.destination = (tip[0] - 3, tip[1] - 29)
        self.started = time.monotonic()
        distance = math.dist(self.origin, self.destination)
        self.duration = min(0.52, max(0.18, distance / 1800))
        if not animate or A.NSWorkspace.sharedWorkspace().accessibilityDisplayShouldReduceMotion():
            self.duration = 0
        self.start_timer()
        self.tick(None)

    def tick(self, timer):
        if self.destination is not None:
            t = min(1, (time.monotonic() - self.started) / self.duration) if self.duration else 1
            ease = t * t * (3 - 2 * t)
            pos = tuple(a + (b - a) * ease for a, b in zip(self.origin, self.destination))
            self.window.setFrameOrigin_(pos)
            if t == 1:
                self.stop_timer()
            return
        mouse = A.NSEvent.mouseLocation()
        if tuple(mouse) == self.last_mouse:
            return
        self.last_mouse = tuple(mouse)
        screen = next((s for s in A.NSScreen.screens() if A.NSPointInRect(mouse, s.frame())), None)
        if screen:
            frame = screen.frame()
            x = min(max(mouse.x + 12, A.NSMinX(frame)), A.NSMaxX(frame) - self.window.frame().size.width)
            y = max(min(mouse.y - 39, A.NSMaxY(frame) - 32), A.NSMinY(frame))
            self.window.setFrameOrigin_((x, y))
