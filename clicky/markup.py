"""Mark a summoned screen in memory, preserving its original image for editing."""

from dataclasses import replace
from io import BytesIO

import AppKit as A
import objc
import Quartz as Q
from PIL import Image, ImageDraw

from .panel import PromptPanel


def annotated(context, strokes):
    if not strokes:
        return context
    with Image.open(BytesIO(context.jpeg)) as source:
        image = source.convert("RGB")
    ink = ImageDraw.Draw(image)
    width = max(3, round(image.width / 350))
    for stroke in strokes:
        points = [(round(x * (image.width - 1)), round(y * (image.height - 1))) for x, y in stroke]
        for color, size in (("#34250e", width + 2), ("#ffb336", width)):
            if len(points) > 1:
                ink.line(points, fill=color, width=size, joint="curve")
            else:
                x, y = points[0]
                r = size + 1
                ink.ellipse((x - r, y - r, x + r, y + r), fill=color)
    with BytesIO() as out:
        image.save(out, "JPEG", quality=90)
        return replace(context, jpeg=out.getvalue(), marked=True)


class MarkPanel(PromptPanel):
    def cancelOperation_(self, sender):
        self.contentView().cancel_(sender)


class MarkView(A.NSView):
    def acceptsFirstResponder(self):
        return True

    def drawRect_(self, dirty):
        frame = self.bounds()
        self.snapshot.drawInRect_fromRect_operation_fraction_(
            frame, A.NSZeroRect, A.NSCompositingOperationCopy, 1
        )
        w, h = frame.size
        for stroke in self.strokes:
            path = A.NSBezierPath.bezierPath()
            path.setLineCapStyle_(A.NSRoundLineCapStyle)
            path.setLineJoinStyle_(A.NSRoundLineJoinStyle)
            first, *rest = [(x * w, (1 - y) * h) for x, y in stroke]
            path.moveToPoint_(first)
            for point in rest or [(first[0] + 0.1, first[1])]:
                path.lineToPoint_(point)
            for color, width in ((A.NSColor.blackColor(), 5), (A.NSColor.systemOrangeColor(), 3)):
                color.setStroke()
                path.setLineWidth_(width)
                path.stroke()

    @objc.python_method
    def point(self, event):
        x, y = self.convertPoint_fromView_(event.locationInWindow(), None)
        w, h = self.bounds().size
        return max(0, min(1, x / w)), max(0, min(1, 1 - y / h))

    def mouseDown_(self, event):
        self.strokes.append([self.point(event)])
        self.setNeedsDisplay_(True)

    def mouseDragged_(self, event):
        if self.strokes:
            point = self.point(event)
            if point != self.strokes[-1][-1]:
                self.strokes[-1].append(point)
                self.setNeedsDisplay_(True)

    def keyDown_(self, event):
        if event.keyCode() == 53:
            self.cancel_(self)
        elif event.keyCode() in (36, 76):
            self.done_(self)

    def done_(self, sender):
        self.finish(self.strokes)

    def cancel_(self, sender):
        self.finish(None)

    def clear_(self, sender):
        self.strokes.clear()
        self.setNeedsDisplay_(True)


def open_markup(context, strokes, finish):
    screen = next((s for s in A.NSScreen.screens() if display_frame(s) == tuple(context.display)), None)
    if not screen:
        raise ValueError("The captured display has changed. Summon CursorMix again.")
    frame = screen.frame()
    window = MarkPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, A.NSWindowStyleMaskBorderless, A.NSBackingStoreBuffered, False
    )
    window.setFloatingPanel_(True)
    window.setHidesOnDeactivate_(False)
    window.setLevel_(A.NSScreenSaverWindowLevel)
    window.setReleasedWhenClosed_(False)
    window.setCollectionBehavior_(
        A.NSWindowCollectionBehaviorMoveToActiveSpace | A.NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    view = MarkView.alloc().initWithFrame_(((0, 0), frame.size))
    view.snapshot = A.NSImage.alloc().initWithData_(context.jpeg)
    if not view.snapshot:
        raise ValueError("The captured image could not be opened.")
    view.strokes = [list(stroke) for stroke in strokes]
    view.finish = finish
    window.setContentView_(view)
    controls = A.NSStackView.alloc().initWithFrame_(A.NSZeroRect)
    controls.setSpacing_(8)
    controls.setEdgeInsets_(A.NSEdgeInsetsMake(7, 10, 7, 10))
    controls.setWantsLayer_(True)
    controls.layer().setCornerRadius_(9)
    controls.layer().setBackgroundColor_(A.NSColor.windowBackgroundColor().CGColor())
    controls.addArrangedSubview_(A.NSTextField.labelWithString_("Mark an area to ask about"))
    for title, selector in (("Clear", "clear:"), ("Cancel", "cancel:"), ("Done", "done:")):
        controls.addArrangedSubview_(A.NSButton.buttonWithTitle_target_action_(title, view, selector))
    size = controls.fittingSize()
    controls.setFrame_((((frame.size.width - size.width) / 2, 24), size))
    view.addSubview_(controls)
    A.NSApp.activateIgnoringOtherApps_(True)
    window.makeKeyAndOrderFront_(None)
    window.makeFirstResponder_(view)
    return window


def display_frame(screen):
    bounds = Q.CGDisplayBounds(int(screen.deviceDescription()["NSScreenNumber"]))
    return (*bounds.origin, *bounds.size)
