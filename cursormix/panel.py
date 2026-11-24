import math

import AppKit as A

from .home import display_id, top_center
from .inputs.typed import TypedInput


class PromptPanel(A.NSPanel):
    def canBecomeKeyWindow(self):
        return True

    def cancelOperation_(self, sender):
        self.orderOut_(sender)


class Panel:
    def __init__(self, delegate):
        self.home_display = None
        style = A.NSWindowStyleMaskTitled | A.NSWindowStyleMaskUtilityWindow | A.NSWindowStyleMaskResizable
        self.window = PromptPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (480, 236)), style, A.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Ask CursorMix")
        self.window.setContentMinSize_((480, 236))
        self.window.setFloatingPanel_(True)
        self.window.setHidesOnDeactivate_(False)
        self.window.setLevel_(A.NSFloatingWindowLevel)
        self.window.setReleasedWhenClosed_(False)
        self.window.setCollectionBehavior_(
            A.NSWindowCollectionBehaviorMoveToActiveSpace | A.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self.field = A.NSTextField.alloc().initWithFrame_(((16, 187), (336, 32)))
        self.field.setFont_(A.NSFont.systemFontOfSize_(16))
        self.field.setPlaceholderString_("Ask a question")
        self.field.setTarget_(delegate)
        self.field.setAction_("submit:")
        self.field.setAccessibilityLabel_("Request")
        top_row = A.NSViewWidthSizable | A.NSViewMinYMargin
        self.field.setAutoresizingMask_(top_row)
        self.typed = TypedInput(self.field)
        self.step_question = A.NSTextField.alloc().initWithFrame_(self.field.frame())
        self.step_question.setFont_(self.field.font())
        self.step_question.setPlaceholderString_("Ask about this step")
        self.step_question.setAccessibilityLabel_("Ask about this step")
        self.step_question.setTarget_(delegate)
        self.step_question.setAction_("askAboutStep:")
        self.step_question.setHidden_(True)
        self.step_question.setAutoresizingMask_(top_row)
        self.window.contentView().addSubview_(self.step_question)
        self.mark = A.NSButton.buttonWithTitle_target_action_("Mark Screen…", delegate, "markScreen:")
        self.mark.setFrame_(((356, 187), (108, 32)))
        self.mark.setAutoresizingMask_(A.NSViewMinXMargin | A.NSViewMinYMargin)
        self.mark.setEnabled_(False)
        self.stop = A.NSButton.buttonWithTitle_target_action_("Stop", delegate, "stopRequest:")
        self.stop.setFrame_(self.mark.frame())
        self.stop.setHidden_(True)
        self.stop.setAutoresizingMask_(A.NSViewMinXMargin | A.NSViewMinYMargin)
        self.window.contentView().addSubview_(self.stop)
        self.step_help = A.NSButton.buttonWithTitle_target_action_(
            "Help with Step", delegate, "helpWithStep:"
        )
        self.step_help.setFrame_(self.mark.frame())
        self.step_help.setHidden_(True)
        self.step_help.setAutoresizingMask_(A.NSViewMinXMargin | A.NSViewMinYMargin)
        self.window.contentView().addSubview_(self.step_help)
        scroll = A.NSScrollView.alloc().initWithFrame_(((16, 12), (448, 162)))
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setAutoresizingMask_(A.NSViewWidthSizable | A.NSViewHeightSizable)
        self.output = A.NSTextView.alloc().initWithFrame_(((0, 0), (432, 162)))
        self.output.setEditable_(False)
        self.output.setAutoresizingMask_(A.NSViewWidthSizable)
        self.output.setDrawsBackground_(False)
        self.output.setFont_(A.NSFont.systemFontOfSize_(14))
        self.output.setAccessibilityLabel_("Answer")
        scroll.setDrawsBackground_(False)
        scroll.setDocumentView_(self.output)
        self.scroll = scroll
        self.target_menu = A.NSPopUpButton.alloc().initWithFrame_pullsDown_(((16, 12), (448, 28)), False)
        self.target_menu.setAccessibilityLabel_("Matching controls")
        self.target_menu.setTarget_(delegate)
        self.target_menu.setAction_("selectPointTarget:")
        self.target_menu.setHidden_(True)
        self.target_menu.setAutoresizingMask_(A.NSViewWidthSizable | A.NSViewMaxYMargin)
        self.window.contentView().addSubview_(self.target_menu)
        self.suggestions = []
        for _ in range(2):
            button = A.NSButton.buttonWithTitle_target_action_("", delegate, "selectSuggestion:")
            button.setFrame_(((16, 12), (448, 25)))
            button.setHidden_(True)
            button.setAutoresizingMask_(A.NSViewWidthSizable | A.NSViewMaxYMargin)
            self.window.contentView().addSubview_(button)
            self.suggestions.append(button)
        self.window.contentView().addSubview_(self.field)
        self.window.contentView().addSubview_(scroll)
        self.window.setInitialFirstResponder_(self.field)

        def key(event):
            if self.window.isKeyWindow() and event.keyCode() == 53:
                delegate.dismiss_(None)
                return None
            return event

        self.monitor = A.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(A.NSEventMaskKeyDown, key)

    def set_suggestions(self, prompts):
        font = self.suggestions[0].font()
        width, height = self.window.contentView().bounds().size
        width -= 32
        prompts = [
            p
            for p in prompts
            if A.NSString.stringWithString_(p).sizeWithAttributes_({A.NSFontAttributeName: font}).width <= width - 40
        ]
        for i, button in enumerate(self.suggestions):
            title = prompts[i] if i < len(prompts) else ""
            if title:
                button.setFrameOrigin_((16, 12 + (len(prompts) - i - 1) * 26))
            button.setTitle_(title)
            button.setToolTip_(title or None)
            button.setHidden_(not title)
        reserved = 26 * len(prompts) + 2 if prompts else 0
        if not self.target_menu.isHidden():
            reserved = 32
        self.scroll.setFrame_(((16, 12 + reserved), (width, height - 62 - reserved)))

    def set_target_choices(self, labels):
        self.target_menu.removeAllItems()
        if labels:
            self.target_menu.addItemWithTitle_("Choose a control…")
            self.target_menu.addItemsWithTitles_(labels)
            self.target_menu.selectItemAtIndex_(0)
        self.target_menu.setHidden_(not labels)
        self.set_suggestions([])

    def place(self):
        size = self.window.frame().size
        if self.home_display is not None:
            screen = next((s for s in A.NSScreen.screens() if display_id(s) == self.home_display), None)
            if screen:
                cx, top = top_center(screen)
                frame = screen.visibleFrame()
                x = max(frame.origin.x, min(cx - size.width / 2, A.NSMaxX(frame) - size.width))
                self.window.setFrameOrigin_((round(x), max(frame.origin.y, top - 24 - size.height)))
                return
            self.home_display = None
        mouse = A.NSEvent.mouseLocation()
        screen = next(
            (s for s in A.NSScreen.screens() if A.NSPointInRect(mouse, s.frame())), A.NSScreen.mainScreen()
        )
        frame = screen.visibleFrame()
        x = max(frame.origin.x, min(mouse.x + 12, A.NSMaxX(frame) - size.width))
        y = max(frame.origin.y, min(mouse.y - size.height - 10, A.NSMaxY(frame) - size.height))
        self.window.setFrameOrigin_((x, y))

    def show(self, select_text=True):
        self.place()
        A.NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)
        self.window.makeFirstResponder_(self.field)
        if select_text:
            self.field.selectText_(None)
        else:
            editor = self.field.currentEditor()
            if editor:
                editor.setSelectedRange_((editor.textStorage().length(), 0))

    def avoid(self, target):
        frame = self.window.frame()
        if target is None or not self.window.isVisible() or not A.NSIntersectsRect(frame, target):
            return
        screen = self.window.screen()
        if not screen:
            return
        visible = screen.visibleFrame()
        width, height = frame.size
        candidates = (
            (A.NSMinX(target) - width - 12, frame.origin.y),
            (A.NSMaxX(target) + 12, frame.origin.y),
            (frame.origin.x, A.NSMinY(target) - height - 12),
            (frame.origin.x, A.NSMaxY(target) + 12),
        )
        positions = [p for p in candidates if A.NSContainsRect(visible, (p, frame.size))]
        if positions:
            self.window.setFrameOrigin_(min(positions, key=lambda p: math.dist(p, frame.origin)))
        else:
            self.window.orderOut_(None)

    def close(self):
        A.NSEvent.removeMonitor_(self.monitor)
        self.window.orderOut_(None)
