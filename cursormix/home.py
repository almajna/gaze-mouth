"""A nonactivating Home panel at the top of each display."""

import AppKit as A
import objc


def display_id(screen):
    return int(screen.deviceDescription()["NSScreenNumber"])


def top_center(screen):
    frame = screen.frame()
    top = min(A.NSMaxY(screen.visibleFrame()), A.NSMaxY(frame) - screen.safeAreaInsets().top)
    return A.NSMidX(frame), top


class HomePanel(A.NSPanel):
    def canBecomeKeyWindow(self):
        return False

    def canBecomeMainWindow(self):
        return False


class HomeView(A.NSVisualEffectView):
    @objc.python_method
    def setup(self, owner, screen):
        self.owner = owner
        self.display_id = display_id(screen)
        self.expanded = self.pinned = False
        self.timer = None
        self.setMaterial_(A.NSVisualEffectMaterialHUDWindow)
        self.setState_(A.NSVisualEffectStateActive)
        self.setAppearance_(A.NSAppearance.appearanceNamed_(A.NSAppearanceNameDarkAqua))
        self.setWantsLayer_(True)
        self.layer().setCornerRadius_(9)
        self.layer().setMasksToBounds_(True)
        self.tab = self.button("CursorMix", "openHome:", (0, 0, 90, 19))
        self.tab.setBordered_(False)
        self.tab.setAccessibilityLabel_("Open CursorMix Home")
        self.tab.setFont_(A.NSFont.boldSystemFontOfSize_(11))
        self.indicator = A.NSView.alloc().initWithFrame_(((7, 7), (5, 5)))
        self.indicator.setWantsLayer_(True)
        self.indicator.layer().setCornerRadius_(2.5)
        self.addSubview_(self.indicator)
        self.status = A.NSTextField.wrappingLabelWithString_("")
        self.status.setFrame_(((14, 45), (244, 34)))
        self.status.setMaximumNumberOfLines_(2)
        self.status.setFont_(A.NSFont.systemFontOfSize_(13))
        self.status.setAccessibilityLabel_("CursorMix status")
        self.addSubview_(self.status)
        self.open_button = self.button("Ask", "showAsk:", (8, 9, 128, 28))
        self.history_button = self.button("History", "showHistory:", (136, 9, 76, 28))
        self.cancel_button = self.button("Cancel", "stopRequest:", (212, 9, 72, 28))
        self.routine_button = self.button("Review", "showRoutine:", (212, 9, 72, 28))
        self.close_button = self.button("×", "closeHome:", (260, 52, 25, 26))
        self.close_button.setBordered_(False)
        self.close_button.setAccessibilityLabel_("Close CursorMix Home")
        self.controls = (self.status, self.open_button, self.history_button, self.close_button)
        self.tracking = A.NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            A.NSZeroRect,
            A.NSTrackingMouseEnteredAndExited | A.NSTrackingActiveAlways | A.NSTrackingInVisibleRect,
            self,
            None,
        )
        self.addTrackingArea_(self.tracking)

    @objc.python_method
    def button(self, title, selector, frame):
        button = A.NSButton.buttonWithTitle_target_action_(title, self, selector)
        button.setFrame_((frame[:2], frame[2:]))
        self.addSubview_(button)
        return button

    @objc.python_method
    def layout_on(self, screen, expanded):
        self.expanded = expanded
        width, height = (292, 88) if expanded else (90, 19)
        cx, top = top_center(screen)
        self.window().setFrame_display_(((round(cx - width / 2), top - height - 1), (width, height)), True)
        self.tab.setHidden_(expanded)
        self.indicator.setHidden_(expanded)
        for control in self.controls:
            control.setHidden_(not expanded)
        self.cancel_button.setHidden_(True)
        self.routine_button.setHidden_(True)
        if self.timer:
            self.timer.invalidate()
            self.timer = None
        if expanded:
            self.refresh_(None)
            self.timer = A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.4, self, "refresh:", None, True
            )
        else:
            self.pinned = False

    def refresh_(self, timer):
        delegate = self.owner.delegate
        busy = not delegate.cancel_item.isHidden()
        output = str(delegate.panel.output.string()).strip()
        has_panel = busy or delegate.panel.window.isVisible() or bool(delegate.conversation)
        self.open_button.setTitle_(
            "Show Walkthrough" if delegate.step_help is not None else "Show Request" if has_panel else "Ask"
        )
        status = output if has_panel and output else delegate.status.button().title()
        task = delegate.routine_menu.featured if not has_panel else None
        jobs = delegate.routine_menu.job_summary if not has_panel else ""
        self.routine_button.setHidden_(not (task or jobs))
        if jobs and not (task and task["status"] == "Ready"):
            status = "Routines · " + jobs
            self.routine_button.setTitle_("Jobs")
            self.routine_button.setAccessibilityLabel_("Show running and queued routines")
        elif task:
            ready = task["status"] == "Ready"
            self.routine_button.setTitle_("Start" if ready else "Result")
            self.routine_button.setAccessibilityLabel_(("Start " if ready else "Review ") + task["name"])
            status = f"{task['name']} · {task['status']}"
            if jobs:
                status += "\n" + jobs
            self.routine_key = task["id"]
            self.routine_revision = task["revision"]
        self.status.setStringValue_(status[:210])
        self.cancel_button.setHidden_(not busy)

    def mouseEntered_(self, event):
        if self.owner and not self.expanded:
            self.owner.open(self.display_id, pinned=False)

    def mouseExited_(self, event):
        if self.owner and self.expanded and not self.pinned:
            self.performSelector_withObject_afterDelay_("leave:", None, 0.35)

    def leave_(self, sender):
        if (
            self.owner
            and self.expanded
            and not self.pinned
            and not A.NSPointInRect(A.NSEvent.mouseLocation(), self.window().frame())
        ):
            self.owner.collapse()

    def openHome_(self, sender):
        self.owner.open(self.display_id, pinned=True)

    def closeHome_(self, sender):
        self.owner.collapse()

    def showAsk_(self, sender):
        self.owner.collapse()
        self.owner.delegate.ask_from_home(self.display_id)

    def showRoutine_(self, sender):
        manager = self.owner.delegate.routine_menu
        if self.routine_button.title() == "Jobs":
            menu = A.NSMenu.alloc().initWithTitle_("Routines")
            menu.setDelegate_(manager)
            manager.menuNeedsUpdate_(menu)
            menu.popUpMenuPositioningItem_atLocation_inView_(None, (0, 0), self.routine_button)
            return
        task = manager.featured
        if task and task["id"] == self.routine_key:
            item = A.NSMenuItem.alloc().init()
            item.setRepresentedObject_(task["id"])
            item.setTag_(self.routine_revision)
            self.owner.collapse()
            if task["status"] == "Ready":
                manager.runRoutine_(item)
            else:
                manager.viewResult_(item)

    def showHistory_(self, sender):
        self.owner.collapse()
        self.owner.delegate.history_(sender)

    def stopRequest_(self, sender):
        self.owner.collapse()
        self.owner.delegate.cancelRequest_(sender)


class DisplayHome:
    def __init__(self, delegate):
        self.delegate = delegate
        self.windows = {}
        self.enabled = False
        self.suspended = False

    def sync(self, enabled):
        self.enabled = bool(enabled)
        screens = {display_id(s): s for s in A.NSScreen.screens()} if enabled else {}
        for identifier in self.windows.keys() - screens.keys():
            window = self.windows.pop(identifier)
            view = window.contentView()
            if view.timer:
                view.timer.invalidate()
                view.timer = None
            view.owner = None
            window.close()
        for identifier, screen in screens.items():
            if identifier not in self.windows:
                window = HomePanel.alloc().initWithContentRect_styleMask_backing_defer_(
                    ((0, 0), (66, 19)),
                    A.NSWindowStyleMaskBorderless | A.NSWindowStyleMaskNonactivatingPanel,
                    A.NSBackingStoreBuffered,
                    False,
                )
                window.setReleasedWhenClosed_(False)
                window.setBecomesKeyOnlyIfNeeded_(True)
                window.setHidesOnDeactivate_(False)
                window.setOpaque_(False)
                window.setBackgroundColor_(A.NSColor.clearColor())
                window.setLevel_(A.NSFloatingWindowLevel)
                window.setSharingType_(A.NSWindowSharingNone)
                window.setCollectionBehavior_(
                    A.NSWindowCollectionBehaviorCanJoinAllSpaces
                    | A.NSWindowCollectionBehaviorFullScreenAuxiliary
                    | A.NSWindowCollectionBehaviorIgnoresCycle
                )
                view = HomeView.alloc().initWithFrame_(window.contentView().bounds())
                window.setContentView_(view)
                view.setup(self, screen)
                self.windows[identifier] = window
            window = self.windows[identifier]
            window.contentView().layout_on(screen, window.contentView().expanded)
            if not self.suspended:
                window.orderFrontRegardless()

        self.activity("speaking" if self.delegate.speaker else self.delegate.activity)

    def activity(self, state):
        indicators = {
            "idle": ("Ready", A.NSColor.secondaryLabelColor()),
            "listening": ("Listening", A.NSColor.systemRedColor()),
            "thinking": ("Working", A.NSColor.systemOrangeColor()),
            "transcribing": ("Transcribing", A.NSColor.systemOrangeColor()),
            "speaking": ("Speaking", A.NSColor.systemGreenColor()),
        }
        label, color = indicators.get(state, indicators["idle"])
        if state == "idle" and self.delegate.routine_menu.job_summary:
            label, color = self.delegate.routine_menu.job_summary, A.NSColor.systemOrangeColor()
        for window in self.windows.values():
            view = window.contentView()
            view.indicator.layer().setBackgroundColor_(color.CGColor())
            view.tab.setAccessibilityLabel_("Open CursorMix Home · " + label)

    def open(self, identifier, *, pinned):
        if not self.enabled or self.suspended:
            return
        for screen in A.NSScreen.screens():
            key = display_id(screen)
            window = self.windows.get(key)
            if window and (key == identifier or window.contentView().expanded):
                view = window.contentView()
                view.layout_on(screen, key == identifier)
                view.pinned = key == identifier and pinned

    def collapse(self):
        self.open(None, pinned=False)

    def suspend(self, suspended):
        if self.suspended == suspended:
            return
        if suspended:
            self.collapse()
        self.suspended = suspended
        for window in self.windows.values():
            if suspended:
                window.orderOut_(None)
            else:
                window.orderFrontRegardless()
