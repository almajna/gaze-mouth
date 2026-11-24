"""First-run window: request the three permissions and save the Claude API key."""

import os

import AppKit as A
import ApplicationServices as AX
import AVFoundation as AV
import objc
import Quartz as Q
from Foundation import NSObject

from . import save_local_key
from .hotkey import shortcut_label
from .settings import grants, open_permission

REASONS = {
    "Screen Recording": "See your screen when you summon CursorMix",
    "Accessibility": "Read control names, point at them and click for you",
    "Microphone": "Hear you while you hold the shortcut",
}


def welcome_text(shortcut):
    return (
        f"Press {shortcut_label(shortcut)} from any app and type a question, or hold it and speak. "
        "CursorMix sees your screen, points at controls, walks you through steps and can do tasks for you."
    )


def request(name, asked):
    """Show the macOS prompt for a permission. After the first try, open its Settings pane instead."""
    if name in asked:
        open_permission(name)
        return
    asked.add(name)
    if name == "Screen Recording":
        if not Q.CGRequestScreenCaptureAccess():
            open_permission(name)
    elif name == "Accessibility":
        AX.AXIsProcessTrustedWithOptions({AX.kAXTrustedCheckOptionPrompt: True})
    elif AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio) == AV.AVAuthorizationStatusNotDetermined:
        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeAudio, lambda allowed: None)
    else:
        open_permission(name)


def label(text, frame, *, size=13, bold=False, color=None):
    field = A.NSTextField.wrappingLabelWithString_(text)
    field.setFrame_(frame)
    field.setFont_(A.NSFont.boldSystemFontOfSize_(size) if bold else A.NSFont.systemFontOfSize_(size))
    if color:
        field.setTextColor_(color)
    return field


class Setup(NSObject):
    def initWithShortcut_done_(self, shortcut, done):
        self = objc.super(Setup, self).init()
        if self is None:
            return None
        self.done = done
        self.asked = set()
        self.rows = {}
        self.timer = None
        window = A.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (500, 400)),
            A.NSWindowStyleMaskTitled | A.NSWindowStyleMaskClosable,
            A.NSBackingStoreBuffered,
            False,
        )
        window.setTitle_("Set Up CursorMix")
        window.setReleasedWhenClosed_(False)
        window.setLevel_(A.NSFloatingWindowLevel)
        window.setDelegate_(self)
        view = window.contentView()
        view.addSubview_(label("Welcome to CursorMix", ((20, 360), (460, 22)), size=16, bold=True))
        view.addSubview_(label(welcome_text(shortcut), ((20, 296), (460, 58))))
        y = 252
        for name, reason in REASONS.items():
            view.addSubview_(label(name, ((20, y + 14), (300, 18)), bold=True))
            view.addSubview_(label(reason, ((20, y - 4), (330, 17)), size=11, color=A.NSColor.secondaryLabelColor()))
            status = label("", ((262, y + 14), (100, 18)))
            status.setAlignment_(A.NSTextAlignmentRight)
            status.setAccessibilityLabel_(f"{name} status")
            view.addSubview_(status)
            button = A.NSButton.buttonWithTitle_target_action_("Allow", self, "allow:")
            button.setFrame_(((372, y + 4), (112, 30)))
            button.setIdentifier_(name)
            button.setAccessibilityLabel_(f"Allow {name}")
            view.addSubview_(button)
            self.rows[name] = (status, button)
            y -= 48
        view.addSubview_(label("Claude API key", ((20, 102), (300, 18)), bold=True))
        self.key = A.NSSecureTextField.alloc().initWithFrame_(((20, 70), (344, 26)))
        self.key.setAccessibilityLabel_("Claude API key")
        self.key.setTarget_(self)
        self.key.setAction_("saveKey:")
        view.addSubview_(self.key)
        save = A.NSButton.buttonWithTitle_target_action_("Save Key", self, "saveKey:")
        save.setFrame_(((372, 67), (112, 30)))
        view.addSubview_(save)
        self.key_status = label("", ((20, 46), (460, 17)), size=11, color=A.NSColor.secondaryLabelColor())
        self.key_status.setAccessibilityLabel_("API key status")
        view.addSubview_(self.key_status)
        finish = A.NSButton.buttonWithTitle_target_action_("Done", self, "finish:")
        finish.setFrame_(((372, 10), (112, 30)))
        finish.setKeyEquivalent_("\r")
        view.addSubview_(finish)
        self.window = window
        return self

    def show(self):
        self.refresh_(None)
        if self.timer is None:
            # Permissions are granted in System Settings, outside this window. Poll so the rows keep up.
            self.timer = A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0, self, "refresh:", None, True
            )
        self.window.center()
        A.NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    def refresh_(self, timer):
        for name, allowed in grants().items():
            status, button = self.rows[name]
            status.setStringValue_("Allowed ✓" if allowed else "Not allowed")
            status.setTextColor_(A.NSColor.systemGreenColor() if allowed else A.NSColor.systemOrangeColor())
            button.setEnabled_(not allowed)
            button.setTitle_("Open Settings" if name in self.asked and not allowed else "Allow")
        if not self.key_status.stringValue() or self.key_status.stringValue().startswith(("A key", "No key")):
            saved = bool(os.getenv("ANTHROPIC_API_KEY"))
            self.key.setPlaceholderString_("Paste a new key to replace it" if saved else "sk-ant-…")
            self.key_status.setStringValue_(
                "A key is saved. CursorMix answers with Claude." if saved
                else "No key yet. Paste your key from console.anthropic.com and press Save Key."
            )

    def allow_(self, sender):
        request(str(sender.identifier()), self.asked)
        self.refresh_(None)

    def saveKey_(self, sender):
        value = str(self.key.stringValue()).strip()
        if not value:
            return
        self.key.setStringValue_("")
        os.environ["ANTHROPIC_API_KEY"] = value
        try:
            save_local_key("ANTHROPIC_API_KEY", value)
            self.key_status.setStringValue_("Saved. CursorMix answers with Claude.")
        except OSError:
            self.key_status.setStringValue_("Could not write the key file. This key works until CursorMix quits.")

    def finish_(self, sender):
        self.window.close()

    def windowWillClose_(self, notification):
        if self.timer:
            self.timer.invalidate()
            self.timer = None
        self.done()
