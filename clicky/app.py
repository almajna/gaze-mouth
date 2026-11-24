"""Native menu-bar entry point for the standalone assistant."""

import argparse
import os
import sqlite3
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import AppKit as A
import objc
from Foundation import NSObject
from PyObjCTools import AppHelper

from . import agent, annotations, ax, capture, figure, markup, memory, router, teach
from .floating import FloatingButton, icon_hidden
from .home import DisplayHome, display_id
from .hotkey import DICTATION_SHORTCUTS, SHORTCUTS, Hotkey, shortcut_label
from .inputs.voice import VoiceInput
from .models import (
    ANSWER_STYLES,
    PROVIDERS,
    answer,
    followup_questions,
    selected_provider,
)
from .onboarding import Setup
from .overlay import MultipleTargets, Overlay, locate
from .panel import Panel
from .pointer import COLORS
from .routine_menu import RoutineMenu
from .settings import (
    FEATURES,
    LAUNCH_AGENT,
    PANES,
    grants,
    open_permission,
    preferences,
    set_login,
)
from .voice import DICTATION_LANGUAGES, DictationField, Recording, custom_spellings


def item(menu, title, target, selector):
    command = A.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, selector, "")
    command.setTarget_(target)
    menu.addItem_(command)
    return command


def section(menu, title):
    submenu = A.NSMenu.alloc().initWithTitle_(title)
    parent = item(menu, title, None, None)
    parent.setSubmenu_(submenu)
    return submenu


def alert(title, message, buttons=("OK",)):
    box = A.NSAlert.alloc().init()
    box.setMessageText_(title)
    box.setInformativeText_(message)
    for button in buttons:
        box.addButtonWithTitle_(button)
    A.NSApp.activateIgnoringOtherApps_(True)
    return box.runModal()


class ClickyDelegate(NSObject):
    @objc.python_method
    def settings_store(self):
        return preferences()

    def applicationDidFinishLaunching_(self, notification):
        self.store = self.settings_store()
        self.panel = Panel(self)
        self.point_choices = None
        self.home = DisplayHome(self)
        self.generation = 0
        self.suggestion_generation = 0
        self.capture_queue = ThreadPoolExecutor(1, thread_name_prefix="CursorMix snapshot")
        self.cancel = threading.Event()
        self.overlay = Overlay(on_move=self.panel.avoid)
        self.figure_window = None
        self.annotation = None
        self.mark_window = None
        self.original_context = None
        self.marks = []
        self.context = None
        self.turn_screen = None
        self.conversation = []
        self.history_epoch = 0
        self.request_history_epoch = None
        self.pending_guide = None
        self.active_guide_key = None
        self.conversation_id = None
        self.capture_cancel = threading.Event()
        self.summon_id = 0
        self.hold_timer = None
        self.record_limit = None
        self.recording = None
        self.dictation = None
        self.field_target = None
        self.pending_dictation_message = None
        self.dictation_hotkey = None
        self.speaker = None
        self.activity = "idle"
        self.voice_generation = 0
        self.voice_followup = None
        self.asked_aloud = False
        self.speak_turn = False
        self.previous_app = None
        self.guide_app = None
        self.setup_window = None
        self.status = A.NSStatusBar.systemStatusBar().statusItemWithLength_(A.NSVariableStatusItemLength)
        icon = A.NSImage.imageWithSystemSymbolName_accessibilityDescription_("cursorarrow.rays", "CursorMix")
        icon.setTemplate_(True)
        self.status.button().setImage_(icon)
        self.status.button().setImagePosition_(A.NSImageLeading)
        self.status.button().setToolTip_("CursorMix")
        self.status.button().setTitle_("")
        self.status.setAutosaveName_("Clicky")
        self.menu = A.NSMenu.alloc().initWithTitle_("CursorMix")
        # Dictation stays off. Its items live on a menu that is never shown.
        self.off_menu = A.NSMenu.alloc().initWithTitle_("Off")
        item(self.menu, "Ask", self, "ask:")
        self.home_item = item(self.menu, "Open CursorMix Home", self, "openHome:")
        self.continue_item = item(self.menu, "Continue with Current Screen", self, "continueConversation:")
        self.continue_item.setHidden_(True)
        settings = section(self.menu, "Settings")
        shortcuts = section(settings, "Keyboard Shortcut")
        self.shortcut_items = {}
        for name in SHORTCUTS:
            command = item(shortcuts, shortcut_label(name), self, "shortcut:")
            command.setRepresentedObject_(name)
            self.shortcut_items[name] = command
        dictation_menu = section(self.off_menu, "Dictation Shortcut")
        self.dictation_choices = {}
        for name in ("Off", *DICTATION_SHORTCUTS):
            command = item(
                dictation_menu, "Off" if name == "Off" else shortcut_label(name), self, "dictationKey:"
            )
            command.setRepresentedObject_(name)
            self.dictation_choices[name] = command
        languages = section(self.off_menu, "Dictation Language")
        self.dictation_languages = {}
        for code, name in DICTATION_LANGUAGES.items():
            command = item(languages, name, self, "dictationLanguage:")
            command.setRepresentedObject_(code)
            self.dictation_languages[code] = command
        item(self.off_menu, "Custom Dictation Words…", self, "editWords:")
        colors = section(settings, "Pointer Color")
        self.colors = {name: item(colors, name, self, "pointerColor:") for name in COLORS}
        styles = section(settings, "Answer Style")
        self.styles = {name: item(styles, name, self, "answerStyle:") for name in ANSWER_STYLES}
        providers = section(settings, "Answer Provider")
        self.providers = {s: item(providers, s, self, "provider:") for s in PROVIDERS}
        self.flags = {}
        for key, label in FEATURES.items():
            # The voice toggle sits in the main menu, next to Settings.
            command = item(self.menu if key == "voice" else settings, label, self, "flag:")
            command.setRepresentedObject_(key)
            self.flags[key] = command
        retention = section(settings, "History Retention")
        self.retention = {}
        for days in (1, 7, 30, 90):
            command = item(retention, f"{days} days", self, "days:")
            command.setTag_(days)
            self.retention[days] = command
        permissions = section(settings, "Permissions")
        permissions.setDelegate_(self)
        self.permissions = {}
        for name in PANES:
            command = item(permissions, name, self, "permission:")
            command.setRepresentedObject_(name)
            self.permissions[name] = command
        item(settings, "Set Up CursorMix…", self, "setup:")
        self.passive_app = None
        self.passive_since = 0
        self.nudged = False
        self.passive_timer = None
        self.login = item(settings, "Launch at Login", self, "loginToggle:")
        item(self.menu, "History", self, "history:")
        self.routine_menu = RoutineMenu.alloc().init()
        self.routine_menu.attach(self, section(self.menu, "Routines"))
        self.routine_menu.menu_item.setHidden_(True)
        self.step_help = None
        self.help_item = item(self.menu, "Help with This Step", self, "helpWithStep:")
        self.help_item.setHidden_(True)
        self.cancel_item = item(self.menu, "Cancel Current Request", self, "cancelRequest:")
        self.cancel_item.setHidden_(True)
        self.menu.addItem_(A.NSMenuItem.separatorItem())
        item(self.menu, "Quit CursorMix", self, "quit:")
        self.status.setMenu_(self.menu)
        self.hotkey = None
        startup_error = None
        try:
            self.hotkey = Hotkey(self.hotkey_down, self.hotkey_up)
            self.hotkey.register(self.store.stringForKey_("shortcut"))
        except (KeyError, RuntimeError) as error:
            startup_error = str(error)
        self.sync_menu()
        self.configure_home()
        if self.check_startup:
            # Inspect the actual objects built by normal startup in the user's WindowServer session.
            titles = [
                command.title() for command in self.status.menu().itemArray() if not command.isSeparatorItem()
            ]
            panel = self.panel
            valid = (
                titles
                == [
                    "Ask",
                    "Open CursorMix Home",
                    "Continue with Current Screen",
                    "Settings",
                    FEATURES["voice"],
                    "History",
                    "Routines",
                    "Help with This Step",
                    "Cancel Current Request",
                    "Quit CursorMix",
                ]
                and self.continue_item.isHidden()
                and self.help_item.isHidden()
                and panel.step_help.isHidden()
                and panel.step_question.isHidden()
                and panel.step_question.action() == "askAboutStep:"
                and panel.window.initialFirstResponder() == panel.field
                and panel.window.isFloatingPanel()
                and not panel.window.isVisible()
                and panel.field.target() == self
                and panel.field.action() == "submit:"
                and startup_error is None
            )
            registered = (
                shortcut_label(self.hotkey.shortcut) if self.hotkey and self.hotkey.shortcut else "none"
            )
            print(f"Menu: {', '.join(titles)}. Exclusive shortcut: {registered}.")
            print(f"Answer provider: {selected_provider(self.store.stringForKey_('provider'))}.")
            print(
                startup_error or ("Native startup check passed." if valid else "Native startup check failed.")
            )
            self.exit_code = 0 if valid else 1
            AppHelper.callAfter(end_run)
        elif startup_error:
            alert("Shortcut unavailable", startup_error + " Open Settings in the CursorMix menu to change it.")
        elif not self.trial and not self.store.boolForKey_("didCompleteOnboarding"):
            self.setup_(None)
        if not self.check_startup:
            memory.prune(self.store.integerForKey_("retentionDays"))
            self.configure_passive()
            self.floating = FloatingButton(self)
            self.presence_timer = A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "presenceTick:", None, True
            )
            self.overlay.pointer.enabled = self.store.boolForKey_("floatingPointer")
            self.overlay.pointer.follow()

    def presenceTick_(self, timer):
        self.floating.sync(icon_hidden(self.status))

    def showFloatingMenu_(self, sender):
        self.sync_menu()
        # Open the menu above the button so the pointer does not land on an item.
        frame, size = sender.window().frame(), self.menu.size()
        self.menu.popUpMenuPositioningItem_atLocation_inView_(
            None, (A.NSMaxX(frame) - size.width, A.NSMaxY(frame) + 8 + size.height), None
        )

    @objc.python_method
    def sync_menu(self):
        for label, command in self.shortcut_items.items():
            command.setState_(int(self.hotkey is not None and self.hotkey.shortcut == label))
        dictation = self.dictation_hotkey.shortcut if self.dictation_hotkey else "Off"
        for name, command in self.dictation_choices.items():
            command.setState_(int(name == dictation))
        language = self.store.stringForKey_("dictationLanguage")
        for code, command in self.dictation_languages.items():
            command.setState_(int(code == language))
        color = self.store.stringForKey_("pointerColor")
        color = color if color in COLORS else "Amber"
        self.overlay.pointer.appearance(color=color)
        for name, command in self.colors.items():
            command.setState_(int(name == color))
        style = self.store.stringForKey_("answerStyle")
        for name, command in self.styles.items():
            command.setState_(int(name == (style if style in ANSWER_STYLES else "Standard")))
        for label, command in self.providers.items():
            command.setState_(int(label == self.store.stringForKey_("provider")))
        for key, command in self.flags.items():
            command.setState_(int(self.store.boolForKey_(key)))
        for days, command in self.retention.items():
            command.setState_(int(self.store.integerForKey_("retentionDays") == days))
        self.login.setState_(int(LAUNCH_AGENT.exists()))

    def editWords_(self, sender):
        dialog = A.NSAlert.alloc().init()
        dialog.setMessageText_("Custom Dictation Words")
        dialog.setInformativeText_(
            "Enter one preferred spelling per line. For words CursorMix mishears, use spoken form = written form."
        )
        scroll = A.NSScrollView.alloc().initWithFrame_(((0, 0), (400, 160)))
        scroll.setHasVerticalScroller_(True)
        field = A.NSTextView.alloc().initWithFrame_(scroll.contentView().bounds())
        field.setRichText_(False)
        field.setAutomaticSpellingCorrectionEnabled_(False)
        field.setAutomaticTextReplacementEnabled_(False)
        field.setString_(self.store.stringForKey_("dictationSpellings") or "")
        scroll.setDocumentView_(field)
        dialog.setAccessoryView_(scroll)
        dialog.addButtonWithTitle_("Save")
        dialog.addButtonWithTitle_("Cancel")
        dialog.window().setInitialFirstResponder_(field)
        A.NSApp.activateIgnoringOtherApps_(True)
        while dialog.runModal() == A.NSAlertFirstButtonReturn:
            entries = str(field.string()).strip()
            try:
                custom_spellings(entries)
            except ValueError as error:
                dialog.setInformativeText_(str(error))
                continue
            self.store.setObject_forKey_(entries, "dictationSpellings")
            break

    def setup_(self, sender):
        if self.setup_window is None:
            self.setup_window = Setup.alloc().initWithShortcut_done_(
                self.store.stringForKey_("shortcut"), self.setup_done
            )
        self.setup_window.show()

    @objc.python_method
    def setup_done(self):
        self.store.setBool_forKey_(True, "didCompleteOnboarding")

    def menuWillOpen_(self, menu):
        for name, allowed in grants().items():
            self.permissions[name].setTitle_(f"{name} · {'Allowed' if allowed else 'Open Settings'}")

    def permission_(self, sender):
        open_permission(sender.representedObject())

    def days_(self, sender):
        self.store.setInteger_forKey_(sender.tag(), "retentionDays")
        memory.prune(sender.tag())
        self.sync_menu()

    def flag_(self, sender):
        key = sender.representedObject()
        enabled = not self.store.boolForKey_(key)
        self.store.setBool_forKey_(enabled, key)
        if key == "history" and not enabled:
            self.history_epoch += 1
        if key == "suggestFollowups":
            self.clear_suggestions()
            self.store.removeObjectForKey_("followupsIgnored")
            self.store.removeObjectForKey_("followupsSnoozedUntil")
        if key == "displayHome":
            self.configure_home()
        if key == "floatingPointer":
            self.overlay.pointer.enabled = enabled
            self.overlay.pointer.follow()
        if key == "voice" and not enabled:
            self.stop_voice()
        if key == "spokenReplies" and not enabled:
            self.stop_speech()
        if key in ("passive", "nudges"):
            self.configure_passive()
        if not enabled and key in ("screenContext", "teaching", "agentTasks", "pointing"):
            self.dismiss_(sender)
        self.sync_menu()

    @objc.python_method
    def configure_passive(self):
        if self.nudged:
            self.status.button().setTitle_("")
        if self.passive_timer:
            self.passive_timer.invalidate()
            self.passive_timer = None
        self.passive_app = None
        self.nudged = False
        if self.store.boolForKey_("passive"):
            self.passive_timer = A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                15, self, "passiveTick:", None, True
            )

    def passiveTick_(self, timer):
        active = A.NSWorkspace.sharedWorkspace().frontmostApplication()
        name = str(active.localizedName()) if active else ""
        now = time.monotonic()
        if name != self.passive_app:
            if self.nudged:
                self.status.button().setTitle_("")
            self.passive_app, self.passive_since, self.nudged = name, now, False
        elif (
            name
            and not self.nudged
            and now - self.passive_since >= 600
            and self.store.boolForKey_("nudges")
            and self.cancel_item.isHidden()
        ):
            self.status.button().setTitle_(f"Help with {name}?")
            self.nudged = True

    def shortcut_(self, sender):
        try:
            if self.hotkey is None:
                self.hotkey = Hotkey(self.hotkey_down, self.hotkey_up)
            name = sender.representedObject()
            self.hotkey.register(name)
            self.store.setObject_forKey_(name, "shortcut")
        except RuntimeError as error:
            alert("Shortcut unavailable", str(error))
        self.sync_menu()

    @objc.python_method
    def register_dictation(self, name):
        if name == "Off":
            if self.dictation_hotkey:
                self.dictation_hotkey.close()
                self.dictation_hotkey = None
            return
        hotkey = self.dictation_hotkey or Hotkey(
            self.dictate_down, self.dictate_up, identifier=2, choices=DICTATION_SHORTCUTS
        )
        try:
            hotkey.register(name)
        except (KeyError, RuntimeError):
            if self.dictation_hotkey is None:
                hotkey.close()
            raise
        self.dictation_hotkey = hotkey

    def dictationKey_(self, sender):
        name = sender.representedObject()
        try:
            self.register_dictation(name)
        except RuntimeError as error:
            alert("Dictation shortcut unavailable", str(error))
        else:
            if self.field_target is not None:
                self.stop_voice()
            self.store.setObject_forKey_(name, "dictationShortcut")
        self.sync_menu()

    def dictationLanguage_(self, sender):
        self.store.setObject_forKey_(sender.representedObject(), "dictationLanguage")
        self.sync_menu()

    def pointerColor_(self, sender):
        self.store.setObject_forKey_(sender.title(), "pointerColor")
        self.sync_menu()

    def answerStyle_(self, sender):
        self.store.setObject_forKey_(sender.title(), "answerStyle")
        self.sync_menu()

    def provider_(self, sender):
        self.clear_suggestions()
        self.store.setObject_forKey_(sender.title(), "provider")
        self.sync_menu()

    def loginToggle_(self, sender):
        try:
            set_login(not LAUNCH_AGENT.exists())
        except OSError as error:
            alert("Could not update Launch at Login", str(error))
        self.sync_menu()

    @objc.python_method
    def set_activity(self, activity):
        self.activity = activity
        self.home.activity("speaking" if self.speaker else activity)
        self.overlay.pointer.appearance(
            activity="speaking"
            if self.speaker is not None
            else ("thinking" if activity == "transcribing" else activity)
        )

    @objc.python_method
    def stop_speech(self):
        speaker, self.speaker = self.speaker, None
        if speaker is not None:
            speaker.setDelegate_(None)
            speaker.stopSpeaking()
            self.set_activity(self.activity)

    @objc.python_method
    def speak(self, text):
        self.stop_speech()
        speaker = A.NSSpeechSynthesizer.alloc().init()
        if speaker is None:
            return
        self.speaker = speaker
        speaker.setDelegate_(self)
        if speaker.startSpeakingString_(text):
            self.set_activity(self.activity)
        else:
            self.stop_speech()

    def speechSynthesizer_didFinishSpeaking_(self, speaker, finished):
        if speaker == self.speaker:
            speaker.setDelegate_(None)
            self.speaker = None
            self.set_activity(self.activity)

    @objc.python_method
    def stop_voice(self):
        self.voice_generation += 1
        if self.field_target is not None:
            self.cancel_item.setHidden_(True)
        self.field_target = None
        self.voice_followup = None
        if self.activity in ("listening", "transcribing"):
            self.set_activity("idle")
            self.status.button().setTitle_("")
        if self.hold_timer:
            self.hold_timer.invalidate()
            self.hold_timer = None
        if self.record_limit:
            self.record_limit.invalidate()
            self.record_limit = None
        if self.recording:
            self.recording.close()
            self.recording = None
        self.dictation = None

    @objc.python_method
    def hotkey_down(self):
        if ax.session_locked():
            return
        previous = self.previous_app
        followup = (self.conversation_id, tuple(self.conversation)) if self.panel.window.isVisible() else None
        self.ask_(None)
        if self.store.boolForKey_("voice"):
            if previous is not None and self.previous_app == previous:
                self.voice_followup = followup
            self.hold_timer = A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.25, self, "beginVoice:", None, False
            )

    @objc.python_method
    def show_dictation_message(self, message):
        if ax.session_locked():
            self.pending_dictation_message = message
            return
        active = A.NSWorkspace.sharedWorkspace().frontmostApplication()
        if active and active != A.NSRunningApplication.currentApplication():
            self.previous_app = active
            self.context = None
            self.clear_marks()
            self.turn_screen = None
            self.conversation = []
            self.conversation_id = None
        self.panel.output.setString_(message)
        self.panel.show(select_text=False)

    @objc.python_method
    def dictate_down(self):
        if ax.session_locked():
            return
        self.clear_point_choices()
        self.stop_voice()
        self.stop_speech()
        self.clear_suggestions()
        target = DictationField.current()
        self.cancel.set()
        self.close_drawings()
        self.set_step_help(None)
        self.generation += 1
        self.capture_cancel.set()
        self.pending_guide = None
        self.routine_menu.cancel_pending()
        self.summon_id += 1
        self.home.suspend(False)
        self.field_target = target
        self.set_activity("idle")
        self.overlay.hide()
        self.status.button().setTitle_("")
        if target is None:
            self.cancel_item.setHidden_(True)
            self.show_dictation_message(
                "Place the cursor in a supported text field and hold the dictation shortcut."
            )
            return
        self.cancel_item.setTitle_("Cancel Dictation")
        self.cancel_item.setHidden_(False)
        self.hold_timer = A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.25, self, "beginVoice:", None, False
        )

    @objc.python_method
    def dictate_up(self):
        if self.field_target is not None:
            self.release_voice()
            if self.activity != "transcribing":
                self.stop_voice()

    def beginVoice_(self, timer):
        if timer != self.hold_timer:
            return
        self.hold_timer = None
        self.recording = Recording()
        try:
            self.recording.start()
            self.record_limit = A.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                60.0, self, "endVoiceHold:", None, False
            )
            if self.field_target is None and self.voice_followup:
                self.conversation_id, previous_turns = self.voice_followup
                self.conversation = list(previous_turns)
            self.voice_followup = None
            self.set_activity("listening")
            self.status.button().setTitle_(" Listening")
        except (ValueError, OSError, objc.error) as error:
            if self.field_target is not None:
                self.show_dictation_message(str(error))
            else:
                self.panel.output.setString_(str(error))
            self.stop_voice()

    @objc.python_method
    def hotkey_up(self):
        if self.field_target is None:
            self.release_voice()

    def endVoiceHold_(self, timer):
        if timer == self.record_limit:
            self.release_voice()

    @objc.python_method
    def release_voice(self):
        if self.record_limit:
            self.record_limit.invalidate()
            self.record_limit = None
        if ax.session_locked():
            self.stop_voice()
            return
        if self.hold_timer:
            self.hold_timer.invalidate()
            self.hold_timer = None
            self.voice_followup = None
        if not self.recording:
            return
        recording, self.recording = self.recording, None
        recording.stop()
        ticket, generation, voice = self.summon_id, self.generation, self.voice_generation
        provider = selected_provider(self.store.stringForKey_("provider"))
        spellings = self.store.stringForKey_("dictationSpellings") or ""
        language = self.store.stringForKey_("dictationLanguage") or "default"
        self.set_activity("transcribing")
        self.status.button().setTitle_(" Transcribing")

        def transcribe():
            try:
                text = VoiceInput(recording, provider, spellings, language).read()
                AppHelper.callAfter(self.voice_ready, ticket, generation, voice, text, None)
            except Exception as error:  # noqa: BLE001 — relay transcription failures from the worker to AppKit
                AppHelper.callAfter(self.voice_ready, ticket, generation, voice, "", str(error))

        threading.Thread(target=transcribe, daemon=True, name="CursorMix dictation").start()

    @objc.python_method
    def voice_ready(self, ticket, generation, voice, text, error):
        if (ticket, generation, voice) != (self.summon_id, self.generation, self.voice_generation):
            return
        self.status.button().setTitle_("")
        self.set_activity("idle")
        if self.field_target is not None:
            target, self.field_target = self.field_target, None
            self.cancel_item.setHidden_(True)
            if error or not text:
                self.show_dictation_message(error or "No speech detected.")
            elif target.insert(text):
                return
            else:
                self.show_dictation_message("CursorMix couldn't insert into the original selection.\n\n" + text)
            return
        if error or not text:
            self.panel.output.setString_(error or "No speech detected.")
        elif self.context is None:
            self.dictation = text
        else:
            self.panel.field.setStringValue_(text)
            self.asked_aloud = True
            self.submit_(None)

    @objc.python_method
    def configure_home(self):
        enabled = self.store.boolForKey_("displayHome")
        self.home_item.setHidden_(not enabled)
        self.home.sync(enabled)
        if not enabled:
            self.panel.home_display = None

    def applicationDidChangeScreenParameters_(self, notification):
        if hasattr(self, "home") and hasattr(self, "home_item"):
            self.configure_home()
            if self.panel.home_display is not None and self.panel.window.isVisible():
                self.panel.place()

    def openHome_(self, sender):
        mouse = A.NSEvent.mouseLocation()
        screen = next((s for s in A.NSScreen.screens() if A.NSPointInRect(mouse, s.frame())), A.NSScreen.mainScreen())
        if screen:
            self.home.open(display_id(screen), pinned=True)

    @objc.python_method
    def ask_from_home(self, identifier):
        busy = not self.cancel_item.isHidden()
        if not busy and not self.panel.window.isVisible() and not self.conversation:
            self.ask_(self)
            self.panel.home_display = identifier
            return
        self.panel.home_display = identifier
        if busy:
            self.panel.place()
            self.panel.window.orderFrontRegardless()
        else:
            self.panel.show(select_text=False)

    def ask_(self, sender):
        self.panel.home_display = None
        self.home.suspend(True)
        self.clear_point_choices()
        self.clear_suggestions(ignored=True)
        self.continue_item.setHidden_(True)
        self.stop_voice()
        self.stop_speech()
        self.set_activity("idle")
        active = A.NSWorkspace.sharedWorkspace().frontmostApplication()
        locked = ax.session_locked()
        if (
            not locked
            and active
            and active.processIdentifier() != A.NSRunningApplication.currentApplication().processIdentifier()
        ):
            self.previous_app = active
        self.cancel.set()
        self.close_drawings()
        self.set_step_help(None)
        self.generation += 1
        self.pending_guide = None
        self.routine_menu.cancel_pending()
        self.summon_id += 1
        self.overlay.hide(suspend=True)
        self.capture_cancel.set()
        self.capture_cancel = threading.Event()
        self.cancel_item.setHidden_(True)
        self.status.button().setTitle_("")
        ticket = self.summon_id
        self.context = None
        self.clear_marks()
        self.turn_screen = None
        self.conversation = []
        self.conversation_id = None
        self.panel.window.orderOut_(None)
        self.panel.field.setStringValue_("")
        self.panel.output.setString_("")
        target = self.previous_app or active
        pid, name = (target.processIdentifier(), str(target.localizedName())) if target else (0, "")
        shortcut = self.hotkey.shortcut if sender is None and self.hotkey else None
        if self.store.boolForKey_("screenContext") and not locked:
            future = self.capture_queue.submit(capture.on_summon, pid, name, self.capture_cancel)
        else:
            future = Future()
            issue = "Unlock your Mac to use screen context." if locked else "Screen context is disabled."
            future.set_result(capture.Context(ax.Window(0, ""), issue=issue))
        future.add_done_callback(
            lambda f: AppHelper.callAfter(self.show_context, ticket, f, pid, name, shortcut)
        )

    @objc.python_method
    def show_context(self, ticket, future, pid, name, shortcut):
        if ax.session_locked():
            context = capture.Context(ax.Window(0, ""), issue="Unlock your Mac to use screen context.")
        else:
            try:
                context = future.result()
            except Exception:  # noqa: BLE001 — continue with a text-only panel if the capture worker failed
                context = capture.Context(ax.Window(pid, name), issue="Screen context is unavailable.")
        self.present_context(ticket, context, shortcut)

    @objc.python_method
    def present_context(self, ticket, context, shortcut):
        if ticket != self.summon_id:
            return
        self.context = context
        self.original_context = self.context
        self.home.suspend(False)
        self.panel.mark.setEnabled_(bool(self.context.jpeg and self.context.display))
        self.overlay.pointer.follow()
        self.panel.show()
        if self.pending_dictation_message:
            self.panel.output.setString_(self.pending_dictation_message)
            self.pending_dictation_message = None
        elif context.issue and self.store.boolForKey_("screenContext"):
            self.panel.output.setString_(
                f"{context.issue}\nAllow it under CursorMix menu → Settings → Permissions, then summon again."
            )
        if self.trial:
            self.trial.summoned(shortcut, self.panel.window.isKeyWindow())
        if self.routine_menu.submit_pending():
            return
        if self.pending_guide:
            resume = self.pending_guide
            self.pending_guide = None
            self.panel.field.setStringValue_(resume[1]["goal"])
            self.submit_request(resume)
            return
        if self.dictation:
            self.voice_ready(ticket, self.generation, self.voice_generation, self.dictation, None)
            self.dictation = None

    @objc.python_method
    def close_mark_window(self):
        if self.mark_window:
            self.mark_window.orderOut_(None)
            view = self.mark_window.contentView()
            view.snapshot = view.finish = None
            view.strokes = []
            self.mark_window = None

    @objc.python_method
    def clear_marks(self):
        self.close_mark_window()
        self.marks = []
        self.original_context = None
        self.panel.mark.setEnabled_(False)
        self.panel.mark.setTitle_("Mark Screen…")

    def markScreen_(self, sender):
        if self.mark_window or not self.original_context or not self.original_context.jpeg:
            return
        ticket = self.summon_id

        def finish(strokes):
            if ticket != self.summon_id or not self.mark_window:
                return
            self.close_mark_window()
            if strokes is not None:
                self.marks = strokes
                self.context = markup.annotated(self.original_context, strokes)
                self.panel.mark.setTitle_("Edit Marks…" if strokes else "Mark Screen…")
            self.overlay.pointer.follow()
            self.panel.show(select_text=False)

        self.close_drawings()
        self.overlay.hide(suspend=True)
        self.panel.window.orderOut_(None)
        try:
            self.mark_window = markup.open_markup(self.original_context, self.marks, finish)
        except ValueError as error:
            self.overlay.pointer.follow()
            self.panel.show(select_text=False)
            self.panel.output.setString_(str(error))

    def dismiss_(self, sender):
        self.home.suspend(False)
        self.home.collapse()
        self.panel.home_display = None
        self.clear_point_choices()
        self.clear_suggestions(ignored=True)
        self.continue_item.setHidden_(True)
        self.capture_cancel.set()
        self.stop_voice()
        self.stop_speech()
        self.set_activity("idle")
        self.overlay.hide()
        self.pending_guide = None
        self.routine_menu.cancel_pending()
        self.summon_id += 1
        self.context = None
        self.clear_marks()
        self.turn_screen = None
        self.conversation = []
        self.conversation_id = None
        self.cancel.set()
        self.close_drawings()
        self.set_step_help(None)
        self.generation += 1
        self.panel.window.orderOut_(None)
        self.status.button().setTitle_("")
        self.cancel_item.setHidden_(True)
        if self.previous_app:
            self.previous_app.activateWithOptions_(A.NSApplicationActivateIgnoringOtherApps)
            self.previous_app = None
        if self.trial and self.trial.dismissed(not self.panel.window.isVisible()):
            self.quit_(None)

    def submit_(self, sender):
        self.submit_request()

    @objc.python_method
    def submit_request(self, resume=None, routine=None):
        prompt = self.panel.typed.read()
        if not prompt:
            return
        self.clear_point_choices()
        self.clear_suggestions(ignored=True)
        self.continue_item.setHidden_(True)
        self.stop_voice()
        self.stop_speech()
        # A question asked by voice gets its answer read aloud.
        self.speak_turn = self.asked_aloud or self.store.boolForKey_("spokenReplies")
        self.asked_aloud = False
        self.set_activity("thinking")
        self.cancel.set()
        self.close_drawings()
        self.set_step_help(None)
        self.generation += 1
        self.turn_screen = None
        self.guide_app = None
        self.cancel = threading.Event()
        cancel = self.cancel
        self.cancel_item.setTitle_("Cancel Current Request")
        self.cancel_item.setHidden_(False)
        self.overlay.hide()
        request = self.generation
        self.panel.output.setString_("Waiting for a reply…")
        self.status.button().setTitle_(" ···")
        provider = selected_provider(routine.task["provider"] if routine else self.store.stringForKey_("provider"))
        if self.trial:
            self.trial.submitted(provider)

        context = self.context
        conversation = tuple(self.conversation)
        style = self.store.stringForKey_("answerStyle")
        switches = {
            route: self.store.boolForKey_(key)
            for route, key in (
                ("point_to", "pointing"),
                ("annotate", "pointing"),
                ("teach", "teaching"),
                ("agent_task", "agentTasks"),
            )
        }
        screen_enabled = self.store.boolForKey_("screenContext")

        self.active_guide_key = resume[0] if resume else None
        history_epoch = self.history_epoch
        save_guides = self.store.boolForKey_("history")
        self.request_history_epoch = history_epoch if save_guides else None
        parent_conversation = self.conversation_id

        def stream():
            nonlocal context
            first = True
            result = ""
            failure = None
            cancelled = False
            try:
                if routine:
                    route = "agent_task"
                else:
                    route = "teach" if resume else router.classify(prompt, provider)
                if route == "screen_question":
                    route = "answer"
                if hasattr(self.trial, "routed"):
                    self.trial.routed(route)
                if not switches.get(route, True):
                    if routine:
                        raise ValueError("This feature is disabled in CursorMix Settings.")
                    route = "answer"
                if not screen_enabled and route in ("point_to", "teach", "agent_task"):
                    raise ValueError("Turn on Use Screen Context in CursorMix Settings for this request.")
                if context:
                    AppHelper.callAfter(self.set_turn_screen, request, memory.screen_excerpt(context.window))
                if route == "annotate":
                    summary, marks, links, display = annotations.compose(
                        prompt, provider, context, conversation
                    )
                    if not cancel.is_set():
                        AppHelper.callAfter(
                            self.show_annotations,
                            request,
                            prompt,
                            summary,
                            marks,
                            links,
                            display,
                            context.jpeg,
                        )
                    return
                if route == "figure":
                    diagram = figure.compose(prompt, provider, context, conversation)
                    if not cancel.is_set():
                        AppHelper.callAfter(self.show_figure, request, prompt, diagram)
                    return
                if route == "agent_task":
                    if not context or not context.window.app:
                        raise ValueError("Summon CursorMix with an app in front, then ask again.")
                    if not routine:
                        # One confirmation covers the whole task. It shows what CursorMix is about to do.
                        AppHelper.callAfter(self.task_progress, request, "Working out the steps…")
                        steps = agent.outline(prompt, context, provider)
                        decision = Future()
                        AppHelper.callAfter(self.confirm_task, request, prompt, steps, decision)
                        while not decision.done():
                            if cancel.wait(0.05):
                                return
                        if not decision.result():
                            raise InterruptedError("Cancelled. Nothing was changed.")
                    AppHelper.callAfter(self.background_task, request)
                    own_pid = os.getpid()

                    def confirm(pid, app, description):
                        # Act only while the task's app is in front. Clicking Stop or switching apps ends it.
                        for _ in range(40):
                            focused = ax.frontmost()[0]
                            if cancel.is_set() or focused not in (pid, own_pid):
                                return False
                            if focused == pid:
                                return True
                            AppHelper.callAfter(self.focus_task, request, pid)
                            cancel.wait(0.1)
                        return False

                    def progress(message, target=None):
                        AppHelper.callAfter(self.task_progress, request, message, target)

                    observer = getattr(self.trial, "agent_event", None)
                    try:
                        result = agent.run(prompt, context, provider, cancel, confirm, progress, observer)
                    finally:
                        if observer:
                            observer("stopped", {"cancelled": cancel.is_set()})
                    AppHelper.callAfter(self.present_result, request, result)
                    AppHelper.callAfter(self.finished, request, prompt, None)
                    return
                if route == "teach":
                    if not resume:
                        # A walkthrough happens where the task is done, which may be another app.
                        AppHelper.callAfter(self.task_progress, request, "Finding the right place to start…")
                        context = teach.place(prompt, context, provider, cancel)
                        if cancel.is_set():
                            return
                        AppHelper.callAfter(self.set_guide_place, request, context)
                    AppHelper.callAfter(self.focus_guide, request)

                    def display(message, target, help_requests, explanation):
                        AppHelper.callAfter(
                            self.show_instruction,
                            request,
                            message,
                            target,
                            help_requests,
                            explanation,
                        )

                    key, saved = resume or ("walkthrough/" + uuid.uuid4().hex, None)
                    first_save = not resume

                    def checkpoint(state):
                        nonlocal first_save
                        state["conversation"] = parent_conversation
                        AppHelper.callAfter(self.save_guide, request, history_epoch, key, state, first_save)
                        first_save = False

                    conclusion = teach.guide(
                        prompt,
                        context,
                        provider,
                        cancel,
                        display,
                        answer_style=style,
                        saved=saved,
                        checkpoint=checkpoint if save_guides else None,
                        retarget=lambda target, message: AppHelper.callAfter(
                            self.retarget_step, request, target, message, context.window.pid
                        ),
                    )
                    if not cancel.is_set():
                        AppHelper.callAfter(self.task_progress, request, conclusion)
                        AppHelper.callAfter(self.finished, request, prompt, None)
                    return
                if route == "point_to":
                    try:
                        target = locate(prompt, context, provider)
                    except MultipleTargets as choices:
                        AppHelper.callAfter(self.offer_point_choices, request, prompt, choices.targets)
                        return
                    # A slider or switch is labelled with its value, which is no name to show a person.
                    named = not target.label.replace(".", "").isdigit()
                    AppHelper.callAfter(self.show_target, request, target, None if named else "Here")
                    reply = [
                        f"Look for {target.label} in the highlighted area."
                        if named
                        else "It’s the control highlighted on your screen."
                    ]
                else:
                    pointed = [] if context and switches["point_to"] else None
                    reply = answer(
                        prompt, provider, context, conversation=conversation, style=style, pointed=pointed
                    )
                for token in reply:
                    if request != self.generation:
                        return
                    AppHelper.callAfter(self.append_token, request, token, first)
                    if first and route == "answer" and pointed:
                        # The answer names a control on screen: put the arrow on it while the text streams.
                        threading.Thread(
                            target=self.point_at, args=(request, pointed[0], prompt, context, provider), daemon=True
                        ).start()
                    first = False
                if first:
                    raise RuntimeError("The provider returned an empty answer.")
                AppHelper.callAfter(self.finished, request, prompt, None)
            except Exception as error:  # noqa: BLE001 — finish the native request on provider or framework errors
                cancelled = isinstance(error, InterruptedError)
                failure = str(error)
                AppHelper.callAfter(self.finished, request, prompt, failure)
            finally:
                if routine:
                    status = "Cancelled" if cancel.is_set() or cancelled else "Complete"
                    if failure and status != "Cancelled":
                        status = "Failed"
                    AppHelper.callAfter(self.routine_menu.end_desktop, routine, status, failure or result)

        threading.Thread(target=stream, name="CursorMix answer", daemon=True).start()

    @objc.python_method
    def close_drawings(self):
        if self.annotation is not None:
            self.annotation.close()
            self.annotation = None
        if self.figure_window is not None:
            if self.figure_window.isVisible():
                self.figure_window.close()
            self.figure_window = None

    @objc.python_method
    def show_annotations(self, request, prompt, summary, marks, links, display, jpeg):
        if request != self.generation:
            return
        try:
            self.annotation = annotations.Drawing(
                marks, links, display, jpeg, self.panel.window.screen(), self.panel.avoid
            )
            if self.annotation.live and A.NSApp.isActive():
                self.focus_guide(request)
        except (ValueError, RuntimeError, objc.error) as error:
            self.finished(request, prompt, str(error))
            return
        self.panel.output.setString_(summary)
        self.finished(request, prompt, None)

    @objc.python_method
    def show_figure(self, request, prompt, diagram):
        if request != self.generation:
            return
        try:
            self.figure_window = figure.open_figure(diagram, self.panel.window.screen())
        except (ValueError, RuntimeError, objc.error) as error:
            self.finished(request, prompt, str(error))
            return
        self.panel.output.setString_(diagram["summary"])
        self.finished(request, prompt, None)

    @objc.python_method
    def set_guide_place(self, request, context):
        if request == self.generation and context:
            self.context = context
            self.guide_app = A.NSRunningApplication.runningApplicationWithProcessIdentifier_(context.window.pid)

    @objc.python_method
    def focus_guide(self, request):
        app = self.guide_app or self.previous_app
        if request == self.generation and app:
            app.activateWithOptions_(A.NSApplicationActivateIgnoringOtherApps)

    @objc.python_method
    def background_task(self, request):
        """Hand the keyboard back to the task's app and keep the panel up for progress and Stop."""
        if request == self.generation:
            self.panel.stop.setHidden_(False)
            if self.previous_app:
                self.previous_app.activateWithOptions_(A.NSApplicationActivateIgnoringOtherApps)
            self.panel.window.orderFrontRegardless()

    @objc.python_method
    def focus_task(self, request, pid):
        target = A.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if request == self.generation and target:
            target.activateWithOptions_(A.NSApplicationActivateIgnoringOtherApps)

    @objc.python_method
    def confirm_task(self, request, prompt, steps, decision):
        allowed = False
        try:
            if request != self.generation:
                return
            plan = "\n".join(f"{number}. {step}" for number, step in enumerate(steps, start=1))
            choice = alert(
                "Let CursorMix do this for you?",
                f"“{prompt[:200]}”\n\n{plan}\n\nCursorMix will click and type in your apps. "
                "Press Stop in the CursorMix panel to end it at any time.",
                ("Do It", "Cancel"),
            )
            allowed = choice == A.NSAlertFirstButtonReturn and request == self.generation
        finally:
            decision.set_result(allowed)

    def stopRequest_(self, sender):
        """End the running request and keep the panel open for the next question."""
        self.cancel.set()
        self.generation += 1
        self.stop_speech()
        self.overlay.hide()
        self.set_step_help(None)
        self.set_activity("idle")
        self.status.button().setTitle_("")
        self.cancel_item.setHidden_(True)
        self.panel.stop.setHidden_(True)
        self.panel.output.setString_("Stopped.")

    @objc.python_method
    def present_result(self, request, result):
        if request == self.generation:
            self.overlay.hide()
            self.panel.window.orderFrontRegardless()
            self.panel.output.setString_(result)

    def cancelRequest_(self, sender):
        if self.field_target is not None:
            self.stop_voice()
            return
        self.dismiss_(sender)

    @objc.python_method
    def task_progress(self, request, message, target=None):
        if request == self.generation:
            self.panel.output.setString_(message)
            self.status.button().setTitle_(" " + message.partition(".")[0][:24])
            if target:
                self.panel.avoid(self.overlay.show(target, message))
            else:
                self.overlay.hide()

    @objc.python_method
    def retarget_step(self, request, target, message, pid):
        if request == self.generation and not self.cancel.is_set():
            if target and not ax.session_locked() and ax.frontmost()[0] == pid:
                self.panel.avoid(self.overlay.reposition_visual(target, message))
            else:
                self.overlay.hide()

    @objc.python_method
    def show_instruction(self, request, message, target, help_requests, explanation):
        if request == self.generation:
            self.set_step_help(help_requests)
            self.set_activity("idle" if help_requests is not None else "thinking")
            self.panel.window.orderFrontRegardless()
            self.task_progress(request, message, target)
            if help_requests is None:
                self.stop_speech()
                return
            if explanation:
                self.panel.output.setString_(f"{explanation}\n\n{message}")
            if self.speak_turn:
                self.speak(explanation or message)

    @objc.python_method
    def set_step_help(self, requests):
        self.panel.stop.setHidden_(True)
        self.step_help = requests
        active = requests is not None
        self.help_item.setHidden_(not active)
        self.panel.step_help.setHidden_(not active)
        self.panel.step_help.setEnabled_(active)
        self.panel.step_question.setStringValue_("")
        self.panel.step_question.setHidden_(not active)
        self.panel.step_question.setEnabled_(active)
        self.panel.field.setHidden_(active)
        self.panel.mark.setHidden_(active)

    def helpWithStep_(self, sender):
        self.request_step_help("")

    def askAboutStep_(self, sender):
        question = self.panel.step_question.stringValue().strip()
        if question:
            self.request_step_help(question)

    @objc.python_method
    def request_step_help(self, question):
        if self.step_help is None or self.cancel.is_set() or ax.session_locked():
            return
        self.step_help.put(question)
        self.step_help = None
        self.help_item.setHidden_(True)
        self.panel.step_help.setEnabled_(False)
        self.panel.step_question.setEnabled_(False)
        self.stop_speech()
        self.set_activity("thinking")
        status = "Answering your question…" if question else "Checking the current step…"
        self.task_progress(self.generation, status)
        self.focus_guide(self.generation)

    @objc.python_method
    def clear_point_choices(self):
        if self.point_choices is not None:
            self.point_choices = None
            self.panel.set_target_choices([])

    @objc.python_method
    def offer_point_choices(self, request, prompt, targets):
        if request != self.generation or self.cancel.is_set():
            return
        self.point_choices = (request, prompt, targets)
        labels = []
        region = self.context.display or self.context.window.frame
        for index, target in enumerate(targets, start=1):
            x, y, w, h = target.frame
            position = ""
            if region:
                left, top, width, height = region
                position = f" · {(x+w/2-left)/width:.0%} across, {(y+h/2-top)/height:.0%} down"
            labels.append(f"{index}. {target.label[:64]}{position}")
        self.set_activity("idle")
        self.status.button().setTitle_(" Choose a control")
        self.panel.output.setString_("I found several matching controls. Which one should I point to?")
        self.panel.set_target_choices(labels)
        self.panel.window.orderFrontRegardless()

    def selectPointTarget_(self, sender):
        if self.point_choices is None:
            return
        request, prompt, targets = self.point_choices
        index = sender.indexOfSelectedItem() - 1
        if request != self.generation or self.cancel.is_set() or not 0 <= index < len(targets):
            return
        target = targets[index]
        self.clear_point_choices()
        self.panel.output.setString_(f"Look for {target.label} in the highlighted area.")
        self.show_target(request, target)
        self.finished(request, prompt, None)

    @objc.python_method
    def show_target(self, request, target, caption=None):
        if request == self.generation and not self.cancel.is_set():
            self.panel.avoid(self.overlay.show(target, caption))

    @objc.python_method
    def point_at(self, request, label, prompt, context, provider):
        """Resolve the control an answer named and show the arrow on it. A miss leaves the answer as it is."""
        try:
            target = locate(label, context, provider, instruction=prompt)
        except MultipleTargets as choices:
            target = choices.targets[0]
        except Exception:  # noqa: BLE001 — the arrow is an addition to an answer that already stands
            return
        AppHelper.callAfter(self.show_target, request, target, label[:60])

    @objc.python_method
    def append_token(self, request, token, first):
        if request != self.generation:
            return
        output = self.panel.output
        if first:
            output.setString_("")
        storage = output.textStorage()
        storage.replaceCharactersInRange_withString_((storage.length(), 0), token)
        output.scrollRangeToVisible_((storage.length(), 0))
        if self.trial and not self.trial.saw_reply and self.panel.window.isVisible():
            self.panel.window.displayIfNeeded()
            self.trial.saw_reply = bool(output.string().strip())

    @objc.python_method
    def set_turn_screen(self, request, screen):
        if request == self.generation:
            self.turn_screen = screen

    @objc.python_method
    def save_guide(self, request, epoch, key, state, create):
        if request != self.generation or epoch != self.history_epoch or not self.store.boolForKey_("history"):
            return
        try:
            memory.save_walkthrough(key, state, self.store.integerForKey_("retentionDays"), create=create)
            self.active_guide_key = key
        except (OSError, sqlite3.Error) as exc:
            alert("Could not save walkthrough progress", str(exc))

    @objc.python_method
    def finished(self, request, prompt, error):
        if request != self.generation:
            return
        self.set_step_help(None)
        self.status.button().setTitle_("")
        self.set_activity("idle")
        if (self.overlay.window or (self.annotation and self.annotation.live)) and not error:
            self.cancel_item.setTitle_("Clear Guidance")
        else:
            self.cancel_item.setHidden_(True)
        if error:
            self.overlay.hide()
            self.present_result(request, error)
            return
        turn = {"request": memory.compact(prompt), "reply": str(self.panel.output.string())[:800]}
        if self.turn_screen:
            turn["screen_at_answer"] = self.turn_screen
        self.conversation.append(turn)
        self.conversation = self.conversation[-4:]
        self.continue_item.setHidden_(False)
        self.schedule_suggestions()
        if self.speak_turn:
            self.speak(self.panel.output.string())
        if self.trial:
            self.trial.completed = True
        try:
            if self.request_history_epoch == self.history_epoch and self.store.boolForKey_("history"):
                self.conversation_id = memory.remember(
                    prompt,
                    self.panel.output.string(),
                    self.store.integerForKey_("retentionDays"),
                    conversation=self.conversation_id,
                    screen=self.turn_screen,
                    completed_walkthrough=self.active_guide_key,
                )
            elif self.active_guide_key:
                memory.discard_walkthrough(self.active_guide_key)
            self.active_guide_key = None
        except (OSError, sqlite3.Error) as exc:
            alert("Could not save history", str(exc))

    @objc.python_method
    def clear_suggestions(self, *, ignored=False):
        self.suggestion_generation += 1
        if ignored and any(not b.isHidden() for b in self.panel.suggestions):
            count = self.store.integerForKey_("followupsIgnored") + 1
            if count >= 8:
                self.store.setDouble_forKey_(time.time() + 3600, "followupsSnoozedUntil")
                count = 0
            self.store.setInteger_forKey_(count, "followupsIgnored")
        self.panel.set_suggestions([])

    @objc.python_method
    def schedule_suggestions(self):
        if (
            not self.store.boolForKey_("suggestFollowups")
            or not self.panel.window.isVisible()
            or time.time() < self.store.doubleForKey_("followupsSnoozedUntil")
        ):
            return
        ticket = self.suggestion_generation
        turns = tuple({"request": t["request"], "reply": t["reply"]} for t in self.conversation)
        provider = selected_provider(self.store.stringForKey_("provider"))
        available = ["answers"]
        if self.store.boolForKey_("screenContext") and self.context and self.context.jpeg:
            available.append("screen questions")
            available.extend(
                label
                for key, label in (
                    ("pointing", "pointing and annotations"),
                    ("teaching", "walkthroughs"),
                    ("agentTasks", "desktop actions"),
                )
                if self.store.boolForKey_(key)
            )
        AppHelper.callLater(0.35, self.start_suggestions, ticket, turns, provider, available)

    @objc.python_method
    def start_suggestions(self, ticket, turns, provider, available):
        if ticket != self.suggestion_generation or not self.panel.window.isVisible():
            return

        def fetch():
            try:
                choices = followup_questions(turns, provider, available)
            except (OSError, ValueError, RuntimeError):
                return
            AppHelper.callAfter(self.show_suggestions, ticket, choices)

        threading.Thread(target=fetch, name="CursorMix follow-ups", daemon=True).start()

    @objc.python_method
    def show_suggestions(self, ticket, choices):
        if ticket == self.suggestion_generation and self.panel.window.isVisible():
            self.panel.set_suggestions(choices)

    def selectSuggestion_(self, sender):
        if sender not in self.panel.suggestions or sender.isHidden() or not sender.title():
            return
        prompt = str(sender.title())
        self.clear_suggestions()
        self.store.removeObjectForKey_("followupsIgnored")
        self.panel.field.setStringValue_(prompt)
        self.submit_(sender)

    def continueConversation_(self, sender):
        if self.conversation:
            self.resume_conversation(self.conversation_id, self.conversation)

    def history_(self, sender):
        try:
            days = self.store.integerForKey_("retentionDays")
            saved = memory.conversations(days)
            if not saved:
                alert("CursorMix History", "Completed requests and saved walkthroughs appear here.")
                return
            original_app = A.NSWorkspace.sharedWorkspace().frontmostApplication()
            box = A.NSAlert.alloc().init()
            box.setMessageText_("CursorMix History")
            view = A.NSView.alloc().initWithFrame_(((0, 0), (420, 336)))
            search = A.NSSearchField.alloc().initWithFrame_(((0, 305), (420, 26)))
            search.setPlaceholderString_("Search saved conversations")
            search.setAccessibilityLabel_("Search saved conversations")
            search.setSendsSearchStringImmediately_(True)
            search.setTarget_(self)
            search.setAction_("searchHistory:")
            view.addSubview_(search)
            picker = A.NSPopUpButton.alloc().initWithFrame_pullsDown_(((0, 268), (420, 28)), False)
            picker.setAccessibilityLabel_("Saved conversations")
            picker.setTarget_(self)
            picker.setAction_("previewHistory:")
            view.addSubview_(picker)
            scroll = A.NSScrollView.alloc().initWithFrame_(((0, 0), (420, 260)))
            scroll.setHasVerticalScroller_(True)
            body = A.NSTextView.alloc().initWithFrame_(((0, 0), (400, 260)))
            body.setEditable_(False)
            body.setAccessibilityLabel_("Conversation excerpts")
            scroll.setDocumentView_(body)
            view.addSubview_(scroll)
            box.setAccessoryView_(view)
            for label in ("Done", "Resume Walkthrough", "Clear History…"):
                box.addButtonWithTitle_(label)
            self.history_picker = picker
            self.history_preview = body
            self.history_resume = box.buttons()[1]
            try:
                self.show_history_matches(saved)
                box.window().setInitialFirstResponder_(search)
                A.NSApp.activateIgnoringOtherApps_(True)
                choice = box.runModal()
            finally:
                self.history_preview = self.history_picker = self.history_resume = None
            if choice == A.NSAlertSecondButtonReturn and picker.selectedItem():
                key = picker.selectedItem().representedObject()
                messages = memory.turns(key, days)
                if messages:
                    if (
                        not ax.session_locked()
                        and original_app
                        and original_app != A.NSRunningApplication.currentApplication()
                    ):
                        self.previous_app = original_app
                    self.resume_conversation(key, messages)
                else:
                    alert("Conversation unavailable", "Its saved excerpts have expired or were cleared.")
            elif (
                choice == A.NSAlertThirdButtonReturn
                and alert(
                    "Clear all CursorMix history?",
                    "This removes saved conversations and unfinished walkthroughs.",
                    ("Cancel", "Clear"),
                )
                == A.NSAlertSecondButtonReturn
            ):
                memory.clear()
                self.history_epoch += 1
                self.active_guide_key = None
                self.pending_guide = None
                self.clear_suggestions()
                self.continue_item.setHidden_(True)
                self.conversation_id = None
                self.turn_screen = None
                self.conversation = []
                self.voice_followup = None
        except (OSError, sqlite3.Error) as error:
            alert("Could not open history", str(error))

    @objc.python_method
    def show_history_matches(self, saved):
        picker = self.history_picker
        previous = picker.selectedItem()
        key = previous.representedObject() if previous else None
        picker.removeAllItems()
        for conversation, title, timestamp in saved:
            date = time.strftime("%b %d, %H:%M", time.localtime(timestamp))
            entry = picker.menu().addItemWithTitle_action_keyEquivalent_(f"{date} · {title}", None, "")
            entry.setRepresentedObject_(conversation)
        picker.setEnabled_(bool(saved))
        self.history_resume.setEnabled_(bool(saved))
        if saved:
            index = picker.indexOfItemWithRepresentedObject_(key) if key else -1
            picker.selectItemAtIndex_(max(0, index))
            self.previewHistory_(picker)
        else:
            self.history_preview.setString_("No matching conversations.")
            self.history_resume.setTitle_("Resume")

    def searchHistory_(self, sender):
        if self.history_picker is None:
            return
        try:
            self.show_history_matches(
                memory.conversations(self.store.integerForKey_("retentionDays"), str(sender.stringValue()))
            )
        except (OSError, sqlite3.Error):
            self.show_history_matches([])
            self.history_preview.setString_("The saved excerpts could not be read.")

    def previewHistory_(self, sender):
        try:
            messages = memory.turns(
                sender.selectedItem().representedObject(), self.store.integerForKey_("retentionDays")
            )
            excerpts = []
            for message in messages:
                screen = message.get("screen_at_answer", {})
                location = " · ".join(filter(None, (screen.get("app"), screen.get("window"))))
                excerpts.append("\n".join(filter(None, (message["request"], location, message["reply"]))))
            guide = memory.walkthrough(
                sender.selectedItem().representedObject(), self.store.integerForKey_("retentionDays")
            )
            self.history_resume.setTitle_("Resume Walkthrough" if guide else "Resume")
            if guide:
                excerpts = [
                    guide["goal"],
                    guide["starting"]["app"],
                    f"{len(guide['verified'])} verified steps",
                    guide["pending"]["instruction"] if guide["pending"] else "Checking the next step…",
                ]
            text = "\n\n".join(excerpts)
        except (OSError, sqlite3.Error):
            text = "The saved excerpts could not be read."
        self.history_preview.setString_(text)

    @objc.python_method
    def resume_conversation(self, key, messages):
        guide = memory.walkthrough(key, self.store.integerForKey_("retentionDays"))
        if guide:
            if not all(self.store.boolForKey_(flag) for flag in ("teaching", "screenContext")):
                alert(
                    "Walkthrough unavailable",
                    "Enable Step-by-Step Guidance and Use Screen Context in Settings.",
                )
                return
            self.ask_(self)
            self.pending_guide = (key, guide)
            self.conversation_id = guide.get("conversation")
            if self.conversation_id:
                self.conversation = memory.turns(
                    self.conversation_id, self.store.integerForKey_("retentionDays")
                )[-4:]
            return
        self.ask_(self)
        self.conversation_id = key
        self.conversation = messages[-4:]
        self.continue_item.setHidden_(False)
        self.panel.field.setStringValue_("")
        self.panel.output.setString_(f"{messages[-1]['request']}\n{messages[-1]['reply']}")

    def quit_(self, sender):
        A.NSApp.terminate_(sender)

    def applicationWillTerminate_(self, notification):
        self.routine_menu.stop()
        if getattr(self, "floating", None):
            self.presence_timer.invalidate()
            self.floating.close()
        self.home.sync(False)
        self.clear_suggestions()
        self.capture_cancel.set()
        self.stop_voice()
        self.stop_speech()
        self.cancel.set()
        self.close_drawings()
        self.set_step_help(None)
        if self.hotkey:
            self.hotkey.close()
        if self.dictation_hotkey:
            self.dictation_hotkey.close()
        self.overlay.close()
        self.capture_queue.shutdown(wait=False, cancel_futures=True)
        self.panel.close()


def end_run():
    # stop: takes effect after the current NSEvent; a timer callback may have no pending event.
    app = A.NSApplication.sharedApplication()
    app.stop_(None)
    wake = A.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
        A.NSEventTypeApplicationDefined, (0, 0), 0, 0, 0, None, 0, 0, 0
    )
    app.postEvent_atStart_(wake, False)


def main(trial=None, *, delegate_type=ClickyDelegate, argv=None):
    parser = argparse.ArgumentParser(description="CursorMix menu-bar assistant")
    parser.add_argument("--check-startup", action="store_true", help="inspect native startup, then exit")
    args = parser.parse_args(argv)
    app = A.NSApplication.sharedApplication()
    app.setActivationPolicy_(A.NSApplicationActivationPolicyAccessory)
    edit = A.NSMenu.alloc().initWithTitle_("Main")
    changes = section(edit, "Edit")
    for title, action, key in (
        ("Cut", "cut:", "x"),
        ("Copy", "copy:", "c"),
        ("Paste", "paste:", "v"),
        ("Select All", "selectAll:", "a"),
    ):
        changes.addItemWithTitle_action_keyEquivalent_(title, action, key)
    app.setMainMenu_(edit)
    delegate = delegate_type.alloc().init()
    delegate.owns_scheduler = trial is None and delegate_type is ClickyDelegate
    delegate.trial = trial
    delegate.check_startup = args.check_startup
    delegate.exit_code = 1
    app.setDelegate_(delegate)
    app.run()
    if args.check_startup:
        sys.exit(delegate.exit_code)
