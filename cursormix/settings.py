import os
import plistlib
import sys
import tempfile
from functools import cache
from pathlib import Path

import yaml
from Foundation import NSBundle, NSUserDefaults

from .hotkey import SHORTCUTS

DOMAIN = "local.clicky.assistant"
LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / f"{DOMAIN}.plist"


def preferences(domain=DOMAIN):
    store = (
        NSUserDefaults.standardUserDefaults()
        if NSBundle.mainBundle().bundleIdentifier() == domain
        else NSUserDefaults.alloc().initWithSuiteName_(domain)
    )
    defaults = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    if defaults["shortcut"] not in SHORTCUTS:
        raise ValueError("Choose a supported CursorMix shortcut in config.yaml.")
    store.registerDefaults_(defaults)
    return store


def set_login(enabled):
    if not enabled:
        LAUNCH_AGENT.unlink(missing_ok=True)
        return
    LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
    packaged = NSBundle.mainBundle().bundleIdentifier() == DOMAIN
    launch = {
        "Label": DOMAIN,
        "ProgramArguments": [sys.executable] if packaged else [sys.executable, "-m", "cursormix"],
        "WorkingDirectory": str(Path(__file__).resolve().parent.parent),
        "RunAtLoad": True,
        "LimitLoadToSessionType": "Aqua",
    }
    with tempfile.NamedTemporaryFile(dir=LAUNCH_AGENT.parent, delete=False) as output:
        try:
            plistlib.dump(launch, output)
            output.flush()
            os.replace(output.name, LAUNCH_AGENT)
        finally:
            Path(output.name).unlink(missing_ok=True)


FEATURES = {
    "displayHome": "Show CursorMix Home at Top of Display",
    "floatingPointer": "Show Floating Pointer",
    "screenContext": "Use Screen Context",
    "suggestFollowups": "Suggest Follow-Ups",
    "pointing": "Point to Controls",
    "teaching": "Step-by-Step Guidance",
    "agentTasks": "Desktop Actions",
    "history": "Save Request History",
    "passive": "Watch Frontmost App Name",
    "nudges": "Offer Help After 10 Minutes",
    "voice": "Voice: Hold Shortcut to Talk, Hear the Answer",
    "spokenReplies": "Speak Typed Answers Too",
}

PANES = {
    "Screen Recording": "Privacy_ScreenCapture",
    "Accessibility": "Privacy_Accessibility",
    "Microphone": "Privacy_Microphone",
}


@cache
def installed_panes():
    """Map unique localized pane names to the Mac's built-in Settings UI extensions."""
    names = {}
    plugins = Path("/System/Library/ExtensionKit/Extensions")
    for path in plugins.glob("*.appex"):
        bundle = NSBundle.bundleWithPath_(str(path))
        if not bundle:
            continue
        attributes = bundle.objectForInfoDictionaryKey_("EXAppExtensionAttributes") or {}
        if attributes.get("EXExtensionPointIdentifier") != "com.apple.Settings.extension.ui":
            continue
        name = bundle.objectForInfoDictionaryKey_("CFBundleDisplayName")
        identifier = bundle.bundleIdentifier()
        if name and identifier:
            names[str(name)] = None if name in names else str(identifier)
    return {name: identifier for name, identifier in sorted(names.items()) if identifier}


def grants():
    import ApplicationServices as AX
    import AVFoundation as AV
    import Quartz as Q

    return {
        "Screen Recording": bool(Q.CGPreflightScreenCaptureAccess()),
        "Accessibility": bool(AX.AXIsProcessTrusted()),
        "Microphone": AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeAudio)
        == AV.AVAuthorizationStatusAuthorized,
    }


def open_permission(name):
    from AppKit import NSURL, NSWorkspace

    url = "x-apple.systempreferences:com.apple.preference.security?" + PANES[name]
    NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url))
