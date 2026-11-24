"""Exercise dictation registration and reconfiguration without recording microphone input."""

import argparse
import subprocess
import sys
import uuid

import AppKit as A
import objc
from objc import super

from cursormix.app import ClickyDelegate, main
from cursormix.hotkey import DICTATION_SHORTCUTS, SHORTCUTS, Hotkey
from cursormix.settings import preferences


def probe(name, busy):
    choices = SHORTCUTS if name in SHORTCUTS else DICTATION_SHORTCUTS
    key = Hotkey(lambda: None, choices=choices)
    try:
        try:
            key.register(name)
        except RuntimeError as error:
            if not busy or "already reserved" not in str(error):
                raise
        else:
            if busy:
                raise RuntimeError(f"A second process acquired {name}.")
    finally:
        key.close()


class DictationCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = f"local.clicky.dictation-check.{uuid.uuid4().hex}"
        store = preferences(self.suite)
        store.registerDefaults_({"dictationShortcut": "Control Option D"})
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        try:
            assert self.dictation_hotkey.shortcut == "Control Option D"
            for name, state in (
                ("Control Option Space", "busy"),
                ("Control Option D", "busy"),
                ("Control Shift D", "free"),
            ):
                subprocess.run([sys.executable, "-m", "eval.cursormix.dictation_check", name, state], check=True)
            self.dictationKey_(self.dictation_choices["Control Shift D"])
            assert preferences(self.suite).stringForKey_("dictationShortcut") == "Control Shift D"
            self.dictationKey_(self.dictation_choices["Off"])
            for name, state in (
                ("Control Option D", "free"),
                ("Control Shift D", "free"),
                ("Control Option Space", "busy"),
            ):
                subprocess.run([sys.executable, "-m", "eval.cursormix.dictation_check", name, state], check=True)
            print(
                "Both shortcuts registered exclusively. Changing and disabling dictation released its shortcuts."
            )
        except Exception as error:  # noqa: BLE001 — exit the native startup check with the verification failure
            self.exit_code = 1
            print(f"Dictation check failed: {error}", file=sys.stderr)
        finally:
            self.store.removePersistentDomainForName_(self.suite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", choices=(*SHORTCUTS, *DICTATION_SHORTCUTS))
    parser.add_argument("state", nargs="?", choices=("busy", "free"))
    args = parser.parse_args()
    if args.name and args.state:
        A.NSApplication.sharedApplication()
        probe(args.name, args.state == "busy")
    elif args.name or args.state:
        parser.error("Supply both a shortcut and an expected registration state.")
    else:
        main(delegate_type=DictationCheck, argv=["--check-startup"])
