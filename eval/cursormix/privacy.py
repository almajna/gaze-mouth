"""Observe capture calls and image writes during real native startup, passive ticks, summon and dismissal."""

import argparse
import json
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import objc
from objc import super
from PyObjCTools import AppHelper

from cursormix import capture, memory
from cursormix.app import ClickyDelegate, end_run, main
from cursormix.settings import preferences


class Observation:
    def __init__(self):
        self.phase = "startup"
        self.calls = []
        self.writes = set()
        self.acquired = self.released = False

    def summoned(self, shortcut, opened):
        pass

    def dismissed(self, hidden):
        return False

    def profile(self, frame, event, arg):
        if event == "call" and frame.f_code is capture.on_summon.__code__:
            self.calls.append(self.phase)

    def audit(self, event, args):
        if event == "open" and isinstance(args[0], (str, bytes)) and args[2] & (os.O_WRONLY | os.O_RDWR):
            self.writes.add(Path(os.fsdecode(args[0])))


class PrivacyDelegate(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        store = preferences("local.clicky.privacy-eval." + uuid.uuid4().hex)
        store.registerDefaults_({"passive": True, "nudges": True, "history": False, "voice": False})
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        self.trial.phase = "passive"
        AppHelper.callAfter(self.exercise)

    @objc.python_method
    def exercise(self):
        self.passiveTick_(self.passive_timer)
        self.passive_since -= 601
        self.passiveTick_(self.passive_timer)
        self.trial.phase = "summoned"
        self.ask_(self)

    @objc.python_method
    def show_context(self, ticket, future, pid, name, shortcut):
        super().show_context(ticket, future, pid, name, shortcut)
        self.trial.acquired = bool(self.context and self.context.jpeg)
        self.trial.phase = "dismissed"
        self.dismiss_(None)
        self.trial.released = self.context is None
        self.passiveTick_(self.passive_timer)
        AppHelper.callAfter(end_run)


def measure():
    obs = Observation()
    sys.addaudithook(obs.audit)
    threading.setprofile_all_threads(obs.profile)
    try:
        with tempfile.TemporaryDirectory() as temp:
            memory.DATABASE = Path(temp) / "history.sqlite3"
            main(obs, delegate_type=PrivacyDelegate, argv=[])
    finally:
        threading.setprofile_all_threads(None)
    image_writes = 0
    for path in obs.writes:
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".bmp"):
            image_writes += 1
        elif path.is_file():
            with path.open("rb") as f:
                if f.read(8).startswith((b"\x89PNG", b"\xff\xd8\xff", b"II*\0", b"MM\0*")):
                    image_writes += 1
    return {
        "ok": obs.acquired and obs.released and obs.calls == ["summoned"] and not image_writes,
        "capture_phases": obs.calls,
        "image_writes": image_writes,
        "released": obs.released,
        "acquired": obs.acquired,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure()))
