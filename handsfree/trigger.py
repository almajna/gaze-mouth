"""Shared jaw trigger, calibration loading, and audio cue playback."""

import json
import subprocess
import sys

POP_SOUND = "/System/Library/Sounds/Pop.aiff"
COOLDOWN_MS = 600


def play(sound):
    try:
        subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


class Trigger:
    """Fires when cond holds >= debounce_ms; then needs cooldown_ms AND a mouth release before re-arming."""

    def __init__(self, debounce_ms, cooldown_ms=COOLDOWN_MS):
        self.d, self.cool = debounce_ms, cooldown_ms
        self.onset, self.armed, self.released, self.last_fire = None, True, True, -10**12

    def update(self, t, active):
        if not active:
            self.onset, self.released = None, True
        if not self.armed and self.released and t - self.last_fire >= self.cool:
            self.armed = True
        if active and self.armed:
            if self.onset is None:
                self.onset = t
            if t - self.onset >= self.d:
                onset = self.onset
                self.armed, self.released, self.last_fire, self.onset = False, False, t, None
                return onset
        return None


def load_trigger_config(path):
    if not path.exists():
        sys.exit(f"No validated trigger config at {path}.\n"
                 "Run `.venv-gf/bin/python -m eval.eval_offline`; it is written only when DEPLOYABLE: YES.")
    cfg = json.loads(path.read_text())
    for k in ("jawOpen", "lip_gap", "debounce_ms"):
        if k not in cfg:
            sys.exit(f"{path} is missing '{k}'")
    return cfg
