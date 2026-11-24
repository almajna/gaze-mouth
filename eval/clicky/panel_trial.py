"""Run from the project root with python -m eval.clicky.panel_trial."""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from clicky.app import main
from clicky.hotkey import shortcut_label
from clicky.models import selected_provider
from clicky.settings import preferences


class Trial:
    def __init__(self):
        self.shortcut = None
        self.provider = None
        self.opened = False
        self.entered = False
        self.saw_reply = False
        self.completed = False
        self.passed = False

    def summoned(self, shortcut, key_window):
        self.shortcut = shortcut
        self.opened = bool(shortcut and key_window)
        self.entered = self.saw_reply = self.completed = False

    def submitted(self, provider):
        self.provider = provider
        self.entered = self.opened
        self.saw_reply = self.completed = False

    def dismissed(self, hidden):
        self.passed = all((self.entered, self.saw_reply, self.completed, hidden))
        if self.passed:
            directory = Path(__file__).resolve().parents[2] / "data/clicky/panel"
            directory.mkdir(parents=True, exist_ok=True)
            now = datetime.now(UTC)
            path = directory / f"{now:%Y%m%dT%H%M%S%fZ}.json"
            with path.open("x") as file:
                json.dump(
                    {
                        "utc": now.isoformat(),
                        "provider": self.provider,
                        "shortcut": self.shortcut,
                        "hotkey_opened_key_window": self.opened,
                        "submitted": self.entered,
                        "rendered_text": self.saw_reply,
                        "stream_completed": self.completed,
                        "escape_hid_window": bool(hidden),
                    },
                    file,
                    indent=2,
                )
            print(f"Panel trial passed. Observation saved to {path}", flush=True)
        else:
            print(
                "Repeat the shortcut, submit a question, wait for the answer to finish, then press Esc.",
                flush=True,
            )
        return self.passed


def run():
    store = preferences()
    provider = selected_provider(store.stringForKey_("provider"))
    keys = {"OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY"}
    if provider in keys and not os.getenv(keys[provider]):
        sys.exit(f"Panel trial requires {keys[provider]} in the launching shell. No trial recorded.")
    shortcut = shortcut_label(store.stringForKey_("shortcut"))
    print(
        f"With another app focused, press {shortcut}, type a question for {provider}, and press Enter. "
        "After the reply finishes, press Esc.",
        flush=True,
    )
    trial = Trial()
    main(trial)
    if not trial.passed:
        sys.exit("Panel trial ended before all observations were recorded.")


if __name__ == "__main__":
    run()
