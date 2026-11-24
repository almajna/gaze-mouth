"""Replay frozen typed requests through the real field action and native text view."""

import argparse
import json
import re
import tempfile
import uuid
from pathlib import Path
from time import perf_counter

import objc
from objc import super
from PyObjCTools import AppHelper

from clicky import memory
from clicky.app import ClickyDelegate, end_run, main
from clicky.settings import preferences

from . import corpus


class Measurements:
    def __init__(self, provider, items):
        self.provider, self.items = provider, iter(items)
        self.results = []
        self.completed = self.saw_reply = False

    def routed(self, route):
        self.actual_route = route

    def submitted(self, provider):
        self.actual_route = None
        self.saw_reply = False
        self.first_ms = None


class ReplayDelegate(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        store = preferences("local.clicky.eval." + uuid.uuid4().hex)
        store.registerDefaults_(
            {
                "provider": self.trial.provider,
                "history": False,
                "passive": False,
                "voice": False,
                "spokenReplies": False,
                "agentTasks": False,
                "teaching": False,
                "pointing": False,
            }
        )
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        AppHelper.callAfter(self.advance)

    @objc.python_method
    def advance(self):
        item = next(self.trial.items, None)
        if item is None:
            end_run()
            return
        self.item = item
        self.conversation = []
        self.context = corpus.context(item["context"])
        self.panel.show()
        self.panel.field.setStringValue_(item["request"])
        self.trial.start = perf_counter()
        self.panel.field.sendAction_to_(self.panel.field.action(), self)

    @objc.python_method
    def append_token(self, request, token, first):
        super().append_token(request, token, first)
        if self.trial.saw_reply and self.trial.first_ms is None:
            self.panel.output.displayIfNeeded()
            self.trial.first_ms = (perf_counter() - self.trial.start) * 1000

    @objc.python_method
    def finished(self, request, prompt, error):
        if request != self.generation:
            return
        failure = None
        if error:
            # Keep logs free of arbitrary messages containing prompts or screen data.
            status = re.search(r"returned HTTP (\d{3})", error)
            if status:
                failure = "provider_http_" + status.group(1)
            elif error == "The provider returned an empty answer.":
                failure = "empty_answer"
            elif "closed the stream" in error or "interrupted the response" in error:
                failure = "stream_interrupted"
            elif error.startswith("Could not reach "):
                failure = "connection_failed"
            else:
                failure = "request_error"
        elif not self.trial.saw_reply:
            failure = "no_visible_text"
        elif self.trial.actual_route != self.item["route"]:
            failure = "route_mismatch"
        self.trial.results.append({
            "id": self.item["id"],
            "session": self.item["session"],
            "route": self.item["route"],
            "ms": self.trial.first_ms,
            "actual_route": self.trial.actual_route,
            "ok": failure is None,
            **({"failure": failure} if failure else {}),
        })
        AppHelper.callAfter(self.advance)


def measure(provider, items):
    trial = Measurements(provider, items)
    with tempfile.TemporaryDirectory() as tmp:
        memory.DATABASE = Path(tmp) / "history.sqlite3"
        main(trial, delegate_type=ReplayDelegate, argv=[])
    return trial.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("Anthropic", "OpenAI", "Local"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    corpus.verify("latency")
    result = measure(args.provider, corpus.rows("latency"))
    args.output.write_text(json.dumps(result))
