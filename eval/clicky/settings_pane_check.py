"""Check native Settings destinations and planner outcomes against a recorded Finder window."""

import argparse
from threading import Event

from AppKit import NSWorkspace
from Foundation import NSURL, NSBundle

from clicky import actions, agent
from clicky.capture import Context
from clicky.memory import observed_window
from clicky.settings import installed_panes

from .annotation_check import recorded_finder


def check(provider):
    context = Context(recorded_finder().window)
    initial = observed_window(context.window)
    catalog = installed_panes()
    workspace = NSWorkspace.sharedWorkspace()
    misses = []
    for name in ("Appearance", "Time Machine"):
        task = f"Open the {name} pane in System Settings so I can look at its options."
        url = NSURL.URLWithString_("x-apple.systempreferences:" + catalog[name])
        handler = workspace.URLForApplicationToOpenURL_(url)
        assert handler and NSBundle.bundleWithURL_(handler).bundleIdentifier() == "com.apple.systempreferences"
        inspection = {"kind": "inspect_settings", "result": {"installed_panes": list(catalog)}}
        kind, value, _ = agent.next_step(task, context, [], provider)
        if kind == "inspect_settings" or (kind == "open_settings" and value not in catalog):
            kind, value, _ = agent.next_step(task, context, [inspection], provider)
        planned = (kind, value) == ("open_settings", name)
        if not planned:
            misses.append(name)
        # Both observations come from the same Finder recording. Planning an action
        # must never be enough evidence that the Settings pane became visible.
        past = [{"kind": kind, "value": value, "observed_before": initial, "observed_after": initial}]
        status, _ = agent.verify_result(task, initial, past, context, provider)
        assert status in ("unfinished", "unclear"), status
        print(f"{name}: native handler found, direct proposal {planned}, unchanged Finder judged {status}.", flush=True)

    cancelled = Event()
    cancelled.set()
    for name, token, expected in (
        ("Appearance?query", Event(), actions.ActionUnavailable),
        ("Appearance", cancelled, InterruptedError),
    ):
        try:
            actions.perform("open_settings", name, None, context.window, token)
        except expected:
            continue
        raise AssertionError(f"Unexpected dispatch for {name}")
    print(f"{len(catalog)} unique installed panes. Unknown and cancelled destinations rejected.")
    if misses:
        raise SystemExit("Direct navigation was not proposed for " + ", ".join(misses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("Anthropic", "Local"), default="Anthropic", nargs="?")
    check(parser.parse_args().provider)
