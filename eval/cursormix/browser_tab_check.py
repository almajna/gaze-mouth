"""Verify named tab selection and stale approvals in a disposable Chrome window."""

import os
import subprocess
import tempfile
import time
from threading import Event

import AppKit as A

from cursormix import actions, ax

from . import corpus


def check():
    root = corpus.ROOT / "data/clicky/task-workspace"
    first, second = root / "Clicky-copy.txt", root / "Clicky-source.txt"
    if not all(p.is_file() and not p.is_symlink() for p in (first, second)):
        raise ValueError("Restore the two real text files in the task workspace first.")
    bundle = A.NSWorkspace.sharedWorkspace().URLForApplicationWithBundleIdentifier_("com.google.Chrome")
    if not bundle:
        raise RuntimeError("Google Chrome is unavailable.")
    executable = A.NSBundle.bundleWithURL_(bundle).executablePath()
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    with tempfile.TemporaryDirectory(prefix="clicky-tab-check-") as directory:
        launch = [
            str(executable),
            f"--user-data-dir={directory}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--force-renderer-accessibility",
        ]
        browser = subprocess.Popen(
            [*launch, "--new-window", first.as_uri(), second.as_uri()],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cancel = Event()

        def view(count, active=None):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if browser.poll() is not None or ax.session_locked():
                    break
                window = ax.read_window(browser.pid, "Google Chrome")
                tabs = ax.browser_tabs(window)
                if len(tabs) == count and any(
                    tab["selected"]
                    and tab["title"] in (first.name, second.name)
                    and tab["title"] in window.title
                    and (active is None or tab["title"] == active)
                    for tab in tabs.values()
                ):
                    return window, tabs
                time.sleep(0.15)
            raise RuntimeError("The disposable browser did not reach the expected tab state.")

        def select(window, tabs, name):
            index = next(i for i, tab in tabs.items() if tab["title"] == name)
            actions.perform("switch_tab", name, index, window, cancel)

        try:
            window, tabs = view(2)
            select(window, tabs, second.name)
            baseline, tabs = view(2, second.name)
            original = next(i for i, tab in tabs.items() if tab["title"] == first.name)
            select(baseline, tabs, first.name)
            closing, _ = view(2, first.name)
            actions.perform("shortcut", "cmd+w", -1, closing, cancel)
            view(1, second.name)
            subprocess.run([*launch, first.as_uri()], check=True, env=env, capture_output=True, timeout=15)
            reopened, tabs = view(2, first.name)
            select(reopened, tabs, second.name)
            current, tabs = view(2, second.name)
            replacement = next(i for i, tab in tabs.items() if tab["title"] == first.name)
            assert baseline.ref == current.ref and baseline.title == current.title
            assert baseline.elements[original].ref != current.elements[replacement].ref
            try:
                actions.perform("switch_tab", first.name, original, baseline, cancel)
            except actions.ActionUnavailable:
                view(2, second.name)
            else:
                raise AssertionError("The previous approval selected a replacement tab.")
            print(
                "Named selection verified. The same-title replacement stayed unselected after a stale approval."
            )
        finally:
            if browser.poll() is None:
                browser.terminate()
                try:
                    browser.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    browser.kill()
                    browser.wait()


if __name__ == "__main__":
    check()
