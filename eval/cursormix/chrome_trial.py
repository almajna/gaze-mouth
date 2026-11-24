"""Collect the two frozen tab-switch observations in an isolated Chrome profile."""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import AppKit as A
import Quartz as Q

from cursormix import actions, ax

from . import corpus
from .agent_trial import OBS, matches


def run(provider, *, resume=False):
    corpus.verify("agent")
    records = corpus.agent_records(OBS, corpus.source_revision(), provider) if resume else {}
    modes = corpus.pending_agent_modes("chrome-switch", records)
    if not modes:
        print("chrome-switch: both attempts already recorded for this build and provider.")
        return
    if ax.session_locked():
        raise SystemExit("The Mac is locked. No browser trial was started.")
    task = next(row for row in corpus.rows("agent") if row["id"] == "chrome-switch")
    bundle = A.NSWorkspace.sharedWorkspace().URLForApplicationWithBundleIdentifier_("com.google.Chrome")
    if not bundle:
        raise RuntimeError("Google Chrome is unavailable.")
    executable = A.NSBundle.bundleWithURL_(bundle).executablePath()
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    with tempfile.TemporaryDirectory(prefix="clicky-chrome-") as temp:
        args = [
            str(executable),
            f"--user-data-dir={Path(temp) / 'profile'}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--force-renderer-accessibility",
            "--new-window",
            "https://example.org",
            "https://www.apple.com",
        ]
        browser = subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for mode in modes:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if ax.session_locked():
                        raise RuntimeError("The Mac locked during browser setup.")
                    if browser.poll() is not None:
                        raise RuntimeError("The isolated Chrome process exited.")
                    pid = browser.pid
                    app = A.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
                    if app and app.isFinishedLaunching():
                        app.activateWithOptions_(A.NSApplicationActivateIgnoringOtherApps)
                        if ax.frontmost()[0] == pid:
                            window = ax.read_window(pid, "Google Chrome")
                            if matches(task["start"], window):
                                break
                            if matches(task["end"], window):
                                actions.keyboard(pid, 19, Q.kCGEventFlagMaskCommand)
                    time.sleep(0.3)
                else:
                    raise RuntimeError("The isolated Chrome window did not reach the frozen Apple-tab start.")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "eval.cursormix.agent_trial",
                        task["id"],
                        provider,
                        mode,
                        "--summon",
                        "--verify-navigation",
                    ],
                    check=True,
                    timeout=90,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("Anthropic", "OpenAI", "Local"), default="Anthropic", nargs="?")
    parser.add_argument("--resume", action="store_true", help="Keep prior valid-start attempts on this build.")
    args = parser.parse_args()
    run(args.provider, resume=args.resume)
