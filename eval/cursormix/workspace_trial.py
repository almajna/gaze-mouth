"""Prepare fixed Finder and Settings starts and replay the selected native desktop tasks."""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import AppKit as A

from cursormix import ax

from . import corpus
from .agent_trial import DEFAULT_WORKSPACE, OBS, matches

CASES = (
    "calculator",
    "calendar",
    "finder-subfolder",
    "finder-parent",
    "finder-info",
    "rename",
    "duplicate",
    "make-folder",
    "terminal",
    "open-file",
    "settings-sound",
    "settings-keyboard",
    "settings-display",
    "safari-url",
    "draft",
)


def prerequisite(task_id, source):
    if not source.is_file():
        return f"The source fixture is missing at {source}."
    if task_id in ("open-file", "safari-url", "draft"):
        url, bundle, description = {
            "open-file": (
                A.NSURL.fileURLWithPath_(str(source)),
                "com.apple.TextEdit",
                "TextEdit for text files",
            ),
            "safari-url": (
                A.NSURL.URLWithString_("https://example.org"),
                "com.apple.Safari",
                "Safari for HTTPS",
            ),
            "draft": (
                A.NSURL.URLWithString_("mailto:"),
                "com.apple.mail",
                "Mail for email drafts",
            ),
        }[task_id]
        handler = A.NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(url)
        actual = (
            A.NSBundle.bundleWithURL_(handler).bundleIdentifier() if handler else None
        )
        if actual != bundle:
            return f"This frozen task requires {description}. The current handler is {actual or 'unavailable'}."
    return None


def at_fixed_start(case, window, root, expected):
    if not matches(case["start"], window, root):
        return False
    if window.app != "Finder":
        return True
    locations = ax.finder_locations(window, "Clicky-source.txt")
    if locations.get("directory_containing_displayed_files") != str(expected):
        return False
    return case["id"] != "finder-info" or any(
        e.label == "Clicky-source.txt" and e.selected for e in window.elements
    )


def prepare(case, root):
    source = root / "Clicky-source.txt"
    folder = root / "Clicky material"
    if ax.session_locked():
        raise RuntimeError("An unlocked desktop is required to prepare a trial.")
    target = folder if case["id"] == "finder-parent" else source
    if case["id"] == "settings-display":
        subprocess.run(
            [
                "open",
                "-b",
                "com.apple.systempreferences",
                "x-apple.systempreferences:com.apple.Keyboard-Settings.extension",
            ],
            check=True,
        )
    else:
        subprocess.run(
            ["open", str(target)] if target == folder else ["open", "-R", str(source)],
            check=True,
        )
    expected = folder if target == folder else root
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if ax.session_locked():
            raise RuntimeError("The desktop locked during setup.")
        pid, name = ax.frontmost()
        if name == case["start"]["app"]:
            window = ax.read_window(pid, name)
            if at_fixed_start(case, window, root, expected):
                if matches(case["end"], window, root):
                    raise RuntimeError("The end state already holds in this fixture.")
                return
        time.sleep(0.25)
    raise RuntimeError(f"The {case['id']} task did not reach its fixed start.")


def run(provider, ids, *, preflight=False, resume=False):
    corpus.verify("agent")
    cases = {r["id"]: r for r in corpus.rows("agent")}
    records = corpus.agent_records(OBS, corpus.source_revision(), provider) if resume else {}
    source = DEFAULT_WORKSPACE / "Clicky-source.txt"
    if preflight:
        print("Desktop:", "locked" if ax.session_locked() else "unlocked")
    for task_id in ids:
        modes = corpus.pending_agent_modes(task_id, records)
        if not modes:
            print(f"{task_id}: both attempts already recorded for this build and provider.")
            continue
        if preflight:
            reason = prerequisite(task_id, source)
            print(f"{task_id}: {reason or 'source and handler prerequisites met'} Pending: {', '.join(modes)}.")
            continue
        if ax.session_locked():
            raise SystemExit("The Mac is locked. No desktop task was started.")
        if not source.is_file():
            raise SystemExit(f"The source fixture is missing at {source}.")
        for mode in modes:
            with tempfile.TemporaryDirectory(prefix="clicky-agent-") as staging:
                root = Path(staging).resolve() / "task-workspace"
                folder = root / "Clicky material"
                folder.mkdir(parents=True)
                for name in (
                    "Clicky-source.txt",
                    "Clicky-rename.txt",
                    "Clicky-copy.txt",
                ):
                    shutil.copy2(source, root / name)
                shutil.copy2(source, folder / source.name)
                # A per-document handler can differ from the original file's handler.
                reason = prerequisite(task_id, root / source.name)
                if reason:
                    print(f"Skipped {task_id}. {reason}", flush=True)
                    break
                prepare(cases[task_id], root)
                flag = (
                    "--verify-workspace"
                    if task_id
                    in ("rename", "duplicate", "make-folder", "terminal", "draft")
                    else "--verify-navigation"
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "eval.cursormix.agent_trial",
                        task_id,
                        provider,
                        mode,
                        "--summon",
                        "--workspace",
                        str(root),
                        flag,
                    ],
                    check=True,
                    timeout=150,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("Anthropic", "OpenAI", "Local"))
    parser.add_argument("tasks", nargs="+", choices=CASES)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Inspect the lock, source file and default handlers without launching apps.",
    )
    parser.add_argument("--resume", action="store_true", help="Run only modes without a recorded valid start on this build.")
    args = parser.parse_args()
    run(args.provider, list(dict.fromkeys(args.tasks)), preflight=args.preflight, resume=args.resume)
