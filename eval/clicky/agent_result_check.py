"""Check desktop outcome reporting on recorded Finder states and real local file operations."""

import argparse
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from threading import Event

from clicky import actions, agent, ax
from clicky.capture import Context
from clicky.memory import observed_window

from .annotation_check import recorded_finder
from .corpus import ROOT
from .workspace_trial import prepare


def live_copy(provider):
    """Ask the real planner to duplicate one real file and confirm the byte match."""
    with tempfile.TemporaryDirectory(prefix="clicky-proofreading-") as stage:
        root = Path(stage).resolve() / "Proofreading"
        root.mkdir()
        source = root / "Clicky-source.txt"
        shutil.copy2(ROOT / "data/clicky/task-workspace" / source.name, source)
        destination = root / "Readme for proofreading.txt"
        prepare({
            "id": "proofreading",
            "start": {"app": "Finder", "title_contains": root.name},
            "end": {"app": "Finder", "file": destination.name},
        }, root)
        pid, name = ax.frontmost()
        def approve(target_pid, app, description):
            if target_pid != pid or ax.frontmost()[0] != pid or destination.exists() or destination.is_symlink():
                return False
            if not description.startswith("Command · "):
                return False
            argv = shlex.split(description.removeprefix("Command · "))
            allowed = argv[:1] == ["cp"] and argv[-2:] == [str(source), str(destination)] and all(
                opt in ("-n", "-p", "-v", "--") for opt in argv[1:-2]
            )
            if not allowed:
                print("Declined a command outside the two fixture paths.", flush=True)
            return allowed

        command = shlex.join(["cp", str(source), str(destination)])
        prompt = f"Run this command: {command}\nAfterward, tell me whether the contents match."
        events = []
        agent.run(prompt, Context(ax.read_window(pid, name)), provider, Event(), approve,
                  lambda message: None, lambda event, details: events.append(event))
        assert events.count("dispatch") == 1 and events[-1] == "done"
        assert source.read_bytes() == destination.read_bytes()
        print("The planner confirmed the copy after one dispatch. Both files match byte for byte.")


def main(provider, with_image=False):
    start, moved, inside, returned = (recorded_finder(k) for k in ("start", "moved", "inside", "returned"))
    starting = observed_window(start.window)
    goal = "Open the Clicky material folder, then return to my starting folder with the original file selected."
    history = []
    failures = []

    def check(name, task, context, expected):
        # Desktop tasks first use the current Accessibility state. They request
        # an app-only image only if that first verification is inconclusive.
        initial = context if with_image else Context(context.window)
        status, detail = agent.verify_result(task, starting, history, initial, provider)
        if status == "unclear" and initial.jpeg is None:
            status, detail = agent.verify_result(task, starting, history, context, provider)
        if (status == "verified") != expected:
            failures.append((name, status, detail))
        print(f"{name}: {status}", flush=True)

    selected = next(e.label for e in start.window.elements if e.selected and e.role == "AXTextField")
    check("Already selected", f"Select {selected} in Finder.", start, True)
    for name, context in (("Initial window", start), ("Moved window", moved), ("Entered folder", inside)):
        if context is inside:
            history.append({"kind": "click", "observed_after": observed_window(context.window)})
        check(name, goal, context, False)
    history.append({"kind": "click", "observed_after": observed_window(returned.window)})
    check("Completed return", goal, returned, True)
    check("Second trip outstanding", goal + " Do the round trip twice.", returned, False)


    files = ROOT / "data/clicky/task-workspace"
    sources = [files / name for name in ("Clicky-source.txt", "Clicky-edit.txt")]
    history.clear()
    with tempfile.TemporaryDirectory(prefix="clicky-copy-result-") as directory:
        task = f"Copy both {sources[0].name} and {sources[1].name} into {directory} with identical contents."
        for index, source in enumerate(sources):
            command = shlex.join(["cp", str(source), directory])
            result = actions.terminal(command, Event())
            destination = Path(directory) / source.name
            assert destination.read_bytes() == source.read_bytes()
            assert str(destination) in result and "contents match" in result
            history.append({"kind": "command", "value": command, "result": result, "observed_after": starting})
            check(f"Verified {index + 1} of two file copies", task, start, expected=index == 1)
        command = shlex.join(["mv", str(destination), str(Path(directory) / "renamed.txt")])
        result = actions.terminal(command, Event())
        assert not destination.exists() and (Path(directory) / "renamed.txt").read_bytes() == sources[1].read_bytes()
        assert f"Original path {destination} is absent." in result
        assert "same filesystem identity" in result
        renamed = Path(directory) / "renamed.txt"
        skipped = actions.terminal(shlex.join(["mv", "-n", str(renamed), str(Path(directory) / sources[0].name)]), Event())
        assert renamed.exists() and "same filesystem identity" not in skipped
        folder = Path(directory) / "notes for review"
        marker = folder / "pending.txt"
        for utility, target in (("mkdir", folder), ("touch", marker)):
            relative = os.path.relpath(target, Path.home())
            result = actions.terminal(shlex.join([utility, relative]), Event())
            assert target.exists() and "was absent before the command and now exists" in result
            argv = [utility, "-p", relative] if utility == "mkdir" else [utility, relative]
            result = actions.terminal(shlex.join(argv), Event())
            assert "already existed before the command" in result
    if failures:
        raise AssertionError(failures)
    print("Eight desktop outcomes, move identity, a skipped move and relative-path creation checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", nargs="?", choices=("Anthropic", "OpenAI", "Local"), default="Anthropic")
    parser.add_argument("--with-image", action="store_true", help="also exercise verification with the image supplied from the outset")
    parser.add_argument("--live-copy", action="store_true", help="exercise the planner and result check in a temporary Finder folder")
    args = parser.parse_args()
    if args.live_copy:
        live_copy(args.provider)
    else:
        main(args.provider, args.with_image)
