"""Exercise native file transfers on disposable copies of real project text."""

import argparse
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

from cursormix import actions, agent
from cursormix.capture import Context
from cursormix.memory import observed_window

from .annotation_check import recorded_finder
from .corpus import ROOT


def main(provider, other_volume=None, cancel_source=None):
    recording = recorded_finder("start")
    context = Context(recording.window)
    starting = observed_window(context.window)
    original = ROOT / "data/clicky/task-workspace/Clicky-source.txt"
    with tempfile.TemporaryDirectory(prefix="clicky-file-review-") as stage:
        root = Path(stage).resolve()
        source = root / "Reading copy.txt"
        draft = root / "Copy for editor.txt"
        renamed = root / "Annotated edition.txt"
        shutil.copy2(original, source)

        def transfer(src, dest):
            return actions.FileTransfer.prepare(json.dumps({"source": str(src), "destination": str(dest)}))

        stopped = Event()
        stopped.set()
        for kind in ("copy_file", "move_file"):
            pending = transfer(source, draft)
            try:
                actions.perform(kind, pending, -1, recording.window, stopped)
            except InterruptedError:
                pass
            else:
                raise AssertionError("A cancelled transfer was dispatched.")
            assert not draft.exists() and source.read_bytes() == original.read_bytes()

            # Occupy the destination after preview with a real, pre-existing file.
            shutil.copy2(original, draft)
            inode = draft.stat().st_ino
            try:
                actions.perform(kind, pending, -1, recording.window, Event())
            except actions.ActionUnavailable:
                pass
            else:
                raise AssertionError("A transfer replaced an occupied destination.")
            assert draft.stat().st_ino == inode and source.exists()
            draft.unlink()

        changed = transfer(source, draft)
        source.rename(root / "Original reading copy.txt")
        shutil.copy2(original, source)
        try:
            actions.perform("move_file", changed, -1, recording.window, Event())
        except actions.ActionUnavailable:
            pass
        else:
            raise AssertionError("A different file was moved after approval.")
        assert source.is_file() and not draft.exists()
        print("Cancellation, an occupied destination and a changed source preserved the originals.", flush=True)

        def attempt(action, kind, barrier):
            barrier.wait(timeout=5)
            try:
                actions.perform(kind, action, -1, context.window, Event())
            except (OSError, ValueError):
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as pool:
            for kind in ("copy_file", "move_file"):
                for trial in range(25):
                    folder = root / f"{kind}-{trial}"
                    folder.mkdir()
                    sources = [folder / name for name in ("first.txt", "second.txt")]
                    target = folder / "winner.txt"
                    for path in sources:
                        shutil.copy2(original, path)
                    attempts = [transfer(path, target) for path in sources]
                    barrier = Barrier(2)
                    workers = [pool.submit(attempt, action, kind, barrier) for action in attempts]
                    outcomes = [worker.result() for worker in workers]
                    assert outcomes.count(True) == 1 and target.read_bytes() == original.read_bytes()
                    assert all(path.exists() == (not ok or kind == "copy_file") for path, ok in zip(sources, outcomes))
                    assert not list(folder.glob(".clicky-copy-*"))
        print("All 25 concurrent copy trials and 25 concurrent move trials kept one winner.", flush=True)

        history = []
        for kind, src, dest, verb in (("copy_file", source, draft, "Copy"), ("move_file", draft, renamed, "Rename")):
            request = f'{verb} the file "{src}" to "{dest}".'
            proposed, value, _ = agent.next_step(request, context, [], provider)
            assert proposed == kind, f"Planner proposed {proposed} for {verb.lower()}"
            action = actions.FileTransfer.prepare(value)
            assert (action.source, action.destination) == (src, dest)
            result = actions.perform(kind, action, -1, context.window, Event())
            assert dest.read_bytes() == original.read_bytes()
            assert src.exists() == (kind == "copy_file")
            history.append({"kind": kind, "value": action.preview(), "result": result, "observed_after": starting})
            status, detail = agent.verify_result(request, starting, history, context, provider)
            assert status == "verified", detail
            print(f"{verb} used the approved paths and verified the real result.", flush=True)

        if cancel_source:
            binary = cancel_source.resolve(strict=True)
            before = binary.stat()
            dest = root / "Cancelled copy"
            pending = transfer(binary, dest)
            cancel = Event()
            with ThreadPoolExecutor(max_workers=1) as pool:
                copying = pool.submit(actions.perform, "copy_file", pending, -1, context.window, cancel)
                deadline = time.monotonic() + 10
                while not any(root.glob(".clicky-copy-*")):
                    if copying.done() or time.monotonic() >= deadline:
                        raise AssertionError("Could not observe the copy in progress.")
                    time.sleep(0.0005)
                cancel.set()
                try:
                    copying.result(timeout=90)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError("The cancelled copy finished.")
            assert not dest.exists() and not list(root.glob(".clicky-copy-*"))
            assert actions.same_file_version(before, binary.stat())
            print(f"An in-flight copy of {before.st_size} bytes cancelled before publication.")

        if other_volume:
            with tempfile.TemporaryDirectory(dir=other_volume, prefix="clicky-review-") as remote:
                dest = Path(remote).resolve() / renamed.name
                assert renamed.stat().st_dev != dest.parent.stat().st_dev
                move = transfer(renamed, dest)
                actions.perform("move_file", move, -1, context.window, Event())
                assert not renamed.exists() and dest.read_bytes() == original.read_bytes()
                print("The cross-volume move preserved the contents and removed the original.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("Anthropic", "OpenAI", "Local"))
    parser.add_argument("--other-volume", type=Path, help="writable directory on a different mounted volume")
    parser.add_argument("--cancel-source", type=Path, help="existing large file to exercise cancellation during copying")
    args = parser.parse_args()
    main(args.provider, args.other_volume, args.cancel_source)
