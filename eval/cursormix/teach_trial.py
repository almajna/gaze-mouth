"""Record independent native before, unrelated and after states for the ten teaching tasks."""

import argparse
import fcntl
import json
from datetime import UTC, datetime
from uuid import uuid4

import AppKit as A
import Quartz as Q

from cursormix import ax, capture
from cursormix.hotkey import Hotkey, shortcut_label
from cursormix.settings import preferences

from . import corpus
from .corpus import write_once

PLANS = corpus.TEACH_PLANS


def wait_for_press(pressed):
    app = A.NSApplication.sharedApplication()
    pressed.clear()
    while not pressed:
        deadline = A.NSDate.dateWithTimeIntervalSinceNow_(0.15)
        event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
            A.NSEventMaskAny, deadline, A.NSDefaultRunLoopMode, True
        )
        if event:
            app.sendEvent_(event)


def point():
    location = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return location.x, location.y


def step_checksum(directory, step):
    for key in ("before", "unrelated", "after"):
        path = corpus.inside(step[key])
        if path.parent != directory:
            raise ValueError("A step refers to another recording directory.")
        meta = json.loads(path.read_text())
        if corpus.inside(meta["image"]).parent != directory:
            raise ValueError("A step image belongs to another recording directory.")
    return corpus.digest("teach-checkpoint", json.dumps({"steps": [step]}, sort_keys=True).encode())


def save_step(directory, number, step):
    record = {"step": step, "checksum": step_checksum(directory, step)}
    write_once(directory / f"step-{number:02d}.json", json.dumps(record).encode())


def completed_steps(directory, instructions):
    steps = []
    for number, path in enumerate(sorted(directory.glob("step-*.json")), 1):
        if number > len(instructions) or path.name != f"step-{number:02d}.json":
            raise ValueError("The recorded steps are out of sequence.")
        record = json.loads(path.read_text())
        step = record["step"]
        if step["instruction"] != instructions[number - 1] or record["checksum"] != step_checksum(directory, step):
            raise ValueError(f"The instruction or captured evidence for step {number} has changed.")
        steps.append(step)
    return steps


def record(task, app_name, resume=None):
    if resume:
        directory = corpus.inside(resume)
        task = directory.parent.name
        if task not in PLANS or directory.parent.parent != corpus.inside("teach"):
            raise ValueError("Choose the printed teach/TASK/SESSION directory.")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        directory = corpus.inside(f"teach/{task}/{stamp}")
        directory.mkdir(parents=True, mode=0o700)
    with (directory / ".recorder.lock").open("ab") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another recorder is already using this session.") from None
        manifest = directory / "session.json"
        if not resume:
            write_once(manifest, json.dumps({"plan": PLANS[task], "app": app_name or PLANS[task][0]}).encode())
        elif not manifest.exists():
            raise ValueError("This session has no resumable step annotations. Existing captures are preserved.")
        session = json.loads(manifest.read_text())
        if session["plan"] != PLANS[task] or (app_name and app_name != session["app"]):
            raise ValueError("The task or reference application differs from the original recording.")
        _, request, *instructions = PLANS[task]
        steps = completed_steps(directory, instructions)
        relative = directory.relative_to(corpus.DATA)
        print(f"{len(steps)} of {len(instructions)} steps saved.", flush=True)
        print(f".venv-cursormix/bin/python -m eval.cursormix.teach_trial --resume {relative}", flush=True)
        if len(steps) < len(instructions):
            if ax.session_locked():
                raise ValueError("Unlock the Mac and run the printed resume command to capture the next step.")
            if resume:
                print("Restore the starting state for the first unfinished step before recording.", flush=True)
            capture_steps(directory, session["app"], instructions, steps)
        case = {"id": task, "session": directory.name, "request": request, "context": steps[0]["before"], "steps": steps}
        gate = next(name for name, task_id in corpus.TEACH_PARTS.items() if task_id == task)
        corpus.validate_rows(gate, [case])
        path = directory / "task.json"
        write_once(path, (json.dumps(case) + "\n").encode())
        print("Recorded", task, "under", directory)
        print(".venv-cursormix/bin/python -m eval.cursormix.teach_trial --seal-recording", path.relative_to(corpus.DATA))


def capture_steps(directory, expected_app, instructions, steps):
    app = A.NSApplication.sharedApplication()
    app.setActivationPolicy_(A.NSApplicationActivationPolicyProhibited)
    pressed = []
    hotkey = Hotkey(lambda: pressed.append(True))
    shortcut = preferences().stringForKey_("shortcut")
    hotkey.register(shortcut)

    def await_app(message):
        while True:
            print(message, f"Press {shortcut_label(shortcut)} when ready.", flush=True)
            wait_for_press(pressed)
            pid, name = ax.frontmost()
            if name == expected_app and not ax.session_locked():
                return pid, name
            print("Focus the reference application:", expected_app, flush=True)

    def snapshot(name, message):
        pid, app = await_app(message)
        pos = point()
        state = capture.on_summon(pid, app)
        if ax.frontmost() != (pid, app) or ax.session_locked():
            raise ValueError("The foreground application changed during capture. Resume this step.")
        if not state.jpeg or not state.display:
            raise ValueError(state.issue or "Screen capture unavailable.")
        image = directory / (name + ".jpg")
        write_once(image, state.jpeg)
        meta = directory / (name + ".json")
        write_once(meta,
            json.dumps(
                {
                    "app": app,
                    "title": state.window.title,
                    "frame": state.window.frame,
                    **({"selected_text": state.window.selected_text} if state.window.selected_text is not None else {}),
                    "display": state.display,
                    "image": str(image.relative_to(corpus.DATA)),
                    "elements": [{"frame": e.frame, **e.description()} for e in state.window.elements],
                }
            ).encode()
        )
        return str(meta.relative_to(corpus.DATA)), pos, state.display

    try:
        for i in range(len(steps) + 1, len(instructions) + 1):
            instruction = instructions[i - 1]
            prefix = f"{i}-{uuid4().hex}"
            print(f"Step {i} of {len(instructions)}. {instruction}", flush=True)
            before, (x, y), screen = snapshot(
                f"{prefix}-before", "Point at the target's top-left corner before doing the step."
            )
            while True:
                await_app("Point at the bottom-right corner of the same target.")
                r, b = point()
                left, top, width, height = screen
                if left <= x < r <= left + width and top <= y < b <= top + height:
                    break
                print("The target must fit within the recorded display.", flush=True)
            unrelated, _, _ = snapshot(
                f"{prefix}-unrelated", "Make an unrelated change that leaves this instruction unfinished."
            )
            after, _, _ = snapshot(
                f"{prefix}-after", "Restore the baseline, complete the instruction, and leave its outcome visible."
            )
            step = {
                "instruction": instruction,
                "before": before,
                "unrelated": unrelated,
                "after": after,
                "target_frame": [x, y, r - x, b - y],
            }
            save_step(directory, i, step)
            steps.append(step)
            print(f"Saved step {i} of {len(instructions)}.", flush=True)
    finally:
        hotkey.close()


def assemble():
    records = {r["id"]: r for r in corpus.teaching_parts()}
    for path in sorted(corpus.inside("teach").glob("*/*/task.json")):
        row = json.loads(path.read_text())
        records.setdefault(row["id"], row)
    missing = sorted(set(PLANS) - records.keys())
    if missing:
        raise SystemExit("Still need recordings for " + ", ".join(missing))
    ordered = [records[task] for task in PLANS]
    corpus.validate_rows("teach", ordered)
    content = "".join(json.dumps(row) + "\n" for row in ordered).encode()
    write_once(corpus.inside("teach.jsonl"), content)
    print("Run python -m eval.cursormix.corpus teach to validate and seal the complete set.")


def seal_recording(relative):
    path = corpus.inside(relative)
    row = json.loads(path.read_text())
    task = row["id"]
    if task not in PLANS or path != corpus.inside(f"teach/{task}/{row['session']}/task.json"):
        raise ValueError("Choose the task.json from the recorded task and session directory.")
    if len(row["steps"]) != len(PLANS[task]) - 2 or row["context"] != row["steps"][0]["before"]:
        raise ValueError("The recording must start at the first step and cover the complete task.")
    for step in row["steps"]:
        for key in ("before", "unrelated", "after"):
            if corpus.inside(step[key]).parent != path.parent:
                raise ValueError("Each state must belong to this recording session.")
    gate = next(name for name, task_id in corpus.TEACH_PARTS.items() if task_id == task)
    corpus.validate_rows(gate, [row])
    complete = corpus.inside("teach.jsonl")
    if complete.exists():
        corpus.verify("teach")
        if row not in corpus.rows("teach"):
            raise ValueError("The complete teaching set already uses a different recording.")
    manifest = corpus.inside(gate + ".jsonl")
    source = manifest if manifest.exists() else corpus.CATALOG / manifest.name
    if source.exists():
        content = source.read_bytes()
        if [json.loads(line) for line in content.splitlines() if line.strip()] != [row]:
            raise ValueError(f"{gate} already has a different reference recording.")
    else:
        content = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    checksum = (corpus.digest(gate, content) + "\n").encode()
    seal = corpus.seal(gate)
    if seal.exists() and seal.read_bytes() != checksum:
        raise ValueError("The recording differs from the existing seal.")
    write_once(manifest, content)
    write_once(seal, checksum)
    corpus.verify(gate)
    print(f"{task} validated and sealed. The teaching replay now includes it.")


def list_recordings():
    sealed = {row["id"] for row in corpus.teaching_parts()}
    for task, (_, request, *instructions) in PLANS.items():
        if task in sealed:
            print(f"{task} [sealed] {request}")
            continue
        print(f"{task} [recording needed] {request}")
        for manifest in sorted(corpus.inside(f"teach/{task}").glob("*/session.json")):
            directory = manifest.parent
            relative = directory.relative_to(corpus.DATA)
            if (directory / "task.json").exists():
                print(f"  Ready for review: .venv-cursormix/bin/python -m eval.cursormix.teach_trial --seal-recording {relative}/task.json")
            else:
                count = len(completed_steps(directory, instructions))
                print(f"  {count}/{len(instructions)} saved: .venv-cursormix/bin/python -m eval.cursormix.teach_trial --resume {relative}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    action = p.add_mutually_exclusive_group()
    action.add_argument("task", nargs="?", choices=PLANS)
    p.add_argument("--app", help="localized application name to require in the foreground")
    action.add_argument("--resume", metavar="SESSION", help="continue from the first unfinished step")
    action.add_argument("--assemble", action="store_true")
    action.add_argument("--seal-recording", metavar="TASK_JSON", help="path relative to data/clicky/frozen")
    args = p.parse_args()
    if args.assemble:
        assemble()
    elif args.seal_recording:
        seal_recording(args.seal_recording)
    elif args.task or args.resume:
        record(args.task, args.app, args.resume)
    else:
        list_recordings()
