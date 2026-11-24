"""Exercise recording recovery on private copies of a real, sealed walkthrough."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import corpus, teach_trial


def replay(directory, limit):
    source = json.loads((directory / "original.json").read_text())
    instructions = corpus.TEACH_PLANS[source["id"]][2:]
    saved = teach_trial.completed_steps(directory, instructions)
    for number in range(len(saved) + 1, limit + 1):
        # The recorder supplies the fixed plan's instruction at capture time.
        # All images, target annotations and transitions come from the real session.
        step = dict(source["steps"][number - 1], instruction=instructions[number - 1])
        teach_trial.save_step(directory, number, step)
    if limit < len(instructions):
        os._exit(23)


def main():
    gate = "teach-settings-shortcuts"
    corpus.verify(gate)
    source = corpus.rows(gate)[0]
    with tempfile.TemporaryDirectory(prefix="clicky-recording-recovery-") as tmp:
        root = Path(tmp).resolve()
        for name in ("cursormix", "eval/cursormix"):
            shutil.copytree(corpus.ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "logs"))
        relative = Path("teach") / source["id"] / source["session"]
        data = root / "data/clicky/frozen"
        folder = data / relative
        shutil.copytree(corpus.DATA / relative, folder)
        (folder / "task.json").rename(folder / "original.json")
        plan = corpus.TEACH_PLANS[source["id"]]
        (folder / "session.json").write_text(json.dumps({"plan": plan, "app": plan[0]}))
        args = [sys.executable, "-m", "eval.cursormix.teach_recording_check", str(folder)]
        resumed = [sys.executable, "-m", "eval.cursormix.teach_trial", "--resume", str(relative)]

        result = subprocess.run([*args, "3"], cwd=root, check=False)
        assert result.returncode == 23 and len(list(folder.glob("step-*.json"))) == 3
        assert not (folder / "task.json").exists()
        # Leave an image from an interrupted step without publishing a step record.
        step = source["steps"][0]
        before = json.loads((data / step["before"]).read_text())
        after = json.loads((data / step["after"]).read_text())
        image = data / before["image"]
        shutil.copyfile(image, folder / "unfinished-before.jpg")
        baseline = {p.name: p.read_bytes() for p in folder.glob("step-*.json")}
        real_bytes = image.read_bytes()
        shutil.copyfile(data / after["image"], image)
        refused = subprocess.run(resumed, cwd=root, capture_output=True, text=True, timeout=15, check=False)
        assert refused.returncode and "has changed" in refused.stderr and not (folder / "task.json").exists()
        image.write_bytes(real_bytes)
        subprocess.run([*args, str(len(source["steps"]))], cwd=root, check=True)
        for name, data in baseline.items():
            assert (folder / name).read_bytes() == data
        subprocess.run(resumed, cwd=root, check=True, timeout=15)
        result_path = folder / "task.json"
        first_result = result_path.read_bytes()
        result = json.loads(first_result)
        assert len(result["steps"]) == len(source["steps"])
        for actual, ref in zip(result["steps"], source["steps"]):
            assert all(actual[k] == v for k, v in ref.items() if k != "instruction")
        subprocess.run(resumed, cwd=root, check=True, timeout=15)
        assert result_path.read_bytes() == first_result
    print("Three committed steps survived process exit. Changed evidence was refused. All six steps finalized twice identically.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        replay(Path(sys.argv[1]), int(sys.argv[2]))
    else:
        main()
