"""Read and seal human-annotated Clicky recordings under data/. Never generate cases."""

import argparse
import json
import math
import os
import shutil
import subprocess
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

from cursormix.ax import Element, Window, selection_excerpt
from cursormix.capture import Context

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/clicky/frozen"
CATALOG = Path(__file__).with_name("catalog")
TEACH_PLANS = json.loads((CATALOG / "teach-plans.json").read_text())
TEACH_PARTS = {
    ("teach-recorded" if task == "vscode-replace" else "teach-" + task): task for task in TEACH_PLANS
}


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        try:
            os.link(tmp.name, path)
            folder = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"An existing {path.name} differs. Its contents were preserved.") from None


def seal(gate):
    return Path(__file__).with_name("seals") / (gate + ".sha256")


COUNTS = {"latency": 50, "qa": 50, "point": 100, "teach": 10, "agent": 20, "router": 100}
PUBLIC_ROUTER_COUNT = 76
APPS = {
    "Finder",
    "Safari",
    "Google Chrome",
    "Visual Studio Code",
    "Terminal",
    "Figma",
    "System Settings",
    "Mail",
    "Notes",
    "Slack",
    "Calendar",
}


def inside(relative):
    path = (DATA / relative).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError("Fixture path escapes the frozen corpus.")
    return path


def digest(gate, manifest_bytes=None):
    manifest = gate + ".jsonl"
    if manifest_bytes is None:
        manifest_bytes = inside(manifest).read_bytes()
    names = {manifest}
    for line in manifest_bytes.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        contexts = [row["context"]] if "context" in row else []
        for step in row.get("steps", []):
            contexts.extend(step[k] for k in ("before", "after", "unrelated"))
        names.update(contexts)
        for name in contexts:
            image = json.loads(inside(name).read_text()).get("image")
            if image:
                names.add(image)
    h = sha256()
    for name in sorted(names):
        content = manifest_bytes if name == manifest else inside(name).read_bytes()
        h.update(name.encode() + b"\0" + sha256(content).digest())
    return h.hexdigest()


def source_revision():
    h = sha256()
    for path in sorted((ROOT / "cursormix").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml", ".lock"):
            h.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def agent_records(directory, revision, provider):
    checksum = seal("agent").read_text().strip()
    records = {}
    for path in directory.glob("*.json"):
        r = json.loads(path.read_text())
        if (r["revision"], r["seal"], r["provider"]) == (revision, checksum, provider):
            key = r["id"], r["mode"]
            if key not in records or records[key]["time"] < r["time"]:
                records[key] = r
    return records


def pending_agent_modes(task, records):
    return [mode for mode in ("cancel", "run") if not records.get((task, mode), {}).get("start")]


def context(name):
    metadata = json.loads(inside(name).read_text())
    window = Window(
        0, metadata["app"], metadata["title"], metadata.get("frame"),
        selected_text=selection_excerpt(metadata.get("selected_text")),
    )
    for e in metadata["elements"]:
        window.elements.append(
            Element(
                e["role"],
                e["label"],
                e.get("frame"),
                focused=bool(e.get("focused")),
                selected=e.get("selected"),
            )
        )
    jpeg = None
    if metadata.get("image"):
        with Image.open(inside(metadata["image"])) as image, BytesIO() as buf:
            image.convert("RGB").save(buf, "JPEG", quality=90)
            jpeg = buf.getvalue()
    return Context(window, metadata.get("display"), jpeg)


def rows(gate):
    path = inside(gate + ".jsonl")
    if not path.is_file():
        raise ValueError(f"Missing {gate}.jsonl")
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_rows(gate, items)
    return items


def validate_rows(gate, items):
    expected_count = 1 if gate in TEACH_PARTS else {"point": 90, "point-1x": 10}.get(gate, COUNTS.get(gate))
    public_router = gate == "router" and len(items) == PUBLIC_ROUTER_COUNT and all(
        str(row.get("session", "")).startswith("public-") and row.get("negative") is True
        and row.get("source") and row.get("license") for row in items
    )
    if public_router:
        expected_count = PUBLIC_ROUTER_COUNT
    ids = {row["id"] for row in items}
    if len(items) != expected_count or len(ids) != len(items):
        raise ValueError(f"Expected {expected_count} distinct {gate} items.")
    point_gate = gate in ("point", "point-1x")
    if gate in TEACH_PARTS:
        if ids != {TEACH_PARTS[gate]}:
            raise ValueError(f"Unexpected teaching task in {gate}.")
        gate = "teach"
    elif gate == "teach" and ids != set(TEACH_PLANS):
        raise ValueError("The teaching set must include all ten specified workflows.")
    for row in items:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "session", "request")):
            raise ValueError("Each recording needs an ID, real session, and request.")
        if gate in ("latency", "qa", "point", "point-1x", "teach"):
            ctx = context(row["context"])
            if gate in ("qa", "point", "point-1x", "teach") and not ctx.jpeg:
                raise ValueError(f"Missing screenshot for {row['id']}.")
            if gate in ("point", "point-1x", "teach") and not ctx.display:
                raise ValueError(f"Missing screenshot or display for {row['id']}.")
            if point_gate and (
                type(row["target"]) is not int
                or not 0 <= row["target"] < len(ctx.window.elements)
                or not ctx.window.elements[row["target"]].frame
            ):
                raise ValueError("Missing ground-truth frame.")
        if gate == "teach":
            if len(row["steps"]) < 5:
                raise ValueError("Each guided task needs five or more observed steps.")
            for step in row["steps"]:
                states = [step[k] for k in ("before", "after", "unrelated")]
                if len(set(states)) != 3 or not step["instruction"].strip():
                    raise ValueError(
                        "Each step requires three separately recorded states and an instruction."
                    )
                before = context(states[0])
                for name in states:
                    meta = json.loads(inside(name).read_text())
                    if not meta.get("image") or not inside(meta["image"]).is_file():
                        raise ValueError("Teaching states require real recorded images.")
                if ("target_frame" in step) == ("target" in step):
                    raise ValueError("Mark one reference control per step.")
                if "target" in step:
                    index = step["target"]
                    if (
                        type(index) is not int
                        or not 0 <= index < len(before.window.elements)
                        or not before.window.elements[index].frame
                    ):
                        raise ValueError("The recorded target has no accessibility frame.")
                else:
                    rect = step["target_frame"]
                    if (
                        len(rect) != 4
                        or not all(type(v) in (int, float) and math.isfinite(v) for v in rect)
                        or min(rect[2:]) <= 0
                    ):
                        raise ValueError("Invalid reference rectangle.")
        if gate == "agent" and not (row["start"] and row["end"]):
            raise ValueError("Agent tasks require independent start and end predicates.")
    if gate == "router" and not public_router and {r["route"] for r in items} != {
        "answer",
        "screen_question",
        "point_to",
        "teach",
        "agent_task",
    }:
        raise ValueError("Router set must cover all five classes.")
    if gate == "latency" and {r["route"] for r in items} != {"answer", "screen_question"}:
        raise ValueError("Latency set must contain quick answers and screen questions.")
    if gate == "qa" and not APPS <= {context(r["context"]).window.app for r in items}:
        raise ValueError("Q&A set lacks required applications.")
    if point_gate:
        scale = 2 if gate == "point" else 1
        for row in items:
            meta = json.loads(inside(row["context"]).read_text())
            if row["scale"] != scale or meta.get("scale") != scale:
                raise ValueError("AX recording scale differs from annotation.")


def verify(gate):
    if not seal(gate).exists() or seal(gate).read_text().strip() != digest(gate):
        raise ValueError(f"Frozen {gate} seal missing or mismatched.")


def teaching_parts():
    items = []
    for manifest in TEACH_PARTS:
        if inside(manifest + ".jsonl").exists() or seal(manifest).exists():
            verify(manifest)
            items.extend(rows(manifest))
    return items


def install_media():
    manifest = json.loads((CATALOG / "media.json").read_text())
    for asset in manifest:
        image = inside(asset["path"])
        image.parent.mkdir(parents=True, exist_ok=True)
        if not image.exists():
            data = subprocess.check_output(["curl", "-fsSL", "--max-time", "30", asset["url"]])
            if sha256(data).hexdigest() != asset["sha256"]:
                raise ValueError("Public reference image content changed: " + asset["url"])
            image.write_bytes(data)
        elif sha256(image.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Existing reference image differs: {image}")
        metadata = inside(asset["context"])
        text = json.dumps(asset["metadata"], ensure_ascii=False, sort_keys=True) + "\n"
        if metadata.exists() and metadata.read_text() != text:
            raise ValueError(f"Existing metadata differs: {metadata}")
        if not metadata.exists():
            metadata.write_text(text)
    empty = inside("public-screens/text-only.json")
    if not empty.exists():
        empty.write_text('{"app":"","title":"","elements":[]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gates", nargs="*")
    parser.add_argument(
        "--install", action="store_true", help="restore bundled annotated request sets to data/"
    )
    args = parser.parse_args()
    if args.install:
        DATA.mkdir(parents=True, exist_ok=True)
        if not args.gates or set(args.gates) & {"latency", "qa"}:
            install_media()
        for source in CATALOG.glob("*.jsonl"):
            if args.gates and source.stem not in args.gates:
                continue
            to = DATA / source.name
            if to.exists() and to.read_bytes() != source.read_bytes():
                parser.error(f"Existing {to} differs from the bundled set.")
            if not to.exists():
                shutil.copyfile(source, to)
    for gate in args.gates or COUNTS:
        if gate not in (*COUNTS, "point-1x", *TEACH_PARTS):
            parser.error(f"Unknown gate {gate}")
        rows(gate)
        if seal(gate).exists():
            verify(gate)
        else:
            seal(gate).parent.mkdir(exist_ok=True)
            with seal(gate).open("x") as f:
                f.write(digest(gate) + "\n")
        print(f"{gate} validated and sealed.")
