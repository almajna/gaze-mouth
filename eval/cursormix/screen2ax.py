"""Restore the original Retina screenshots and AX frames for the public pointing cases."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "https://huggingface.co/datasets/macpaw-research/Screen2AX-Tree/resolve/be62204c771ab82b40640252408d1e98102abf0c/data/train-00000-of-00002.parquet"
SHA256 = "db83eb693235c5cfcb634cdf1f7b666cf7e03062cd5d8256b67de165e47da06e"


def nodes(tree):
    yield tree
    for child in tree.get("children", []):
        yield from nodes(child)


def frame(node):
    if node.get("position") and node.get("size"):
        return list(map(float, (node["position"] + ";" + node["size"]).split(";")))
    return None


def install(path):
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SHA256:
            raise ValueError("The Screen2AX source file differs from the pinned release.")
    cases = Path(__file__).with_name("catalog") / "point.jsonl"
    indices = {json.loads(line)["source_row"] for line in cases.read_text().splitlines()}
    destination = ROOT / "data/clicky/frozen/screen2ax"
    destination.mkdir(exist_ok=True, parents=True)
    rows = (r for batch in pq.ParquetFile(path).iter_batches(batch_size=4) for r in batch.to_pylist())
    for i, record in enumerate(rows):
        if i not in indices:
            continue
        tree = json.loads(record["accessibility"])
        x, y, w, h = frame(tree)
        data = record["image"]["bytes"]
        with Image.open(BytesIO(data)) as image:
            if image.size != (w * 2, h * 2):
                raise ValueError("The recorded pixel dimensions do not confirm Retina scale.")
        # The image covers the original window. Child origins in this dataset
        # are window-relative, including descendants of nested containers.
        elements = []
        for j, node in enumerate(nodes(tree)):
            bounds = frame(node)
            if bounds and j:
                bounds[0] += x
                bounds[1] += y
            label = " ".join(
                dict.fromkeys(
                    str(node[k])
                    for k in ("name", "description", "value")
                    if isinstance(node.get(k), (int, float, str)) and str(node[k])
                )
            )
            elements.append({"role": node["role"], "label": label[:280], "frame": bounds})
        meta = {
            "app": "",
            "title": tree.get("name") or "",
            "display": [x, y, w, h],
            "frame": [x, y, w, h],
            "scale": 2,
            "elements": elements,
            "image": f"screen2ax/{i}.image",
        }
        for dest, content in (
            (destination / f"{i}.image", data),
            (destination / f"{i}.json", (json.dumps(meta, sort_keys=True) + "\n").encode()),
        ):
            if dest.exists() and dest.read_bytes() != content:
                raise ValueError(f"Existing recording differs: {dest}")
            if not dest.exists():
                dest.write_bytes(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, help="already downloaded first shard")
    args = parser.parse_args()
    if args.parquet:
        install(args.parquet)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.parquet"
            subprocess.run(["curl", "-fsSL", "--max-time", "300", "-o", str(source), SOURCE], check=True)
            install(source)
