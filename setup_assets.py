#!/usr/bin/env python3
"""Install or verify the pinned public assets without replacing existing files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parent
OPTIONAL = ("models/FastVLM-0.5B-bf16/", "models/Qwen3-4B-Instruct-2507-4bit/")


def matches(path, checksum):
    kind, expected = checksum.split(":")
    with path.open("rb") as stream:
        if kind == "sha256":
            actual = hashlib.file_digest(stream, "sha256")
        else:
            actual = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(block)
    return actual.hexdigest() == expected


def git(*args, **kwargs):
    return subprocess.run(["git", *map(str, args)], check=True, **kwargs)


def repository(path, url, revision, check):
    if not path.exists():
        if check:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            checkout = Path(directory) / "checkout"
            git("init", "--quiet", checkout)
            git("-C", checkout, "fetch", "--quiet", "--depth=1", url, revision)
            git("-C", checkout, "checkout", "--quiet", "--detach", revision)
            if path.exists():
                raise FileExistsError(path)
            checkout.rename(path)
    actual = git("-C", path, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"expected checkout {revision}, found {actual}")


def download(path, url, checksum, check):
    if not path.exists():
        if check:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            incoming = Path(directory) / "download"
            subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--silent",
                            "--show-error", "--output", str(incoming), url], check=True)
            if not matches(incoming, checksum):
                raise ValueError("download checksum mismatch")
            path.hardlink_to(incoming)
    if not matches(path, checksum):
        raise ValueError("existing file checksum mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify local assets without downloading")
    parser.add_argument("--planner", action="store_true", help="include the optional Qwen3 and FastVLM assets")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "asset_manifest.json").read_text())
    failed = False
    unavailable = []
    for install, entries in ((repository, manifest["repositories"]), (download, manifest["files"])):
        for name, (source, digest) in entries.items():
            if not args.planner and name.startswith(OPTIONAL):
                continue
            if install is download and any(name.startswith(repo + "/") for repo in unavailable):
                continue
            try:
                install(ROOT / name, source, digest, args.check)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                print(f"{name}: {exc}", file=sys.stderr)
                failed = True
                if install is repository:
                    unavailable.append(name)
            else:
                print(f"OK {name}")
    print("Local training and calibration assets must be copied from an authorized local backup "
          "or regenerated. See ASSETS.md for exact paths and commands.")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
