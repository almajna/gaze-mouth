"""Replay recorded FLEURS sentences through the local Clicky dictation path."""

import argparse
import json
import subprocess
import tarfile
import tempfile
import unicodedata
from hashlib import sha256
from pathlib import Path

import requests

from clicky.inputs.voice import VoiceInput
from clicky.voice import Recording

CONFIGS = ("ar_eg", "en_us", "fr_fr", "de_de", "it_it", "ja_jp", "pt_br", "es_419")
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def obtain(root, config, split):
    folder = root / f"{config}-{split}"
    if folder.is_dir():
        return
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data/{config}"
    metadata = requests.get(f"{base}/{split}.tsv", timeout=45)
    metadata.raise_for_status()
    with requests.get(
        f"{base}/audio/{split}.tar.gz", stream=True, timeout=45
    ) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            first = next(
                member
                for member in archive
                if member.isfile() and member.name.endswith(".wav")
            )
            source = Path(first.name).name
            audio = archive.extractfile(first).read()
    rows = (line.split("\t") for line in metadata.content.decode("utf-8").splitlines())
    row = next(row for row in rows if row[1] == source)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        stage = Path(tmp) / "row"
        stage.mkdir()
        (stage / "source.wav").write_bytes(audio)
        meta = {
            "id": int(row[0]),
            "expected": row[2],
            "source": source,
            "sha256": sha256(audio).hexdigest(),
        }
        (stage / "record.json").write_text(json.dumps(meta, ensure_ascii=False))
        stage.rename(folder)


def letters(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text).casefold()
        if unicodedata.category(c)[0] in "LMN"
    )


def edit_distance(expected, actual):
    previous = list(range(len(actual) + 1))
    for index, letter in enumerate(expected, 1):
        row = [index]
        for column, other in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[column] + 1, previous[column - 1] + (letter != other)))
        previous = row
    return previous[-1]


def replay(root):
    totals = {}
    for split in ("train", "test"):
        correct = [0, 0]
        edits = [0, 0]
        characters = 0
        for config in CONFIGS:
            folder = root / f"{config}-{split}"
            meta = json.loads((folder / "record.json").read_text())
            source = folder / "source.wav"
            expected = letters(meta["expected"])
            characters += len(expected)
            if sha256(source.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError(f"The recording changed for {folder.name}.")
            results = []
            for index, language in enumerate((config.split("_")[0], "auto")):
                recording = Recording()
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(source),
                            "-ar",
                            "16000",
                            "-ac",
                            "1",
                            str(recording.path),
                        ],
                        check=True,
                    )
                    transcript = VoiceInput(
                        recording, "Local", language=language
                    ).read()
                finally:
                    recording.close()
                distance = edit_distance(expected, letters(transcript))
                edits[index] += distance
                exact = distance == 0
                results.append("exact" if exact else "different")
                correct[index] += exact
            print(
                f"{split} {config} {meta['id']}: selected {results[0]}, automatic {results[1]}",
                flush=True,
            )
        totals[split] = (correct, edits, characters)
    for split, ((selected, automatic), edits, characters) in totals.items():
        print(
            f"{split}: {len(CONFIGS) - selected}/{len(CONFIGS)} selected-language and "
            f"{len(CONFIGS) - automatic}/{len(CONFIGS)} automatic sentence mismatches. "
            f"Character edits per {characters} reference characters: {edits[0]} selected, {edits[1]} automatic."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/clicky/development/dictation-fleurs")
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch the first recorded WAV from each pinned split archive.",
    )
    args = parser.parse_args()
    if args.download:
        args.data.mkdir(parents=True, exist_ok=True)
        for config in CONFIGS:
            for split in ("train", "test"):
                obtain(args.data, config, split)
    replay(args.data)
