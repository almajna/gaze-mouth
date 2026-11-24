#!/usr/bin/env python3
"""Score recorder archives using pretrained visual CTC and a 4-gram KenLM model."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from context_decoder.word_search import DEFAULT_LM
from eval_silent_speech import errors, words
from silent_avsr.features import DATA
from silent_avsr.sentence_video import PERIOD_NS, RATE_HZ


def load_clip(path):
    with np.load(path, allow_pickle=False) as clip:
        meta = json.loads(str(clip["metadata_json"]))
        crops, grid = clip["mouth_88x88"], clip["crop_timestamps_ns"]
    if (crops.dtype != np.uint8 or crops.shape[1:] != (88, 88) or not len(crops)
            or meta["video_fps"] != RATE_HZ
            or not np.array_equal(grid, np.arange(len(crops)) * PERIOD_NS)):
        raise ValueError(f"{path}: expected an 88x88 grayscale clip on the 25 Hz timestamp grid")
    return meta, crops


def report(rows):
    negatives = [row for row in rows if row["kind"] == "negative"]
    if negatives:
        false_positives = sum(bool(words(row["prediction"])) for row in negatives)
        print(f"Negative false positives: {false_positives}/{len(negatives)}")
    else:
        print("Negative false positives: no negative clips in this evaluation")
    speech = [row for row in rows if row["kind"] == "sentence"]
    sessions = sorted({row["session"] for row in speech})
    if len(sessions) < 2:
        print("Cross-session WER: at least two recorded sessions are needed")
    for session in sessions:
        takes = [row for row in speech if row["session"] == session]
        edits = sum(row["word_errors"] for row in takes)
        total = sum(row["reference_words"] for row in takes)
        print(f"Session {session} WER: {edits / total:.2%} ({edits}/{total} words)")
    edits = sum(row["word_errors"] for row in speech)
    total = sum(row["reference_words"] for row in speech)
    if total:
        print(f"Overall WER: {edits / total:.2%} ({edits}/{total} words, {len(speech)} clips)")
    else:
        print("Overall WER: no reference words")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "eval_holdout")
    parser.add_argument("--backend", choices=("usr2", "auto_avsr"), default="usr2")
    parser.add_argument("--decode", choices=("ctc", "joint"), default="ctc",
                        help="CTC with KenLM, or USR 2.0 attention plus CTC")
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path, help="Speaker LoRA adapter directory")
    parser.add_argument("--kenlm", type=Path, default=DEFAULT_LM)
    parser.add_argument("--unigrams", type=Path)
    parser.add_argument("--lm-alpha", type=float, default=0.1)
    parser.add_argument("--lm-beta", type=float, default=0.0)
    parser.add_argument("--greedy", action="store_true", help="Use framewise greedy CTC decoding")
    parser.add_argument("--min-ctc-tokens", type=int, default=0,
                        help="Joint decoding: return no text when greedy CTC finds fewer tokens (non-speech gate)")
    parser.add_argument("--json", type=Path, help="Write per-clip hypotheses and scores")
    args = parser.parse_args()
    if args.decode == "joint" and (args.backend != "usr2" or args.greedy):
        parser.error("Joint decoding requires --backend usr2 and cannot use --greedy")
    paths = sorted(path for path in args.data.rglob("*.npz")
                   if not any(part.startswith(".") for part in path.relative_to(args.data).parts))
    if not paths:
        parser.error(f"No recorded clips in {args.data}. Record the holdout first.")
    from silent_avsr import PretrainedVSR, SentenceDecoder

    import torch
    torch.set_num_threads(4)
    if args.decode == "joint":
        from context_decoder.usr_joint import USRJointDecoder
        decoder = USRJointDecoder(PretrainedVSR(args.backend, args.device, args.checkpoint, args.adapter))
    else:
        decoder = SentenceDecoder(args.backend, args.device, args.checkpoint, args.kenlm, args.unigrams,
                                  args.adapter, args.lm_alpha, args.lm_beta)
    rows = []
    for path in paths:
        meta, crops = load_clip(path)
        if args.greedy:
            hypotheses = []
            prediction = decoder.search.greedy(decoder.vsr.log_probs(crops))
        else:
            hypotheses = decoder.decode(crops)
            if args.decode == "joint" and decoder.last["ctc_greedy_tokens"] < args.min_ctc_tokens:
                hypotheses = []
            prediction = hypotheses[0].text if hypotheses else ""
        kind = meta.get("kind", "sentence")
        if kind not in ("sentence", "negative"):
            raise ValueError(f"{path}: unexpected clip kind {kind!r}")
        reference = meta["sentence"]
        edits, count = errors(reference, prediction)
        if kind == "sentence" and not count:
            raise ValueError(f"{path}: empty sentence reference")
        rows.append(dict(clip=str(path.relative_to(args.data)), session=meta["session"], kind=kind,
                         reference=reference, prediction=prediction, word_errors=edits,
                         reference_words=count, hypotheses=[asdict(h) for h in hypotheses],
                         **({"decode": dict(decoder.last)} if args.decode == "joint" else {})))
    report(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
