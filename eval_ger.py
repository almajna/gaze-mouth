#!/usr/bin/env python3
"""Score the Qwen stage on saved n-best lists (eval_wer.py --json rows): top-1, best of the list, and final WER.

Default: generative correction (GERCorrector, optional LoRA). --select: pick one of the hypotheses (LMSelector).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from context_decoder.ger import DEFAULT_MODEL, LM_WEIGHT, GERCorrector, LMSelector
from eval_silent_speech import errors
from eval_wer import report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beams", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path, help="GER LoRA directory; omit for zero-shot Qwen")
    ap.add_argument("--weights", type=Path, nargs="*", default=[None],
                    help="Adapter weight files to score in turn (checkpoints of --adapter)")
    ap.add_argument("--select", action="store_true", help="Pick among the hypotheses with Qwen as a language model")
    ap.add_argument("--lm-weight", type=float, default=LM_WEIGHT)
    ap.add_argument("--json", type=Path, help="Write per-clip rows (for the last weights file)")
    args = ap.parse_args()
    beams = json.loads(args.beams.read_text())
    if args.select and (args.adapter or args.weights != [None]):
        ap.error("--select uses Qwen without an adapter")
    corrector = LMSelector(args.model, args.lm_weight) if args.select else GERCorrector(args.model, args.adapter)
    corrector.correct(beams[0]["hypotheses"])  # warm-up, not timed
    speech = [beam for beam in beams if beam["kind"] == "sentence"]
    total = sum(beam["reference_words"] for beam in speech)
    top1 = sum(beam["word_errors"] for beam in speech)
    oracle = sum(min([errors(beam["reference"], h["text"])[0] for h in beam["hypotheses"]] or [beam["reference_words"]])
                 for beam in speech)
    for weights in args.weights:
        if weights is not None:
            corrector.model.load_weights(str(weights), strict=False)
            corrector.reset()
        rows, latency = [], []
        for beam in beams:
            prediction = corrector.correct(beam["hypotheses"])
            latency.append(corrector.last["latency_ms"])
            edits, count = errors(beam["reference"], prediction)
            rows.append(dict(clip=beam["clip"], session=beam["session"], kind=beam["kind"], reference=beam["reference"],
                             first_pass=beam["prediction"], first_pass_word_errors=beam["word_errors"],
                             prediction=prediction, word_errors=edits, reference_words=count,
                             ger_ms=corrector.last["latency_ms"], prompt_tokens=corrector.last["prompt_tokens"],
                             generated_tokens=corrector.last["generated_tokens"], hypotheses=beam["hypotheses"]))
        print(f"== Qwen stage: {f'selection, LM weight {args.lm_weight}' if args.select else weights or args.adapter or 'zero-shot correction'}")
        report(rows)
        scored = [r for r in rows if r["kind"] == "sentence"]
        ger = sum(r["word_errors"] for r in scored)
        better = sum(r["word_errors"] < r["first_pass_word_errors"] for r in scored)
        worse = sum(r["word_errors"] > r["first_pass_word_errors"] for r in scored)
        print(f"First pass top-1 {top1 / total:.2%} ({top1}/{total})  best of list {oracle / total:.2%} ({oracle}/{total})  "
              f"GER {ger / total:.2%} ({ger}/{total})  clips better {better} / worse {worse} / same {len(scored) - better - worse}")
        print(f"GER latency median {np.median(latency):.0f} ms, p95 {np.percentile(latency, 95):.0f} ms", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
