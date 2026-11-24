"""Benchmark the open-vocabulary lip decoder on real recorded clips. No camera, no generated sensor data.

Reports, in this order:
  1. false positives: text produced for real non-speech clips (__NULL__ takes, session negatives)
  2. cross-session error on real silent takes (the session the encoder was not trained on)
  3. latency on real clips: encoder + lattice against the 120 ms budget, then Qwen reconstruction
  4. reconstructor accuracy on text-derived lattices (synthetic; not a lip-reading result)

  .venv-gf/bin/python bench_open_vocab.py
"""
from pathlib import Path
import argparse
import json
import statistics
import time

from silent_avsr import OpenVocabLipDecoder, features
from silent_avsr.lattice import edit_distance
from silent_avsr.lexicon import phrase_phonemes

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "open_vocab_lip"
BUDGET_MS = 120.0
NULL = "__NULL__"


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT)
    ap.add_argument("--repeats", type=int, default=5, help="warm passes per clip for latency")
    ap.add_argument("--skip-llm", action="store_true", help="encoder and lattice only")
    args = ap.parse_args()

    metrics_path = args.checkpoint_dir / "train_ctc_base_metrics.json"
    if not (args.checkpoint_dir / "encoder.safetensors").is_file() or not metrics_path.is_file():
        raise SystemExit("No trained encoder. Run: .venv-gf/bin/python -m silent_avsr.train_ctc --mode base --epochs 60")
    training = json.loads(metrics_path.read_text())
    held_out = training["test_session"]
    decoder = OpenVocabLipDecoder(checkpoint_dir=args.checkpoint_dir)
    has_adapter = (args.checkpoint_dir / "reconstructor_lora" / "adapters.safetensors").is_file()
    use_llm = not args.skip_llm
    print(f"encoder trained on session {training['train_session']} ({training['epochs']} epochs); "
          f"held-out session {held_out}; device {decoder.device}; "
          f"reconstructor {'Qwen2.5-3B + LoRA' if has_adapter else 'Qwen2.5-3B zero-shot'}"
          f"{'' if use_llm else ' (skipped)'}")

    takes = [features.load_silent_take(*pair) for pair in features.silent_takes()
             if pair[0].parent.name == held_out]
    negatives = [features.load_session_clip(p) for p in features.session_clips() if p.stem.startswith("neg_")]
    dual = [features.load_session_clip(p) for p in features.session_clips()[::9]]
    decoder.lattice(None, takes[0]["mouth_roi_30hz"])  # one-off MPS start-up
    context = {}  # no screen context was recorded with these takes
    if use_llm:
        decoder.decode(None, takes[0]["mouth_roi_30hz"], context)

    def run(clip):
        if use_llm:
            result = decoder.decode(None, clip["mouth_roi_30hz"], context)
            return result["phoneme_lattice"]["hypotheses"][0]["phonemes"], result["text"]
        lattice, _, _ = decoder.lattice(None, clip["mouth_roi_30hz"])
        return list(lattice.best), ""

    report = dict(checkpoint=str(args.checkpoint_dir), held_out_session=held_out, reconstructor_lora=has_adapter)

    # 1. False positives -----------------------------------------------------------------------
    print("\n[1] False positives on real non-speech clips (mouth-ROI stream)")
    for name, clips in (("__NULL__ takes, held-out session", [t for t in takes if t["phrase"] == NULL]),
                        ("session negatives (blink, swallow, look down)", negatives)):
        outputs = [run(clip) for clip in clips]
        emitted = sum(bool(phones) for phones, _ in outputs)
        with_text = sum(bool(text) for _, text in outputs)
        print(f"    {name:48s} phonemes emitted {emitted}/{len(clips)}   text returned {with_text}/{len(clips)}")
        report[f"false_positive:{name}"] = dict(clips=len(clips), phonemes_emitted=emitted, text_returned=with_text)

    # 2. Cross-session real takes ----------------------------------------------------------------
    print(f"\n[2] Real silent takes from the held-out session {held_out} (one speaker, 20 phrases)")
    speech = [t for t in takes if t["phrase"] != NULL]
    errors = reference = exact_phones = exact_text = 0
    examples = []
    for take in speech:
        target = phrase_phonemes(take["phrase"])
        phones, text = run(take)
        errors += edit_distance(phones, target)
        reference += len(target)
        exact_phones += phones == target
        exact_text += text == take["phrase"]
        examples.append((take["phrase"], " ".join(phones), text))
    print(f"    phoneme error rate            : {errors / reference:.1%} over {len(speech)} takes")
    print(f"    exact phoneme sequence        : {exact_phones}/{len(speech)}")
    if use_llm:
        print(f"    exact phrase after Qwen        : {exact_text}/{len(speech)}  (no screen context recorded)")
        for phrase, phones, text in examples[:6]:
            print(f"      {phrase!r:20} lips: {phones[:44]:44} -> {text!r}")
    report["cross_session"] = dict(takes=len(speech), phoneme_error_rate=errors / reference,
                                   exact_phoneme_sequences=exact_phones, exact_phrases=exact_text if use_llm else None)

    # 3. Latency -------------------------------------------------------------------------------
    print(f"\n[3] Latency on real clips, warm, {args.repeats} passes each (budget {BUDGET_MS:.0f} ms for encoder + lattice)")
    rows = []
    for kind, clips in (("take, ROI stream", takes[::6]), ("session clip, both streams", dual)):
        for clip in clips:
            first = sum(decoder.lattice(clip["landmarks_30hz"], clip["mouth_roi_30hz"])[1:])
            warm = [decoder.lattice(clip["landmarks_30hz"], clip["mouth_roi_30hz"])[1:] for _ in range(args.repeats)]
            rows.append(dict(kind=kind, frames=len(clip["mouth_roi_30hz"]), first_ms=first,
                             encoder_ms=statistics.median(e for e, _ in warm),
                             lattice_ms=statistics.median(l for _, l in warm),
                             total_ms=statistics.median(e + l for e, l in warm), worst_ms=max(e + l for e, l in warm)))
    for kind in ("take, ROI stream", "session clip, both streams"):
        subset = [r for r in rows if r["kind"] == kind]
        frames = [r["frames"] for r in subset]
        print(f"    {kind:27s} n={len(subset):2d}  {min(frames)}-{max(frames)} frames   "
              f"encoder {statistics.median(r['encoder_ms'] for r in subset):5.1f}  "
              f"lattice {statistics.median(r['lattice_ms'] for r in subset):4.1f}  "
              f"total median {statistics.median(r['total_ms'] for r in subset):5.1f}  "
              f"worst {max(r['worst_ms'] for r in subset):5.1f}  "
              f"first call at a new length {max(r['first_ms'] for r in subset):5.1f} ms")
    worst = max(max(r["worst_ms"], r["first_ms"]) for r in rows)
    print(f"    slowest encoder + lattice call: {worst:.1f} ms -> {'within' if worst < BUDGET_MS else 'OVER'} budget")
    report["latency"] = dict(clips=rows, slowest_ms=worst, within_budget=worst < BUDGET_MS)
    if use_llm:
        llm, total = [], []
        for take in speech[:20]:
            result = decoder.decode(None, take["mouth_roi_30hz"], context)
            llm.append(decoder.last_timing["reconstruct_ms"])
            total.append(result["latency_ms"])
        print(f"    Qwen reconstruction: median {statistics.median(llm):.0f} ms, p95 {percentile(llm, 0.95):.0f} ms; "
              f"end to end median {statistics.median(total):.0f} ms, p95 {percentile(total, 0.95):.0f} ms ({len(llm)} real takes)")
        report["latency"].update(reconstruct_median_ms=statistics.median(llm), reconstruct_p95_ms=percentile(llm, 0.95),
                                 end_to_end_median_ms=statistics.median(total), end_to_end_p95_ms=percentile(total, 0.95))

    # 4. Reconstructor on synthetic lattices -----------------------------------------------------
    if use_llm:
        from context_decoder import synth_phoneme_data as synth
        from context_decoder.train_reconstructor_lora import evaluate, evaluate_challenge
        print("\n[4] Reconstructor on text-derived lattices (synthetic; not a lip-reading result)")
        challenge = evaluate_challenge(decoder.reconstructor)
        held = evaluate(decoder.reconstructor, [r for r in synth.generate(12000, seed=0)
                                                if r["split"] == "test" and r["domain"] != "vocab"][:200])
        print(f"    same lattice, two screen contexts: both right in {challenge['pairs_both_right']}/{challenge['pairs']} pairs "
              f"({challenge['exact']}/{challenge['total']} items)")
        print(f"    held-out phrases, noisy lattices  : {held['exact']}/{held['total']} exact ({held['accuracy']:.1%})  {held['by_domain']}")
        report["synthetic"] = dict(challenge=challenge, held_out=held)

    out = ROOT / "logs" / f"open_vocab_bench_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
