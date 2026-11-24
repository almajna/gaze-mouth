"""LoRA fine-tuning of Qwen2.5-3B (mlx-lm) to turn noisy phoneme lattices + screen context into text.

Generates the synthetic dataset if --data is empty, trains with the loss masked to the answer,
then reports exact-match accuracy on held-out phrases and on the homophene challenge pairs,
before and after. All of these numbers are for text-derived lattices, not for real lip reading.

  .venv-gf/bin/python -m context_decoder.train_reconstructor_lora
"""
from pathlib import Path
import argparse
import json
import math
import time

from .prompt import messages
from .reconstructor import DEFAULT_MODEL, ROOT, ScreenGroundedReconstructor
from .synth_phoneme_data import challenge_records, read_jsonl, write_dataset

DEFAULT_OUT = ROOT / "models" / "open_vocab_lip"
MAX_SEQ_LENGTH = 768


def to_chat(record):
    return dict(messages=messages(record["lattice"], record["screen_context"], record["text"]))


def train_lora(model, tokenizer, train_records, valid_records, adapter_dir, iters, batch_size=4,
               num_layers=24, rank=16, scale=20.0, learning_rate=1e-4, seed=0, report_every=20, resume=None):
    """Adds LoRA layers to `model` in place, trains them, and writes an mlx-lm adapter directory
    (adapters.safetensors + adapter_config.json). Returns the reported train and validation losses."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    class History(TrainingCallback):
        def __init__(self):
            self.train, self.valid = [], []

        def on_train_loss_report(self, info):
            self.train.append((info["iteration"], float(info["train_loss"])))

        def on_val_loss_report(self, info):
            self.valid.append((info["iteration"], float(info["val_loss"])))

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    lora_parameters = dict(rank=rank, scale=scale, dropout=0.0)
    mx.random.seed(seed)
    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_parameters)
    if resume:  # continue from an earlier adapter with the same shape
        model.load_weights(str(resume), strict=False)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(dict(
        fine_tune_type="lora", num_layers=num_layers, lora_parameters=lora_parameters, iters=iters,
        batch_size=batch_size, learning_rate=learning_rate, seed=seed, mask_prompt=True), indent=2))
    dataset = lambda rows: CacheDataset(ChatDataset([to_chat(r) for r in rows], tokenizer, mask_prompt=True))
    history = History()
    valid_batches = max(1, min(10, len(valid_records) // batch_size)) if valid_records else 0
    model.train()
    train(model=model, optimizer=optim.AdamW(learning_rate=learning_rate), train_dataset=dataset(train_records),
          val_dataset=dataset(valid_records) if valid_batches else None,
          args=TrainingArgs(batch_size=batch_size, iters=iters, val_batches=valid_batches,
                            steps_per_report=min(report_every, iters), steps_per_eval=max(iters, 1),
                            steps_per_save=10 ** 9, adapter_file=str(adapter_dir / "adapters.safetensors"),
                            max_seq_length=MAX_SEQ_LENGTH),
          training_callback=history)
    model.eval()
    return dict(train=history.train, valid=history.valid)


def evaluate(reconstructor, records):
    """Exact-match accuracy overall and per domain, plus median latency."""
    by_domain, latencies, misses = {}, [], []
    for record in records:
        guess = reconstructor.reconstruct(record["lattice"], record["screen_context"])
        latencies.append(reconstructor.last["latency_ms"])
        hit, total = by_domain.get(record["domain"], (0, 0))
        by_domain[record["domain"]] = (hit + (guess == record["text"]), total + 1)
        if guess != record["text"] and len(misses) < 12:
            misses.append((record["text"], guess))
    hits = sum(h for h, _ in by_domain.values())
    return dict(exact=hits, total=len(records), accuracy=hits / max(1, len(records)),
                by_domain={k: f"{h}/{t}" for k, (h, t) in sorted(by_domain.items())},
                median_latency_ms=round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None,
                misses=misses)


def evaluate_challenge(reconstructor, records=None):
    """Same lattice, two screen contexts. A pair counts only if both contexts give their own phrase."""
    records = records or challenge_records()
    guesses = [reconstructor.reconstruct(r["lattice"], r["screen_context"]) for r in records]
    right = [g == r["text"] for g, r in zip(guesses, records)]
    pairs = len(records) // 2
    return dict(exact=sum(right), total=len(records), pairs_both_right=sum(right[2 * i] and right[2 * i + 1] for i in range(pairs)),
                pairs=pairs, context_changed_output=sum(guesses[2 * i] != guesses[2 * i + 1] for i in range(pairs)),
                misses=[(r["text"], g) for g, r, ok in zip(guesses, records, right) if not ok])


def report(title, held_out, challenge):
    print(f"\n{title}")
    print(f"  homophene pairs, both contexts right : {challenge['pairs_both_right']}/{challenge['pairs']}")
    print(f"  homophene items exact                : {challenge['exact']}/{challenge['total']}"
          f"   (context changed the output in {challenge['context_changed_output']}/{challenge['pairs']} pairs)")
    print(f"  held-out phrases exact               : {held_out['exact']}/{held_out['total']} ({held_out['accuracy']:.1%})"
          f"   {held_out['by_domain']}")
    print(f"  median reconstruction latency        : {held_out['median_latency_ms']} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_OUT / "synth", help="dataset directory (created if empty)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "reconstructor_lora", help="adapter directory")
    ap.add_argument("--n", type=int, default=12000, help="examples to generate when --data is empty")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--iters", type=int, help="overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=4, help="8 runs out of memory on a 24 GB Mac")
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--resume", type=Path, help="adapters.safetensors to continue from (same rank and layers)")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=200, help="held-out phrases to score")
    ap.add_argument("--skip-baseline", action="store_true", help="skip the zero-shot evaluation")
    args = ap.parse_args()

    from mlx_lm import load
    if not (args.data / "train.jsonl").is_file():
        print("generated", write_dataset(args.data, args.n, args.seed))
    train_records, valid_records = read_jsonl(args.data / "train.jsonl"), read_jsonl(args.data / "valid.jsonl")
    test_records = read_jsonl(args.data / "test.jsonl")
    # Score command/chat phrases and vocabulary drills separately: a drill without an on-screen
    # hint has several equally valid homophene answers, so its exact-match rate is capped.
    test_records = ([r for r in test_records if r["domain"] != "vocab"][:args.eval_n]
                    + [r for r in test_records if r["domain"] == "vocab"][:args.eval_n // 2])
    iters = args.iters or math.ceil(len(train_records) * args.epochs / args.batch_size)
    print(f"train {len(train_records)}  valid {len(valid_records)}  held-out scored {len(test_records)}  iters {iters}")

    baseline = None
    if not args.skip_baseline:
        zero_shot = ScreenGroundedReconstructor(args.model)
        baseline = dict(held_out=evaluate(zero_shot, test_records), challenge=evaluate_challenge(zero_shot))
        del zero_shot

    model, tokenizer = load(str(args.model))
    start = time.time()
    history = train_lora(model, tokenizer, train_records, valid_records, args.out, iters, args.batch_size,
                         args.num_layers, args.rank, learning_rate=args.learning_rate, seed=args.seed,
                         resume=args.resume)
    seconds = round(time.time() - start, 1)
    del model

    tuned = ScreenGroundedReconstructor(args.model, adapter_path=args.out)
    result = dict(held_out=evaluate(tuned, test_records), challenge=evaluate_challenge(tuned))
    if baseline:
        report("Zero-shot Qwen2.5-3B (synthetic lattices)", baseline["held_out"], baseline["challenge"])
    report(f"After LoRA, {iters} iterations in {seconds}s (synthetic lattices)", result["held_out"], result["challenge"])
    (args.out / "train_metrics.json").write_text(json.dumps(dict(
        iters=iters, seconds=seconds, train_examples=len(train_records), loss=history,
        zero_shot=baseline, lora=result), indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
