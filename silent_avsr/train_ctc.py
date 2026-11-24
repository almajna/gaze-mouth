"""Phoneme-CTC training on the real silent takes in data/silent_train.

  --mode base   train a new encoder (Auto-AVSR front-end frozen) and save it to --checkpoint-dir
  --mode lora   speaker fine-tuning: freeze the saved encoder, train a LoRA adapter + CTC head

Trains on one recording session and reports on the other. These takes are one speaker and 20
phrases, so the result says whether the pipeline learns from real lips, not how well it reads
open vocabulary. The takes have no landmarks, so only the mouth-ROI stream is trained.
"""
from pathlib import Path
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from .decoder import DEFAULT_FRONTEND, default_device
from .encoder import DualStreamPhonemeEncoder, EncoderConfig
from .features import ROI_SIDE, ROOT, load_session_clip, load_silent_take, session_clips, silent_takes
from .lattice import PhonemeLatticeDecoder, edit_distance
from .lexicon import phrase_phonemes
from .lora_adapter import LoRAAdapter
from .phonemes import encode

NULL = "__NULL__"
DEFAULT_CHECKPOINT = ROOT / "models" / "open_vocab_lip"


def load_examples(sessions=None):
    """Real takes as (T, 1, 96, 96) uint8 at 30 Hz with phoneme targets. __NULL__ has no target."""
    examples = []
    for folder, meta in silent_takes():
        if sessions and folder.parent.name not in sessions:
            continue
        take = load_silent_take(folder, meta, side=96)
        phones = [] if take["phrase"] == NULL else phrase_phonemes(take["phrase"])
        examples.append(dict(roi=torch.from_numpy(take["mouth_roi_30hz"]), target=encode(phones),
                             phones=phones, phrase=take["phrase"], session=take["session"], name=take["name"]))
    return examples


def crop(roi, rng=None):
    """Random 88x88 crop for training, centre crop otherwise."""
    span = roi.shape[-1] - ROI_SIDE
    dy, dx = (rng.randint(0, span), rng.randint(0, span)) if rng else (span // 2, span // 2)
    return roi[..., dy:dy + ROI_SIDE, dx:dx + ROI_SIDE]


def collate(batch, rng=None):
    lengths = torch.tensor([len(e["roi"]) for e in batch])
    roi = torch.zeros(len(batch), int(lengths.max()), 1, ROI_SIDE, ROI_SIDE, dtype=torch.uint8)
    for i, e in enumerate(batch):
        roi[i, :len(e["roi"])] = crop(e["roi"], rng)
    targets = torch.tensor([i for e in batch for i in e["target"]], dtype=torch.long)
    return roi, lengths, targets, torch.tensor([len(e["target"]) for e in batch])


def ctc_loss(model, batch, device, rng=None):
    roi, lengths, targets, target_lengths = collate(batch, rng)
    out = model(None, roi.to(device), lengths)
    # CTC runs on the CPU: it is cheap here and the MPS implementation is not complete.
    return F.ctc_loss(out.log_probs.transpose(0, 1).float().cpu(), targets, lengths, target_lengths,
                      zero_infinity=True)


def train_epochs(model, parameters, examples, epochs, lr, batch_size, device, seed, log=print):
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-2)
    history = []
    for epoch in range(epochs):
        model.train()
        order = examples[:]
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            loss = ctc_loss(model, batch, device, rng)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(order))
        log(f"epoch {epoch + 1}/{epochs}  train CTC loss {history[-1]:.4f}")
    model.eval()
    return history


@torch.inference_mode()
def mean_loss(model, examples, device, batch_size=8):
    model.eval()
    total = sum(ctc_loss(model, examples[i:i + batch_size], device).item() * len(examples[i:i + batch_size])
                for i in range(0, len(examples), batch_size))
    return total / len(examples)


@torch.inference_mode()
def evaluate(model, examples, device, decoder=None):
    """Failure modes first: emissions on __NULL__ takes, then phoneme error rate on speech takes."""
    decoder = decoder or PhonemeLatticeDecoder()
    model.eval()
    null_fired = null_total = errors = reference = exact = top_k = speech = 0
    for e in examples:
        out = model(None, crop(e["roi"])[None].to(device))
        lattice = decoder.decode(out.log_probs[0])
        if e["phrase"] == NULL:
            null_total += 1
            null_fired += not lattice.is_empty
            continue
        speech += 1
        errors += edit_distance(list(lattice.best), e["phones"])
        reference += len(e["phones"])
        exact += list(lattice.best) == e["phones"]
        top_k += any(list(h.phonemes) == e["phones"] for h in lattice.hypotheses)
    return dict(null_emissions=null_fired, null_takes=null_total, speech_takes=speech,
                phoneme_error_rate=errors / reference if reference else None,
                exact_sequences=exact, in_top_k=top_k)


@torch.inference_mode()
def negative_emissions(model, device, decoder=None):
    """Non-speech session clips (blink, swallow, look down), ROI stream only."""
    decoder = decoder or PhonemeLatticeDecoder()
    fired = total = 0
    for path in session_clips():
        if not path.stem.startswith("neg_"):
            continue
        clip = load_session_clip(path)
        out = model(None, torch.from_numpy(clip["mouth_roi_30hz"])[None].to(device))
        fired += not decoder.decode(out.log_probs[0]).is_empty
        total += 1
    return dict(negative_clip_emissions=fired, negative_clips=total)


def report(title, metrics):
    print(f"\n{title}")
    print(f"  emissions on __NULL__ takes : {metrics['null_emissions']}/{metrics['null_takes']}")
    if "negative_clips" in metrics:
        print(f"  emissions on negative clips : {metrics['negative_clip_emissions']}/{metrics['negative_clips']}")
    per = metrics["phoneme_error_rate"]
    print(f"  phoneme error rate          : {'n/a' if per is None else f'{per:.1%}'} "
          f"({metrics['speech_takes']} speech takes)")
    print(f"  exact phoneme sequences     : {metrics['exact_sequences']}/{metrics['speech_takes']}"
          f"  (in top-K: {metrics['in_top_k']}/{metrics['speech_takes']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("base", "lora"), required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--train-session", help="default: the earlier session in data/silent_train")
    ap.add_argument("--test-session", help="default: the later session")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 (base) or 1e-3 (lora)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--skip-negatives", action="store_true", help="skip the session negative clips")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    sessions = sorted({folder.parent.name for folder, _ in silent_takes()})
    if len(sessions) < 2:
        raise SystemExit("Need two recorded sessions in data/silent_train")
    train_session = args.train_session or sessions[0]
    test_session = args.test_session or sessions[-1]
    if train_session == test_session:
        raise SystemExit("Train and test sessions must differ")
    train, test = load_examples({train_session}), load_examples({test_session})
    print(f"train {train_session}: {len(train)} takes   test {test_session}: {len(test)} takes   device {args.device}")

    if args.mode == "base":
        model = DualStreamPhonemeEncoder(EncoderConfig())
        if DEFAULT_FRONTEND.is_file():
            model.load_auto_avsr_frontend(DEFAULT_FRONTEND)
            print(f"front-end weights: {DEFAULT_FRONTEND}")
        model.freeze_frontend().to(args.device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        adapter = None
    else:
        model = DualStreamPhonemeEncoder.load(args.checkpoint_dir, args.device).freeze_frontend()
        adapter = LoRAAdapter(model, rank=args.rank, alpha=2 * args.rank)
        model.to(args.device)
        parameters = adapter.parameters()
    print(f"trainable parameters: {sum(p.numel() for p in parameters):,}")

    before = evaluate(model, test, args.device)
    start = time.time()
    history = train_epochs(model, parameters, train, args.epochs, args.lr or (3e-4 if adapter is None else 1e-3),
                           args.batch_size, args.device, args.seed)
    after = evaluate(model, test, args.device)
    if not args.skip_negatives:
        after.update(negative_emissions(model, args.device))
    report(f"Before training, held-out session {test_session}", before)
    report(f"After {args.epochs} epoch(s), held-out session {test_session}", after)

    if adapter is None:
        model.save(args.checkpoint_dir)
        saved = args.checkpoint_dir / "encoder.safetensors"
    else:
        adapter.save(args.checkpoint_dir / "speaker_lora")
        saved = args.checkpoint_dir / "speaker_lora" / "adapter.safetensors"
    metrics = dict(mode=args.mode, train_session=train_session, test_session=test_session, epochs=args.epochs,
                   train_loss=history, seconds=round(time.time() - start, 1), before=before, after=after)
    (args.checkpoint_dir / f"train_ctc_{args.mode}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {saved}")


if __name__ == "__main__":
    main()
