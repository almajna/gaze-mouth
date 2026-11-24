#!/usr/bin/env python3
"""Train a speaker LoRA adapter on recorded 25 Hz USR 2.0 sentence clips.

Joint loss (CTC + label-smoothed attention), LoRA on every encoder layer and on the decoder's
cross-attention, optional BatchNorm statistics pass. Part of the clips is held back for validation;
the adapter in --output is the epoch with the lowest validation joint loss.
"""
import argparse
import json
import math
from pathlib import Path
import random
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eval.eval_silent_speech import errors
from eval.eval_wer import load_clip
from silent_avsr.features import DATA, ROOT
from silent_avsr.lora_adapter import LoRAAdapter
from silent_avsr.pretrained import PretrainedVSR
from silent_avsr.sentence_video import normalize_contrast
from silent_avsr.usr_tokenizer import USRTokenizer

# Every Linear of the 12 encoder layers and the q/k/v/out projections of the decoder's cross-attention.
INCLUDE = r"(?:encoder\.encoders\.\d+\..*|decoder\.decoders\.\d+\.src_attn\.linear_(?:q|k|v|out))"
ENCODER_ONLY = r"encoder\.encoders\.\d+\..*"


def augment(crops, rng):
    # A consistent spatial shift, flip and exposure across a clip preserve motion.
    dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
    gain, offset = rng.uniform(.9, 1.1), rng.uniform(-8, 8)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = np.stack([cv2.warpAffine(frame, matrix, (88, 88),
                                      borderMode=cv2.BORDER_REFLECT_101) for frame in crops])
    if rng.random() < .5:
        shifted = shifted[:, :, ::-1]
    adjusted = np.clip(shifted.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    normalized = normalize_contrast(np.ascontiguousarray(adjusted))
    # Upstream's adaptive time mask: one mask of 0 to 9 frames per started second, zero after normalization.
    for _ in range(math.ceil(len(normalized) / 25)):
        width = rng.randint(0, 9)
        if 0 < width < len(normalized):
            start = rng.randrange(0, len(normalized) - width)
            normalized[start:start + width] = 0
    return normalized


def set_mode(model, training):
    """Dropout follows `training`; BatchNorm always uses its running statistics."""
    model.eval()
    if training:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()


def joint_loss(model, frames, target, ctc_weight, smoothing):
    """Loss per token for one clip, with its parts and the (T, vocabulary) CTC log-probabilities."""
    device = next(model.decoder.parameters()).device
    encoded = model.encoder(xs_v=torch.from_numpy(frames)[None].to(device))
    log_probs = model.ctc_v.ctc_lo(encoded).log_softmax(-1).transpose(0, 1).float().cpu()
    y = torch.tensor(target, dtype=torch.long)
    # MPS has no dependable CTC loss, so this term runs on the CPU.
    ctc = F.ctc_loss(log_probs, y[None], torch.tensor([log_probs.shape[0]]), torch.tensor([len(target)]),
                     blank=0, reduction="sum", zero_infinity=True)
    edge = torch.tensor([model.sos], dtype=torch.long)
    ys_in, ys_out = torch.cat([edge, y])[None].to(device), torch.cat([y, edge]).to(device)
    mask = torch.ones(ys_in.shape[1], ys_in.shape[1], dtype=torch.bool, device=device).tril()[None]
    hidden, _ = model.decoder(ys_in, mask, encoded, None)
    logp = model.decoder.out_layer_v(hidden)[0].float().log_softmax(-1)
    soft = torch.full_like(logp, smoothing / (logp.shape[-1] - 1)).scatter_(1, ys_out[:, None], 1 - smoothing)
    attention = -(soft * logp).sum().cpu()
    tokens = len(target) + 1
    loss = (ctc_weight * ctc + (1 - ctc_weight) * attention) / tokens
    correct = (logp.argmax(-1) == ys_out).sum().item()
    return loss, ctc.item() / tokens, attention.item() / tokens, correct, log_probs[:, 0]


@torch.no_grad()  # not inference_mode: the encoder caches its positional table for later training steps
def validate(vsr, examples, ctc_weight, smoothing):
    """Mean losses without augmentation or dropout, token accuracy, and greedy CTC word errors."""
    set_mode(vsr.model, False)
    sums, correct, tokens, edits, count = np.zeros(3), 0, 0, 0, 0
    for path, crops, target, sentence in examples:
        loss, ctc, attention, right, log_probs = joint_loss(vsr.model, normalize_contrast(crops), target,
                                                            ctc_weight, smoothing)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss on {path}")
        sums += loss.item(), ctc, attention
        correct += right
        tokens += len(target) + 1
        pieces, previous = [], 0
        for token in log_probs[:, :-1].argmax(-1).tolist():
            if token != previous and token != 0:
                pieces.append(vsr.labels[token])
            previous = token
        wrong, reference_words = errors(sentence, "".join(pieces).replace("▁", " "))
        edits += wrong
        count += reference_words
    loss, ctc, attention = sums / len(examples)
    return dict(val_loss=loss, val_ctc=ctc, val_attention=attention, val_token_accuracy=correct / tokens,
                val_greedy_word_errors=edits, val_reference_words=count)


def batchnorm_layers(model):
    return {f"encoder.frontend_v.{name}": module for name, module in model.encoder.frontend_v.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}


@torch.no_grad()
def batchnorm_statistics(model, examples):
    """Replace the front-end's BatchNorm running statistics, in place, with averages over the given clips."""
    layers = batchnorm_layers(model).values()
    device = next(model.decoder.parameters()).device
    for module in layers:
        module.reset_running_stats()
        module.momentum = None  # cumulative average over clips
        module.train()
    for _, crops, _, _ in examples:
        model.encoder(xs_v=torch.from_numpy(normalize_contrast(crops))[None].to(device))
    for module in layers:
        module.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA / "train_sentences")
    ap.add_argument("--output", type=Path, default=ROOT / "models/usr2_speaker_lora_v2")
    ap.add_argument("--device", choices=("cpu", "mps"))
    ap.add_argument("--split", type=Path, default=ROOT / "models/usr2_speaker_lora/training.json",
                    help="JSON with train_clips and val_clips file names")
    ap.add_argument("--folds", type=int, help="Use a seeded k-fold partition of all clips instead of --split")
    ap.add_argument("--fold", type=int, default=0, help="Fold held back for validation with --folds")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2)
    ap.add_argument("--accumulate", type=int, default=1, help="Clips per optimizer update")
    ap.add_argument("--ctc-weight", type=float, default=0.3)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--encoder-only", action="store_true", help="No LoRA on the decoder cross-attention")
    ap.add_argument("--bn-stats", action=argparse.BooleanOptionalAction, default=False,
                    help="Recompute front-end BatchNorm statistics on the training clips before training "
                         "(off: it cost 12 points of validation WER, see docs/plan.md)")
    ap.add_argument("--save-last", action="store_true", help="Keep the last epoch instead of the best one")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    paths = sorted(args.data.glob("*.npz"))
    if not paths:
        ap.error(f"No recorded training clips in {args.data}")
    if args.epochs < 1 or args.lr <= 0 or args.accumulate < 1:
        ap.error("epochs, learning rate and accumulation must be positive")
    if not 0 <= args.ctc_weight <= 1:
        ap.error("the CTC weight must be between 0 and 1")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(4)
    vsr = PretrainedVSR(device=args.device)
    model = vsr.model
    processor = USRTokenizer()
    if any(vsr.labels[index] != piece for piece, index in processor.pieces.items()):
        raise ValueError("USR tokenizer IDs do not match the checkpoint labels")
    examples = {}
    for path in paths:
        meta, crops = load_clip(path)
        if meta.get("split") != "train_sentences":
            raise ValueError(f"{path}: not a training clip (split {meta.get('split')!r})")
        if meta.get("kind") != "sentence":
            continue
        target = processor.encode(meta["sentence"])
        required_frames = len(target) + sum(a == b for a, b in zip(target, target[1:]))
        if required_frames > len(crops):
            raise ValueError(f"{path}: too few frames for its CTC target")
        examples[path.name] = (path, crops, target, meta["sentence"])
    if args.folds:
        if not 0 <= args.fold < args.folds <= len(examples):
            ap.error("--fold must be in 0..folds-1")
        order = sorted(examples)
        random.Random(args.seed).shuffle(order)
        held = set(order[args.fold::args.folds])
        split = f"fold {args.fold} of {args.folds}, seed {args.seed}"
    else:
        lists = json.loads(args.split.read_text())
        held = set(lists["val_clips"])
        if held & set(lists["train_clips"]) or held | set(lists["train_clips"]) != set(examples):
            raise ValueError(f"{args.split}: clip lists do not partition the clips in {args.data}")
        split = str(args.split)
    validation = [examples[name] for name in sorted(held)]
    training = [examples[name] for name in sorted(set(examples) - held)]
    if not validation or not training:
        ap.error("Not enough sentence clips for a train and a validation part")

    include = ENCODER_ONLY if args.encoder_only else INCLUDE
    adapter = LoRAAdapter(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout, include=include, full=(),
                          buffers=sorted(batchnorm_layers(model)) if args.bn_stats else ())
    if args.bn_stats:
        # The LoRA update starts at zero, so this pass sees the pretrained network.
        batchnorm_statistics(model, training)
    print(f"Training {len(adapter.parameters())} adapter tensors "
          f"({sum(p.numel() for p in adapter.parameters())} values) on {len(training)} clips, "
          f"validating on {len(validation)} ({split}), device {vsr.device}", flush=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(training) / args.accumulate)
    total, warmup = updates_per_epoch * args.epochs, max(1, round(updates_per_epoch * args.warmup_epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup
                                                  else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    settings = dict(
        backend="usr2", checkpoint=str(vsr.backend), tokenizer="USR units, fewest-token Viterbi",
        data=str(args.data), split=split, device=vsr.device, epochs=args.epochs, learning_rate=args.lr,
        weight_decay=args.weight_decay, betas=[0.9, 0.98], warmup_epochs=args.warmup_epochs,
        schedule="linear warmup, cosine decay per update", accumulate=args.accumulate, gradient_clip=3.0,
        loss=f"{args.ctc_weight} * CTC + {1 - args.ctc_weight} * attention, per token",
        ctc_weight=args.ctc_weight, label_smoothing=args.smoothing, lora=adapter.config,
        batchnorm="statistics recomputed on the training clips, then frozen" if args.bn_stats else "pretrained",
        dropout_modules="active during training", selection="last epoch" if args.save_last else "lowest validation joint loss",
        augmentation="shift ±4 px (reflect), flip 0.5, gain 0.9-1.1, offset ±8, time masks 0-9 frames per second",
        seed=args.seed, clips=len(examples))

    row = validate(vsr, validation, args.ctc_weight, args.smoothing)
    print(f"Epoch 0/{args.epochs}  before training  val loss {row['val_loss']:.4f} "
          f"(CTC {row['val_ctc']:.3f}, attention {row['val_attention']:.3f})  "
          f"token accuracy {row['val_token_accuracy']:.1%}  greedy CTC word errors "
          f"{row['val_greedy_word_errors']}/{row['val_reference_words']}", flush=True)
    history = [dict(epoch=0, train_loss=None, **row)]
    best = dict(epoch=0, val_loss=row["val_loss"])
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        rng.shuffle(training)
        set_mode(model, True)
        running = 0.0
        optimizer.zero_grad()
        for index, (path, crops, target, _) in enumerate(training, 1):
            loss, *_ = joint_loss(model, augment(crops, rng), target, args.ctc_weight, args.smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss on {path}")
            (loss / args.accumulate).backward()
            running += loss.item()
            if index % args.accumulate == 0 or index == len(training):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        row = validate(vsr, validation, args.ctc_weight, args.smoothing)
        history.append(dict(epoch=epoch, train_loss=running / len(training), **row))
        improved = args.save_last or row["val_loss"] < best["val_loss"]
        if improved:
            best = dict(epoch=epoch, val_loss=row["val_loss"])
            adapter.save(args.output)
        print(f"Epoch {epoch}/{args.epochs}  train loss {running / len(training):.4f}  val loss {row['val_loss']:.4f} "
              f"(CTC {row['val_ctc']:.3f}, attention {row['val_attention']:.3f})  "
              f"token accuracy {row['val_token_accuracy']:.1%}  greedy CTC word errors "
              f"{row['val_greedy_word_errors']}/{row['val_reference_words']}  "
              f"{perf_counter() - started:.0f} s{'  saved' if improved else ''}", flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "training.json").write_text(json.dumps(dict(
            **settings, epochs_run=epoch, saved_epoch=best["epoch"] or None,
            saved_val_loss=best["val_loss"] if best["epoch"] else None, history=history,
            train_clips=sorted(path.name for path, *_ in training),
            val_clips=sorted(path.name for path, *_ in validation)), indent=2) + "\n")
    if best["epoch"]:
        print(f"Saved epoch {best['epoch']}: val loss {best['val_loss']:.4f}, adapter in {args.output}")
    else:
        print("No epoch improved on the untrained validation loss. No adapter was saved.")


if __name__ == "__main__":
    main()
