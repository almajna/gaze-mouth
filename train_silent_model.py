"""Fit a personal silent-command classifier on data/silent_train.

--train-all-collected selects the epoch, CTC alignment weight, temperature and
rejection gates from leave-one-session-out predictions, then refits on all takes.
--eval-heldout applies the frozen checkpoint to data/silent_speech.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar
import torch
from torch import nn
from torch.nn import functional as F

from eval_silent_speech import resample_video
from silent_train_collect import CLASSES, ROOT

TRAIN = ROOT / "data/silent_train"
TEST = ROOT / "data/silent_speech"
CHECKPOINT = ROOT / "models/silent_command_net.pt"
REPO = ROOT / "models/auto_avsr"
NULL = CLASSES.index("__NULL__")
GEOMETRY = 52 * 3
DIM = GEOMETRY + 768


class CommandAlignment(nn.Module):
    """Compare each vocabulary phrase with the frozen Auto-AVSR CTC best path."""
    def __init__(self, projection, units):
        super().__init__()
        import sentencepiece as spm
        processor = spm.SentencePieceProcessor(model_file=str(REPO / "spm/unigram/unigram5000.model"))
        ids = {piece: int(index) for piece, index in (line.split() for line in units)}
        phrases = [[ids[p] for p in processor.encode(label.upper(), out_type=str)] for label in CLASSES[:NULL]]
        symbols = sorted({0, *(i for phrase in phrases for i in phrase)})
        self.projection = projection.eval().requires_grad_(False)
        self.symbols = torch.tensor(symbols)
        self.targets = torch.tensor([symbols.index(i) for phrase in phrases for i in phrase])
        self.lengths = torch.tensor([len(phrase) for phrase in phrases])

    @torch.no_grad()
    def forward(self, visual):
        logp = self.projection(visual).log_softmax(-1)
        best_path = logp.max(dim=1).values.sum()
        # CTC only looks up blank and target symbols. Select after log_softmax
        # so all other words still compete for probability.
        scores = logp[:, self.symbols, None].transpose(1, 2).expand(-1, NULL, -1)
        cost = F.ctc_loss(scores, self.targets, torch.full_like(self.lengths, len(logp)),
                          self.lengths, reduction="none")
        per_token = (-cost - best_path) / self.lengths
        return F.pad(per_token.clamp(-20, 5), (0, 1))  # zero adjustment to NULL


def load_encoder(device):
    sys.path.insert(0, str(REPO))
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

    units = (REPO / "spm/unigram/unigram5000_units.txt").read_text().splitlines()
    model = E2E(len(units) + 2, "video")
    model.load_state_dict(torch.load(REPO / "vsr.pth", map_location="cpu", weights_only=True))
    encoder = nn.Sequential(model.frontend, model.proj_encoder, model.encoder).eval().requires_grad_(False).to(device)
    encoder.add_module("alignment", CommandAlignment(model.ctc.ctc_lo, units))
    return encoder


def trajectories(path, meta, frames, names):
    channels = meta["blendshape_names"]
    if len(channels) != 52 or sorted(channels) != names:
        raise ValueError(f"{path}: inconsistent blendshape channels")
    shapes = np.load(path, allow_pickle=False)
    if shapes.shape != (frames, 52) or not np.isfinite(shapes).all():
        raise ValueError(f"{path}: invalid trajectories")
    # Both streams use the same source timestamps, interpolator and output ticks.
    values = resample_video(shapes[:, [channels.index(n) for n in names], None], meta)[4:-4, :, 0]
    return values


def derivatives(values):
    delta = np.diff(values, axis=0, prepend=values[:1])
    return np.concatenate([values, delta, np.diff(delta, axis=0, prepend=delta[:1])], axis=1)


def observations(folder, meta, names, allow_missing):
    video = np.load(folder / "mouth_96x96.npy", allow_pickle=False)
    if video.dtype != np.uint8 or video.shape[1:] != (96, 96) or meta["video_fps"] != 25 or not len(video):
        raise ValueError(f"{folder}: expected 25 fps uint8 mouth crops")
    frames = resample_video(video, meta)[4:-4, 4:92, 4:92]
    path = folder / "blendshapes.npy"
    shapes = None if allow_missing and not path.exists() else trajectories(path, meta, len(video), names)
    return frames, shapes


@torch.no_grad()
def encode(frames, shapes, encoder, device):
    pixels = np.pad(frames, ((4, 4), (0, 0), (0, 0)), mode="edge")
    x = torch.from_numpy(pixels).to(device).float()
    x = ((x / 255 - .421) / .165)[None, :, None]
    encoded, _ = encoder[2](encoder[1](encoder[0](x)), None)
    return (None if shapes is None else torch.from_numpy(derivatives(shapes)).float(),
            encoded[0, 4:-4].cpu())


def features(folder, meta, encoder, device, names, allow_missing=False):
    return encode(*observations(folder, meta, names, allow_missing), encoder, device)


class CommandNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(.2))
        self.gru = nn.GRU(64, 48, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(96, 1)
        self.output = nn.Sequential(nn.Dropout(.3), nn.Linear(96, len(CLASSES)))

    def batch(self, sequences):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
        lengths = [len(x) for x in sequences]
        x = self.input(pad_sequence(sequences, batch_first=True))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)
        valid = torch.arange(h.size(1), device=h.device)[None, :] < torch.tensor(lengths, device=h.device)[:, None]
        weights = self.attention(h).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        return self.output((h * weights.unsqueeze(-1)).sum(1))

    def forward(self, x):
        return self.batch([x])[0]


def standardized(pair, mean, scale):
    shapes, visual = pair
    if shapes is None:
        # Missing modality is masked in standardized space, as during training.
        shapes = mean[:GEOMETRY].expand(len(visual), -1)
    return (torch.cat([shapes, visual], 1) - mean) / scale


def augment(x, rng):
    length = max(2, round(len(x) / rng.uniform(.85, 1.15)))
    warped = F.interpolate(x.T[None], size=length, mode="linear", align_corners=False)[0].T
    keep = rng.random(length) >= .1
    keep[[0, -1]] = True
    warped = warped[keep]
    warped += torch.randn_like(warped) * .025
    if rng.random() < .3:
        warped[:, :GEOMETRY] = 0
    return warped


def clips(root):
    if root.resolve() != root:
        raise ValueError(f"Expected local dataset directory: {root}")
    for p in sorted(root.glob("*/*/meta.json")):
        if any(f.is_symlink() for f in [p.parent.parent, p.parent, p, p.parent / "mouth_96x96.npy",
                                         p.parent / "blendshapes.npy"]):
            raise ValueError(f"Refusing linked dataset entry {p}")
        meta = json.loads(p.read_text())
        if meta.get("saved") and meta["run"] == p.parent.parent.name:
            yield p.parent, meta
        else:
            raise ValueError(f"Unexpected clip provenance: {p}")


def fingerprint(folder):
    return hashlib.sha256((folder / "mouth_96x96.npy").read_bytes()).hexdigest()


def score_logits(logits, temperature, command_ranking=True):
    scaled = logits.double() / temperature
    probs = scaled.softmax(-1)
    # Score the most likely command against its closest command competitor.
    # NULL remains in the softmax denominator. Its training frequency includes
    # artificial exposures, so its argmax alone is not a calibrated rejection.
    values, ranking = (probs[..., :NULL] if command_ranking else probs).topk(3)
    margin = values[..., 0] - values[..., 1]
    # Command-only free energy. Strong evidence for NULL must not lower it.
    energy = -temperature * torch.logsumexp(scaled[..., :NULL], dim=-1)
    return values[..., 0], margin, energy, ranking


def calibrate(logits, labels, groups):
    logits = logits.double()
    # Equal total weight for clean validation clips and the proxy negatives.
    losses = lambda t: F.cross_entropy(logits / t, labels, reduction="none")
    clean = groups < 2
    def nll(t):
        loss = losses(t)
        return (loss[clean].mean() + loss[~clean].mean()).item() / 2
    temperature = float(minimize_scalar(nll, bounds=(.05, 20.), method="bounded").x)
    prob, margin, energy, rank = score_logits(logits, temperature)
    prob, margin, energy = (v.numpy() for v in (prob, margin, energy))
    commands = groups == 0
    correct = (rank[:, 0] == labels).numpy() & commands
    n_commands = int(commands.sum())
    neg = [groups == i for i in (1, 2, 3)]
    sizes = [int(mask.sum()) for mask in neg]
    best = None
    # Every attainable nonempty set of accepted commands has a weakest passing
    # probability and margin. Cap energy just below the first disallowed negative.
    for pc in np.unique(np.r_[np.finfo(float).eps, prob[commands]]):
        p_ok = prob >= pc
        for mc in np.unique(np.r_[np.finfo(float).eps, margin[commands & p_ok]]):
            keep = p_ok & (margin >= mc)
            limit = float(np.max(energy[groups == 0]))
            for mask, size in zip(neg, sizes):
                e = np.sort(energy[keep & mask])
                allowed = size // 20
                if len(e) > allowed:
                    limit = min(limit, np.nextafter(e[allowed], -np.inf))
            accepted = keep & (energy <= limit)
            if (accepted & commands).any():
                limit = energy[accepted & commands].max()
                accepted &= energy <= limit
            hit = int((accepted & correct).sum())
            passed = int((accepted & commands).sum())
            fp = sum(int((accepted & mask).sum()) for mask in neg)
            # Prefer multiple useful gates over relying entirely on one score.
            quality = (min(passed, int(np.ceil(.95*n_commands))), hit, passed, -fp, mc, pc)
            if best is None or quality > best:
                best = quality
                cal = dict(temperature=temperature, prob_cutoff=float(pc), margin_cutoff=float(mc),
                           energy_cutoff=float(limit), recall=passed/n_commands,
                           correct_recall=hit/n_commands,
                           ungated_top1=float(correct.sum()/n_commands),
                           rejection=[1-int((accepted & mask).sum())/n for mask,n in zip(neg,sizes)])
    return cal


def mix_sequences(a, b, rng):
    other = F.interpolate(b.T[None], size=len(a), mode="linear", align_corners=False)[0].T
    return torch.lerp(a, other, float(rng.uniform(.35, .65)))


def fit_model(train_rows, validation, epochs, checkpoints, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    population = torch.cat([x for views, _ in train_rows for x in views[:1]])
    mean, scale = population.mean(0), population.std(0, unbiased=False)
    scale[scale < 1e-6] = 1
    fit = [([(x-mean)/scale for x in views], y) for views,y in train_rows]
    model = CommandNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    predictions = {}
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(fit))
        for start in range(0, len(order), 12):
            xs, ys = [], []
            for i in order[start:start+12]:
                views, y = fit[i]
                xs.append(augment(views[0], rng))
                ys.append(y)
                if y != NULL:
                    if rng.random() < .35:
                        j = rng.choice([k for k,(_, label) in enumerate(fit) if label not in (y,NULL)])
                        negative = mix_sequences(views[0], fit[j][0][0], rng)
                    else:
                        negative = views[rng.integers(1,3)]
                    xs.append(augment(negative, rng))
                    ys.append(NULL)
            logits = model.batch([x.to(device) for x in xs])
            targets = torch.tensor(ys, device=device)
            losses = F.cross_entropy(logits, targets, label_smoothing=.08, reduction="none")
            # Each real command brings one extra proxy negative. Keep the single
            # rejection class from dominating the twenty individual command logits.
            weights = torch.where(targets == NULL, .35, 1.)
            loss = (losses * weights).sum() / weights.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        if epoch in checkpoints and validation:
            model.eval()
            with torch.no_grad():
                vals = []
                for views,y in validation:
                    for j,view in enumerate(views):
                        x = (view-mean)/scale
                        dual = model(x.to(device)).cpu()
                        x[:, :GEOMETRY] = 0
                        visual = model(x.to(device)).cpu()
                        vals.append((dual, visual, NULL if j else y, j+1 if j else int(y==NULL)))
                predictions[epoch] = vals
    return model, mean, scale, predictions


def train(args, device):
    records = list(clips(TRAIN))
    sessions = sorted({m["run"] for _,m in records})
    if len(sessions) < 2:
        raise SystemExit("Training requires two or more collected sessions.")
    folds = sessions if args.train_all_collected else [args.val_session or sessions[-1]]
    if any(s not in sessions for s in folds):
        raise SystemExit("Unknown validation session")
    hashes = set()
    names = sorted(records[0][1]["blendshape_names"])
    data = {s: [] for s in sessions}
    alignments = {s: [] for s in folds}
    encoder = load_encoder(device)
    rng = np.random.default_rng(args.seed)
    for folder, meta in records:
        digest = fingerprint(folder)
        if digest in hashes or meta["mode"] != "silent":
            raise ValueError(f"Duplicate or invalid training clip: {folder}")
        hashes.add(digest)
        label = CLASSES.index(meta["phrase"])
        frames, shapes = observations(folder, meta, names, False)
        views = [torch.cat(encode(frames, shapes, encoder, device), 1)]
        if label != NULL:
            for idx in (np.arange(len(frames))[::-1], rng.permutation(len(frames))):
                views.append(torch.cat(encode(frames[idx], shapes[idx], encoder, device), 1))
        data[meta["run"]].append((views,label))
        if meta["run"] in alignments:
            alignments[meta["run"]].extend(encoder.alignment(view[:, GEOMETRY:]) for view in views)
    del encoder
    if any({y for _,y in v} != set(range(len(CLASSES))) for v in data.values()):
        raise ValueError("Each calibration fold requires all 21 classes")
    epochs = sorted({max(1,args.epochs//4), max(1,args.epochs//2), args.epochs})
    pooled = {e: [] for e in epochs}
    for session in folds:
        fit = [row for s in sessions if s != session for row in data[s]]
        print(f"Validating on {session}, fitting {len(fit)} takes", flush=True)
        _, _, _, preds = fit_model(fit, data[session], args.epochs, epochs, args.seed, device)
        for e in epochs:
            pooled[e].extend(preds[e])
    best = None
    alignment = torch.stack([v for s in folds for v in alignments[s]])
    for epoch, vals in pooled.items():
        dual, visual, labels, groups = zip(*vals)
        candidates = {}
        for mode, logits in (("dual", dual), ("visual", visual)):
            logits = torch.stack(logits)
            options = []
            for weight in (0., .5, 1., 2.):
                cal = calibrate(logits + weight * alignment, torch.tensor(labels), np.array(groups))
                options.append(dict(cal, alignment_weight=weight))
            candidates[mode] = max(options, key=lambda c: (min(.95, c["recall"]), c["correct_recall"],
                                                            c["recall"], min(c["rejection"])))
        recalls = [c["recall"] for c in candidates.values()]
        correct = [c["correct_recall"] for c in candidates.values()]
        print(f"Out-of-fold epoch {epoch}: accepted command recall dual/visual {recalls}, "
              f"correctly identified and accepted {correct}", flush=True)
        quality = (min(.95, min(recalls)), min(correct), sum(correct), sum(recalls))
        if best is None or quality > best:
            best, best_epoch, calibration = quality, epoch, candidates
    selected = sessions if args.train_all_collected else [s for s in sessions if s not in folds]
    model, mean, scale, _ = fit_model([r for s in selected for r in data[s]], [], best_epoch, [],
                                      args.seed, device)
    for mode, cal in calibration.items():
        print(f"{mode}: command recall {cal['recall']:.1%}; rejection (NULL/reverse/shuffle) "
              f"{cal['rejection']}; meets both targets: {cal['recall'] >= .95 and min(cal['rejection']) >= .95}")
    state = dict(format=3, classes=list(CLASSES), names=names, mean=mean, scale=scale,
                 weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
                 calibration=calibration, clip_hashes=sorted(hashes), sessions=sessions, validation=folds,
                 fit_sessions=selected, epoch=best_epoch, seed=args.seed, encoder_sha256=file_digest(REPO / "vsr.pth"))
    CHECKPOINT.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=CHECKPOINT.parent, delete=False) as tmp:
        try:
            torch.save(state, tmp)
            tmp.close()
            os.replace(tmp.name, CHECKPOINT)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    print(f"Saved {CHECKPOINT} (epoch {best_epoch}); validation calibration: {calibration}")


def file_digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def summary(rows, title):
    negative = [r for r in rows if r["target"] == NULL]
    target = [r for r in rows if r["target"] != NULL]
    def rate(count, total):
        return f"{count}/{total} ({count/total:.1%})" if total else "n/a (0 clips)"
    print(f"\n{title}: false accepts {rate(sum(r['decision'] != NULL for r in negative), len(negative))}")
    if target:
        print("Command error:", rate(sum(r["decision"] != r["target"] for r in target), len(target)))
        print("Accepted and correct:", rate(sum(r["decision"] == r["target"] for r in target), len(target)),
              "Gate recall:", rate(sum(r["decision"] != NULL for r in target), len(target)))
        print("Raw 21-class Top-1:", rate(sum(r["raw_label"] == r["target"] for r in target), len(target)))
        print("Ungated candidate Top-1:", rate(sum(r["rank"][0] == r["target"] for r in target), len(target)),
              "Top-3:", rate(sum(r["target"] in r["rank"] for r in target), len(target)))
    for name, subset in (("targets", target), ("non-targets", negative)):
        if subset:
            print(f"{name} calibrated top candidate probability p05/p50/p95:",
                  np.round(np.percentile([r["conf"] for r in subset], [5,50,95]), 3))
    print(f"Median inference (encoder through decision): {np.median([r['ms'] for r in rows]):.1f} ms")


def predict(observed, model, encoder, state, device):
    pair = encode(*observed, encoder, device)
    mode = "visual" if pair[0] is None else "dual"
    x = standardized(pair, state["mean"], state["scale"])
    cal = state["calibration"][mode]
    logits = model(x.to(device)).cpu()
    net_command_conf = logits.softmax(-1)[:NULL].max().item()
    if cal.get("alignment_weight", 0):
        logits = logits + cal["alignment_weight"] * encoder.alignment(pair[1])
    conf, margin, energy, ranking = score_logits(logits, cal["temperature"], state["format"] >= 3)
    if "prob_cutoff" in cal:
        accept = (conf.item() >= cal["prob_cutoff"] and margin.item() >= cal["margin_cutoff"]
                  and energy.item() <= cal["energy_cutoff"])
    else:  # legacy checkpoint
        accept = conf.item() > cal["cutoff"]
    rank = ranking.tolist()
    return mode, conf.item(), rank, logits.argmax().item(), rank[0] if accept else NULL, net_command_conf


@torch.no_grad()
def evaluate(device):
    if not CHECKPOINT.exists():
        raise SystemExit(f"No trained checkpoint at {CHECKPOINT}. Collect sessions and train first.")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if state["classes"] != list(CLASSES) or state["encoder_sha256"] != file_digest(REPO / "vsr.pth"):
        raise ValueError("Checkpoint vocabulary or pretrained encoder mismatch")
    model = CommandNet().to(device).eval()
    model.load_state_dict(state["weights"])
    encoder = load_encoder(device)
    seen = set(state["clip_hashes"])
    rows = []
    for folder, meta in clips(TEST):
        digest = fingerprint(folder)
        if digest in seen or meta["run"] in state["sessions"]:
            raise ValueError(f"Duplicate clip or overlap with development sessions: {folder}")
        seen.add(digest)
        target = CLASSES.index(meta["phrase"]) if meta["phrase"] in CLASSES else NULL
        observed = observations(folder, meta, state["names"], allow_missing=True)
        if not rows:
            predict(observed, model, encoder, state, device)  # real warmup outside timer
        start = perf_counter()
        mode, conf, rank, raw_label, decision, net_conf = predict(observed, model, encoder, state, device)
        elapsed_ms = (perf_counter()-start)*1000
        rows.append(dict(session=meta["run"], mode=meta["mode"], stream=mode,
                         phrase=meta["phrase"], target=target, conf=conf, rank=rank,
                         raw_label=raw_label, decision=decision, ms=elapsed_ms, net_conf=net_conf))
    if not rows:
        raise SystemExit("No locked clips available")
    for group, members in (("non-speech nulls", {"__NULL__"}),
                           ("out-of-vocabulary phrases", {r["phrase"] for r in rows} - set(CLASSES))):
        negatives = [r for r in rows if r["target"] == NULL and r["phrase"] in members]
        n = len(negatives)
        count = sum(r["decision"] != NULL for r in negatives)
        print(f"{group} false accepts: {count}/{n} ({count/n:.1%})" if n else f"{group}: no locked examples")
    oov = [r for r in rows if r["phrase"] not in CLASSES]
    print(f"OOV command probabilities above 0.95: raw network "
          f"{sum(r['net_conf'] > .95 for r in oov)}/{len(oov)}, calibrated "
          f"{sum(r['conf'] > .95 and r['rank'][0] != NULL for r in oov)}/{len(oov)}")
    for row in oov:
        print(f"  {row['session']} | {row['phrase']} | {CLASSES[row['decision']]} "
              f"| top candidate {CLASSES[row['rank'][0]]} ({row['conf']:.3f})")
    print("Calibration:", state["calibration"])
    for field in ("session", "stream", "mode"):
        for key in sorted({r[field] for r in rows}):
            summary([r for r in rows if r[field] == key], key)
    summary(rows, f"All locked clips on {device}")
    print("Confusion matrix of final decisions. OOV phrases share the __NULL__ row. Class order:")
    print(list(CLASSES))
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for r in rows:
        matrix[r["target"],r["decision"]] += 1
    for label, row in zip(CLASSES, matrix):
        print(f"{label:18}", *row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--eval-heldout", action="store_true")
    action.add_argument("--train-all-collected", action="store_true", help="Tune by leaving each session out, then refit on all sessions")
    parser.add_argument("--val-session", help="entire validation session (default latest training session)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.val_session and args.train_all_collected:
        parser.error("--train-all-collected uses every session as a validation fold")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.eval_heldout:
        evaluate(device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
