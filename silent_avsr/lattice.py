"""CTC prefix beam search -> compact top-K phoneme lattice with log-probabilities and frame spans."""
from dataclasses import dataclass, field
import math

import numpy as np

from .features import RATE_HZ
from .phonemes import BLANK_ID, ID, SYMBOLS, homophenes

NEG_INF = float("-inf")


def logaddexp(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


@dataclass
class Hypothesis:
    phonemes: tuple          # ARPAbet symbols
    log_prob: float          # CTC log-probability of the sequence, summed over alignments
    frames: list             # [start, end] (inclusive) per phoneme, from the best single alignment


@dataclass
class Slot:
    frames: list             # [start, end] of the best hypothesis's phoneme
    alternatives: list       # [(phoneme, log_prob)] best first; log-probs normalized over phonemes


@dataclass
class PhonemeLattice:
    hypotheses: list = field(default_factory=list)
    slots: list = field(default_factory=list)
    num_frames: int = 0
    frame_rate: float = RATE_HZ

    @property
    def best(self):
        return self.hypotheses[0].phonemes if self.hypotheses else ()

    @property
    def is_empty(self):
        return not self.best

    def to_dict(self):
        return dict(
            frame_rate=self.frame_rate, num_frames=self.num_frames,
            hypotheses=[dict(phonemes=list(h.phonemes), log_prob=round(h.log_prob, 4), frames=h.frames)
                        for h in self.hypotheses],
            slots=[dict(frames=s.frames, alternatives=[[p, round(lp, 4)] for p, lp in s.alternatives])
                   for s in self.slots])


def prefix_beam_search(log_probs, beam_width=16, token_limit=8, blank_skip=0.9995, min_token_prob=1e-4):
    """log_probs: (T, V) numpy, blank at 0. Returns [(ids tuple, log_prob)] best first.

    Per frame only the `token_limit` most probable symbols above `min_token_prob` are expanded.
    A frame whose blank probability exceeds `blank_skip` only extends every beam with blank.
    """
    beams = {(): (0.0, NEG_INF)}  # prefix -> (log P ending in blank, log P ending in non-blank)
    order = np.argsort(-log_probs, axis=1)[:, :token_limit].tolist()
    rows = log_probs.tolist()
    log_skip = math.log(blank_skip)
    log_min = math.log(min_token_prob) if min_token_prob > 0 else NEG_INF
    for row, ranked in zip(rows, order):
        blank_lp = row[BLANK_ID]
        if blank_lp > log_skip:
            beams = {p: (logaddexp(pb, pnb) + blank_lp, NEG_INF) for p, (pb, pnb) in beams.items()}
            continue
        tokens = [c for c in ranked if c != BLANK_ID and row[c] > log_min]
        grown = {}
        for prefix, (pb, pnb) in beams.items():
            total = logaddexp(pb, pnb)
            entry = grown.get(prefix)
            if entry is None:
                entry = grown[prefix] = [NEG_INF, NEG_INF]
            entry[0] = logaddexp(entry[0], total + blank_lp)
            last = prefix[-1] if prefix else -1
            for c in tokens:
                lp = row[c]
                if c == last:
                    entry[1] = logaddexp(entry[1], pnb + lp)  # repeat without a blank: same prefix
                    if pb == NEG_INF:
                        continue
                    score = pb + lp                            # repeat after a blank: new symbol
                else:
                    score = total + lp
                longer = prefix + (c,)
                other = grown.get(longer)
                if other is None:
                    grown[longer] = [NEG_INF, score]
                else:
                    other[1] = logaddexp(other[1], score)
        ranked_beams = sorted(grown.items(), key=lambda kv: -logaddexp(kv[1][0], kv[1][1]))[:beam_width]
        beams = {p: (v[0], v[1]) for p, v in ranked_beams}
    return sorted(((p, logaddexp(pb, pnb)) for p, (pb, pnb) in beams.items()), key=lambda kv: -kv[1])


def forced_alignment(log_probs, ids):
    """Best single CTC alignment of `ids`. Returns [[start, end]] (inclusive frames) per symbol."""
    if not ids:
        return []
    t_max, n = len(log_probs), len(ids)
    states = np.full(2 * n + 1, BLANK_ID)
    states[1::2] = ids
    emit = log_probs[:, states]                                  # (T, S)
    skip = np.zeros(2 * n + 1, bool)                             # may jump over the blank before it
    skip[3::2] = states[3::2] != states[1:-2:2]
    score = np.full(2 * n + 1, NEG_INF)
    score[0], score[1] = emit[0, 0], emit[0, 1]
    back = np.zeros((t_max, 2 * n + 1), np.int8)
    columns = np.arange(2 * n + 1)
    for t in range(1, t_max):
        stay = score
        step = np.concatenate(([NEG_INF], score[:-1]))
        jump = np.where(skip, np.concatenate(([NEG_INF, NEG_INF], score[:-2])), NEG_INF)
        options = np.stack((stay, step, jump))
        choice = options.argmax(0)
        back[t] = choice
        score = options[choice, columns] + emit[t]
    s = 2 * n if score[2 * n] >= score[2 * n - 1] else 2 * n - 1
    if score[s] == NEG_INF:
        raise ValueError("Sequence cannot be aligned to this many frames")
    spans = [[None, None] for _ in range(n)]
    for t in range(t_max - 1, -1, -1):
        if s % 2:
            spans[s // 2][0] = t
            if spans[s // 2][1] is None:
                spans[s // 2][1] = t
        s -= int(back[t, s])
    return spans


class PhonemeLatticeDecoder:
    """Turns frame-level phoneme log-probabilities into a top-K hypothesis lattice.

    Each slot keeps every member of the best phoneme's viseme class (e.g. P, B, M), with the
    model's own posterior for each, so downstream text reconstruction sees the homophene set
    even when the encoder happens to prefer one member.
    """

    def __init__(self, beam_width=16, top_k=5, max_alternatives=4, min_alternative_prob=0.05,
                 token_limit=8, frame_rate=RATE_HZ):
        self.beam_width, self.top_k = beam_width, top_k
        self.max_alternatives, self.min_alternative_prob = max_alternatives, min_alternative_prob
        self.token_limit, self.frame_rate = token_limit, frame_rate

    def decode(self, log_probs, length=None) -> PhonemeLattice:
        """log_probs: (T, V) array or tensor of log-softmax scores for one utterance."""
        if hasattr(log_probs, "detach"):
            log_probs = log_probs.detach().float().cpu().numpy()
        log_probs = np.asarray(log_probs, np.float64)
        if log_probs.ndim != 2 or log_probs.shape[1] != len(SYMBOLS):
            raise ValueError(f"log_probs must be (T, {len(SYMBOLS)}), got {log_probs.shape}")
        if length is not None:
            log_probs = log_probs[:int(length)]
        beams = prefix_beam_search(log_probs, self.beam_width, self.token_limit)[:self.top_k]
        hypotheses = [Hypothesis(tuple(SYMBOLS[i] for i in ids), float(score),
                                 forced_alignment(log_probs, list(ids))) for ids, score in beams]
        lattice = PhonemeLattice(hypotheses, [], len(log_probs), self.frame_rate)
        if hypotheses and hypotheses[0].phonemes:
            lattice.slots = self._slots(log_probs, hypotheses[0])
        return lattice

    def _slots(self, log_probs, best):
        slots = []
        for phoneme, (start, end) in zip(best.phonemes, best.frames):
            posterior = np.exp(log_probs[start:end + 1, 1:]).mean(0)  # blank excluded
            posterior = posterior / posterior.sum()
            ranked = [SYMBOLS[i + 1] for i in np.argsort(-posterior)]
            keep = [phoneme]
            for candidate in ranked:
                likely = posterior[ID[candidate] - 1] >= self.min_alternative_prob
                if candidate not in keep and likely and len(keep) < self.max_alternatives:
                    keep.append(candidate)
            keep += [p for p in homophenes(phoneme) if p not in keep]
            keep.sort(key=lambda p: -posterior[ID[p] - 1])
            slots.append(Slot([start, end],
                              [(p, float(np.log(max(posterior[ID[p] - 1], 1e-12)))) for p in keep]))
        return slots
