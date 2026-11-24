"""Deterministic segmentation using the USR checkpoint's published token IDs."""
import re
from pathlib import Path

from .features import ROOT

DEFAULT_UNITS = ROOT / "models/usr2/utils/labels/unigram1000_units.txt"


class USRTokenizer:
    """Fewest-token Viterbi segmentation, preferring longer pieces on ties.

    The units file has no unigram scores, so this cannot reproduce the original
    SentencePiece model's probability-weighted segmentation.
    """

    def __init__(self, path=DEFAULT_UNITS):
        self.pieces = {}
        self.by_initial = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            piece, number = line.split()
            token_id = int(number)
            if piece in self.pieces or token_id in self.pieces.values():
                raise ValueError("Duplicate token or ID in USR units")
            self.pieces[piece] = token_id
            if not piece.startswith("<"):
                self.by_initial.setdefault(piece[0], []).append(piece)
        for candidates in self.by_initial.values():
            candidates.sort(key=lambda piece: (-len(piece), piece))

    def encode(self, text):
        words = re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)*", text.upper().replace("’", "'"))
        if not words:
            raise ValueError("Transcript contains no supported words")
        sequence = "▁" + "▁".join(words)
        best = [None] * (len(sequence) + 1)
        best[-1] = ()
        for start in range(len(sequence) - 1, -1, -1):
            for piece in self.by_initial.get(sequence[start], ()):
                end = start + len(piece)
                if sequence.startswith(piece, start) and best[end] is not None:
                    candidate = (self.pieces[piece],) + best[end]
                    if best[start] is None or len(candidate) < len(best[start]):
                        best[start] = candidate
        if best[0] is None:
            raise ValueError(f"Transcript cannot be represented by the USR vocabulary: {text!r}")
        return list(best[0])
