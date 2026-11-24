"""SentencePiece CTC beam search with a 4-gram KenLM model."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from silent_avsr.features import ROOT

# Compatibility for callers that still decode saved phoneme lattices.
from .phoneme_search import VISEME_OF, common_words, covers, screen_words, search, variants

__all__ = ["CTCWordSearch", "Hypothesis", "VISEME_OF", "common_words", "covers",
           "screen_words", "search", "variants"]
DEFAULT_LM = ROOT / "models/english-4gram.bin"
DEFAULT_UNIGRAMS = ROOT / "models/english-4gram-unigrams.txt"


@dataclass(frozen=True)
class Hypothesis:
    text: str
    ctc_log_prob: float
    fused_log_score: float


class CTCWordSearch:
    def __init__(self, labels, lm_path=DEFAULT_LM, unigram_path=None, alpha=0.1, beta=0.0):
        from kenlm import Model
        from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
        from pyctcdecode.language_model import load_unigram_set_from_arpa

        lm_path = Path(lm_path)
        if not lm_path.is_file():
            raise FileNotFoundError(lm_path)
        self.vocab_size = len(labels)
        kenlm = Model(str(lm_path))
        if kenlm.order != 4:
            raise ValueError(f"Expected a 4-gram KenLM model, got {kenlm.order}-gram")
        if unigram_path is None and lm_path == DEFAULT_LM:
            unigram_path = DEFAULT_UNIGRAMS
        if unigram_path is not None:
            unigrams = Path(unigram_path).read_text().splitlines()
        elif lm_path.suffix == ".arpa":
            unigrams = list(load_unigram_set_from_arpa(str(lm_path)))
        else:
            unigrams = None
        if labels[0] != "<blank>" or labels[-1] != "<eos>":
            raise ValueError("Expected CTC blank first and attention EOS last")
        alphabet = Alphabet.build_alphabet(["", *labels[1:-1]])
        self.decoder = BeamSearchDecoderCTC(alphabet, LanguageModel(kenlm, unigrams, alpha=alpha, beta=beta))

    def set_weights(self, alpha, beta):
        self.decoder.reset_params(alpha=float(alpha), beta=float(beta))

    def greedy(self, log_probs):
        ids = np.argmax(log_probs[:, :-1], axis=1)
        previous = None
        pieces = []
        for token in ids:
            if token != previous and token != 0:
                pieces.append(self.decoder._alphabet.labels[token])
            previous = token
        return "".join(pieces).replace("▁", " ").strip()

    def decode(self, log_probs):
        if log_probs.ndim != 2 or log_probs.shape[1] != self.vocab_size or not len(log_probs):
            raise ValueError("Expected a nonempty (frames, vocabulary) CTC matrix")
        # Keep the reserved attention token out of the search alphabet. Folding its
        # column into blank preserves probability mass for externally supplied scores.
        scores = log_probs[:, :-1].copy()
        scores[:, 0] = np.logaddexp(scores[:, 0], log_probs[:, -1])
        beams = self.decoder.decode_beams(scores, beam_width=100, prune_history=False)
        if len(beams) < 8:
            # Strongly peaked clips may have too few candidates after default pruning.
            beams = self.decoder.decode_beams(scores, beam_width=100, beam_prune_logp=-20.0,
                                              token_min_logp=-10.0, prune_history=False)
        hypotheses = []
        for beam in beams[:8]:
            if isinstance(beam, tuple):  # pyctcdecode 0.5
                text, _, _, ctc, fused = beam
            else:  # pyctcdecode 0.6
                text, ctc, fused = beam.text, beam.logit_score, beam.lm_score
            hypotheses.append(Hypothesis(text, float(ctc), float(fused)))
        return hypotheses
