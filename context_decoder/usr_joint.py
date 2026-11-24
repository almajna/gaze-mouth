"""USR 2.0's attention and CTC joint beam search."""
from dataclasses import dataclass
from time import perf_counter

import torch

from silent_avsr.sentence_video import normalize_contrast


@dataclass(frozen=True)
class JointHypothesis:
    text: str
    log_score: float
    attention_log_score: float
    ctc_log_score: float


class _MemoryProjection:
    """Cross-attention key or value projection of the encoder output, computed once per utterance.

    Upstream recomputes it for every hypothesis at every step although it depends only on the clip.
    """

    def __init__(self, module):
        self.project, self.value, self.active = module.forward, None, False
        module.forward = self

    def __call__(self, memory):
        if not self.active:
            return self.project(memory)
        if self.value is None:
            self.value = self.project(memory[:1])
        return self.value.expand(memory.shape[0], -1, -1)


class USRJointDecoder:
    def __init__(self, vsr, beam_size=10, ctc_weight=0.1, nbest=8, exact_stop=True):
        if vsr.backend != "usr2":
            raise ValueError("Joint decoding requires the USR 2.0 backend")
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.scorers.length_bonus import LengthBonus

        self.vsr, self.nbest, self.exact_stop = vsr, nbest, exact_stop
        model = vsr.model
        model.ctc_v.to("cpu")
        model.decoder.to("cpu")
        scorers = model.scorers()
        scorers["length_bonus"] = LengthBonus(len(vsr.labels))
        self.beam = BatchBeamSearch(
            beam_size=beam_size, vocab_size=len(vsr.labels),
            weights={"decoder": 1.0 - ctc_weight, "ctc": ctc_weight, "length_bonus": 0.0},
            scorers=scorers, sos=model.sos, eos=model.eos,
            token_list=vsr.labels, pre_beam_score_key="decoder",
        ).to("cpu")
        self.memory = [_MemoryProjection(module) for layer in model.decoder.decoders
                       for module in (layer.src_attn.linear_k, layer.src_attn.linear_v)]
        self.last = {}

    def search(self, features):
        """Ended hypotheses, best first. With exact_stop the first `nbest` equal the full search's.

        Every token adds a log-probability (never positive) to both scores and there is no length
        bonus, so a running hypothesis can only lose score. Once `nbest` hypotheses have ended and
        the best running one is below the worst of them, no later hypothesis can enter the list.
        """
        if not self.exact_stop:
            return self.beam(x=features, modality="v", maxlenratio=1.0, minlenratio=0.0)
        beam, ended = self.beam, []
        running = beam.init_hyp(features, "v")
        steps = features.shape[0]
        for step in range(steps):
            running = beam.post_process(step, steps, 1.0, beam.search(running, features, "v"), ended)
            if len(running) == 0:
                break
            if len(ended) >= self.nbest:
                worst_kept = sorted((float(h.score) for h in ended), reverse=True)[self.nbest - 1]
                if float(running.score.max()) < worst_kept:
                    break
        self.last["steps"] = step + 1
        return sorted(ended, key=lambda h: h.score, reverse=True)

    @torch.inference_mode()
    def decode(self, crops):
        started = perf_counter()
        x = torch.from_numpy(normalize_contrast(crops))[None].to(self.vsr.device)
        # Upstream's CTC prefix scorer creates CPU index tensors, so beam search
        # runs on CPU while the visual encoder can still run on MPS.
        features = self.vsr.model.encoder(xs_v=x)[0].cpu()
        encoded = perf_counter()
        # Evidence from the CTC head that someone spoke: the attention decoder writes a sentence for any clip.
        frame_log_probs = self.vsr.model.ctc_v.log_softmax(features[None])[0]
        greedy = torch.unique_consecutive(frame_log_probs.argmax(-1))
        self.last = dict(ctc_greedy_tokens=int(((greedy != 0) & (greedy != self.vsr.model.eos)).sum()),
                         ctc_nonblank=float(1 - frame_log_probs[:, 0].exp().mean()))
        for cache in self.memory:
            cache.value, cache.active = None, True
        try:
            beams = self.search(features)
        finally:
            for cache in self.memory:
                cache.value, cache.active = None, False
        output = []
        for hypothesis in beams[:self.nbest]:
            ids = hypothesis.yseq.tolist()
            text = "".join(self.vsr.labels[i] for i in ids[1:]
                           if i not in (0, self.vsr.model.eos)).replace("▁", " ").strip()
            output.append(JointHypothesis(text, float(hypothesis.score),
                                          float(hypothesis.scores["decoder"]),
                                          float(hypothesis.scores["ctc"])))
        self.last.update(encoder_ms=(encoded - started) * 1e3, beam_ms=(perf_counter() - encoded) * 1e3,
                         frames=len(crops))
        return output
