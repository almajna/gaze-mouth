"""Full pretrained visual speech recognition for 25 Hz sentence clips."""
from pathlib import Path
import sys

import torch

from .features import ROOT
from .sentence_video import AUTO_AVSR, normalize_contrast

USR2 = ROOT / "models/usr2"
USR2_WEIGHTS = USR2 / "baseplus.pth"
AUTO_WEIGHTS = AUTO_AVSR / "vsr.pth"


class PretrainedVSR:
    def __init__(self, backend="usr2", device=None, checkpoint=None, adapter_path=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.backend = backend
        if backend == "usr2":
            repo, default_weights = USR2, USR2_WEIGHTS
        elif backend == "auto_avsr":
            repo, default_weights = AUTO_AVSR, AUTO_WEIGHTS
        else:
            raise ValueError(f"Unknown VSR backend {backend!r}")
        weights = Path(checkpoint) if checkpoint else default_weights
        if not weights.is_file():
            raise FileNotFoundError(weights)

        # The upstream projects both use the same top-level ESPnet package name.
        loaded = sys.modules.get("espnet")
        if loaded is not None and str(repo / "espnet") not in list(loaded.__path__):
            raise RuntimeError("Run each VSR backend in its own Python process")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        if backend == "usr2":
            from types import SimpleNamespace

            from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

            units = repo / "utils/labels/unigram1000_units.txt"
            # Official resnet_transformer_baseplus config for the Base+ checkpoint.
            config = SimpleNamespace(idim=512, adim=768, aheads=12, eunits=3072, elayers=12,
                                     ddim=768, dheads=12, dunits=3072, dlayers=6, gamma_init=0.1)
        else:
            from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

            units, config = repo / "spm/unigram/unigram5000_units.txt", "video"
        self.labels = ["<blank>", *[line.split()[0] for line in units.read_text().splitlines()], "<eos>"]
        model = E2E(len(self.labels), config)
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if backend == "usr2":
            state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
            prefix = "model.backbone."
            if any(key.startswith(prefix) for key in state):
                state = {key.removeprefix(prefix): value for key, value in state.items()
                         if key.startswith(prefix)}
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.eval().to(self.device)
        if adapter_path is not None:
            from .lora_adapter import LoRAAdapter
            self.adapter = LoRAAdapter.load(self.model, adapter_path)
            self.model.eval()

    @torch.inference_mode()
    def log_probs(self, crops):
        """Return (T, vocabulary) visual CTC natural log-probabilities."""
        x = torch.from_numpy(normalize_contrast(crops)).to(self.device)[None]
        model = self.model
        if self.backend == "usr2":
            logits = model.ctc_v.ctc_lo(model.encoder(xs_v=x))
        else:
            encoded, _ = model.encoder(model.proj_encoder(model.frontend(x[:, :, None])), None)
            logits = model.ctc.ctc_lo(encoded)
        # Sentence boundary is reserved for the attention decoder. CTC uses ID 0 for blank.
        logits[..., -1] = -torch.inf
        return logits[0].log_softmax(-1).cpu().numpy()


class SentenceDecoder:
    """Decode an aligned 25 Hz mouth clip into up to eight ranked sentences."""

    def __init__(self, backend="usr2", device=None, checkpoint=None, lm_path=None, unigram_path=None,
                 adapter_path=None, lm_alpha=0.1, lm_beta=0.0):
        from context_decoder.word_search import CTCWordSearch, DEFAULT_LM

        self.vsr = PretrainedVSR(backend, device, checkpoint, adapter_path)
        self.search = CTCWordSearch(self.vsr.labels, lm_path or DEFAULT_LM, unigram_path,
                                    alpha=lm_alpha, beta=lm_beta)

    def decode(self, crops):
        return self.search.decode(self.vsr.log_probs(crops))
