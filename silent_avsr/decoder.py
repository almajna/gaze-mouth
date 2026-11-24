"""OpenVocabLipDecoder: encoder -> phoneme lattice -> screen-grounded text."""
from pathlib import Path
import json
import time

import numpy as np
import torch

from .encoder import DualStreamPhonemeEncoder
from .features import ROI_SIDE, ROOT
from .lattice import PhonemeLatticeDecoder
from .lora_adapter import LoRAAdapter

DEFAULT_QWEN = ROOT / "models" / "Qwen2.5-3B-Instruct-4bit"
DEFAULT_FRONTEND = ROOT / "models" / "auto_avsr" / "vsr.pth"


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class OpenVocabLipDecoder:
    """decode(landmarks_30hz, mouth_roi_30hz, screen_context) -> dict(text, phoneme_lattice, latency_ms)

    checkpoint_dir holds:
      encoder_config.json, encoder.safetensors   trained phoneme encoder
      speaker_lora/                              optional speaker adapter (silent_avsr.LoRAAdapter)
      reconstructor_lora/                        optional mlx-lm adapter for Qwen
      decoder_config.json                        optional: qwen_path, beam_width, top_k

    Without encoder weights the constructor raises, because an untrained encoder emits noise.
    `allow_untrained=True` builds one anyway (Auto-AVSR front-end weights if available) for
    plumbing and latency work; `self.trained` is then False.
    """

    def __init__(self, checkpoint_dir, device=None, reconstructor=None, allow_untrained=False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or default_device()
        config_path = self.checkpoint_dir / "decoder_config.json"
        self.config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        self.trained = (self.checkpoint_dir / "encoder.safetensors").is_file()
        if self.trained:
            self.encoder = DualStreamPhonemeEncoder.load(self.checkpoint_dir, self.device)
        elif allow_untrained:
            self.encoder = DualStreamPhonemeEncoder()
            if DEFAULT_FRONTEND.is_file():
                self.encoder.load_auto_avsr_frontend(DEFAULT_FRONTEND)
            self.encoder.to(self.device).eval()
        else:
            raise FileNotFoundError(
                f"{self.checkpoint_dir / 'encoder.safetensors'} not found. Train one with "
                "`python -m silent_avsr.train_ctc --mode base`, or pass allow_untrained=True.")
        self.speaker_adapter = None
        if (self.checkpoint_dir / "speaker_lora" / "adapter.safetensors").is_file():
            self.speaker_adapter = LoRAAdapter.load(self.encoder, self.checkpoint_dir / "speaker_lora")
            self.speaker_adapter.merge()
            self.encoder.to(self.device).eval()
        self.lattice_decoder = PhonemeLatticeDecoder(beam_width=self.config.get("beam_width", 16),
                                                     top_k=self.config.get("top_k", 5))
        self.reconstructor = reconstructor
        self.last_timing = {}

    def _load_reconstructor(self):
        from context_decoder import ScreenGroundedReconstructor
        adapter = self.checkpoint_dir / "reconstructor_lora"
        return ScreenGroundedReconstructor(
            model_path=self.config.get("qwen_path", str(DEFAULT_QWEN)),
            adapter_path=str(adapter) if (adapter / "adapters.safetensors").is_file() else None)

    @staticmethod
    def _landmarks(x):
        if x is None:
            return None
        x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
        x = x[None] if x.dim() == 2 else x
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"landmarks_30hz must be (T, D_geom) or (1, T, D_geom), got {tuple(x.shape)}")
        return x.float()

    @staticmethod
    def _roi(x):
        if x is None:
            return None
        x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
        if x.dim() == 3:
            x = x[:, None]
        if x.dim() == 4:
            x = x[None]
        if x.dim() != 5 or x.shape[0] != 1 or x.shape[2:] != (1, ROI_SIDE, ROI_SIDE):
            raise ValueError(f"mouth_roi_30hz must be (T, 1, {ROI_SIDE}, {ROI_SIDE}) with an optional "
                             f"batch of 1, got {tuple(x.shape)}")
        if x.dtype != torch.uint8 and x.numel() and float(x.max()) > 1.0 + 1e-3:
            raise ValueError("Float mouth_roi_30hz must be scaled to [0, 1]; pass uint8 for 0-255")
        return x

    @torch.inference_mode()
    def lattice(self, landmarks_30hz, mouth_roi_30hz):
        """Encoder + beam search only. Returns (PhonemeLattice, encoder_ms, lattice_ms)."""
        landmarks, roi = self._landmarks(landmarks_30hz), self._roi(mouth_roi_30hz)
        start = time.perf_counter()
        out = self.encoder(None if landmarks is None else landmarks.to(self.device),
                           None if roi is None else roi.to(self.device))
        log_probs = out.log_probs[0].float().cpu().numpy()  # the copy waits for the device
        encoded = time.perf_counter()
        lattice = self.lattice_decoder.decode(log_probs)
        return lattice, (encoded - start) * 1000, (time.perf_counter() - encoded) * 1000

    def decode(self, landmarks_30hz, mouth_roi_30hz, screen_context):
        lattice, encoder_ms, lattice_ms = self.lattice(landmarks_30hz, mouth_roi_30hz)
        lattice_dict = lattice.to_dict()
        text, reconstruct_ms = "", 0.0
        if not lattice.is_empty:  # nothing mouthed: never ask the language model to invent a phrase
            if self.reconstructor is None:
                self.reconstructor = self._load_reconstructor()
            start = time.perf_counter()
            text = self.reconstructor.reconstruct(lattice_dict, screen_context or {})
            reconstruct_ms = (time.perf_counter() - start) * 1000
        self.last_timing = dict(encoder_ms=encoder_ms, lattice_ms=lattice_ms, reconstruct_ms=reconstruct_ms)
        return dict(text=text, phoneme_lattice=lattice_dict,
                    latency_ms=round(encoder_ms + lattice_ms + reconstruct_ms, 2))
