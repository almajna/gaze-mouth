"""DualStreamPhonemeEncoder: geometry stream + mouth-ROI stream -> gated fusion -> phoneme CTC."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, Optional
import json

import torch
from torch import nn
import torch.nn.functional as F

from .conformer import ConformerBlock
from .features import D_GEOM, ROI_SIDE
from .phonemes import VOCAB_SIZE

ROI_MEAN, ROI_STD = 0.421, 0.165  # Auto-AVSR grayscale statistics


@dataclass
class EncoderConfig:
    d_geom: int = D_GEOM
    d_model: int = 256
    heads: int = 4
    geom_layers: int = 2
    visual_layers: int = 4
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
    modality_dropout: float = 0.0  # training only: chance of hiding one stream when both are present


class EncoderOutput(NamedTuple):
    log_probs: torch.Tensor   # (B, T, vocab) log-softmax, blank = 0
    embeddings: torch.Tensor  # (B, T, d_model) fused articulatory embeddings
    gate: torch.Tensor        # (B, T, d_model) in [0, 1]; 1 = geometry stream, 0 = ROI stream
    lengths: torch.Tensor     # (B,)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn2(self.conv2(F.silu(self.bn1(self.conv1(x)))))
        return F.silu(out + (x if self.downsample is None else self.downsample(x)))


class ResNet18Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._layer(64, 1)
        self.layer2 = self._layer(128, 2)
        self.layer3 = self._layer(256, 2)
        self.layer4 = self._layer(512, 2)

    def _layer(self, planes, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes))
        blocks = [BasicBlock(self.inplanes, planes, stride, downsample), BasicBlock(planes, planes)]
        self.inplanes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x)))).mean((2, 3))


class VisualFrontend(nn.Module):
    """3D-conv stem + 2D ResNet-18 per frame. Parameter names and shapes match the `frontend.*`
    entries of an Auto-AVSR video checkpoint, so those weights load with strict=True."""

    def __init__(self):
        super().__init__()
        self.trunk = ResNet18Trunk()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False), nn.BatchNorm3d(64),
            nn.SiLU(), nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))

    def forward(self, x):  # (B, T, 1, H, W) -> (B, T, 512)
        b, t = x.shape[:2]
        x = self.frontend3D(x.transpose(1, 2))
        x = x.transpose(1, 2).reshape(b * t, *x.shape[1:2], *x.shape[3:])
        return self.trunk(x).view(b, t, -1)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.absent_geom = nn.Parameter(torch.zeros(dim))
        self.absent_visual = nn.Parameter(torch.zeros(dim))

    def forward(self, geom, visual):
        if geom is None:
            geom = self.absent_geom.expand_as(visual)
        if visual is None:
            visual = self.absent_visual.expand_as(geom)
        gate = torch.sigmoid(self.gate(torch.cat((geom, visual), -1)))
        return self.norm(gate * geom + (1 - gate) * visual), gate


class GeometryStem(nn.Module):
    def __init__(self, d_in, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(d_in, dim, 5, padding=2)
        self.conv2 = nn.Conv1d(dim, dim, 5, padding=2)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x, pad_mask=None):
        for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2)):
            if pad_mask is not None:
                x = x.masked_fill(pad_mask[:, :, None], 0.0)
            x = F.silu(norm(conv(x.transpose(1, 2)).transpose(1, 2)))
        return x


class DualStreamPhonemeEncoder(nn.Module):
    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()
        self.config = c = config or EncoderConfig()
        block = lambda: ConformerBlock(c.d_model, c.heads, c.ff_mult, c.conv_kernel, c.dropout)
        # Stream A: pose-normalized geometry.
        self.geom_stem = GeometryStem(c.d_geom, c.d_model)
        self.geom_blocks = nn.ModuleList(block() for _ in range(c.geom_layers))
        # Stream B: mouth ROI.
        self.frontend = VisualFrontend()
        self.visual_proj = nn.Linear(512, c.d_model)
        self.visual_blocks = nn.ModuleList(block() for _ in range(c.visual_layers))
        self.fusion = GatedFusion(c.d_model)
        self.ctc_head = nn.Linear(c.d_model, c.vocab_size)
        self.frontend_frozen = False

    # ---- Auto-AVSR front-end -------------------------------------------------------------
    def load_auto_avsr_frontend(self, checkpoint):
        """Load the `frontend.*` weights of an Auto-AVSR video checkpoint (e.g. models/auto_avsr/vsr.pth)."""
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prefix = "frontend."
        weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.frontend.load_state_dict(weights, strict=True)
        return self

    def freeze_frontend(self, frozen=True):
        """A frozen front-end stays in eval mode so its BatchNorm statistics are not updated."""
        self.frontend_frozen = frozen
        self.frontend.requires_grad_(not frozen)
        if frozen:
            self.frontend.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.frontend_frozen:
            self.frontend.eval()
        return self

    # ---- forward -------------------------------------------------------------------------
    @staticmethod
    def prepare_roi(roi):
        """uint8 0-255 or float 0-1, (B, T, 1, 88, 88) -> normalized float."""
        if roi.dim() != 5 or roi.shape[2] != 1 or roi.shape[-2:] != (ROI_SIDE, ROI_SIDE):
            raise ValueError(f"mouth_roi_30hz must be (B, T, 1, {ROI_SIDE}, {ROI_SIDE}), got {tuple(roi.shape)}")
        roi = roi.float() / 255 if roi.dtype == torch.uint8 else roi.float()
        return (roi - ROI_MEAN) / ROI_STD

    def forward(self, landmarks_30hz=None, mouth_roi_30hz=None, lengths=None) -> EncoderOutput:
        if landmarks_30hz is None and mouth_roi_30hz is None:
            raise ValueError("At least one stream is required")
        reference = landmarks_30hz if landmarks_30hz is not None else mouth_roi_30hz
        b, t = reference.shape[:2]
        if landmarks_30hz is not None:
            if landmarks_30hz.dim() != 3 or landmarks_30hz.shape[-1] != self.config.d_geom:
                raise ValueError(f"landmarks_30hz must be (B, T, {self.config.d_geom}), "
                                 f"got {tuple(landmarks_30hz.shape)}")
            if mouth_roi_30hz is not None and mouth_roi_30hz.shape[:2] != (b, t):
                raise ValueError("The two streams must have the same batch size and frame count")
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=reference.device)
            pad_mask = None
        else:
            lengths = lengths.to(reference.device)
            pad_mask = torch.arange(t, device=reference.device)[None, :] >= lengths[:, None]

        use_geom, use_visual = landmarks_30hz is not None, mouth_roi_30hz is not None
        if self.training and use_geom and use_visual and self.config.modality_dropout > 0:
            draw = torch.rand(()).item()
            use_geom = draw >= self.config.modality_dropout
            use_visual = draw < 1 - self.config.modality_dropout

        geom = visual = None
        if use_geom:
            geom = self.geom_stem(landmarks_30hz.float(), pad_mask)
            for layer in self.geom_blocks:
                geom = layer(geom, pad_mask)
        if use_visual:
            roi = self.prepare_roi(mouth_roi_30hz)
            if pad_mask is not None:  # zero padded frames so they act like the conv's own padding
                roi = roi.masked_fill(pad_mask[:, :, None, None, None], 0.0)
            visual = self.visual_proj(self.frontend(roi))
            for layer in self.visual_blocks:
                visual = layer(visual, pad_mask)

        fused, gate = self.fusion(geom, visual)
        return EncoderOutput(F.log_softmax(self.ctc_head(fused), -1), fused, gate, lengths)

    # ---- persistence ---------------------------------------------------------------------
    def save(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "encoder_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(directory / "encoder.safetensors"))

    @classmethod
    def load(cls, directory, device="cpu"):
        from safetensors.torch import load_file
        directory = Path(directory)
        model = cls(EncoderConfig(**json.loads((directory / "encoder_config.json").read_text())))
        model.load_state_dict(load_file(str(directory / "encoder.safetensors")), strict=True)
        return model.to(device).eval()
