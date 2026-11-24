"""LoRA adapters for speaker-specific fine-tuning of the phoneme encoder."""
from pathlib import Path
import json
import math
import re

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank update: W + (alpha / rank) * B @ A."""

    def __init__(self, base: nn.Linear, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def delta(self):
        return self.scale * (self.lora_b @ self.lora_a)

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        return y + self.scale * F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta()
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta()
            self.merged = False


class LoRAAdapter:
    """Freezes a model, wraps its Linear layers with LoRA, and saves or loads only the adapter.

    `include` is a regex over module names. `full` names modules trained in full and stored with
    the adapter (by default the CTC head, which is small and speaker-sensitive). `buffers` names
    modules whose buffers (BatchNorm running statistics) are stored with the adapter.
    """

    def __init__(self, model, rank=8, alpha=16.0, dropout=0.0, include=r".*", full=("ctc_head",), buffers=()):
        self.model = model
        self.config = dict(rank=rank, alpha=alpha, dropout=dropout, include=include, full=list(full),
                           buffers=list(buffers))
        model.requires_grad_(False)
        self.layers = {}
        pattern = re.compile(include)
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and pattern.fullmatch(name) and name not in full:
                parent_name, _, child = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                if isinstance(parent, LoRALinear):
                    continue  # already wrapped
                wrapped = LoRALinear(module, rank, alpha, dropout)
                setattr(parent, child, wrapped)
                self.layers[name] = wrapped
        if not self.layers:
            raise ValueError(f"No Linear layer matches {include!r}")
        self.full = {name: model.get_submodule(name).requires_grad_(True) for name in full}
        self.buffers = {f"{name}.{key}": value for name in buffers
                        for key, value in model.get_submodule(name).named_buffers(recurse=False)}

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self):
        state = {}
        for name, layer in self.layers.items():
            state[f"{name}.lora_a"], state[f"{name}.lora_b"] = layer.lora_a, layer.lora_b
        for name, module in self.full.items():
            state.update({f"{name}.{k}": v for k, v in module.state_dict().items()})
        state.update(self.buffers)
        return {k: v.detach().cpu().contiguous() for k, v in state.items()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"Adapter keys differ: {sorted(set(state) ^ set(expected))[:6]}")
        with torch.no_grad():
            for name, layer in self.layers.items():
                layer.lora_a.copy_(state[f"{name}.lora_a"])
                layer.lora_b.copy_(state[f"{name}.lora_b"])
        for name, module in self.full.items():
            prefix = f"{name}."
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        with torch.no_grad():
            for name, buffer in self.buffers.items():
                buffer.copy_(state[name])

    def save(self, directory):
        from safetensors.torch import save_file
        if any(layer.merged for layer in self.layers.values()):
            raise RuntimeError("Unmerge before saving: merged weights live in the base layers")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text(json.dumps(self.config, indent=2))
        save_file(self.state_dict(), str(directory / "adapter.safetensors"))

    @classmethod
    def load(cls, model, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        config = json.loads((directory / "adapter_config.json").read_text())
        config["full"] = tuple(config["full"])
        config["buffers"] = tuple(config.get("buffers", ()))
        adapter = cls(model, **config)
        adapter.load_state_dict(load_file(str(directory / "adapter.safetensors")))
        return adapter

    def merge(self):
        for layer in self.layers.values():
            layer.merge()

    def unmerge(self):
        for layer in self.layers.values():
            layer.unmerge()
