"""Dual-stream visual phoneme encoder, CTC phoneme lattice, and the open-vocabulary lip decoder.

Exports resolve lazily so `silent_avsr.phonemes` and `silent_avsr.lexicon` import without torch.
"""
_EXPORTS = {
    "EncoderConfig": "encoder", "DualStreamPhonemeEncoder": "encoder", "VisualFrontend": "encoder",
    "PhonemeLatticeDecoder": "lattice", "PhonemeLattice": "lattice",
    "LoRAAdapter": "lora_adapter", "LoRALinear": "lora_adapter",
    "OpenVocabLipDecoder": "decoder",
    "PretrainedVSR": "pretrained",
    "SentenceDecoder": "pretrained",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
