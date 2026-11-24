"""ScreenGroundedReconstructor: phoneme lattice + screen context -> the intended phrase."""
from pathlib import Path
import time

from .prompt import messages, normalize_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-3B-Instruct-4bit"


class ScreenGroundedReconstructor:
    """Runs a local Qwen2.5-3B-Instruct (MLX) with an optional LoRA adapter, greedily.

    lattice         a silent_avsr.PhonemeLattice or its `to_dict()` form
    screen_context  dict(app_name, ax_role, ax_text, nearby_labels); missing keys are allowed

    Only the MLX backend exists: the weights in models/ are MLX 4-bit and cannot be loaded by
    PyTorch. The static system prompt is evaluated once and its key/value cache is reused.
    """

    def __init__(self, model_path=DEFAULT_MODEL, adapter_path=None, max_tokens=48, cache_system_prompt=True):
        from mlx_lm import load
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Qwen model directory not found: {model_path}")
        self.model, self.tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
        self.model.eval()
        self.model_path, self.adapter_path = str(model_path), adapter_path
        self.max_tokens = max_tokens
        self.last = {}
        self._prefix, self._cache = [], None
        if cache_system_prompt:
            self._build_prefix_cache()

    # ---- prompt ------------------------------------------------------------------------------
    def prompt_tokens(self, lattice, screen_context):
        lattice = lattice.to_dict() if hasattr(lattice, "to_dict") else lattice
        return list(self.tokenizer.apply_chat_template(messages(lattice, screen_context),
                                                       add_generation_prompt=True))

    def _build_prefix_cache(self):
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache
        import mlx.core as mx
        empty = dict(frame_rate=30.0, num_frames=0, hypotheses=[], slots=[])
        a = self.prompt_tokens(empty, dict(app_name="A"))
        b = self.prompt_tokens(empty, dict(app_name="B"))
        shared = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        cache = make_prompt_cache(self.model)
        for _ in generate_step(mx.array(a[:shared]), self.model, max_tokens=0, prompt_cache=cache):
            pass
        if can_trim_prompt_cache(cache) and cache[0].offset == shared:
            self._prefix, self._cache = a[:shared], cache

    # ---- generation --------------------------------------------------------------------------
    def reconstruct(self, lattice, screen_context=None) -> str:
        from mlx_lm.generate import stream_generate, trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        start = time.perf_counter()
        tokens = self.prompt_tokens(lattice, screen_context)
        cached = self._cache is not None and tokens[:len(self._prefix)] == self._prefix
        kwargs = dict(prompt_cache=self._cache) if cached else {}
        pieces = []
        try:
            for response in stream_generate(self.model, self.tokenizer, tokens[len(self._prefix):] if cached else tokens,
                                            max_tokens=self.max_tokens, sampler=make_sampler(temp=0.0), **kwargs):
                pieces.append(response.text)
                if "\n" in response.text:
                    break
        finally:
            if cached:  # drop this request's tokens so the cache holds only the system prefix again
                trim_prompt_cache(self._cache, self._cache[0].offset - len(self._prefix))
        raw = "".join(pieces)
        text = normalize_text(raw.strip().split("\n")[0])
        self.last = dict(raw=raw, prompt_tokens=len(tokens), cached_prefix=len(self._prefix) if cached else 0,
                         latency_ms=(time.perf_counter() - start) * 1000)
        return text
