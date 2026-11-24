"""Generative error correction of lip-reading n-best lists with the local MLX Qwen.

One prompt format for the LoRA training rows and for inference.
"""
from pathlib import Path
import re
import time

from silent_avsr.features import ROOT

DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-3B-Instruct-4bit"
DEFAULT_ADAPTER = ROOT / "models" / "ger" / "lora"
SYSTEM = (
    "Correct silent lip reading. You get what is on screen and up to eight guesses from a lip-reading "
    "model, best first, each with its log score (closer to zero is more likely). The guesses are noisy: "
    "words that look alike on the lips get swapped, and words can be missing or invented. Write the "
    "sentence the user most likely mouthed, as plain lowercase words separated by spaces.")
MAX_GUESSES = 8
MAX_TEXT_CHARS = 240
MAX_LABELS = 12


def words(text):
    """The scorer's word set (eval_silent_speech.words) without importing its audio stack."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower().replace("’", "'"))


def context_lines(screen_context):
    """Four lines from dict(app_name, ax_role, ax_text, nearby_labels); missing fields read (none)."""
    context = dict(screen_context or {})
    labels = context.get("nearby_labels") or []
    if isinstance(labels, str):
        labels = [labels]
    labels = [" ".join(str(label).split()) for label in labels if str(label).strip()][:MAX_LABELS]
    text = " ".join(str(context.get("ax_text") or "").split())[-MAX_TEXT_CHARS:]  # the end is the most recent part
    return [f"App: {str(context.get('app_name') or '').strip() or '(none)'}",
            f"Element: {str(context.get('ax_role') or '').strip() or '(none)'}",
            f"Text: {text or '(none)'}",
            f"Nearby: {' | '.join(labels) or '(none)'}"]


def guess_lines(hypotheses, limit=MAX_GUESSES):
    """`<joint log-score> <lowercase words>` per unique non-empty hypothesis, in the given order."""
    lines, seen = [], set()
    for hypothesis in hypotheses:
        get = hypothesis.get if isinstance(hypothesis, dict) else lambda key: getattr(hypothesis, key)
        text = " ".join(words(get("text")))
        if text and text not in seen:
            seen.add(text)
            lines.append(f"{get('log_score'):.1f} {text}")
    return lines[:limit]


def ger_messages(hypotheses, screen_context=None, text=None):
    """Chat messages; with `text` the assistant turn is appended (a training row)."""
    user = "\n".join(context_lines(screen_context) + ["Guesses:"] + guess_lines(hypotheses))
    chat = [dict(role="system", content=SYSTEM), dict(role="user", content=user)]
    if text is not None:
        chat.append(dict(role="assistant", content=" ".join(words(text))))
    return chat


def clean(raw):
    """First line of the model output as scorer words; any <think> block is dropped."""
    raw = re.sub(r"<think>.*?(?:</think>|\Z)", "", raw, flags=re.S)
    line = next((line for line in raw.splitlines() if line.strip()), "")
    return " ".join(words(line))


class GERCorrector:
    """Qwen with an optional GER LoRA. The prompt cache keeps the longest prefix shared with the last request,
    so the system message and an unchanged screen context are not evaluated again."""

    def __init__(self, model_path=DEFAULT_MODEL, adapter_path=None, max_tokens=32):
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Qwen model directory not found: {model_path}")
        self.model, self.tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
        self.model.eval()
        self.max_tokens = max_tokens
        self._cache, self._held = make_prompt_cache(self.model), []
        self.last = {}

    def tokens(self, hypotheses, screen_context=None):
        return list(self.tokenizer.apply_chat_template(ger_messages(hypotheses, screen_context),
                                                       add_generation_prompt=True))

    def _advance(self, tokens, keep_last):
        """Trim the cache to its common prefix with `tokens`; return the tokens still to evaluate."""
        from mlx_lm.generate import trim_prompt_cache
        shared = next((i for i, (a, b) in enumerate(zip(self._held, tokens)) if a != b),
                      min(len(self._held), len(tokens)))
        shared = min(shared, len(tokens) - keep_last)
        if shared < len(self._held):
            trim_prompt_cache(self._cache, self._cache[0].offset - shared)
        self._held = tokens[:shared]
        return tokens[shared:]

    def reset(self):
        """Empty the prompt cache (needed after the adapter weights change)."""
        from mlx_lm.generate import trim_prompt_cache
        if self._held:
            trim_prompt_cache(self._cache, self._cache[0].offset)
        self._held = []

    def prefill(self, screen_context=None):
        """Evaluate system message and screen context ahead of the utterance. Returns the tokens now cached."""
        from mlx_lm.generate import generate_step
        import mlx.core as mx
        a = self.tokens([dict(text="a", log_score=-1.0)], screen_context)
        b = self.tokens([dict(text="b", log_score=-2.0)], screen_context)
        stable = a[:next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)]
        rest = self._advance(stable, keep_last=0)
        if rest:
            for _ in generate_step(mx.array(rest), self.model, max_tokens=0, prompt_cache=self._cache):
                pass
            self._held = stable
        return len(self._held)

    def correct(self, hypotheses, screen_context=None):
        """The corrected sentence as lowercase words; empty when there is no guess to correct."""
        from mlx_lm.generate import stream_generate, trim_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        start = time.perf_counter()
        if not guess_lines(hypotheses):
            self.last = dict(raw="", prompt_tokens=0, cached_tokens=0, generated_tokens=0, latency_ms=0.0)
            return ""
        tokens = self.tokens(hypotheses, screen_context)
        rest = self._advance(tokens, keep_last=1)
        cached, pieces, generated = len(self._held), [], 0
        try:
            for response in stream_generate(self.model, self.tokenizer, rest, max_tokens=self.max_tokens,
                                            sampler=make_sampler(temp=0.0), prompt_cache=self._cache):
                pieces.append(response.text)
                generated += 1
                if "\n" in response.text:
                    break
        finally:  # keep the prompt, drop what was generated
            trim_prompt_cache(self._cache, self._cache[0].offset - len(tokens))
            self._held = tokens
        raw = "".join(pieces)
        self.last = dict(raw=raw, prompt_tokens=len(tokens), cached_tokens=cached, generated_tokens=generated,
                         latency_ms=(time.perf_counter() - start) * 1000)
        return clean(raw)


# ---- n-best selection with Qwen as a language model --------------------------------------------
SELECT_SYSTEM = "You write down what the user dictates. Reply with the dictated sentence only."
LM_WEIGHT = 0.2  # tuned on out-of-fold beams of the training clips (docs/plan.md, Stage 2)


def as_sentence(text):
    """Hypothesis words written the way a language model expects a sentence: capital, 'I', full stop."""
    tokens = ["I" + word[1:] if word == "i" or word.startswith("i'") else word for word in words(text)]
    if not tokens:
        return ""
    return " ".join([tokens[0][0].upper() + tokens[0][1:]] + tokens[1:]) + "."


class LMSelector:
    """Pick among the visual hypotheses: joint visual log-score + LM_WEIGHT * log P_Qwen(sentence | screen).

    The output is always one of the hypotheses. System prompt and screen context are evaluated
    once (prefill) and their key/value cache is shared by all candidates.
    """

    def __init__(self, model_path=DEFAULT_MODEL, lm_weight=LM_WEIGHT):
        from mlx_lm import load
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Qwen model directory not found: {model_path}")
        self.model, self.tokenizer = load(str(model_path))
        self.model.eval()
        self.lm_weight = lm_weight
        self._prefix, self._state = None, None
        self.last = {}

    def prefill(self, screen_context=None):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        chat = [dict(role="system", content=SELECT_SYSTEM),
                dict(role="user", content="\n".join(context_lines(screen_context)))]
        prefix = list(self.tokenizer.apply_chat_template(chat, add_generation_prompt=True))
        if prefix != self._prefix:
            cache = make_prompt_cache(self.model)
            self.model(mx.array([prefix[:-1]]), cache=cache)
            state = [(layer.keys[..., :layer.offset, :], layer.values[..., :layer.offset, :]) for layer in cache]
            mx.eval(state)
            self._prefix, self._state = prefix, state
        return len(prefix)

    def log_probs(self, texts, screen_context=None):
        """log P(sentence) of each text after the system prompt and screen context."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.cache import make_prompt_cache
        self.prefill(screen_context)
        candidates = [self.tokenizer.encode(as_sentence(text), add_special_tokens=False) for text in texts]
        longest = max(map(len, candidates))
        padded = [tokens + [0] * (longest - len(tokens)) for tokens in candidates]
        cache = make_prompt_cache(self.model)
        for layer, (keys, values) in zip(cache, self._state):
            layer.state = (mx.repeat(keys, len(texts), axis=0), mx.repeat(values, len(texts), axis=0),
                           keys.shape[2])
        inputs = mx.array([[self._prefix[-1]] + tokens[:-1] for tokens in padded])
        logp = nn.log_softmax(self.model(inputs, cache=cache).astype(mx.float32), axis=-1)
        picked = mx.take_along_axis(logp, mx.array(padded)[..., None], axis=-1)[..., 0]
        mask = mx.array([[1.0] * len(tokens) + [0.0] * (longest - len(tokens)) for tokens in candidates])
        return (picked * mask).sum(axis=1).tolist()

    def correct(self, hypotheses, screen_context=None):
        """The chosen hypothesis as lowercase words; empty when there is none."""
        start = time.perf_counter()
        unique = {}
        for hypothesis in hypotheses:
            get = hypothesis.get if isinstance(hypothesis, dict) else lambda key: getattr(hypothesis, key)
            text = " ".join(words(get("text")))
            if text and text not in unique:
                unique[text] = get("log_score")
        if not unique:
            self.last = dict(latency_ms=0.0, candidates=0, prompt_tokens=0, cached_tokens=0, generated_tokens=0)
            return ""
        cached = self._prefix is not None
        lm = self.log_probs(list(unique), screen_context)
        scores = [visual + self.lm_weight * language for visual, language in zip(unique.values(), lm)]
        best = max(range(len(scores)), key=scores.__getitem__)
        self.last = dict(latency_ms=(time.perf_counter() - start) * 1000, candidates=len(unique), lm_log_probs=lm,
                         prompt_tokens=len(self._prefix), cached_tokens=len(self._prefix) if cached else 0,
                         generated_tokens=0, changed=best != 0)
        return list(unique)[best]
