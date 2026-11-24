"""The one prompt format used for both LoRA training data and inference."""
import math
import re

from .phoneme_search import search

SYSTEM = (
    "Decode silent lip reading into the words the user mouthed. The input is ARPAbet phonemes with no "
    "word boundaries. A slash lists look-alike candidates for one sound: lips cannot tell apart "
    "P/B/M, T/D/N, K/G/NG, F/V, S/Z, SH/ZH/CH/JH or TH/DH, similar vowels get confused, and weak "
    "vowels and H may be missing. Readings are dictionary guesses at the words, with look-alike "
    "words separated by slashes; they can be wrong or incomplete. Use the screen context to choose "
    "the words that fit. Reply with the phrase only, lowercase, no punctuation.")
MAX_TEXT_CHARS = 240
MAX_LABELS = 12
MAX_SLOT_ALTERNATIVES = 4
MIN_SLOT_PROB = 0.10
EXTRA_HYPOTHESES = 1
MAX_READINGS = 4
MAX_READING_WORDS = 5


def normalize_context(screen_context):
    """dict(app_name, ax_role, ax_text, nearby_labels) with missing keys filled and sizes capped."""
    context = dict(screen_context or {})
    labels = context.get("nearby_labels") or []
    if isinstance(labels, str):
        labels = [labels]
    text = " ".join(str(context.get("ax_text") or "").split())
    return dict(app_name=str(context.get("app_name") or "").strip(),
                ax_role=str(context.get("ax_role") or "").strip(),
                ax_text=text[-MAX_TEXT_CHARS:],  # the end of a text area is the most recent part
                nearby_labels=[" ".join(str(l).split()) for l in labels if str(l).strip()][:MAX_LABELS])


def render_slots(lattice):
    parts = []
    for slot in lattice["slots"]:
        names = [p for i, (p, log_prob) in enumerate(slot["alternatives"])
                 if i == 0 or math.exp(log_prob) >= MIN_SLOT_PROB][:MAX_SLOT_ALTERNATIVES]
        parts.append("/".join(names))
    return " ".join(parts)


def render_readings(lattice, screen_context):
    """Lines such as 'restart the pod/bot/pot/mod'. Empty when the search finds nothing."""
    return [" ".join("/".join(words[:MAX_READING_WORDS]) for words in reading)
            for reading in search(lattice, screen_context, n_best=MAX_READINGS)]


def user_message(lattice, screen_context):
    context = normalize_context(screen_context)
    lines = [f"App: {context['app_name'] or '(none)'}",
             f"Element: {context['ax_role'] or '(none)'}",
             f"Text: {context['ax_text'] or '(none)'}",
             f"Nearby: {' | '.join(context['nearby_labels']) or '(none)'}",
             f"Phonemes: {render_slots(lattice)}"]
    others = [" ".join(h["phonemes"]) for h in lattice["hypotheses"][1:1 + EXTRA_HYPOTHESES] if h["phonemes"]]
    if others:
        lines.append("Also possible: " + " ; ".join(others))
    readings = render_readings(lattice, context)
    if readings:
        lines += ["Readings:"] + readings
    return "\n".join(lines)


def messages(lattice, screen_context, text=None):
    chat = [dict(role="system", content=SYSTEM), dict(role="user", content=user_message(lattice, screen_context))]
    if text is not None:
        chat.append(dict(role="assistant", content=normalize_text(text)))
    return chat


def normalize_text(text):
    """Lowercase words separated by single spaces; punctuation other than apostrophes removed."""
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()))
