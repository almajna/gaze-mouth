"""Lexicon search: phoneme lattice -> candidate word sequences ("readings").

A 3B language model with a small adapter reads raw ARPAbet poorly, so this gives it dictionary
guesses to choose from. Words are indexed by viseme sequence (pod / bot / pot share an entry),
weak vowels and H are optional, and words visible on screen come first. The search only has
word frequencies, no grammar, so its guesses are hints: measured on training-split lattices the
true phrase is among the top four readings for about a quarter of the command phrases.
"""
from functools import lru_cache
from itertools import accumulate
from pathlib import Path
import json
import math
import re

from silent_avsr.lexicon import EXTRA, word_phonemes
from silent_avsr.phonemes import VISEME_OF, VOWELS, strip_stress

VOCAB_JSON = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-3B-Instruct-4bit" / "vocab.json"
SHORT_WORDS = frozenset("a i to of in it is on we my no do go up me he so or at as an be if us by".split())
VOWEL_CLASSES = frozenset(VISEME_OF[v] for v in VOWELS)
DROP_COST = 1.0        # a weak vowel or H of the word is missing from the lattice
VOWEL_SWAP_COST = 3.0  # a vowel the lattice does not list at all
INSERT_COST = 4.0      # a lattice sound that belongs to no word
WORD_COST = 0.5        # per word: prefers fewer, longer words
SCREEN_COST = 1.5      # unigram cost of a word that is visible on screen
UNKNOWN_COST = 9.5     # unigram cost of a lexicon word without a frequency rank
MAX_OPTIONAL = 3
MAX_WORDS_PER_POSITION = 8
SUFFIXES = ("ing", "ed", "es", "s")


@lru_cache(maxsize=1)
def common_words():
    """(words, cumulative sampling weights). Common English words are taken to be the CMUdict words
    that are a single token in the Qwen vocabulary; earlier tokens are more frequent. Without the
    model directory this falls back to short CMUdict words with equal weight."""
    import cmudict
    pronounced = cmudict.dict()
    if VOCAB_JSON.is_file():
        vocab = json.loads(VOCAB_JSON.read_text())
        ranked = sorted((index, token[1:]) for token, index in vocab.items()
                        if token.startswith("\u0120") and token[1:].isascii() and token[1:].isalpha()
                        and token[1:].islower() and (len(token) > 3 or token[1:] in SHORT_WORDS)
                        and token[1:] in pronounced)
        words = [word for _, word in ranked]
        weights = [1.0 / (rank + 200) ** 0.8 for rank in range(len(words))]
    else:
        words = sorted(w for w in pronounced if w.isalpha() and 3 <= len(w) <= 7)
        weights = [1.0] * len(words)
    return words, list(accumulate(weights))


class Node:
    __slots__ = ("children", "words", "best", "sequence")

    def __init__(self):
        self.children, self.words, self.sequence = {}, None, None
        self.best = math.inf  # cheapest word at or below this node: look-ahead for pruning


def variants(phones):
    """Viseme sequences a word can show: each unstressed vowel and each H may be missing."""
    optional = [i for i, p in enumerate(phones) if p.endswith("0") or p == "HH"][:MAX_OPTIONAL]
    for mask in range(1 << len(optional)):
        dropped = {optional[j] for j in range(len(optional)) if mask >> j & 1}
        sequence = tuple(VISEME_OF[strip_stress(p)] for i, p in enumerate(phones) if i not in dropped)
        if sequence:
            yield sequence, len(dropped)


def insert(root, terminals, word, cost):
    for sequence, dropped in variants(word_phonemes(word)):
        total = cost + DROP_COST * dropped
        node = root
        for viseme in sequence:
            child = node.children.get(viseme)
            if child is None:
                child = node.children[viseme] = Node()
            node = child
            if total < node.best:
                node.best = total
        if node.words is None:
            node.words, node.sequence = {}, sequence
            terminals[sequence] = node
        if total < node.words.get(word, math.inf):
            node.words[word] = total


@lru_cache(maxsize=1)
def lexicon():
    """(trie root, {viseme sequence: terminal node}, {word: unigram cost}) over common words plus
    the developer-term table."""
    words, cumulative = common_words()
    total = cumulative[-1]
    costs, previous = {}, 0.0
    for word, running in zip(words, cumulative):
        costs[word] = -math.log((running - previous) / total)
        previous = running
    for word in EXTRA:
        costs.setdefault(word, UNKNOWN_COST)
    root, terminals = Node(), {}
    for word, cost in costs.items():
        insert(root, terminals, word, cost)
    return root, terminals, costs


def screen_words(screen_context):
    """Lowercase words visible on screen, with simple stems ('pods' also offers 'pod') and the
    parts of CamelCase names ('ModBot' offers 'mod' and 'bot')."""
    context = screen_context or {}
    labels = context.get("nearby_labels") or []
    text = " ".join([str(context.get("ax_text") or ""), str(context.get("app_name") or "")]
                    + [str(l) for l in ([labels] if isinstance(labels, str) else labels)])
    found = set()
    for token in re.findall(r"[A-Za-z]+", text):
        for piece in [token] + re.findall(r"[A-Z][a-z]+|[a-z]+", token):
            piece = piece.lower()
            found.add(piece)
            for suffix in SUFFIXES:
                if piece.endswith(suffix) and len(piece) - len(suffix) >= 3:
                    found.add(piece[:-len(suffix)])
    return {w for w in found if len(w) >= 2 or w in ("a", "i")}


def slot_costs(lattice):
    """Per slot: {viseme class: -log P(class)} from the slot's alternatives, and whether its
    best alternative is a vowel."""
    costs = []
    for slot in lattice["slots"]:
        mass = {}
        for phoneme, log_prob in slot["alternatives"]:
            viseme = VISEME_OF[phoneme]
            mass[viseme] = mass.get(viseme, 0.0) + math.exp(log_prob)
        costs.append(({v: -math.log(min(1.0, p)) for v, p in mass.items() if p >= 0.02},
                      slot["alternatives"][0][0] in VOWELS))
    return costs


def search(lattice, screen_context=None, n_best=4, beam=400):
    """Best word sequences for a lattice. Each reading is a list of positions; each position is a
    list of look-alike words, most plausible first (on-screen words, then common words)."""
    lattice = lattice.to_dict() if hasattr(lattice, "to_dict") else lattice
    root, terminals, unigram = lexicon()
    visible = screen_words(screen_context)
    overlay, overlay_terminals = Node(), {}  # on-screen words, including names the lexicon lacks
    for word in visible:
        if word_phonemes(word):
            insert(overlay, overlay_terminals, word, SCREEN_COST)
    roots = (root, overlay)

    histories = [None]  # index -> (previous index, terminal node); 0 is the empty history
    interned = {}
    # (node or None at a word boundary, history index) -> (cost so far, cost used for pruning)
    active = {(None, 0): (0.0, 0.0)}

    def offer(table, key, cost, rank):
        if rank < table.get(key, (math.inf, math.inf))[1]:
            table[key] = (cost, rank)

    for costs, is_vowel in slot_costs(lattice):
        grown = {}
        for (node, history), (cost, rank) in active.items():
            offer(grown, (node, history), cost + INSERT_COST, rank + INSERT_COST)  # sound in no word
            for start in (roots if node is None else (node,)):
                for viseme, child in start.children.items():
                    step = costs.get(viseme)
                    if step is None:
                        if not (is_vowel and viseme in VOWEL_CLASSES):
                            continue
                        step = VOWEL_SWAP_COST
                    moved = cost + step
                    if child.children:
                        offer(grown, (child, history), moved, moved + child.best + WORD_COST)
                    if child.words:
                        key = (history, id(child))
                        index = interned.get(key)
                        if index is None:
                            index = interned[key] = len(histories)
                            histories.append((history, child))
                        done = moved + min(child.words.values()) + WORD_COST
                        offer(grown, (None, index), done, done)
        active = dict(sorted(grown.items(), key=lambda kv: kv[1][1])[:beam])

    def look_alikes(terminal):
        words = {}
        for table in (terminals, overlay_terminals):
            other = table.get(terminal.sequence)
            for word, cost in (other.words.items() if other else ()):
                words[word] = min(cost, words.get(word, math.inf))
        ranked = sorted(words, key=lambda w: (w not in visible, words[w], w))
        return ranked[:MAX_WORDS_PER_POSITION]

    readings, seen = [], set()
    for (node, history), _ in sorted(active.items(), key=lambda kv: kv[1][1]):
        if node is not None or history == 0:
            continue
        chain = []
        while history:
            history, terminal = histories[history]
            chain.append(terminal)
        positions = [look_alikes(terminal) for terminal in reversed(chain)]
        key = tuple(tuple(p) for p in positions)
        if key not in seen:
            seen.add(key)
            readings.append(positions)
            if len(readings) == n_best:
                break
    return readings


def covers(readings, text):
    """True if some reading offers every word of `text`, in order."""
    words = text.split()
    return any(len(reading) == len(words) and all(w in position for w, position in zip(words, reading))
               for reading in readings)
