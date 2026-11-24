"""Text to ARPAbet (with stress digits). CMUdict, a small developer-term table, then a letter fallback."""
from functools import lru_cache
import re

from .phonemes import PHONEMES, strip_stress

# Words CMUdict lacks (or pronounces unhelpfully) that show up in commands and developer prompts.
EXTRA = {
    "npm": "EH1 N P IY1 EH1 M", "json": "JH EY1 S AH0 N", "localhost": "L OW1 K AH0 L HH OW2 S T",
    "pytest": "P AY1 T EH2 S T", "kubectl": "K UW1 B K AH0 N T R OW2 L", "async": "EY1 S IH2 NG K",
    "await": "AH0 W EY1 T", "refactor": "R IY0 F AE1 K T ER0", "docstring": "D AA1 K S T R IH2 NG",
    "typescript": "T AY1 P S K R IH2 P T", "javascript": "JH AA1 V AH0 S K R IH2 P T",
    "rebase": "R IY0 B EY1 S", "readme": "R IY1 D M IY2", "figma": "F IH1 G M AH0",
    "xcode": "EH1 K S K OW2 D", "api": "EY2 P IY2 AY1", "url": "Y UW2 AA2 R EH1 L",
    "sql": "S IY1 K W AH0 L", "css": "S IY2 EH2 S EH1 S", "html": "EY2 CH T IY2 EH2 M EH1 L",
    "cli": "S IY2 EH2 L AY1", "ssh": "EH2 S EH2 S EY1 CH", "repo": "R IY1 P OW0",
    "github": "G IH1 T HH AH2 B", "wifi": "W AY1 F AY2", "config": "K AH0 N F IH1 G",
    "middleware": "M IH1 D AH0 L W EH2 R", "endpoint": "EH1 N D P OY2 N T", "backend": "B AE1 K EH2 N D",
    "frontend": "F R AH1 N T EH2 N D", "codebase": "K OW1 D B EY2 S", "dataset": "D EY1 T AH0 S EH2 T",
    "linter": "L IH1 N T ER0", "lint": "L IH1 N T", "regex": "R EH1 G EH2 K S", "auth": "AO1 TH",
    "dag": "D AE1 G", "homebrew": "HH OW1 M B R UW2", "ghostty": "G OW1 S T IY0", "iterm": "AY1 T ER2 M",
    "jupyter": "JH UW1 P AH0 T ER0", "venmo": "V EH1 N M OW0", "trello": "T R EH1 L OW0",
    "jira": "JH IH1 R AH0", "airflow": "EH1 R F L OW2", "timeout": "T AY1 M AW2 T",
    "todo": "T UH0 D UW1", "docs": "D AA1 K S", "doc": "D AA1 K", "app": "AE1 P", "apps": "AE1 P S",
    "emoji": "IH0 M OW1 JH IY0", "screenshot": "S K R IY1 N SH AA2 T", "bot": "B AA1 T",
    "cache": "K AE1 SH", "facetime": "F EY1 S T AY2 M", "dropdown": "D R AA1 P D AW2 N",
    "checkbox": "CH EH1 K B AA2 K S", "sidebar": "S AY1 D B AA2 R", "toolbar": "T UW1 L B AA2 R",
    "username": "Y UW1 Z ER0 N EY2 M", "login": "L AO1 G IH2 N", "signup": "S AY1 N AH2 P",
    "upload": "AH1 P L OW2 D", "unmute": "AH0 N M Y UW1 T", "redo": "R IY0 D UW1",
    "minimize": "M IH1 N AH0 M AY2 Z", "serverless": "S ER1 V ER0 L AH0 S", "webhook": "W EH1 B HH UH2 K",
    "staging": "S T EY1 JH IH0 NG", "dev": "D EH1 V", "prod": "P R AA1 D", "env": "EH1 N V",
    "sudo": "S UW1 D UW0", "cron": "K R AA1 N", "yaml": "Y AE1 M AH0 L", "zsh": "Z IY1 SH EH1 L",
    "ok": "OW2 K EY1", "okay": "OW2 K EY1", "limiter": "L IH1 M AH0 T ER0", "resizer": "R IY0 S AY1 Z ER0",
    "retries": "R IY1 T R AY2 Z", "hotfix": "HH AA1 T F IH2 K S", "numpy": "N AH1 M P AY2",
    "tailwind": "T EY1 L W IH2 N D", "div": "D IH1 V", "lodash": "L OW1 D AE2 SH",
    "showtimes": "SH OW1 T AY2 M Z", "postgres": "P OW1 S T G R EH2 S", "sourdough": "S AW1 ER0 D OW2",
    "recents": "R IY1 S AH0 N T S", "kubernetes": "K UW2 B ER0 N EH1 T IY0 Z", "ingress": "IH1 N G R EH2 S",
    "formatter": "F AO1 R M AE2 T ER0", "remotes": "R IH0 M OW1 T S",
}
LETTERS = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1", "f": "EH1 F", "g": "JH IY1",
    "h": "EY1 CH", "i": "AY1", "j": "JH EY1", "k": "K EY1", "l": "EH1 L", "m": "EH1 M", "n": "EH1 N",
    "o": "OW1", "p": "P IY1", "q": "K Y UW1", "r": "AA1 R", "s": "EH1 S", "t": "T IY1", "u": "Y UW1",
    "v": "V IY1", "w": "D AH1 B AH0 L Y UW0", "x": "EH1 K S", "y": "W AY1", "z": "Z IY1",
}
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen " \
       "fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


@lru_cache(maxsize=1)
def _cmudict():
    import cmudict
    return cmudict.dict()


def number_words(n):
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10 - 2]] + ([ONES[n % 10]] if n % 10 else [])
    return [ONES[int(d)] for d in str(n)]


def normalize(text):
    """Lowercase words; digits become words; punctuation except apostrophes is dropped."""
    words = []
    for token in re.findall(r"[a-z']+|\d+", text.lower()):
        token = token.strip("'")
        if token.isdigit():
            words += number_words(int(token))
        elif token:
            words.append(token)
    return words


def word_phonemes(word):
    """Stress-marked ARPAbet for one normalized word. Unknown words are spelled letter by letter."""
    if word in EXTRA:
        return EXTRA[word].split()
    entries = _cmudict().get(word)
    if entries:
        return list(entries[0])
    if word.endswith("'s") and _cmudict().get(word[:-2]):
        return list(_cmudict()[word[:-2]][0]) + ["Z"]
    return [p for letter in word if letter in LETTERS for p in LETTERS[letter].split()]


def in_vocabulary(word):
    return word in EXTRA or word in _cmudict()


def g2p(text):
    """List of (word, [stress-marked phonemes]) for a phrase."""
    pairs = [(word, word_phonemes(word)) for word in normalize(text)]
    for _, phones in pairs:
        assert all(strip_stress(p) in PHONEMES for p in phones), phones
    return pairs


def phrase_phonemes(text):
    """Flat stress-free ARPAbet for a phrase: the CTC target."""
    return [strip_stress(p) for _, phones in g2p(text) for p in phones]
