"""ARPAbet inventory and viseme (homophene) classes. Pure Python, no torch."""

BLANK = "<blank>"
BLANK_ID = 0

VOWELS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")
CONSONANTS = ("B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S",
              "SH", "T", "TH", "V", "W", "Y", "Z", "ZH")
PHONEMES = tuple(sorted(VOWELS + CONSONANTS))  # the 39 CMUdict phonemes
SYMBOLS = (BLANK,) + PHONEMES
VOCAB_SIZE = len(SYMBOLS)  # 40
ID = {symbol: i for i, symbol in enumerate(SYMBOLS)}

# Phonemes inside one class look the same on the lips. Consonant classes follow the usual
# place-of-articulation grouping. The vowel groups are coarser and are this project's choice.
VISEME_CLASSES = (
    ("P", "B", "M"),
    ("T", "D", "N"),
    ("K", "G", "NG"),
    ("F", "V"),
    ("S", "Z"),
    ("SH", "ZH", "CH", "JH"),
    ("TH", "DH"),
    ("L",), ("R",), ("W",), ("Y",), ("HH",),
    ("IY", "IH"),
    ("EY", "EH", "AE"),
    ("AA", "AO", "AH"),
    ("UW", "UH"),
    ("OW",), ("ER",), ("AY",), ("AW",), ("OY",),
)
VISEME_OF = {p: i for i, members in enumerate(VISEME_CLASSES) for p in members}
assert sorted(VISEME_OF) == list(PHONEMES)


def strip_stress(phoneme):
    """'AH0' -> 'AH'."""
    return phoneme.rstrip("012")


def homophenes(phoneme):
    """Every phoneme in the same viseme class, the phoneme itself included."""
    return VISEME_CLASSES[VISEME_OF[phoneme]]


def viseme_sequence(phonemes):
    return tuple(VISEME_OF[strip_stress(p)] for p in phonemes)


def encode(phonemes):
    return [ID[strip_stress(p)] for p in phonemes]


def decode(ids):
    return [SYMBOLS[i] for i in ids]
