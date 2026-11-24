"""Checks against the real USR vocabulary and sentence lists."""
import re
import unittest
from pathlib import Path

from silent_avsr.usr_tokenizer import DEFAULT_UNITS, USRTokenizer


SENTENCE_LISTS = (Path("train_sentences.txt"), Path("eval_holdout_sentences.txt"))


@unittest.skipUnless(DEFAULT_UNITS.is_file(), "requires the published USR tokenizer units")
class USRTokenizerTests(unittest.TestCase):
    @unittest.skipUnless(all(path.is_file() for path in SENTENCE_LISTS), "requires local sentence lists")
    def test_real_transcripts_round_trip_to_checkpoint_ids(self):
        tokenizer = USRTokenizer()
        inverse = {index: piece for piece, index in tokenizer.pieces.items()}
        for path in SENTENCE_LISTS:
            for text in path.read_text().splitlines():
                with self.subTest(text=text):
                    ids = tokenizer.encode(text)
                    self.assertTrue(all(index in inverse and index != 0 for index in ids))
                    decoded = "".join(inverse[index] for index in ids).replace("▁", " ").strip()
                    expected = " ".join(re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)*",
                                                   text.upper().replace("’", "'")))
                    self.assertEqual(decoded, expected)
                    self.assertEqual(ids, tokenizer.encode(text))

    def test_whole_word_uses_longest_known_token(self):
        tokenizer = USRTokenizer()
        self.assertEqual(tokenizer.encode("THE"), [tokenizer.pieces["▁THE"]])


if __name__ == "__main__":
    unittest.main()
