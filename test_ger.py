"""GER prompt and corrector checks on saved beams of real training clips (no camera, no holdout)."""
import json
from pathlib import Path
import unittest

from context_decoder import ger

BEAMS = Path(__file__).parent / "models/usr2_speaker_lora_v2_folds/oof_beams.json"


@unittest.skipUnless(BEAMS.is_file(), "out-of-fold beams have not been generated")
class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(BEAMS.read_text())

    def test_guess_lines_are_unique_scored_and_lowercase(self):
        for row in self.rows[:40]:
            lines = ger.guess_lines(row["hypotheses"])
            self.assertLessEqual(len(lines), ger.MAX_GUESSES)
            texts = [line.split(" ", 1)[1] for line in lines]
            self.assertEqual(len(texts), len(set(texts)))
            for line, text in zip(lines, texts):
                float(line.split(" ", 1)[0])
                self.assertEqual(text, " ".join(ger.words(text)))

    def test_training_and_inference_share_the_prompt(self):
        row = self.rows[0]
        context = dict(app_name="Notes", ax_role="AXTextArea", ax_text="x " * 400, nearby_labels=["a"] * 30)
        train = ger.ger_messages(row["hypotheses"], context, row["reference"])
        self.assertEqual(train[:2], ger.ger_messages(row["hypotheses"], context))
        self.assertEqual(train[2], dict(role="assistant", content=" ".join(ger.words(row["reference"]))))
        user = train[1]["content"].split("\n")
        self.assertEqual(user[:2], ["App: Notes", "Element: AXTextArea"])
        self.assertLessEqual(len(user[2]), len("Text: ") + ger.MAX_TEXT_CHARS)
        self.assertEqual(user[3].count("|"), ger.MAX_LABELS - 1)
        self.assertEqual(user[4], "Guesses:")

    def test_missing_context_reads_none(self):
        user = ger.ger_messages(self.rows[0]["hypotheses"])[1]["content"]
        self.assertTrue(user.startswith("App: (none)\nElement: (none)\nText: (none)\nNearby: (none)\nGuesses:\n"))

    def test_clean_keeps_the_first_line_and_drops_thinking(self):
        self.assertEqual(ger.clean("<think>maybe\nthis</think>\nDon't  STOP, now.\nextra"), "don't stop now")
        self.assertEqual(ger.clean(""), "")


@unittest.skipUnless(BEAMS.is_file() and ger.DEFAULT_MODEL.is_dir(), "needs the beams and the Qwen directory")
class CorrectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(BEAMS.read_text())
        adapter = ger.DEFAULT_ADAPTER if (ger.DEFAULT_ADAPTER / "adapter_config.json").is_file() else None
        cls.corrector = ger.GERCorrector(adapter_path=adapter)

    def test_no_guess_returns_empty_text_without_the_model(self):
        self.assertEqual(self.corrector.correct([]), "")
        self.assertEqual(self.corrector.correct([dict(text=" ", log_score=-1.0)]), "")
        self.assertEqual(self.corrector.last["generated_tokens"], 0)

    def test_prompt_cache_does_not_change_the_answer(self):
        context = dict(app_name="Terminal", ax_role="AXTextArea", ax_text="$ git status", nearby_labels=["zsh"])
        first, second = self.rows[0]["hypotheses"], self.rows[1]["hypotheses"]
        self.corrector.reset()
        cold = [self.corrector.correct(first, context)]
        self.assertEqual(self.corrector.last["cached_tokens"], 0)
        self.corrector.reset()
        cold.append(self.corrector.correct(second, None))
        self.corrector.reset()
        self.assertGreater(self.corrector.prefill(context), 50)
        warm = [self.corrector.correct(first, context)]
        self.assertGreater(self.corrector.last["cached_tokens"], 50)
        warm.append(self.corrector.correct(second, None))  # context changed: the cache is trimmed back
        self.assertEqual(cold, warm)
        self.assertEqual(warm[0], " ".join(ger.words(warm[0])))


@unittest.skipUnless(BEAMS.is_file() and ger.DEFAULT_MODEL.is_dir(), "needs the beams and the Qwen directory")
class SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(BEAMS.read_text())
        cls.selector = ger.LMSelector()

    def test_the_answer_is_one_of_the_hypotheses(self):
        for row in self.rows[:12]:
            texts = [" ".join(ger.words(h["text"])) for h in row["hypotheses"]]
            self.assertIn(self.selector.correct(row["hypotheses"]), texts)
            self.assertEqual(self.selector.last["candidates"], len(set(texts) - {""}))

    def test_no_guess_returns_empty_text(self):
        self.assertEqual(self.selector.correct([]), "")
        self.assertEqual(self.selector.correct([dict(text=" ", log_score=-1.0)]), "")

    def test_weight_zero_keeps_the_visual_order(self):
        weight, self.selector.lm_weight = self.selector.lm_weight, 0.0
        try:
            for row in self.rows[:6]:
                self.assertEqual(self.selector.correct(row["hypotheses"]),
                                 " ".join(ger.words(row["hypotheses"][0]["text"])))
        finally:
            self.selector.lm_weight = weight

    def test_scores_do_not_depend_on_the_other_candidates(self):
        texts = [h["text"] for h in self.rows[3]["hypotheses"]]
        together = self.selector.log_probs(texts)
        for text, score in zip(texts[:3], together):
            self.assertAlmostEqual(self.selector.log_probs([text])[0], score, delta=0.5)

    def test_screen_context_changes_the_prefix_and_is_cached(self):
        context = dict(app_name="Terminal", ax_role="AXTextArea", ax_text="git status")
        first = self.selector.prefill(context)
        state = self.selector._state
        self.assertEqual(self.selector.prefill(context), first)
        self.assertIs(self.selector._state, state)
        self.assertNotEqual(self.selector.prefill(None), first)


if __name__ == "__main__":
    unittest.main()
