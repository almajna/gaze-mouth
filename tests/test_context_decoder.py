"""context_decoder checks, plus the end-to-end OpenVocabLipDecoder check on a real take.

The lattices here are text-derived (that is what this package trains on); the end-to-end test
uses a real recorded take. Run: .venv-gf/bin/python -m unittest tests.test_context_decoder -v
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import random
import unittest

from context_decoder import prompt, synth_phoneme_data as synth, word_search
from silent_avsr.phonemes import PHONEMES, VISEME_OF, VOWELS, homophenes, strip_stress, viseme_sequence

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "models" / "Qwen2.5-3B-Instruct-4bit"
CHECKPOINT = ROOT / "models" / "open_vocab_lip"
ADAPTER = CHECKPOINT / "reconstructor_lora"
HAVE_QWEN = (QWEN / "model.safetensors").is_file()
HAVE_ADAPTER = HAVE_QWEN and (ADAPTER / "adapters.safetensors").is_file()
HAVE_ENCODER = (CHECKPOINT / "encoder.safetensors").is_file()
# Regression floors for the trained adapter, not quality bars. Measured on 2026-11-21 (docs/plan.md):
# 4/20 homophene pairs with both contexts right, 18/40 items, context changed the output in 15/20
# pairs, 28% of held-out command phrases exact. The floors sit a little below those numbers.
MIN_CHALLENGE_PAIRS = 3
MIN_CHALLENGE_ITEMS = 14
MIN_CONTEXT_CHANGES = 12
MIN_HELD_OUT_ACCURACY = 0.20


def check_lattice(case, lattice):
    case.assertEqual(lattice["frame_rate"], 30.0)
    case.assertTrue(1 <= len(lattice["hypotheses"]) <= 5)
    scores = [h["log_prob"] for h in lattice["hypotheses"]]
    case.assertEqual(scores, sorted(scores, reverse=True))
    case.assertEqual(lattice["hypotheses"][0]["phonemes"], [s["alternatives"][0][0] for s in lattice["slots"]])
    previous_end = -1
    for slot in lattice["slots"]:
        start, end = slot["frames"]
        case.assertTrue(previous_end < start <= end < lattice["num_frames"])
        previous_end = end
        names = [p for p, _ in slot["alternatives"]]
        log_probs = [lp for _, lp in slot["alternatives"]]
        case.assertTrue(set(names) <= set(PHONEMES))
        case.assertEqual(log_probs, sorted(log_probs, reverse=True))
        case.assertTrue(set(homophenes(names[0])) <= set(names))
        case.assertLessEqual(sum(math.exp(lp) for lp in log_probs), 1.0 + 1e-3)


class ChallengeTests(unittest.TestCase):
    def test_pairs_are_homophenous_and_share_one_lattice(self):
        self.assertGreaterEqual(len(synth.CHALLENGE), 16)
        records = synth.challenge_records()
        for i, (text_a, context_a, text_b, context_b) in enumerate(synth.CHALLENGE):
            self.assertNotEqual(text_a, text_b)
            self.assertEqual(viseme_sequence(synth.flat_phonemes(text_a)),
                             viseme_sequence(synth.flat_phonemes(text_b)), (text_a, text_b))
            a, b = records[2 * i], records[2 * i + 1]
            self.assertEqual(json.dumps(a["lattice"]), json.dumps(b["lattice"]))
            self.assertEqual(json.dumps(a["lattice"]), json.dumps(synth.ambiguous_lattice(text_b)))
            self.assertNotEqual(a["screen_context"], b["screen_context"])
            self.assertEqual((a["text"], b["text"]), (text_a, text_b))

    def test_ambiguous_lattice_does_not_favour_the_truth(self):
        lattice = synth.ambiguous_lattice("restart the pod")
        check_lattice(self, lattice)
        truth = [strip_stress(p) for p in synth.flat_phonemes("restart the pod")]
        self.assertEqual(len(lattice["slots"]), len(truth))
        for phoneme, slot in zip(truth, lattice["slots"]):
            names = [p for p, _ in slot["alternatives"]]
            self.assertEqual(set(names), set(homophenes(phoneme)))
            probabilities = [math.exp(lp) for _, lp in slot["alternatives"]]
            self.assertLess(max(probabilities) - min(probabilities), 1e-3)
        self.assertIn("P/B/M", prompt.render_slots(lattice))
        self.assertIn("T/D/N", prompt.render_slots(lattice))

    def test_challenge_phrases_never_reach_the_training_data(self):
        texts = {r["text"] for r in synth.generate(1500, seed=3)}
        self.assertFalse(texts & synth.CHALLENGE_PHRASES)


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = synth.generate(1200, seed=0)

    def test_deterministic_for_a_seed(self):
        self.assertEqual(json.dumps(synth.generate(200, seed=0)), json.dumps(self.records[:200]))
        self.assertNotEqual(json.dumps(synth.generate(200, seed=1)), json.dumps(self.records[:200]))

    def test_covers_all_domains_with_full_screen_context(self):
        self.assertEqual({r["domain"] for r in self.records}, {"os", "terminal", "dev", "chat", "browser", "vocab"})
        for record in self.records:
            context = record["screen_context"]
            self.assertEqual(sorted(context), ["app_name", "ax_role", "ax_text", "nearby_labels"])
            if record["domain"] != "vocab":  # vocabulary drills may have an empty screen
                self.assertTrue(context["app_name"] and context["ax_role"].startswith("AX") and context["ax_text"])
            self.assertIsInstance(context["nearby_labels"], list)
            self.assertEqual(record["text"], prompt.normalize_text(record["text"]))
            check_lattice(self, record["lattice"])

    def test_splits_are_disjoint_by_phrase(self):
        by_split = {}
        for record in self.records:
            by_split.setdefault(record["split"], set()).add(record["text"])
        self.assertEqual(set(by_split), {"train", "valid", "test"})
        self.assertFalse(by_split["train"] & by_split["test"])
        self.assertFalse(by_split["train"] & by_split["valid"])
        self.assertFalse(by_split["valid"] & by_split["test"])
        self.assertGreater(len(by_split["train"]), 5 * len(by_split["test"]))

    def test_vocabulary_drills_show_their_words_on_screen_about_half_the_time(self):
        drills = [r for r in self.records if r["domain"] == "vocab"]
        self.assertGreater(len(drills), 400)
        shown = sum(any(word in (r["screen_context"]["ax_text"] + " "
                                 + " ".join(r["screen_context"]["nearby_labels"]).lower()).split()
                        for word in r["text"].split()) for r in drills)
        self.assertTrue(0.35 < shown / len(drills) < 0.70, shown / len(drills))
        self.assertGreater(len({w for r in drills for w in r["text"].split()}), 800)

    def test_some_lattices_have_no_preference_inside_a_class(self):
        def flat(record):  # every slot (near-)uniform over its alternatives
            return all(math.exp(s["alternatives"][0][1]) - math.exp(s["alternatives"][-1][1]) < 1e-3
                       for s in record["lattice"]["slots"])
        share = sum(flat(r) for r in self.records) / len(self.records)
        self.assertTrue(0.2 < share < 0.4, share)

    def test_click_commands_name_a_label_that_is_on_screen(self):
        fixed = set(synth.OS_FIXED) | set(synth.BROWSER_FIXED)  # "select all" is not a click on a label
        clicks = [r for r in self.records if r["domain"] in ("os", "browser") and r["text"] not in fixed
                  and r["text"].split()[0] in ("click", "press", "tap", "select", "choose", "hit", "double")]
        self.assertGreater(len(clicks), 30)
        for record in clicks:
            labels = [prompt.normalize_text(l) for l in record["screen_context"]["nearby_labels"]]
            self.assertTrue(any(record["text"].endswith(label) for label in labels), record["text"])

    def test_noise_free_lattice_keeps_every_phoneme_in_its_viseme_class(self):
        for text in ("git push origin main", "open terminal", "see you at noon"):
            truth = [strip_stress(p) for p in synth.flat_phonemes(text)]
            lattice = synth.noisy_lattice(text, random.Random(7), severity=0.0)
            self.assertEqual(len(lattice["slots"]), len(truth))
            for phoneme, slot in zip(truth, lattice["slots"]):
                self.assertEqual(VISEME_OF[slot["alternatives"][0][0]], VISEME_OF[phoneme])
                self.assertIn(phoneme, [p for p, _ in slot["alternatives"]])

    def test_homophene_confusions_happen(self):
        tops = {}
        for seed in range(300):
            lattice = synth.noisy_lattice("pat", random.Random(seed), severity=0.0)  # P AE T
            for index in (0, 2):
                tops.setdefault(index, set()).add(lattice["slots"][index]["alternatives"][0][0])
        self.assertEqual(tops[0], {"P", "B", "M"})
        self.assertEqual(tops[2], {"T", "D", "N"})

    def test_unstressed_vowels_are_the_ones_that_get_deleted(self):
        self.assertEqual(synth.flat_phonemes("banana"), ["B", "AH0", "N", "AE1", "N", "AH0"])
        kept = {True: 0, False: 0}
        trials = 600
        for seed in range(trials):
            lattice = synth.noisy_lattice("banana", random.Random(seed), severity=1.0)
            vowels = [s["alternatives"][0][0] for s in lattice["slots"] if s["alternatives"][0][0] in VOWELS]
            kept[len(vowels) >= 3] += 1
        # Two unstressed vowels at 28% each: all three vowels survive about half the time.
        self.assertTrue(0.35 < kept[True] / trials < 0.70, kept)
        clean = synth.noisy_lattice("banana", random.Random(0), severity=0.0)
        self.assertEqual(len(clean["slots"]), 6)

    def test_write_dataset_round_trip(self):
        with TemporaryDirectory() as directory:
            sizes = synth.write_dataset(directory, 300, seed=5)
            self.assertEqual(sizes["train"] + sizes["valid"] + sizes["test"], 300)
            self.assertEqual(sizes["challenge"], 2 * len(synth.CHALLENGE))
            rows = synth.read_jsonl(Path(directory) / "train.jsonl")
            self.assertEqual(len(rows), sizes["train"])
            self.assertTrue(all(r["split"] == "train" for r in rows))
            self.assertEqual(len(synth.read_jsonl(Path(directory) / "challenge.jsonl")), sizes["challenge"])


KUBERNETES = dict(app_name="Terminal", ax_role="AXTextArea", ax_text="$ kubectl get pods\napi-7d9f 0/1 CrashLoopBackOff",
                  nearby_labels=["zsh", "prod-cluster"])
DISCORD = dict(app_name="Discord", ax_role="AXTextArea", ax_text="Kai: ModBot stopped answering commands",
               nearby_labels=["#general", "ModBot"])


class WordSearchTests(unittest.TestCase):
    def test_look_alike_words_share_a_position_and_on_screen_words_come_first(self):
        lattice = synth.ambiguous_lattice("restart the pod")
        self.assertEqual(json.dumps(lattice), json.dumps(synth.ambiguous_lattice("restart the bot")))
        for context, expected, other in ((KUBERNETES, "pod", "pot"), (DISCORD, "bot", "pod")):
            readings = word_search.search(lattice, context, n_best=4)
            self.assertTrue(word_search.covers(readings[:1], f"restart the {expected}"), readings[:1])
            last = readings[0][-1]
            self.assertEqual(last[0], expected)   # "pods" and "ModBot" are on screen
            self.assertIn(other, last)            # the look-alike stays available
        without_screen = word_search.search(lattice, None)
        self.assertTrue(any({"pod", "pot", "but"} <= set(reading[-1]) for reading in without_screen), without_screen)

    def test_screen_words_include_stems_and_parts_of_names(self):
        words = word_search.screen_words(dict(ax_text="Restarting 3 pods", nearby_labels=["ModBot", "Squash and merge"],
                                              app_name="Google Chrome"))
        self.assertTrue({"pods", "pod", "restarting", "restart", "modbot", "mod", "bot", "merge", "chrome"} <= words)
        self.assertEqual(word_search.screen_words(None), set())

    def test_weak_vowels_and_h_are_optional(self):
        banana = dict(word_search.variants(["B", "AH0", "N", "AE1", "N", "AH0"]))
        full = tuple(word_search.VISEME_OF[p] for p in ("B", "AH", "N", "AE", "N", "AH"))
        self.assertEqual(banana[full], 0)
        self.assertEqual(banana[full[:1] + full[2:]], 1)       # first weak vowel missing
        self.assertEqual(banana[full[:1] + full[2:5]], 2)      # both missing
        self.assertNotIn(full[:3] + full[4:], banana)          # the stressed vowel is never dropped
        lattice = synth.ambiguous_lattice("open terminal")
        lattice["slots"] = [s for i, s in enumerate(lattice["slots"]) if i != 2]  # the weak vowel of "open"
        self.assertTrue(word_search.covers(word_search.search(lattice, None), "open terminal"))

    def test_search_is_deterministic_and_handles_an_empty_lattice(self):
        lattice = synth.ambiguous_lattice("merge it")
        context = dict(nearby_labels=["Merge pull request"])
        self.assertEqual(word_search.search(lattice, context), word_search.search(lattice, context))
        self.assertTrue(word_search.covers(word_search.search(lattice, context), "merge it"))
        empty = dict(hypotheses=[], slots=[], num_frames=30, frame_rate=30.0)
        self.assertEqual(word_search.search(empty, context), [])
        self.assertFalse(word_search.covers([], "merge it"))

    def test_common_words_come_from_the_model_vocabulary(self):
        words, cumulative = word_search.common_words()
        self.assertEqual(len(words), len(cumulative))
        self.assertGreater(len(words), 5000)
        self.assertTrue(all(b > a for a, b in zip(cumulative, cumulative[1:])))
        self.assertTrue({"the", "open", "restart", "merge"} <= set(words))


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.lattice = synth.ambiguous_lattice("git push")

    def test_user_message_carries_context_and_lattice(self):
        context = dict(app_name="Terminal", ax_role="AXTextArea", ax_text="$ git status\nOn branch main",
                       nearby_labels=["zsh", "main"])
        text = prompt.user_message(self.lattice, context)
        self.assertIn("App: Terminal", text)
        self.assertIn("Element: AXTextArea", text)
        self.assertIn("Text: $ git status On branch main", text)
        self.assertIn("Nearby: zsh | main", text)
        self.assertIn("Phonemes: K/G/NG IY/IH T/D/N P/B/M UW/UH SH/ZH/CH/JH", text)
        self.assertIn("Also possible: ", text)
        readings = text.split("Readings:\n")[1].splitlines()
        self.assertTrue(1 <= len(readings) <= prompt.MAX_READINGS)
        self.assertTrue(readings[0].startswith("git/"), readings[0])  # "git" is on screen
        self.assertTrue(all(len(position.split("/")) <= prompt.MAX_READING_WORDS
                            for line in readings for position in line.split()))

    def test_no_readings_block_when_the_search_finds_nothing(self):
        empty = dict(hypotheses=[dict(phonemes=[], log_prob=0.0, frames=[])], slots=[], num_frames=30, frame_rate=30.0)
        self.assertNotIn("Readings", prompt.user_message(empty, None))

    def test_missing_context_and_long_text(self):
        text = prompt.user_message(self.lattice, None)
        for field in ("App", "Element", "Text", "Nearby"):
            self.assertIn(f"{field}: (none)", text)
        long = prompt.normalize_context(dict(ax_text="x" * 500 + " THE END", nearby_labels=[str(i) for i in range(40)]))
        self.assertEqual(len(long["ax_text"]), prompt.MAX_TEXT_CHARS)
        self.assertTrue(long["ax_text"].endswith("THE END"))
        self.assertEqual(len(long["nearby_labels"]), prompt.MAX_LABELS)

    def test_training_and_inference_share_the_prompt(self):
        context = dict(app_name="Terminal")
        training = prompt.messages(self.lattice, context, "Git push!")
        inference = prompt.messages(self.lattice, context)
        self.assertEqual(training[:2], inference)
        self.assertEqual(training[2], dict(role="assistant", content="git push"))
        self.assertEqual(prompt.normalize_text("Don't  STOP, now."), "don't stop now")


@unittest.skipUnless(HAVE_QWEN, "models/Qwen2.5-3B-Instruct-4bit is required")
class ReconstructorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from context_decoder import ScreenGroundedReconstructor
        cls.reconstructor = ScreenGroundedReconstructor(QWEN)
        cls.record = synth.challenge_records()[0]

    def test_returns_normalized_text_and_restores_the_prompt_cache(self):
        r = self.reconstructor
        self.assertGreater(len(r._prefix), 50)  # the system prompt is cached
        first = r.reconstruct(self.record["lattice"], self.record["screen_context"])
        self.assertIsInstance(first, str)
        self.assertEqual(first, prompt.normalize_text(first))
        self.assertEqual(r._cache[0].offset, len(r._prefix))
        self.assertGreater(r.last["latency_ms"], 0)
        self.assertEqual(r.last["cached_prefix"], len(r._prefix))
        other = synth.challenge_records()[5]
        r.reconstruct(other["lattice"], other["screen_context"])
        self.assertEqual(r.reconstruct(self.record["lattice"], self.record["screen_context"]), first)

    def test_accepts_missing_context_and_lattice_objects(self):
        class Wrapped:
            def to_dict(inner):
                return self.record["lattice"]

        self.assertIsInstance(self.reconstructor.reconstruct(Wrapped(), None), str)

    def test_missing_model_directory_raises(self):
        from context_decoder import ScreenGroundedReconstructor
        with self.assertRaises(FileNotFoundError):
            ScreenGroundedReconstructor(ROOT / "models" / "no-such-model")


@unittest.skipUnless(HAVE_QWEN, "models/Qwen2.5-3B-Instruct-4bit is required")
class LoRASmokeTests(unittest.TestCase):
    def test_one_epoch_on_synthetic_data_then_save_and_load(self):
        import mlx.core as mx
        from mlx_lm import load
        from context_decoder import ScreenGroundedReconstructor
        from context_decoder.train_reconstructor_lora import train_lora

        records = synth.generate(40, seed=11)
        train_records, valid_records = records[:32], records[32:]
        model, tokenizer = load(str(QWEN))
        with TemporaryDirectory() as directory:
            history = train_lora(model, tokenizer, train_records, valid_records, directory,
                                 iters=len(train_records) // 4, batch_size=4, num_layers=8, rank=8,
                                 report_every=2)  # one epoch
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["adapter_config.json", "adapters.safetensors"])
            losses = [loss for _, loss in history["train"]]
            self.assertTrue(all(math.isfinite(loss) for loss in losses))
            self.assertLess(losses[-1], losses[0])
            self.assertLess(history["valid"][-1][1], history["valid"][0][1])

            reloaded = ScreenGroundedReconstructor(QWEN, adapter_path=directory, cache_system_prompt=False)
            tokens = mx.array(reloaded.prompt_tokens(records[0]["lattice"], records[0]["screen_context"]))[None]
            trained_logits = model(tokens)[0, -1]
            reloaded_logits = reloaded.model(tokens)[0, -1]
            base_logits = load(str(QWEN))[0](tokens)[0, -1]
            self.assertLess(mx.abs(trained_logits - reloaded_logits).max().item(), 1e-2)
            self.assertGreater(mx.abs(trained_logits - base_logits).max().item(), 0.5)  # the adapter matters
            self.assertIsInstance(reloaded.reconstruct(records[0]["lattice"], records[0]["screen_context"]), str)


@unittest.skipUnless(HAVE_ADAPTER, "a trained adapter in models/open_vocab_lip/reconstructor_lora is required")
class TrainedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from context_decoder import ScreenGroundedReconstructor
        from context_decoder.train_reconstructor_lora import evaluate, evaluate_challenge
        cls.reconstructor = ScreenGroundedReconstructor(QWEN, adapter_path=ADAPTER)
        cls.challenge = evaluate_challenge(cls.reconstructor)
        held_out = [r for r in synth.generate(12000, seed=0)
                    if r["split"] == "test" and r["domain"] != "vocab"][:100]
        cls.held_out = evaluate(cls.reconstructor, held_out)
        print(f"\n  homophene pairs both right {cls.challenge['pairs_both_right']}/{cls.challenge['pairs']}, "
              f"items {cls.challenge['exact']}/{cls.challenge['total']}, "
              f"held-out phrases {cls.held_out['exact']}/{cls.held_out['total']}, "
              f"median {cls.held_out['median_latency_ms']} ms")

    def test_screen_context_disambiguates_homophenes(self):
        """Git/terminal context vs chat/browser context on one and the same lattice."""
        self.assertGreaterEqual(self.challenge["pairs_both_right"], MIN_CHALLENGE_PAIRS, self.challenge["misses"])
        self.assertGreaterEqual(self.challenge["exact"], MIN_CHALLENGE_ITEMS, self.challenge["misses"])
        self.assertGreaterEqual(self.challenge["context_changed_output"], MIN_CONTEXT_CHANGES)

    def test_held_out_phrases(self):
        self.assertGreaterEqual(self.held_out["accuracy"], MIN_HELD_OUT_ACCURACY, self.held_out["misses"])

    def test_cached_system_prompt_gives_the_same_answers(self):
        from context_decoder import ScreenGroundedReconstructor
        plain = ScreenGroundedReconstructor(QWEN, adapter_path=ADAPTER, cache_system_prompt=False)
        records = synth.challenge_records()[:10]
        same = sum(plain.reconstruct(r["lattice"], r["screen_context"])
                   == self.reconstructor.reconstruct(r["lattice"], r["screen_context"]) for r in records)
        self.assertGreaterEqual(same, 9)  # 4-bit weights: cached and uncached logits differ by rounding


@unittest.skipUnless(HAVE_QWEN and HAVE_ENCODER, "a trained encoder in models/open_vocab_lip and Qwen are required")
class EndToEndTests(unittest.TestCase):
    def test_decode_a_real_take(self):
        from silent_avsr import OpenVocabLipDecoder, features
        decoder = OpenVocabLipDecoder(checkpoint_dir=CHECKPOINT)
        speech = [pair for pair in features.silent_takes() if pair[1]["phrase"] != "__NULL__"]
        take = features.load_silent_take(*speech[-1])
        context = dict(app_name="Safari", ax_role="AXWebArea", ax_text="Search results", nearby_labels=["Images", "News"])
        decoder.decode(take["landmarks_30hz"], take["mouth_roi_30hz"], context)  # warm-up
        result = decoder.decode(take["landmarks_30hz"], take["mouth_roi_30hz"], context)
        self.assertEqual(sorted(result), ["latency_ms", "phoneme_lattice", "text"])
        self.assertIsInstance(result["text"], str)
        lattice = result["phoneme_lattice"]
        self.assertEqual(sorted(lattice), ["frame_rate", "hypotheses", "num_frames", "slots"])
        self.assertEqual(lattice["num_frames"], len(take["mouth_roi_30hz"]))
        json.dumps(result)
        timing = decoder.last_timing
        self.assertLess(timing["encoder_ms"] + timing["lattice_ms"], 120.0)
        self.assertAlmostEqual(result["latency_ms"], sum(timing.values()), delta=0.01)
        print(f"\n  real take {take['name']} ({take['phrase']!r}) -> {result['text']!r}  "
              f"encoder {timing['encoder_ms']:.1f} + lattice {timing['lattice_ms']:.1f} + "
              f"reconstruct {timing['reconstruct_ms']:.1f} ms")


if __name__ == "__main__":
    unittest.main()
