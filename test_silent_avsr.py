"""silent_avsr checks on real recorded clips (data/silent_train takes, data/session_* clips).

No camera, no generated sensor arrays. Run: .venv-gf/bin/python -m unittest test_silent_avsr -v
"""
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import json
import math
import statistics
import sys
import unittest

import numpy as np
import torch

from silent_avsr import (DualStreamPhonemeEncoder, EncoderConfig, LoRAAdapter, OpenVocabLipDecoder,
                         PhonemeLatticeDecoder)
from silent_avsr import features, train_ctc
from silent_avsr.lattice import forced_alignment, logaddexp, prefix_beam_search
from silent_avsr.lexicon import phrase_phonemes
from silent_avsr.phonemes import BLANK_ID, SYMBOLS, VISEME_OF, VOCAB_SIZE, homophenes

ROOT = Path(__file__).resolve().parent
AUTO_AVSR = ROOT / "models" / "auto_avsr"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TAKES = features.silent_takes()
CLIPS = features.session_clips()
HAVE_DATA = len(TAKES) >= 8 and len(CLIPS) >= 4
HAVE_FRONTEND = (AUTO_AVSR / "vsr.pth").is_file()
LATENCY_BUDGET_MS = 120.0


def build_encoder():
    torch.manual_seed(0)
    model = DualStreamPhonemeEncoder(EncoderConfig())
    if HAVE_FRONTEND:
        model.load_auto_avsr_frontend(AUTO_AVSR / "vsr.pth")
    return model.to(DEVICE).eval()


def tensor(x):
    return None if x is None else torch.from_numpy(x)[None].to(DEVICE)


@unittest.skipUnless(HAVE_DATA, "real clips in data/ are required")
class RealClipCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_encoder()
        cls.take = features.load_silent_take(*TAKES[0])
        # Two real dual-stream clips of different length.
        clips = [features.load_session_clip(p) for p in CLIPS[:6]]
        clips.sort(key=lambda c: len(c["landmarks_30hz"]))
        cls.short_clip, cls.clip = clips[0], clips[-1]

    @torch.inference_mode()
    def run_model(self, landmarks, roi, model=None):
        return (model or self.model)(tensor(landmarks), tensor(roi))


class PhonemeTests(unittest.TestCase):
    def test_inventory(self):
        self.assertEqual(VOCAB_SIZE, 40)
        self.assertEqual(SYMBOLS[BLANK_ID], "<blank>")
        self.assertEqual(set(homophenes("P")), {"P", "B", "M"})
        self.assertEqual(set(homophenes("T")), {"T", "D", "N"})
        self.assertEqual(VISEME_OF["K"], VISEME_OF["G"])

    def test_lexicon(self):
        self.assertEqual(phrase_phonemes("open terminal"),
                         "OW P AH N T ER M AH N AH L".split())
        self.assertEqual(phrase_phonemes("git push"), "G IH T P UH SH".split())
        self.assertEqual(phrase_phonemes("page 2"), phrase_phonemes("page two"))


class FeatureTests(RealClipCase):
    def test_real_clip_shapes(self):
        geometry, roi = self.clip["landmarks_30hz"], self.clip["mouth_roi_30hz"]
        self.assertEqual(geometry.shape[1], features.D_GEOM)
        self.assertEqual(features.D_GEOM, 163)
        self.assertEqual(roi.shape[1:], (1, 88, 88))
        self.assertEqual(len(geometry), len(roi))
        self.assertEqual(roi.dtype, np.uint8)
        self.assertTrue(np.isfinite(geometry).all())
        self.assertGreater(roi.std(), 5)  # a real image, not a flat crop

    def test_take_is_resampled_to_30hz(self):
        folder, meta = TAKES[0]
        source = np.load(folder / "mouth_96x96.npy")
        roi = self.take["mouth_roi_30hz"]
        self.assertEqual(roi.shape[1:], (1, 88, 88))
        duration = (len(source) - 1) / meta["video_fps"]
        self.assertEqual(len(roi), math.floor(duration * 30 + 1e-6) + 1)
        np.testing.assert_array_equal(roi[0, 0], source[0, 4:92, 4:92])
        self.assertIsNone(self.take["landmarks_30hz"])

    def test_geometry_ignores_head_rotation_scale_and_shift(self):
        with np.load(CLIPS[0]) as d:
            landmarks, blendshapes = d["landmarks"], d["blendshapes"]
            names, width, height = list(d["blendshape_names"]), int(d["width"]), int(d["height"])
        reference = features.geometric_features(landmarks, blendshapes, names, width, height)
        angle, scale = math.radians(17), 1.3
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        size = np.array([width, height, width], float)
        moved = (scale * (landmarks * size) @ rotation.T + [40.0, -25.0, 0.0]) / size
        again = features.geometric_features(moved.astype(np.float32), blendshapes, names, width, height)
        np.testing.assert_allclose(again, reference, atol=2e-4)

    def test_missing_face_frames_are_filled(self):
        with np.load(CLIPS[0]) as d:
            landmarks, blendshapes = d["landmarks"].copy(), d["blendshapes"].copy()
            names, width, height = list(d["blendshape_names"]), int(d["width"]), int(d["height"])
        landmarks[3], blendshapes[3] = np.nan, np.nan  # how capture.py stores a lost face
        geometry = features.geometric_features(landmarks, blendshapes, names, width, height)
        self.assertTrue(np.isfinite(geometry).all())
        np.testing.assert_array_equal(geometry[3], geometry[2])


@unittest.skipUnless(HAVE_FRONTEND, "models/auto_avsr/vsr.pth is required")
class FrontendTests(RealClipCase):
    def test_matches_upstream_auto_avsr_frontend(self):
        sys.path.insert(0, str(AUTO_AVSR))
        try:
            from espnet.nets.pytorch_backend.frontend.resnet import video_resnet
        finally:
            sys.path.remove(str(AUTO_AVSR))
        state = torch.load(AUTO_AVSR / "vsr.pth", map_location="cpu", weights_only=True)
        upstream = video_resnet().eval()
        upstream.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith("frontend.")})
        ours = copy.deepcopy(self.model.frontend).cpu().eval()
        roi = DualStreamPhonemeEncoder.prepare_roi(torch.from_numpy(self.take["mouth_roi_30hz"])[None])
        with torch.inference_mode():
            expected, actual = upstream(roi), ours(roi)
        self.assertEqual(actual.shape, (1, roi.shape[1], 512))
        self.assertLess((expected - actual).abs().max().item(), 1e-4)


class EncoderTests(RealClipCase):
    def test_dual_stream_shapes(self):
        out = self.run_model(self.clip["landmarks_30hz"], self.clip["mouth_roi_30hz"])
        frames = len(self.clip["landmarks_30hz"])
        self.assertEqual(out.log_probs.shape, (1, frames, 40))
        self.assertEqual(out.embeddings.shape, (1, frames, 256))
        self.assertEqual(out.gate.shape, (1, frames, 256))
        self.assertEqual(out.log_probs.device.type, DEVICE)
        totals = out.log_probs.exp().sum(-1).cpu()
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-4))
        self.assertTrue(((out.gate >= 0) & (out.gate <= 1)).all().item())

    def test_either_stream_may_be_absent(self):
        both = self.run_model(self.clip["landmarks_30hz"], self.clip["mouth_roi_30hz"]).log_probs
        roi_only = self.run_model(None, self.clip["mouth_roi_30hz"]).log_probs
        geometry_only = self.run_model(self.clip["landmarks_30hz"], None).log_probs
        self.assertEqual(roi_only.shape, both.shape)
        self.assertEqual(geometry_only.shape, both.shape)
        self.assertTrue(torch.isfinite(roi_only).all().item() and torch.isfinite(geometry_only).all().item())
        with self.assertRaises(ValueError):
            self.model(None, None)

    def test_bad_shapes_are_rejected(self):
        geometry, roi = tensor(self.clip["landmarks_30hz"]), tensor(self.clip["mouth_roi_30hz"])
        with self.assertRaises(ValueError):
            self.model(geometry[:, :-1], roi)
        with self.assertRaises(ValueError):
            self.model(geometry[..., :100], roi)
        with self.assertRaises(ValueError):
            self.model(None, roi[..., :80, :80])

    def test_uint8_and_unit_float_roi_agree(self):
        roi = self.take["mouth_roi_30hz"]
        a = self.run_model(None, roi).log_probs
        b = self.run_model(None, roi.astype(np.float32) / 255).log_probs
        self.assertLess((a - b).abs().max().item(), 1e-3)

    def test_padding_does_not_change_the_shorter_clip(self):
        short, long = self.short_clip, self.clip
        n_short, n_long = len(short["landmarks_30hz"]), len(long["landmarks_30hz"])
        self.assertLess(n_short, n_long)
        geometry = torch.zeros(2, n_long, features.D_GEOM)
        roi = torch.zeros(2, n_long, 1, 88, 88, dtype=torch.uint8)
        geometry[0], roi[0] = torch.from_numpy(long["landmarks_30hz"]), torch.from_numpy(long["mouth_roi_30hz"])
        geometry[1, :n_short] = torch.from_numpy(short["landmarks_30hz"])
        roi[1, :n_short] = torch.from_numpy(short["mouth_roi_30hz"])
        # Fill the padding with the long clip's frames: a leak would then show up.
        geometry[1, n_short:], roi[1, n_short:] = geometry[0, n_short:], roi[0, n_short:]
        with torch.inference_mode():
            batched = self.model(geometry.to(DEVICE), roi.to(DEVICE), torch.tensor([n_long, n_short]))
        alone = self.run_model(short["landmarks_30hz"], short["mouth_roi_30hz"])
        difference = (batched.log_probs[1, :n_short] - alone.log_probs[0]).abs().max().item()
        self.assertLess(difference, 2e-3)


class LatticeTests(RealClipCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with torch.inference_mode():
            out = cls.model(tensor(cls.clip["landmarks_30hz"]), tensor(cls.clip["mouth_roi_30hz"]))
        cls.log_probs = out.log_probs[0].cpu().double().numpy()

    def test_beam_search_equals_exhaustive_enumeration(self):
        # Seven real frames, restricted to blank plus the three likeliest phonemes, renormalized.
        frames = self.log_probs[:7]
        columns = [BLANK_ID] + list(np.argsort(-frames[:, 1:].mean(0))[:3] + 1)
        small = frames[:, columns]
        small = small - np.log(np.exp(small).sum(1, keepdims=True))
        exact = {}
        for path in product(range(4), repeat=len(small)):
            score = sum(small[t, s] for t, s in enumerate(path))
            collapsed = tuple(s for i, s in enumerate(path) if s and (i == 0 or s != path[i - 1]))
            exact[collapsed] = logaddexp(exact.get(collapsed, float("-inf")), score)
        ranked = sorted(exact.items(), key=lambda kv: -kv[1])
        wide = prefix_beam_search(small, beam_width=10 ** 6, token_limit=4, blank_skip=1.0, min_token_prob=0)
        self.assertEqual(len(wide), len(exact))
        for (sequence, score), (expected_sequence, expected_score) in zip(wide[:25], ranked[:25]):
            self.assertEqual(sequence, expected_sequence)
            self.assertAlmostEqual(score, expected_score, places=9)
        self.assertAlmostEqual(logaddexp(*[s for _, s in ranked[:2]]),
                               logaddexp(*[s for _, s in wide[:2]]), places=9)
        # A width-16 beam can only lose probability mass, never invent it.
        for sequence, score in prefix_beam_search(small, beam_width=16, token_limit=4, blank_skip=1.0,
                                                  min_token_prob=0):
            self.assertLessEqual(score, exact[sequence] + 1e-9)

    def test_lattice_structure(self):
        lattice = PhonemeLatticeDecoder(beam_width=16, top_k=5).decode(self.log_probs)
        self.assertEqual(lattice.num_frames, len(self.log_probs))
        self.assertEqual(lattice.frame_rate, 30.0)
        self.assertTrue(1 <= len(lattice.hypotheses) <= 5)
        scores = [h.log_prob for h in lattice.hypotheses]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len({h.phonemes for h in lattice.hypotheses}), len(lattice.hypotheses))
        for hypothesis in lattice.hypotheses:
            self.assertEqual(len(hypothesis.frames), len(hypothesis.phonemes))
            previous_end, previous = -1, None
            for phoneme, (start, end) in zip(hypothesis.phonemes, hypothesis.frames):
                self.assertTrue(0 <= start <= end < lattice.num_frames)
                # A repeated phoneme needs a blank frame between its two spans.
                self.assertGreater(start, previous_end + (phoneme == previous))
                previous_end, previous = end, phoneme
        json.dumps(lattice.to_dict())

    def test_slots_keep_the_whole_homophene_class(self):
        lattice = PhonemeLatticeDecoder().decode(self.log_probs)
        self.assertEqual(len(lattice.slots), len(lattice.best))
        for phoneme, slot, span in zip(lattice.best, lattice.slots, lattice.hypotheses[0].frames):
            names = [p for p, _ in slot.alternatives]
            self.assertEqual(slot.frames, span)
            self.assertIn(phoneme, names)
            self.assertTrue(set(homophenes(phoneme)) <= set(names), (phoneme, names))
            scores = [lp for _, lp in slot.alternatives]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(all(lp <= 1e-9 for lp in scores))

    def test_forced_alignment_of_a_repeat(self):
        ids = [5, 5, 9]
        spans = forced_alignment(self.log_probs[:12], ids)
        self.assertEqual(len(spans), 3)
        self.assertGreater(spans[1][0], spans[0][1] + 1)
        with self.assertRaises(ValueError):
            forced_alignment(self.log_probs[:3], [5, 5, 9])  # needs at least four frames

    def test_blank_frames_give_an_empty_lattice(self):
        # The same real posterior with the blank logit raised: what a no-speech clip looks like.
        silent = self.log_probs.copy()
        silent[:, BLANK_ID] += 50
        silent -= np.log(np.exp(silent).sum(1, keepdims=True))
        lattice = PhonemeLatticeDecoder().decode(silent)
        self.assertTrue(lattice.is_empty)
        self.assertEqual(lattice.slots, [])
        self.assertEqual(lattice.to_dict()["hypotheses"][0]["phonemes"], [])

    def test_rejects_wrong_vocabulary(self):
        with self.assertRaises(ValueError):
            PhonemeLatticeDecoder().decode(self.log_probs[:, :30])


class LoRATests(RealClipCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        session = sorted({folder.parent.name for folder, _ in TAKES})[0]
        cls.examples = train_ctc.load_examples({session})
        cls.base_state = copy.deepcopy(cls.model.state_dict())
        cls.trained = build_encoder().freeze_frontend()
        cls.adapter = LoRAAdapter(cls.trained, rank=8, alpha=16)
        cls.trained.to(DEVICE)
        cls.loss_before = train_ctc.mean_loss(cls.trained, cls.examples, DEVICE)
        cls.history = train_ctc.train_epochs(cls.trained, cls.adapter.parameters(), cls.examples, epochs=1,
                                             lr=1e-3, batch_size=8, device=DEVICE, seed=0, log=lambda _: None)
        cls.loss_after = train_ctc.mean_loss(cls.trained, cls.examples, DEVICE)

    def test_fresh_adapter_leaves_outputs_unchanged(self):
        model = build_encoder()
        before = self.run_model(None, self.take["mouth_roi_30hz"], model).log_probs
        adapter = LoRAAdapter(model, rank=4, alpha=8)
        model.to(DEVICE)
        after = self.run_model(None, self.take["mouth_roi_30hz"], model).log_probs
        self.assertLess((before - after).abs().max().item(), 1e-5)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name or name.startswith("ctc_head") for name in trainable), trainable)
        self.assertFalse(any(name.startswith("frontend") for name in trainable))
        self.assertLess(sum(p.numel() for p in adapter.parameters()),
                        0.1 * sum(p.numel() for p in model.parameters()))

    def test_one_epoch_on_real_takes_lowers_ctc_loss(self):
        self.assertEqual(len(self.history), 1)
        self.assertTrue(math.isfinite(self.loss_after))
        self.assertLess(self.loss_after, self.loss_before)

    def test_base_weights_stay_frozen(self):
        state = self.trained.state_dict()
        for name, value in self.base_state.items():
            if name.startswith("ctc_head"):
                continue
            parent, _, leaf = name.rpartition(".")
            current = state.get(name, state.get(f"{parent}.base.{leaf}"))
            self.assertTrue(torch.equal(current.cpu(), value.cpu()), name)

    def test_save_load_round_trip(self):
        expected = self.run_model(None, self.take["mouth_roi_30hz"], self.trained).log_probs
        with TemporaryDirectory() as directory:
            self.adapter.save(directory)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["adapter.safetensors", "adapter_config.json"])
            fresh = build_encoder()
            self.assertGreater((self.run_model(None, self.take["mouth_roi_30hz"], fresh).log_probs
                                - expected).abs().max().item(), 1e-3)
            LoRAAdapter.load(fresh, directory)
            fresh.to(DEVICE).eval()
            restored = self.run_model(None, self.take["mouth_roi_30hz"], fresh).log_probs
        self.assertLess((restored - expected).abs().max().item(), 1e-5)

    def test_merge_matches_and_unmerge_restores(self):
        expected = self.run_model(None, self.take["mouth_roi_30hz"], self.trained).log_probs
        self.adapter.merge()
        try:
            merged = self.run_model(None, self.take["mouth_roi_30hz"], self.trained).log_probs
            with self.assertRaises(RuntimeError), TemporaryDirectory() as directory:
                self.adapter.save(directory)
        finally:
            self.adapter.unmerge()
        restored = self.run_model(None, self.take["mouth_roi_30hz"], self.trained).log_probs
        self.assertLess((merged - expected).abs().max().item(), 1e-3)
        self.assertLess((restored - expected).abs().max().item(), 1e-3)


class DecoderTests(RealClipCase):
    def test_missing_encoder_weights_raise(self):
        with TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            OpenVocabLipDecoder(checkpoint_dir=directory)

    def test_checkpoint_dir_round_trip_and_lattice(self):
        with TemporaryDirectory() as directory:
            self.model.save(directory)
            decoder = OpenVocabLipDecoder(checkpoint_dir=directory, device=DEVICE)
            self.assertTrue(decoder.trained)
            lattice, encoder_ms, lattice_ms = decoder.lattice(self.clip["landmarks_30hz"],
                                                              self.clip["mouth_roi_30hz"])
        expected = PhonemeLatticeDecoder().decode(
            self.run_model(self.clip["landmarks_30hz"], self.clip["mouth_roi_30hz"]).log_probs[0])
        self.assertEqual(lattice.best, expected.best)
        self.assertGreater(encoder_ms, 0)
        self.assertGreater(lattice_ms, 0)

    def test_input_validation(self):
        with TemporaryDirectory() as directory:
            decoder = OpenVocabLipDecoder(checkpoint_dir=directory, device=DEVICE, allow_untrained=True)
        self.assertFalse(decoder.trained)
        roi = self.take["mouth_roi_30hz"]
        with self.assertRaises(ValueError):
            decoder.lattice(None, roi.astype(np.float32))  # 0-255 floats are ambiguous: refuse
        with self.assertRaises(ValueError):
            decoder.lattice(None, roi[:, :, :80, :80])
        lattice, _, _ = decoder.lattice(None, roi[:, 0])  # (T, 88, 88) is accepted
        self.assertEqual(lattice.num_frames, len(roi))

    def test_no_speech_returns_empty_text_without_the_language_model(self):
        class MustNotRun:
            def reconstruct(self, lattice, screen_context):
                raise AssertionError("the reconstructor ran on an empty lattice")

        with TemporaryDirectory() as directory:
            decoder = OpenVocabLipDecoder(checkpoint_dir=directory, device=DEVICE, allow_untrained=True,
                                          reconstructor=MustNotRun())
        with torch.no_grad():
            decoder.encoder.ctc_head.bias[BLANK_ID] += 50  # an encoder that only ever emits blank
        result = decoder.decode(None, self.take["mouth_roi_30hz"], {"app_name": "Terminal"})
        self.assertEqual(sorted(result), ["latency_ms", "phoneme_lattice", "text"])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["phoneme_lattice"]["slots"], [])
        self.assertGreater(result["latency_ms"], 0)


@unittest.skipUnless(DEVICE == "mps", "Apple Silicon MPS is required for the latency budget")
class LatencyTests(RealClipCase):
    def test_encoder_plus_lattice_under_120_ms_on_real_clips(self):
        with TemporaryDirectory() as directory:
            decoder = OpenVocabLipDecoder(checkpoint_dir=directory, device="mps", allow_untrained=True)
        inputs = [(c["name"], c["landmarks_30hz"], c["mouth_roi_30hz"])
                  for c in (features.load_session_clip(p) for p in CLIPS[::max(1, len(CLIPS) // 6)][:6])]
        inputs += [(t["name"], None, t["mouth_roi_30hz"])
                   for t in (features.load_silent_take(*pair) for pair in TAKES[::max(1, len(TAKES) // 10)][:10])]
        decoder.lattice(*inputs[0][1:])  # one-off MPS start-up
        rows = []
        for name, landmarks, roi in inputs:
            first = sum(decoder.lattice(landmarks, roi)[1:])  # may include a new-length compile
            warm = [sum(decoder.lattice(landmarks, roi)[1:]) for _ in range(5)]
            rows.append((name, len(roi), first, statistics.median(warm), max(warm)))
        for name, frames, first, median, worst in rows:
            print(f"\n  {name:34s} {frames:4d} frames  first {first:6.1f}  median {median:6.1f}  max {worst:6.1f} ms",
                  end="")
        print()
        for name, frames, first, median, worst in rows:
            self.assertLess(median, LATENCY_BUDGET_MS, name)
            self.assertLess(first, LATENCY_BUDGET_MS, f"{name}: first call at a new length")


if __name__ == "__main__":
    unittest.main()
