"""Headless mode and dispatcher checks. No camera or desktop input events."""
from concurrent.futures import Future
from pathlib import Path
from queue import SimpleQueue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock, patch
import wave

import handsfree_mac_agent as h


class InlinePool:
    def submit(self, fn, *args):
        future = Future()
        future.set_result(fn(*args))
        return future


class OpenVocabRoutingTests(unittest.TestCase):
    def setUp(self):
        from silent_avsr import features

        fixtures = [pair for pair in features.silent_takes() if pair[1]["phrase"] != "__NULL__"]
        if not fixtures:
            self.skipTest("requires a recorded silent take")
        clip = features.load_silent_take(*fixtures[-1])
        self.args = clip["landmarks_30hz"], clip["mouth_roi_30hz"], {"app_name": "Safari"}
        h.optional_lip_decoder.cache_clear()
        self.addCleanup(h.optional_lip_decoder.cache_clear)

    def test_checkpoint_dict_and_empty_text_use_primary(self):
        with (patch("silent_avsr.OpenVocabLipDecoder") as decoder,
              patch.object(h, "fallback_lip_decoder") as fallback,
              patch.object(h.sys, "stderr") as log):
            decoder.return_value.decode.side_effect = [{"text": "  Open notes  "}, {"text": ""}]
            self.assertEqual(h.decode_open_vocabulary_utterance(*self.args), "Open notes")
            self.assertEqual(h.decode_open_vocabulary_utterance(*self.args), "")
            decoder.assert_called_once_with(checkpoint_dir=h.MODELS / "open_vocab_lip")
            self.assertIs(decoder.return_value.decode.call_args.args[1], self.args[1])
            fallback.assert_not_called()
            self.assertTrue(any("Using OpenVocabLipDecoder" in call.args[0] for call in log.write.call_args_list))

    def test_missing_checkpoint_logs_the_fallback(self):
        with (patch("silent_avsr.OpenVocabLipDecoder", side_effect=FileNotFoundError("missing encoder")),
              patch.object(h, "fallback_lip_decoder") as fallback,
              patch.object(h.sys, "stderr") as log):
            fallback.return_value.return_value = "  open notes  "
            self.assertEqual(h.decode_open_vocabulary_utterance(*self.args), "open notes")
            self.assertEqual(fallback.return_value.call_args.args[0].shape[1:], (96, 96))
            self.assertTrue(any("Using Auto-AVSR fallback" in call.args[0] and "missing encoder" in call.args[0]
                                for call in log.write.call_args_list))

    def test_silent_route_uses_the_trained_visual_stream(self):
        clip = Path("data/session_1/top_left_00.npz")
        if not clip.is_file():
            self.skipTest("requires recorded face observations")
        with h.np.load(clip) as recorded:
            valid = recorded["face_present"]
            width, height = int(recorded["width"]), int(recorded["height"])
            features = h.np.stack([h.lip_geometry(lm, width, height, shapes) for lm, shapes in
                                   zip(recorded["landmarks"][valid], recorded["blendshapes"][valid])])
        context = {"ask_action": True}
        roi = self.args[1][:, 0]
        segment = h.Segment(0, 0, 0, None, roi, None, None,
                            landmarks_30hz=h.train.derivatives(features), screen_context=context)
        with patch.object(h, "decode_open_vocabulary_utterance", return_value="open notes") as decode:
            self.assertEqual(h.Decoder.silent(None, segment)[0], "open notes")
        decode.assert_called_once()
        landmarks, crops, screen = decode.call_args.args
        self.assertIsNone(landmarks)
        self.assertIs(crops, roi)
        self.assertIs(screen, context)


class ModeTests(unittest.TestCase):
    def setUp(self):
        cfg = json.loads(h.bench.TRIGGER_CONFIG_PATH.read_text())
        self.cap = h.Capture(cfg, [], [], SimpleQueue(), 1, (800, 600), None)
        stream = self.cap.stream = Mock(active=True)
        stream.start.side_effect = lambda: setattr(stream, "active", True)
        stream.stop.side_effect = lambda: setattr(stream, "active", False)
        self.ctrl = h.Controller(self.cap, Mock(), h.Actions(True, {}, (800, 600)),
                                 InlinePool(), None, 70, False,
                                 SimpleNamespace(calibration=SimpleNamespace(has_calibrated=True)))

    def test_sensor_modes(self):
        folder, _ = next(h.train.clips(h.train.TRAIN))
        with wave.open(str(folder / "audio_16k.wav")) as wav:
            pcm = h.np.frombuffer(wav.readframes(320), dtype="<i2")[:, None]
        clip = next(Path("data/session_3").glob("top_left_*.npz"))
        with h.np.load(clip) as d:
            jaw = d["blendshapes"][:, list(d["blendshape_names"]).index("jawOpen")]
            gaps = h.bench.lip_gap({"lm": d["landmarks"]})
            frames = list(zip(d["timestamps_ms"], jaw, gaps))
        for mode in (*h.ACTIVE_MODES, "SLEEP"):
            self.ctrl.change_mode(mode)
            self.assertEqual(self.cap.stream.active, mode == "ACTIVE")
            self.cap.audio_callback(pcm, len(pcm), SimpleNamespace(inputBufferAdcTime=0, currentTime=0), "")
            self.assertEqual(bool(self.cap.audio), self.cap.stream.active)
            self.cap.reset()
            self.cap.armed = True
            fired = [self.cap.jaw_click(t, j >= self.cap.cfg["jawOpen"] and g >= self.cap.cfg["lip_gap"])
                     for t, j, g in frames]
            self.assertEqual(any(v is not None for v in fired), mode in {"ACTIVE", "STEALTH"})
        self.ctrl.toggle()
        self.assertEqual(self.cap.control_mode, "STEALTH")

    def test_active_segment_uses_audio_from_the_recorded_preroll(self):
        folder = Path("data/silent_speech/20261119-212523-861351/11_whisper")
        meta = json.loads((folder / "meta.json").read_text())
        with wave.open(str(folder / "audio_16k.wav")) as wav:
            pcm = h.np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        self.cap.reset()
        self.cap.ring.extend(zip(meta["source_frame_times_ms"], h.np.load(folder / "mouth_96x96.npy"),
                                 h.np.load(folder / "blendshapes.npy")))
        self.cap.audio.append((meta["audio_start_ms"] / 1000, pcm, ""))
        self.cap.onset = meta["vad_segments_ms"][0][0]
        self.cap.release = meta["vad_segments_ms"][-1][1]
        segment = self.cap.segment()
        left = max(0, round((self.cap.onset - h.LIP_PADDING_MS - meta["audio_start_ms"]) * 16))
        right = round((self.cap.release + h.LIP_PADDING_MS - meta["audio_start_ms"]) * 16)
        self.assertAlmostEqual(segment.start, meta["audio_start_ms"] + left / 16)
        self.assertGreater(len(segment.pcm), 0)
        h.np.testing.assert_array_equal(segment.pcm, pcm[left:right])

    def test_eye_onboarding_rejects_movement_and_reloads_nine_point_calibration(self):
        from gazefollower.face_alignment import MediaPipeFaceAlignment
        from gazefollower.gaze_estimator import MGazeNetGazeEstimator

        aligner = MediaPipeFaceAlignment(enable_filter=True)
        self.addCleanup(aligner.release)
        estimator = MGazeNetGazeEstimator()
        self.addCleanup(estimator.release)
        self.cap.eye_onboarding = True
        root = h.bench.ROOT / "data"
        start = round(h.time.monotonic() * 1000)
        flow = h.EyeOnboarding(start / 1000)

        def replay(clip, phase, *, check_dropout=False):
            nonlocal start
            video = h.cv2.VideoCapture(str(clip.with_suffix(".mp4")))
            phase_start, gap_until, paused_at = start, None, 0.
            try:
                with h.np.load(clip.with_suffix(".npz")) as data:
                    aspect = int(data["width"]) / int(data["height"])
                    seen = list(zip(data["landmarks"], data["head_matrix"], data["face_present"]))
                    for (landmarks, matrix, present), offset in zip(seen, data["timestamps_ms"]):
                        ok, frame = video.read()
                        self.assertTrue(ok, clip)
                        t = phase_start + int(offset)
                        if gap_until is not None and t < gap_until:
                            continue
                        resumed = gap_until is not None and check_dropout
                        if resumed:
                            self.assertFalse(flow.eyes_detected(t / 1000))
                            self.assertEqual(flow.elapsed, paused_at)
                        rgb = h.cv2.cvtColor(frame, h.cv2.COLOR_BGR2RGB)
                        eyes = aligner.detect(t * 1_000_000, rgb)
                        self.cap.last_ts, self.cap.vertical = t, None
                        if present:  # what Capture.on_frame measures from FaceLandmarker during onboarding
                            self.cap.measure_vertical(t, landmarks, matrix, aspect)
                        self.cap.on_gaze(eyes, estimator.detect(rgb, eyes))
                        _, kind, sample = self.cap.events.get_nowait()
                        self.assertEqual(kind, "eye_sample")
                        flow.record(*sample)
                        if resumed:
                            self.assertEqual(flow.elapsed, 0.)
                            check_dropout = False
                        elif check_dropout and flow.elapsed > 0.:
                            paused_at = flow.elapsed
                            gap_until = t + 320
                        if flow.phase != phase:
                            start = t
                            return True
                start = t
                return False
            finally:
                video.release()

        self.assertTrue(replay(root / "session_1/neg_blink_00", "face"))
        self.assertEqual(flow.phase, "dot")
        self.assertEqual(flow.remaining(start / 1000), flow.DOT_SECONDS)
        start += 500
        self.assertFalse(replay(root / "session_1/neg_look_down_00", "dot"))
        self.assertFalse(flow.anchors)
        self.assertTrue(flow.moved)
        start += 500
        flow.begin("dot", start / 1000)
        self.assertTrue(replay(root / "session_1/top_right_00", "dot", check_dropout=True))
        self.assertEqual(len(flow.anchors), 1)
        self.assertEqual(len(flow.vertical_anchors()[0]), 3)
        flow.anchors.clear()
        flow.vertical.clear()
        # The corpus has corner and center clips. Use distinct recorded
        # fixations to exercise the nine transitions and persistence only.
        clips = ("session_1/bottom_left_01", "session_1/bottom_left_02", "session_1/bottom_right_01",
                 "session_1/neg_blink_00", "session_1/top_left_00", "session_1/top_left_01",
                 "session_1/top_right_00", "session_2/bottom_right_01", "session_3/bottom_right_02")
        start += 500
        flow.begin("dot", start / 1000)
        for index, label in enumerate(clips):
            self.assertEqual(flow.index, index)
            self.assertTrue(replay(root / label, "dot"), label)
            self.assertEqual(flow.phase, "flash")
            self.assertEqual(flow.remaining(start / 1000), 0.0)
            self.assertEqual(len(flow.anchors), index + 1)
            flow.advance(start / 1000 + flow.FLASH_SECONDS / 2)
            self.assertEqual(flow.phase, "flash")
            start += 500
            flow.advance(start / 1000)
        self.assertEqual(flow.phase, "save")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trigger_config.json"
            path.write_bytes(h.bench.TRIGGER_CONFIG_PATH.read_bytes())
            vertical = flow.vertical_anchors()
            self.assertEqual(h.np.shape(vertical), (9, 3))
            fitted, cfg = h.save_eye_calibration(path, flow.anchors, self.cap.bounds, vertical)
            reloaded = h.saved_eye_model(h.load_trigger_config(path), self.cap.bounds)
            self.assertEqual(cfg["eye_calibration"]["vertical"], vertical)
            self.assertEqual(reloaded.eye_open, h.EYE_OPEN_SHARE * min(v[2] for v in vertical))
            self.assertEqual(reloaded.vertical is None, fitted.vertical is None)
            # A calibration saved before this change has no vertical anchors and still loads.
            del cfg["eye_calibration"]["vertical"]
            legacy = h.saved_eye_model(cfg, self.cap.bounds)
            self.assertIsNone(legacy.vertical)
            self.assertFalse(hasattr(legacy, "eye_open"))
            self.assertEqual(legacy.predict(flow.anchors[0], None), fitted.predict(flow.anchors[0], None))
            self.assertEqual({key: value for key, value in self.cap.cfg.items() if key != "eye_calibration"},
                             {key: value for key, value in cfg.items() if key != "eye_calibration"})
            self.assertEqual(reloaded.anchors[1].tolist(),
                             [[.5, .5], [.15, .15], [.5, .15], [.85, .15], [.85, .5], [.85, .85],
                              [.5, .85], [.15, .85], [.15, .5]])
            self.assertEqual(reloaded.spec[0], "eye_quadratic")
            self.assertEqual(reloaded.predict(flow.anchors[0], None), fitted.predict(flow.anchors[0], None))
        flow.begin("complete", start / 1000)
        self.assertEqual(flow.remaining(start / 1000), 3.0)
        self.assertEqual(flow.remaining(start / 1000 + 1.5), 1.5)
        self.assertEqual(flow.remaining(start / 1000 + 3.0), 0.0)

    def recorded_gaze(self, clip, start, aligner, estimator, model=None):
        """(ms, capture events) for the fixation part of a session clip through Capture.on_gaze. The
        FaceLandmarker output saved beside the video stands in for Capture.on_frame."""
        with h.np.load(clip.with_suffix(".npz")) as data:
            aspect = int(data["width"]) / int(data["height"])
            jaw = data["blendshapes"][:, list(data["blendshape_names"]).index("jawOpen")]
            frames = list(zip(data["timestamps_ms"].astype(int), data["cue"], data["landmarks"], data["head_matrix"],
                              data["face_present"], jaw))
        video = h.cv2.VideoCapture(str(clip.with_suffix(".mp4")))
        self.addCleanup(video.release)
        for offset, cue, landmarks, matrix, present, jaw_open in frames:
            ok, frame = video.read()
            if not ok or cue == "gesture":
                break
            t = start + int(offset)
            rgb = h.cv2.cvtColor(frame, h.cv2.COLOR_BGR2RGB)
            self.cap.last_ts, self.cap.present, self.cap.jaw, self.cap.vertical = t, bool(present), float(jaw_open), None
            if present:
                self.cap.measure_vertical(t, landmarks, matrix, aspect)
            eyes = aligner.detect(t * 1_000_000, rgb)
            info = estimator.detect(rgb, eyes)
            if model is not None and info.status and info.features is not None:
                ok, xy = model.predict(info.features, info.raw_gaze_coordinates)
                if ok:
                    info.calibrated_gaze_coordinates = h.np.multiply(xy, self.cap.bounds)
            self.cap.on_gaze(eyes, info)
            events = []
            while not self.cap.events.empty():
                events.append(self.cap.events.get_nowait()[1:])
            yield t, events

    def test_iris_and_eyelid_signals_on_recorded_fixations(self):
        for session in sorted(Path("data").glob("session_*")):
            iris, opening = {}, {}
            for clip in sorted(session.glob("*.npz")):
                with h.np.load(clip) as d:
                    aspect = int(d["width"]) / int(d["height"])
                    row = "centre" if str(d["label"]) == "neg_blink" else str(d["label"]).split("_")[0]
                    keep = d["face_present"] & (d["cue"] != "gesture")
                    keep[:6] = False
                    iris.setdefault(row, []).extend(h.iris_vertical(lm, aspect) for lm in d["landmarks"][keep])
                    opening.setdefault(row, []).extend(h.eye_opening(lm, aspect) for lm in d["landmarks"][keep])
            with self.subTest(session=session.name):
                # Iris offset over eye-corner width: the bottom dots sit apart from the top and centre.
                bottom = h.np.median(iris["bottom"])
                self.assertGreater(bottom - h.np.median(iris["top"]), .01)
                self.assertGreater(bottom - h.np.median(iris["centre"]), .01)
                # Eyelid opening: nearly every bottom-dot frame is above the blink threshold, and
                # the closed-eye frames of the blink clips, where the iris reads lower than on
                # any bottom dot, are below it.
                self.assertGreater(h.np.percentile(opening["bottom"], 5), h.EYE_OPEN_DEFAULT)
                closed = h.np.array(opening["centre"])[h.np.array(iris["centre"]) > 0]
                self.assertGreater(len(closed), 10)
                self.assertGreater(h.np.mean(closed < h.EYE_OPEN_DEFAULT), .9)

    def test_gaze_paced_scroll_and_click_drift_on_recorded_clips(self):
        from gazefollower.face_alignment import MediaPipeFaceAlignment
        from gazefollower.gaze_estimator import MGazeNetGazeEstimator

        aligner = MediaPipeFaceAlignment(enable_filter=True)
        self.addCleanup(aligner.release)
        estimator = MGazeNetGazeEstimator()
        self.addCleanup(estimator.release)
        session, start = Path("data/session_1"), round(h.time.monotonic() * 1000)

        # Calibrate on the session's first repetition, as onboarding stores its anchors.
        self.cap.eye_onboarding = True
        anchors, verticals, targets = [], [], []
        for label in ("top_left", "top_right", "bottom_left", "bottom_right", "neg_blink", "neg_swallow"):
            seen = [value for _, events in self.recorded_gaze(session / f"{label}_00", start, aligner, estimator)
                    for kind, value in events if kind == "eye_sample" and value[1] is not None and value[4] is not None][6:]
            anchors.append(h.np.median([value[1] for value in seen], axis=0).tolist())
            verticals.append(h.np.median([value[4] for value in seen], axis=0).tolist())
            targets.append(h.recorded_target(label))
            start += 10_000
        self.cap.eye_onboarding = False
        model = h.calibrated_eye_model(anchors, verticals, targets)
        self.assertIsNotNone(model.vertical)
        self.assertLessEqual(model.vertical.error, model.vertical.GAIN * model.vertical_baseline)
        self.cap.use_eye_model(model)
        self.cap.eye_tracking = True
        self.assertEqual(self.cap.eye_open, h.EYE_OPEN_SHARE * min(v[2] for v in verticals))

        ctrl, pacer, (width, height) = self.ctrl, self.ctrl.pacer, self.cap.bounds
        screen = (0, 0, width, height)
        ctrl.index.publish(ctrl.index.generation, (1, "Replay", "Recorded clips", (), (None, screen)))
        ctrl.reading_screens, ctrl.auto_scroll = (screen,), True
        clips, fixations = {}, []
        for label in ("neg_blink_01", "bottom_left_01", "neg_swallow_01", "top_left_01"):
            steps, heights, held = [], [], []
            clips[label] = steps, heights, held
            for t, events in self.recorded_gaze(session / label, start, aligner, estimator, model):
                for kind, value in events:
                    if kind in ("spatial", "lost"):
                        ctrl.event(kind, value)
                    if kind == "spatial" and value[1] is not None and label == "bottom_left_01":
                        fixations.append(value)
                smooth = ctrl.smooth[1] / height if ctrl.smooth is not None and ctrl.last_seen == t / 1000 else None
                if smooth is not None:
                    heights.append((t, smooth))
                # A raised hand held for Ask / Action pauses scrolling for as long as it is up.
                hand = label == "bottom_left_01" and len(heights) in range(20, 26)
                self.cap.hand_hold.pose = "hand_action" if hand else None
                for now in range(t, t + 50, 10):
                    ctrl.index.focus_at, ctrl.last_scroll = now / 1000, None
                    ctrl.update_auto_scroll(now / 1000)
                    if ctrl.last_scroll is not None:
                        steps.append((now, *ctrl.last_scroll))
                        held.append(hand)
                        # Nothing is posted while the filtered gaze reads inside the middle zone.
                        self.assertFalse(smooth is not None and pacer.QUIET[0] < smooth < pacer.QUIET[1], (label, now))
                last = t
            start = last + 50

        steps, heights, _ = clips["neg_blink_01"]
        self.assertEqual(steps, [])
        steps, heights, held = clips["bottom_left_01"]
        self.assertGreaterEqual(h.np.median([y for _, y in heights]), .85)
        self.assertTrue(steps and all(amount < 0 for _, amount, _ in steps))
        entered = next(t for t, y in heights if y >= pacer.DOWN)
        self.assertGreaterEqual(steps[0][0] - entered, 1000 * pacer.DWELL)
        self.assertLess(steps[0][0] - entered, 1000 * pacer.DWELL + 100)
        self.assertFalse(any(held))
        # The raised hand splits the clip into two scroll runs. Each run posts at one point inside
        # the window, and the second starts only after a fresh dwell.
        raised = [t for t, _ in heights][19:25]
        before = [step for step in steps if step[0] < raised[0]]
        after = [step for step in steps if step[0] > raised[-1]]
        self.assertEqual(len(before) + len(after), len(steps))
        for run in (before, after):
            self.assertEqual(len({point for _, _, point in run}), 1)
            self.assertTrue(h.contains(screen, run[0][2]))
        self.assertGreaterEqual(after[0][0] - raised[-1], 1000 * pacer.DWELL)
        gaps = h.np.diff([now for now, _, _ in before])
        self.assertLessEqual(1000 / h.np.median(gaps), pacer.RATE + 1)
        self.assertGreaterEqual(1000 / h.np.median(gaps), 30)
        steps, heights, _ = clips["neg_swallow_01"]
        middle = next(t for t, y in heights if pacer.QUIET[0] < y < pacer.QUIET[1])
        self.assertFalse([now for now, _, _ in steps if now >= middle and now - middle < 1000 * pacer.DWELL])
        steps, heights, _ = clips["top_left_01"]
        self.assertTrue(steps and all(amount > 0 for _, amount, _ in steps))

        # A jaw click that lands on the control the fixation snapped to teaches the drift correction.
        run = [value for value in fixations if value[1] == fixations[-1][1]]
        self.assertGreaterEqual(run[-1][2] - run[0][2], .15)
        fixed = run[-1][1]
        item = h.IndexedElement(object(), "AXButton", (fixed[0] + 20, fixed[1] - 60, 40, 20), "Recorded fixation")
        ctrl.index.publish(ctrl.index.generation, (1, "Replay", "Recorded clips", (item,), (None, screen)))
        ctrl.fixated = None
        ctrl.event("spatial", run[0])
        ctrl.event("spatial", run[-1])
        self.assertIs(ctrl.snapped, item)
        ctrl.event("click", fixed)
        ctrl.click_cached(item, 1)
        ctrl.finish_click()
        residual = h.np.subtract(item.center, fixed)
        learned = h.np.array(self.cap.drift.correction(fixed))
        self.assertTrue(h.np.all(learned * residual > 0))
        self.assertTrue(h.np.all(abs(learned) <= h.DRIFT_LEARNING_RATE * abs(residual) + 1e-9))
        corrected = self.cap.drift.apply(fixed)
        self.assertLess(h.math.dist(self.cap.drift.invert(corrected), fixed), .05)

    def test_dispatch_without_gaze(self):
        self.ctrl.decoder.recognize.return_value = ("talking mode", "")
        self.ctrl.event("listen", None)
        self.assertIsNone(self.ctrl.anchor.point)
        self.ctrl.event("segment", object())
        self.ctrl.completed()
        self.assertEqual(self.cap.control_mode, "ACTIVE")
        self.ctrl.decoder.recognize.return_value = ("open Safari", "")
        with patch.object(h, "window_index", return_value=([], {})):
            self.ctrl.event("listen", None)
            self.ctrl.event("segment", object())
            self.ctrl.completed()
            self.ctrl.tick()
        self.assertEqual(self.ctrl.history[0]["action"], {"tool": "open_app", "app_name": "Safari"})

    def test_clipboard(self):
        board = h.K.NSPasteboard.pasteboardWithUniqueName()
        kind = h.K.NSPasteboardTypeString
        self.addCleanup(board.releaseGlobally)
        board.setString_forType_("previous", kind)
        pasted = []
        def shortcut(key, flags):
            if key == 8:
                board.clearContents()
                board.setString_forType_("selection", kind)
            elif key == 9:
                pasted.append(board.stringForType_(kind))
        with patch.object(h, "keypress", shortcut):
            self.assertEqual(h.clipboard_shortcut(board, 8), "selection")
            h.clipboard_shortcut(board, 9, "rewritten")
        self.assertEqual(pasted, ["rewritten"])
        self.assertEqual(board.stringForType_(kind), "previous")

    def test_command(self):
        self.assertEqual(h.run_safe_command("osascript -e 'return 6 * 7'"), "Exit 0: 42")
        self.assertEqual(h.run_safe_command("echo success"), "Exit 0: success")
        for command in ("sudo ls", "rm file", "dd if=file", "chmod -R 777 .", "shutdown now"):
            with self.assertRaises(ValueError):
                h.run_safe_command(command)

    def test_two_step_observation(self):
        c = self.ctrl
        first = {"tool": "open_app", "app_name": "Safari"}
        second = {"tool": "open_url", "url": "youtube.com"}
        c.command = "open Safari and go to youtube.com"
        c.work_target = h.context(None)
        c.history, c.work_steps, c.work_total, c.work_next = [], [first, second], 2, 0
        c.decoder.generate.side_effect = [json.dumps({"steps": [second]}), '{"steps":[]}']
        with patch.object(h, "window_index", side_effect=[
                ([], {"window": title}) for title in ("initial", "Safari", "Safari", "YouTube")]):
            c.tick()
            self.assertAlmostEqual(c.observe_at - h.time.monotonic(), .55, delta=.05)
            c.observe_at = 0
            c.tick()
            c.tick()
            c.observe_at = 0
            c.tick()
            c.tick()
        for i, title in enumerate(("Safari", "YouTube")):
            messages = c.decoder.generate.call_args_list[i].args[0]
            info = json.loads(messages[-1]["content"])["context"]
            self.assertEqual(info["window"], title)
            self.assertEqual(len(info["history"]), i+1)
            self.assertIn("Would", info["history"][-1]["result"])
        self.assertEqual(c.mode, "READY")


if __name__ == "__main__":
    unittest.main()
