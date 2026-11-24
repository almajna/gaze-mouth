"""Offline sentence recorder and decoding checks using existing recorded video."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from eval.eval_wer import load_clip
from training.record_sentences import save_clip
from silent_avsr.features import DATA
from silent_avsr.sentence_video import Landmarks68, MEAN_FACE, PERIOD_NS, normalize_contrast

NEGATIVE = DATA / "session_1/neg_blink_00.mp4"


@unittest.skipUnless(NEGATIVE.is_file(), "requires the recorded negative clip")
class SentenceBlitzTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capture = cv2.VideoCapture(str(NEGATIVE))
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        with np.load(NEGATIVE.with_suffix(".npz"), allow_pickle=False) as archive:
            cls.timestamps = archive["timestamps_ms"].astype(np.int64) * 1_000_000
        detect = Landmarks68()
        cls.frames = np.asarray(frames)
        cls.landmarks = [detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in frames]
        cls.reference = np.load(MEAN_FACE, allow_pickle=False)

    def test_recorded_clip_round_trip(self):
        metadata = dict(kind="negative", sentence="", session=NEGATIVE.parent.name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "001.npz"
            save_clip(path, self.frames, self.timestamps, self.landmarks, self.reference, metadata)
            meta, crops = load_clip(path)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["raw_frames"], self.frames)
                np.testing.assert_array_equal(archive["raw_timestamps_ns"],
                                              self.timestamps - self.timestamps[0])
                np.testing.assert_array_equal(np.diff(archive["crop_timestamps_ns"]), PERIOD_NS)
                self.assertEqual(archive["landmarks_68"].shape, (len(self.frames), 68, 2))
            self.assertEqual(meta["kind"], "negative")
            self.assertEqual(crops.shape, ((self.timestamps[-1] - self.timestamps[0]) // PERIOD_NS + 1, 88, 88))
            video = cv2.VideoCapture(str(path.with_suffix(".mp4")))
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(video.get(cv2.CAP_PROP_FPS), 25.0)
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(crops))
            finally:
                video.release()
        normalized = normalize_contrast(crops)
        np.testing.assert_allclose(normalized, (crops.astype(np.float32) / 255.0 - .421) / .165)


if __name__ == "__main__":
    unittest.main()
