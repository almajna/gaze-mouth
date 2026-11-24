"""Replay recorded inputs and optionally benchmark the current macOS AX window."""

import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

from audio_arbitration import ContextHotwordBiaser, ModeArbitrator, SpeechSource, JAW_CLICK_MIN
from gaze_engine import GazeLoupe, fixation_threshold, recorded_head_rays, replay_fixations

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SPEECH = DATA / "silent_speech"
SPEECH_RUNS = ("20261119-211044-711080", "20261119-212523-861351")
GAZE_CLIP = DATA / "session_3" / "top_left_00.mp4"


def latency(samples):
    ordered = sorted(samples)
    return {
        "samples": len(ordered),
        "median_ms": round(median(ordered) / 1e6, 6),
        "p95_ms": round(ordered[math.ceil(.95 * len(ordered)) - 1] / 1e6, 6),
    }


def timed(call, *args):
    start = perf_counter_ns()
    value = call(*args)
    return value, perf_counter_ns() - start


def rate(events, total):
    return {"events": events, "total": total, "percent": round(100 * events / total, 2) if total else None}


def recorded_speech():
    for run in SPEECH_RUNS:
        for trial in range(1, 21):
            mode = "silent" if trial <= 10 else "whisper"
            folder = SPEECH / run / f"{trial:02}_{mode}"
            meta = json.loads((folder / "meta.json").read_text())
            if not meta["saved"] or (meta["run"], meta["trial"], meta["mode"]) != (run, trial, mode):
                raise ValueError(f"Unexpected speech recording metadata in {folder}")
            yield folder, meta


def speech_replay(rows):
    import numpy as np

    by_session = {}
    excluded = []
    for folder, meta in rows:
        if not (folder / "blendshapes.npy").is_file():
            excluded.append(str(folder.relative_to(SPEECH)))
            continue
        jaw = np.load(folder / "blendshapes.npy", allow_pickle=False)[:, meta["blendshape_names"].index("jawOpen")]
        times = np.asarray(meta["source_frame_times_ms"])
        if len(jaw) != len(times) or not np.isfinite(jaw).all() or (np.diff(times) < 0).any():
            raise ValueError(f"Invalid jaw observations in {folder}")
        arb = ModeArbitrator()
        candidates = clicks = unobserved_onsets = 0
        for onset, release in meta["vad_segments_ms"]:
            active = (times >= onset) & (times <= release)
            if not active.any():
                unobserved_onsets += 1
                continue
            segment = jaw[active]
            # Both capture modes record visual VAD edges. Audio detector edges
            # are unavailable, including on whispered clips.
            arb.speech_started(SpeechSource.VISUAL, float(segment[0]), onset)
            for t, j in zip(times[active], segment):
                candidates += int(j >= JAW_CLICK_MIN)
                clicks += arb.try_jaw_click(float(j), float(t))
            arb.speech_ended(SpeechSource.VISUAL, release)
        by_session.setdefault((meta["run"], meta["mode"]), []).append(
            (bool(clicks), bool(candidates), candidates, unobserved_onsets))
    if not by_session:
        raise ValueError("No speech recordings contain a jaw trace")
    return {
        "false_positive_clips": rate(sum(hit for group in by_session.values() for hit, *_ in group),
                                     sum(map(len, by_session.values()))),
        "excluded_missing_jaw_trace": excluded,
        "by_session_and_capture_mode": [
            {"session": run, "mode": mode,
             "false_positive_clips": rate(sum(hit for hit, *_ in group), len(group)),
             "false_positive_clips_with_candidates": rate(sum(hit for hit, *_ in group),
                                                          sum(risk for _, risk, *_ in group)),
             "wide_jaw_frames": sum(frames for _, _, frames, _ in group),
             "unobserved_onsets": sum(missing for _, _, _, missing in group)}
            for (run, mode), group in by_session.items()
        ],
    }


def gaze_sessions():
    import eval_offline as ev
    from live_app import load_trigger_config

    sessions = {name: ev.load_session(name)[0] for name in ev.DEFAULT_SESSIONS}
    cfg = load_trigger_config(ev.TRIGGER_CONFIG_PATH)
    negatives = [clip for clips in sessions.values() for clip in clips if clip["label"] in ev.NEG_LABELS]
    feats = {c["name"]: (c["jaw"], ev.lip_gap(c)) for c in negatives}
    scored = ev.eval_trigger(negatives, (cfg["jawOpen"], cfg["lip_gap"], cfg["debounce_ms"]), feats)
    negative_rate = rate(len({name for name, *_ in scored["fp"]}), scored["n_neg"])
    negative_rate["trigger_events"] = scored["fp"]
    corners = {name: [c for c in clips if c["label"] in ev.CORNERS] for name, clips in sessions.items()}
    cross = []
    cols = [0, 1, 2, 3]
    for source, train in corners.items():
        for target, test in corners.items():
            if source == target:
                continue
            result, reason = ev.gaze_split(train, test, cols)
            if result is None:
                raise ValueError(f"{source} -> {target}: {reason}")
            frames, clips, _, _ = result
            cross.append({
                "train": source, "test": target,
                "frame_errors": rate(int(frames.sum() - frames.trace()), int(frames.sum())),
                "clip_errors": rate(int(clips.sum() - clips.trace()), int(clips.sum())),
                "unscored_clips": len(test) - int(clips.sum()),
            })
    return negative_rate, cross, corners["session_2"]


def gaze_latency(train, repeats):
    import cv2
    import mediapipe as mp
    import numpy as np
    import eval_offline as ev
    from capture import MODEL_PATH, extract, make_landmarker

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    cols = [0, 1, 2, 3]
    features = [ev.valid_gaze(clip, cols)[0] for clip in train]
    labels = np.repeat([clip["label"] for clip in train], [len(x) for x in features])
    model = ev.nearest_centroid_fit(np.vstack(features), labels, cols)
    with np.load(GAZE_CLIP.with_suffix(".npz"), allow_pickle=False) as recording:
        times = recording["timestamps_ms"]
    pipeline, loupe = [], []
    for _ in range(repeats):
        video = cv2.VideoCapture(str(GAZE_CLIP))
        try:
            with make_landmarker() as detector:
                for i, timestamp in enumerate(times):
                    ok, frame = video.read()
                    if not ok:
                        raise ValueError(f"Missing frame {i} in {GAZE_CLIP}")
                    start = perf_counter_ns()
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    lm, _, _, matrix, present = extract(detector.detect_for_video(image, int(timestamp)))
                    if not present:
                        continue
                    features = ev.gaze_features(dict(lm=lm[None], hm=matrix[None],
                                                     aspect=frame.shape[1] / frame.shape[0]))
                    if not np.isfinite(features).all():
                        continue
                    ev.nearest_centroid_predict(model, features)
                    if i == 0:
                        continue
                    pipeline.append(perf_counter_ns() - start)
                    # Use two recorded landmarks in image-pixel coordinates.
                    center, target = lm[[468, 33], :2] * (frame.shape[1], frame.shape[0])
                    crop = GazeLoupe(*center)
                    start = perf_counter_ns()
                    crop.loupe_to_screen(*crop.screen_to_loupe(*target))
                    loupe.append(perf_counter_ns() - start)
        finally:
            video.release()
    return {"clip": str(GAZE_CLIP.relative_to(ROOT)), "mediapipe_iris_head": latency(pipeline),
            "loupe_coordinate_roundtrip": latency(loupe)}


def window_benchmark(repeats):
    import AppKit
    import ApplicationServices as ax
    from mac_ui_bridge import WindowIndexer, unwrap_ax

    if not ax.AXIsProcessTrusted():
        raise RuntimeError("Enable Accessibility for the launching terminal")
    workspace = AppKit.NSWorkspace.sharedWorkspace()

    def focused_window():
        app = workspace.frontmostApplication()
        if app is None:
            return None
        root = ax.AXUIElementCreateApplication(app.processIdentifier())
        ax.AXUIElementSetMessagingTimeout(root, .1)
        return unwrap_ax(ax.AXUIElementCopyAttributeValue(root, "AXFocusedWindow", None))

    index = WindowIndexer()
    entries = index.refresh()
    window = focused_window()
    if window is None or not entries:
        raise RuntimeError("Focus a window with accessible controls before measuring")
    prompt = ContextHotwordBiaser().initial_prompt(
        {"role": item.role, "name": item.title} for item in entries
    )
    samples, lookups, counts = [], [], []
    for _ in range(repeats):
        current, elapsed = timed(index.refresh)
        if focused_window() != window or not current:
            raise RuntimeError("Keep the same accessible window focused throughout the measurement")
        samples.append(elapsed)
        counts.append(len(current))
        for item in current:
            target, lookup_elapsed = timed(index.snap, item.center)
            if target is None or target.center != item.center:
                raise RuntimeError("Cached snap did not resolve an indexed center")
            lookups.append(lookup_elapsed)
    cached = latency(lookups)
    return {"implementation": "mac_ui_bridge.WindowIndexer.snap",
            "entry_counts": counts, "refresh": latency(samples), "cached_lookup": cached,
            "cached_p95_below_2_ms": cached["p95_ms"] < 2.0}, prompt


def silent_benchmark(rows, device, repeats):
    import numpy as np
    import torch
    from eval_silent_speech import resample_video, visual_decoder

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch cannot access MPS")
    torch.set_num_threads(4)
    decode = visual_decoder(device)
    clips = [(np.load(folder / "mouth_96x96.npy", allow_pickle=False), meta)
             for folder, meta in rows if meta["mode"] == "silent"]

    def run(video, meta):
        decode(resample_video(video, meta))
        if device == "mps":
            torch.mps.synchronize()

    run(*clips[0])
    return {"model": "Auto-AVSR open-vocabulary VSR", "device": device, "clips": len(clips),
            **latency([timed(run, *clip)[1] for _ in range(repeats) for clip in clips])}


def whisper_benchmark(rows, prompt, context, repeats):
    import mlx.core as mx
    from mlx_whisper import transcribe
    from eval_silent_speech import MODELS, audio_input, errors

    model = MODELS / "whisper-large-v3-turbo"
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model)
    clips = [(audio_input(folder, meta)[0], meta["phrase"])
             for folder, meta in rows if meta["mode"] == "whisper"]

    def run(audio, initial_prompt):
        result = transcribe(audio, path_or_hf_repo=str(model), language="en", temperature=0.,
                            condition_on_previous_text=False, verbose=None, initial_prompt=initial_prompt)
        mx.synchronize()
        return result["text"]

    run(clips[0][0], None)
    run(clips[0][0], prompt)
    samples = ([], [])
    ratios = []
    word_errors = [0, 0]
    words = [0, 0]
    prompts = (None, prompt)
    for repeat in range(repeats):
        for i, (audio, reference) in enumerate(clips):
            order = (0, 1) if (i + repeat) % 2 == 0 else (1, 0)
            pair = [0, 0]
            for variant in order:
                prediction, pair[variant] = timed(run, audio, prompts[variant])
                samples[variant].append(pair[variant])
                if repeat == 0:
                    wrong, total = errors(reference, prediction)
                    word_errors[variant] += wrong
                    words[variant] += total
            ratios.append(pair[0] / pair[1])
    return {"model": model.name, "clips": len(clips), "prompt_context": context,
            "without_prompt": latency(samples[0]),
            "with_prompt": latency(samples[1]), "median_paired_speedup": round(median(ratios), 4),
            "without_prompt_word_errors": rate(word_errors[0], words[0]),
            "with_prompt_word_errors": rate(word_errors[1], words[1])}


def fixture():
    # This existing test replays a real WAV and jaw trace without camera or UI input.
    name = "test_handsfree_modes.ModeTests.test_sensor_modes"
    result = subprocess.run([sys.executable, "-m", "unittest", name], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"{name} failed")
    return {"fixture": name, "passed": True}


def measure(fn, *args):
    try:
        return {"status": "ok", **fn(*args)}
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def spatial_replay():
    """Leave each session out of ray-center and hold-speed calibration."""
    import numpy as np

    clips = list(recorded_head_rays())
    if not clips:
        raise ValueError("No recorded face clips for spatial replay")
    sessions = sorted({session for session, _, _ in clips})
    negative, look_down, cross_session, all_errors = [], [], [], []
    for test in sessions:
        neutral = [frames for session, label, frames in clips if session != test and label == "neg_blink"]
        points = [pose.intersection for frames in neutral for _, pose in frames if pose]
        neutral_ray = np.median(points, axis=0)
        threshold = fixation_threshold([[(t, pose.intersection) for t, pose in frames if pose]
                                        for frames in neutral])
        negatives = false_unlocks = clips_tested = wrong = no_lock = 0
        look_down_clips = look_down_releases = 0
        for session, label, frames in clips:
            if session != test:
                continue
            anchors = list(replay_fixations(frames, threshold))
            all_errors.extend(p.reprojection_px for _, p in frames if p is not None)
            if label in ("neg_blink", "neg_swallow"):
                negatives += 1
                false_unlocks += any(a is not None and b is None for a, b in zip(anchors, anchors[1:]))
            elif label == "neg_look_down":
                look_down_clips += 1
                look_down_releases += any(a is not None and b is None for a, b in zip(anchors, anchors[1:]))
            if label.startswith("neg_"):
                continue
            clips_tested += 1
            locked = next((point for point in anchors if point is not None), None)
            if locked is None:
                no_lock += 1
                continue
            dx, dy = np.subtract(locked, neutral_ray)
            wrong += int(label.endswith("right") != (dx > 0) or label.startswith("bottom") != (dy > 0))
        negative.append({"test_session": test, "blink_and_swallow_clips": negatives,
                         "clips_with_false_fixation_release": false_unlocks})
        look_down.append({"test_session": test, "look_down_clips": look_down_clips,
                          "clips_with_fixation_release": look_down_releases})
        cross_session.append({"test_session": test, "corner_clips": clips_tested, "wrong_quadrant": wrong,
                              "no_fixation": no_lock, "failure_rate": (wrong + no_lock) / clips_tested})
    return {"negative_fixation": negative, "cross_session_head_ray_quadrant": cross_session,
            "look_down_fixation": look_down,
            "reprojection_px_p95": float(np.percentile(all_errors, 95)), "solved_frames": len(all_errors),
            "note": "Corner labels direct eye fixation. The test measures head-ray transfer to those quadrants, not intentional head-pointing or hand-gesture accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="write one JSON report to stdout")
    parser.add_argument("--repeats", type=int, default=1, help="passes over latency recordings")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="Auto-AVSR device")
    parser.add_argument("--live-ax", action="store_true", help="read the focused macOS window and run paired Whisper decoding")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    with redirect_stdout(sys.stderr):
        negative, cross, train = None, None, None
        report = {"environment": {"platform": platform.platform(), "python": platform.python_version(),
                                  "repeats": args.repeats}}
        try:
            negative, cross, train = gaze_sessions()
            report["negative_clips"] = {"status": "ok", **negative}
        except (ImportError, OSError, ValueError, SystemExit) as exc:
            report["negative_clips"] = {"status": "unavailable", "reason": str(exc)}
        rows = []
        try:
            rows = list(recorded_speech())
            report["speech_vs_jaw"] = measure(speech_replay, rows)
        except (OSError, ValueError) as exc:
            report["speech_vs_jaw"] = {"status": "unavailable", "reason": str(exc)}
        report["cross_session_gaze"] = ({"status": "ok", "splits": cross} if cross is not None else
                                        {"status": "unavailable", "reason": "Needs valid gaze sessions"})
        report["spatial_head_replay"] = measure(spatial_replay)
        report["gaze_latency"] = (measure(gaze_latency, train, args.repeats) if train else
                                  {"status": "unavailable", "reason": "Needs a valid gaze training session"})
        report["sensor_mode_fixture"] = measure(fixture)
        report["ax_window_index"] = {
            "status": "skipped", "reason": "Pass --live-ax from a trusted macOS desktop session"}
        # Use a fixed vocabulary from the earlier recording session for the
        # offline A/B. The test session supplies audio and scoring references.
        prompt = ", ".join(dict.fromkeys(meta["phrase"] for _, meta in rows
                           if meta["run"] == SPEECH_RUNS[0] and meta["mode"] == "silent"))
        context = f"silent command vocabulary from {SPEECH_RUNS[0]}"
        if args.live_ax:
            prompt = ""
            context = "focused window AX labels"
            try:
                ax_result, prompt = window_benchmark(max(5, args.repeats))
                report["ax_window_index"] = {"status": "ok", **ax_result}
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                report["ax_window_index"] = {"status": "unavailable", "reason": str(exc)}
        whisper_rows = [(folder, meta) for folder, meta in rows if meta["run"] == SPEECH_RUNS[1]]
        report["whisper_hotwords"] = (measure(whisper_benchmark, whisper_rows, prompt, context, args.repeats)
            if whisper_rows and prompt else {"status": "unavailable", "reason": f"Needs recorded audio and {context}"})
        report["silent_lip_reading"] = (measure(silent_benchmark, rows, args.device, args.repeats) if rows else
                                        {"status": "unavailable", "reason": "Needs recorded speech clips"})

    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        for section, result in report.items():
            print(section.replace("_", " ").capitalize())
            print(json.dumps(result, indent=2, allow_nan=False))
    core = ("negative_clips", "speech_vs_jaw", "cross_session_gaze", "spatial_head_replay", "gaze_latency", "sensor_mode_fixture")
    return int(any(report[key]["status"] != "ok" for key in core))


if __name__ == "__main__":
    raise SystemExit(main())
