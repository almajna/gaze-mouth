#!/usr/bin/env python3
"""Silent sentence dictation: camera -> 25 Hz mouth crops -> USR 2.0 + speaker LoRA -> joint beams -> GER -> text.

Focus the preview window. Space starts and stops an utterance (as in record_sentences.py), Q quits.
--replay CLIP.npz ... runs recorded raw frames through the same path without a camera, checks the
crops against the stored ones and prints the latency of every stage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from time import monotonic_ns, perf_counter, sleep

import cv2
import numpy as np
import torch

from context_decoder.ger import DEFAULT_ADAPTER, GERCorrector, LMSelector, words
from context_decoder.usr_joint import USRJointDecoder
from silent_avsr.features import ROOT
from silent_avsr.pretrained import PretrainedVSR
from silent_avsr.sentence_video import MEAN_FACE, Landmarks68, align_mouth, sentence_crops

WINDOW = "Silent dictation"
MAX_UTTERANCE_NS = 30_000_000_000
MIN_CTC_TOKENS = 5  # non-speech gate: fewer greedy CTC tokens than this returns no text (docs/plan.md, Stage 3)


def screen_context():
    """(context for the GER prompt, pid, secure) of the frontmost app other than this process.

    Needs the Accessibility permission for the focused element; without it only the app name is read.
    """
    import AppKit as K
    import ApplicationServices as AX
    import Quartz as Q
    windows = Q.CGWindowListCopyWindowInfo(
        Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID) or []
    pid = next((int(w["kCGWindowOwnerPID"]) for w in windows
                if w.get("kCGWindowLayer") == 0 and int(w["kCGWindowOwnerPID"]) != os.getpid()), None)
    app = K.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if pid else None
    if app is None:
        return {}, None, False
    context = dict(app_name=str(app.localizedName() or ""))
    if not AX.AXIsProcessTrusted():
        return context, pid, False

    def attribute(element, name):
        if element is None:
            return None
        status, value = AX.AXUIElementCopyAttributeValue(element, name, None)
        return value if status == 0 else None

    root = AX.AXUIElementCreateApplication(pid)
    AX.AXUIElementSetMessagingTimeout(root, 0.1)
    focused = attribute(root, "AXFocusedUIElement")
    secure = attribute(focused, "AXSubrole") == "AXSecureTextField"
    context["ax_role"] = str(attribute(focused, "AXRole") or "")
    if not secure:
        selected, value = attribute(focused, "AXSelectedText"), attribute(focused, "AXValue")
        if isinstance(selected, str) and selected.strip():
            context["ax_text"] = selected
        elif isinstance(value, str):
            caret = attribute(focused, "AXSelectedTextRange")
            ok, span = AX.AXValueGetValue(caret, AX.kAXValueCFRangeType, None) if caret is not None else (False, None)
            context["ax_text"] = value[:span.location] if ok and span.location else value
    title = attribute(attribute(root, "AXFocusedWindow"), "AXTitle")
    context["nearby_labels"] = [str(title)] if isinstance(title, str) and title.strip() else []
    return context, pid, secure


def paste(text, pid):
    """Bring the target app back to the front and paste as one edit; the clipboard string is restored."""
    import AppKit as K
    import Quartz as Q
    app = K.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if app is None or not app.activateWithOptions_(K.NSApplicationActivateIgnoringOtherApps):
        raise RuntimeError("Could not activate the target app")
    sleep(0.15)
    _, now, secure = screen_context()
    if now != pid or secure:
        raise RuntimeError("Target app is not in front, or a secure field has focus")
    board = K.NSPasteboard.generalPasteboard()
    previous = board.stringForType_(K.NSPasteboardTypeString)
    board.clearContents()
    board.setString_forType_(text, K.NSPasteboardTypeString)
    owned = board.changeCount()
    for down in (True, False):
        event = Q.CGEventCreateKeyboardEvent(None, 9, down)  # V
        Q.CGEventSetFlags(event, Q.kCGEventFlagMaskCommand)
        Q.CGEventPost(Q.kCGHIDEventTap, event)
    sleep(0.2)
    if previous is not None and board.changeCount() == owned:
        board.clearContents()
        board.setString_forType_(previous, K.NSPasteboardTypeString)


class Pipeline:
    def __init__(self, adapter, ger_adapter, device, ger="select", min_ctc_tokens=MIN_CTC_TOKENS, fast=False):
        torch.set_num_threads(4)
        self.reference = np.load(MEAN_FACE, allow_pickle=False)
        self.vsr = PretrainedVSR(device=device, adapter_path=adapter)
        if adapter:
            self.vsr.adapter.merge()  # no extra low-rank products at inference
        # fast: greedy attention decoding (one hypothesis, so nothing for Qwen to choose from)
        self.decoder = USRJointDecoder(self.vsr, beam_size=1, nbest=1) if fast else USRJointDecoder(self.vsr)
        ger = "off" if fast else ger
        self.min_ctc_tokens = min_ctc_tokens
        # MLX is used from one thread only: load, prefill and generation all run on this worker.
        # "select": Qwen picks one of the eight hypotheses; "generate": Qwen (+ GER LoRA) rewrites; "off": first pass.
        self.mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ger") if ger != "off" else None
        load = LMSelector if ger == "select" else lambda: GERCorrector(adapter_path=ger_adapter)
        self.ger = self.mlx.submit(load).result() if ger != "off" else None

    def warm_up(self, crops):
        """One untimed pass so that the first utterance does not pay for lazy initialisation."""
        hypotheses = self.decoder.decode(crops)
        if self.ger:
            self.mlx.submit(self.ger.correct, hypotheses).result()

    def prefill(self, context):
        """Evaluate system prompt and screen context while the user is still mouthing."""
        return self.mlx.submit(self.ger.prefill, context) if self.ger else None

    def finish(self, frames, timestamps, landmarks, context=None, first_pass=None):
        """Everything after the stop event. Returns text, first-pass text and per-stage milliseconds."""
        t0 = perf_counter()
        points = np.stack([np.full((68, 2), np.nan, np.float32) if fit is None else fit for fit in landmarks])
        crops, _ = sentence_crops(frames, timestamps, points, self.reference)
        t1 = perf_counter()
        hypotheses = self.decoder.decode(crops)
        t2 = perf_counter()
        stats = dict(self.decoder.last)
        spoke = stats["ctc_greedy_tokens"] >= self.min_ctc_tokens
        visual = " ".join(words(hypotheses[0].text)) if spoke and hypotheses else ""
        if first_pass:
            first_pass(visual)
        text, ger = visual, {}
        if visual and self.ger:
            text = self.mlx.submit(self.ger.correct, hypotheses, context).result()
            ger = dict(self.ger.last)
        t3 = perf_counter()
        return dict(text=text, first_pass=visual, spoke=spoke, crops=crops, hypotheses=hypotheses,
                    ms=dict(crops=(t1 - t0) * 1e3, encoder=stats["encoder_ms"], beam=stats["beam_ms"],
                            first_pass_total=(t2 - t0) * 1e3, ger=(t3 - t2) * 1e3, total=(t3 - t0) * 1e3),
                    ctc_greedy_tokens=stats["ctc_greedy_tokens"], beam_steps=stats.get("steps"),
                    frames=len(crops), prompt_tokens=ger.get("prompt_tokens"), cached_tokens=ger.get("cached_tokens"),
                    generated_tokens=ger.get("generated_tokens"))


def replay(pipeline, paths, output):
    detect, rows = Landmarks68(), []
    for path in paths:
        with np.load(path, allow_pickle=False) as clip:
            meta = json.loads(str(clip["metadata_json"]))
            frames, stamps, stored = clip["raw_frames"], clip["raw_timestamps_ns"], clip["mouth_88x88"]
        prefill = pipeline.prefill(None)
        landmarks = []
        for frame in frames:  # what the camera loop does while the user mouths
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fit = detect(gray)
            try:
                if fit is not None:
                    align_mouth(gray, fit, pipeline.reference)
            except ValueError:
                fit = None
            landmarks.append(fit)
        if prefill:
            prefill.result()
        result = pipeline.finish(list(frames), list(stamps), landmarks)
        same = np.array_equal(result.pop("crops"), stored)
        hypotheses = result.pop("hypotheses")
        rows.append(dict(clip=path.name, reference=meta["sentence"], crops_equal_stored=bool(same),
                         hypotheses=[h.text for h in hypotheses], **result))
        ms = result["ms"]
        print(f"{path.name}  {result['frames']} frames  crops {ms['crops']:.0f}  encoder {ms['encoder']:.0f}  "
              f"beam {ms['beam']:.0f} ({result['beam_steps']} steps)  first pass at {ms['first_pass_total']:.0f}  "
              f"GER {ms['ger']:.0f}  total {ms['total']:.0f} ms  crops equal {same}\n"
              f"    ref  {meta['sentence']}\n    lips {result['first_pass']}\n    text {result['text']}", flush=True)
    if not all(row["crops_equal_stored"] for row in rows):
        print("WARNING: replayed crops differ from the stored crops on at least one clip")
    for key in ("crops", "encoder", "beam", "first_pass_total", "ger", "total"):
        values = [row["ms"][key] for row in rows]
        print(f"{key:17s} median {np.median(values):6.0f} ms   p95 {np.percentile(values, 95):6.0f} ms   max {max(values):6.0f} ms")
    if output:
        output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


def live(pipeline, camera_index, allow_paste):
    detect = Landmarks68()
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        recording, status, shown = False, "Space starts", ""
        frames, stamps, landmarks, context, target, secure = [], [], [], {}, None, False
        while True:
            ok, frame = camera.read()
            stamp = monotonic_ns()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fit = detect(gray)
            try:
                if fit is not None:
                    align_mouth(gray, fit, pipeline.reference)
            except ValueError:
                fit = None
            if recording:
                frames.append(frame)
                stamps.append(stamp)
                landmarks.append(fit)
            view = frame.copy()
            colour = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(view, status if fit is not None else "Face out of view", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(view, shown[-60:], (12, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if recording and (key == 32 or stamp - stamps[0] >= MAX_UTTERANCE_NS):
                recording = False
                try:
                    result = pipeline.finish(frames, stamps, landmarks, context,
                                             lambda text: print(f"[lips] {text}", flush=True))
                except ValueError as exc:
                    status, result = f"Not decoded: {exc}", None
                if result:
                    ms, shown = result["ms"], result["text"] or "(no speech)"
                    print(f"[text] {shown}   first pass {ms['first_pass_total']:.0f} ms, total {ms['total']:.0f} ms "
                          f"(crops {ms['crops']:.0f}, encoder {ms['encoder']:.0f}, beam {ms['beam']:.0f}, "
                          f"GER {ms['ger']:.0f}), app {context.get('app_name') or '-'}", flush=True)
                    status = "Space starts"
                    if allow_paste and result["text"] and target and not secure:
                        try:
                            paste(result["text"], target)
                        except RuntimeError as exc:
                            print(f"[paste] {exc}", flush=True)
            elif key == 32:
                context, target, secure = screen_context()
                pipeline.prefill(context)
                frames, stamps, landmarks, recording, status = [frame], [stamp], [fit], True, "Space stops"
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, default=ROOT / "models/usr2_speaker_lora_v2")
    ap.add_argument("--ger-adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--ger", choices=("select", "generate", "off"), default="select",
                    help="Qwen stage: pick one of the hypotheses (default, best on validation), rewrite with the "
                         "GER LoRA, or none")
    ap.add_argument("--fast", action="store_true",
                    help="Greedy decoding without the Qwen stage: under 180 ms, about 10 WER points worse on validation")
    ap.add_argument("--device", choices=("cpu", "mps"))
    ap.add_argument("--min-ctc-tokens", type=int, default=MIN_CTC_TOKENS)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--paste", action="store_true", help="Paste the text into the app that was in front (off by default)")
    ap.add_argument("--replay", type=Path, nargs="+", help="Recorder archives to run instead of the camera")
    ap.add_argument("--warm-up", type=Path, default=ROOT / "data/train_sentences/001.npz",
                    help="Recorded clip decoded once at start-up")
    ap.add_argument("--json", type=Path, help="With --replay: write per-clip results")
    args = ap.parse_args()
    ger_adapter = args.ger_adapter if (args.ger_adapter / "adapter_config.json").is_file() else None
    if ger_adapter is None and args.ger == "generate":
        print(f"No GER adapter in {args.ger_adapter}: using Qwen without one", flush=True)
    pipeline = Pipeline(args.adapter, ger_adapter, args.device, args.ger, args.min_ctc_tokens, args.fast)
    if args.warm_up.is_file():
        with np.load(args.warm_up, allow_pickle=False) as clip:
            pipeline.warm_up(clip["mouth_88x88"])
    if args.replay:
        replay(pipeline, args.replay, args.json)
    else:
        live(pipeline, args.camera, args.paste)


if __name__ == "__main__":
    main()
