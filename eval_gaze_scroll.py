"""Offline check of vertical gaze, the eye filter and gaze-paced scrolling on the recorded sessions.
No camera and no generated gaze: every number comes from the clips in data/session_*.

Reports, in this order:
  1. scroll events on clips where the user looks at the centre, and on the look-down negatives
  2. scrolling on the bottom and top fixation clips: started or not, delay, time spent scrolling
  3. stop delay when a centre clip follows a bottom clip
  4. vertical error of the calibration with and without iris fusion, cross-session first
  5. the eye filter against GazeFollower's previous setting

Targets are the recorded dot positions in the capture view (30 px inside each corner, centre for
the blink and swallow negatives), so heights are fractions of that view.

  .venv-gf/bin/python eval_gaze_scroll.py        # the first run reads every clip (a few minutes)
"""
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np

from gaze_engine import EYE_FILTER, EYE_OPEN_DEFAULT, FixationLock, OneEuroFilter, ScrollPacer, eye_opening, iris_vertical

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "logs" / "gaze_scroll_signals.npz"
COLUMNS = ("t_ms", "raw_x", "raw_y", "openness")
SETTLE_FRAMES = 6  # frames at the start of a clip while the eyes are still arriving


def signals(refresh=False):
    """Per-frame signals for every session clip. MGazeNet raw gaze and GazeFollower's eye openness
    need the video and are cached; iris offset, eyelid opening and head pitch come from the
    FaceLandmarker landmarks stored with each clip and are worked out on every run."""
    import gaze_follower_test as gaze

    paths = sorted((ROOT / "data").glob("session_*/*.npz"))
    if CACHE.is_file() and not refresh:
        with np.load(CACHE) as z:
            table = {key: z[key] for key in (*COLUMNS, "session", "label", "rep", "frame", "cue")}
    else:
        from gazefollower.face_alignment import MediaPipeFaceAlignment
        from gazefollower.gaze_estimator import MGazeNetGazeEstimator

        aligner, estimator = MediaPipeFaceAlignment(enable_filter=True), MGazeNetGazeEstimator()
        rows, keys, base = [], [], 0
        for clip in paths:
            with np.load(clip) as d:
                stamps, cues, label, rep = d["timestamps_ms"].astype(int), d["cue"], str(d["label"]), int(d["rep"])
            video = cv2.VideoCapture(str(clip.with_suffix(".mp4")))
            for k, stamp in enumerate(stamps):
                ok, frame = video.read()
                if not ok:
                    raise RuntimeError(f"Cannot read frame {k} of {clip}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                eyes = aligner.detect((base + int(stamp)) * 1_000_000, rgb)
                info = estimator.detect(rgb, eyes)
                raw = np.asarray(info.raw_gaze_coordinates, float).ravel() if eyes.status and info.status else np.full(2, np.nan)
                openness = min(eyes.left_eye_openness, eyes.right_eye_openness) if eyes.status else np.nan
                rows.append((stamp, *(raw if raw.size == 2 else (np.nan, np.nan)), openness))
                keys.append((clip.parent.name, label, rep, k, str(cues[k])))
            video.release()
            base += int(stamps[-1]) + 2000  # the aligner's own filter needs increasing timestamps
        aligner.release()
        estimator.release()
        table = dict(zip(COLUMNS, np.asarray(rows, float).T))
        table.update(session=np.array([k[0] for k in keys]), label=np.array([k[1] for k in keys]),
                     rep=np.array([k[2] for k in keys]), frame=np.array([k[3] for k in keys]),
                     cue=np.array([k[4] for k in keys]))
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **table)
    seen = []
    for clip in paths:
        with np.load(clip) as d:
            aspect = int(d["width"]) / int(d["height"])
            for landmarks, matrix, present in zip(d["landmarks"], d["head_matrix"], d["face_present"]):
                iris, opening = (iris_vertical(landmarks, aspect), eye_opening(landmarks, aspect)) if present else (None, None)
                seen.append((np.nan if iris is None else iris, gaze.head_pose(matrix)[1] if present else np.nan,
                             np.nan if opening is None else opening))
    if len(seen) != len(table["t_ms"]):
        raise RuntimeError("The clips changed since the cache was written; run with --refresh")
    table.update(zip(("iris", "pitch", "opening"), np.asarray(seen, float).T))
    return table


VIEW = (1280, 720)
SCREEN = (1352., 878.)  # pixel scale for the filter and fixation settings (the saved calibration's screen)
TARGET_X = {"left": 30 / VIEW[0], "right": 1 - 30 / VIEW[0]}
TARGET_Y = {"top": 30 / VIEW[1], "bottom": 1 - 30 / VIEW[1]}
PREVIOUS_FILTER = (.5, .0005, 1.)  # GazeFollower's One Euro setting that the agent used before
TICK = .01                         # the controller loop runs about every 10 ms


def row_of(label):
    return label.split("_")[0] if label.split("_")[0] in TARGET_Y else "down" if label == "neg_look_down" else "center"


def target_of(label):
    if row_of(label) in TARGET_Y:
        return TARGET_X[label.split("_")[1]], TARGET_Y[row_of(label)]
    return (.5, .5) if row_of(label) == "center" else None


class Clips:
    """The usable frames of every clip, as the live capture would pass them on."""
    def __init__(self, z, blink_threshold):
        self.z = z
        self.seen = ((z["cue"] != "gesture") & np.isfinite(z["raw_x"]) & np.isfinite(z["iris"])
                     & np.isfinite(z["pitch"]) & (z["openness"] > blink_threshold))
        self.usable = self.seen & (z["opening"] >= EYE_OPEN_DEFAULT)
        self.keys = sorted({(str(s), str(l), int(r)) for s, l, r in zip(z["session"], z["label"], z["rep"])})

    def frames(self, key, settled=True):
        z = self.z
        rows = (z["session"] == key[0]) & (z["label"] == key[1]) & (z["rep"] == key[2]) & (z["cue"] != "gesture")
        index = np.flatnonzero(rows & (z["frame"] >= SETTLE_FRAMES if settled else rows))
        return index, self.usable[index]

    def anchor(self, key):
        index, usable = self.frames(key)
        index = index[usable]
        z = self.z
        return (np.median(np.column_stack([z["raw_x"][index], z["raw_y"][index]]), axis=0),
                np.median(np.column_stack([z["iris"][index], z["pitch"][index], z["opening"][index]]), axis=0))


def calibrate(agent, clips, keys, fused=True):
    keys = [k for k in keys if target_of(k[1]) is not None]
    anchors = [clips.anchor(k) for k in keys]
    return agent.calibrated_eye_model([a[0] for a in anchors], [a[1] for a in anchors] if fused else None,
                                      [target_of(k[1]) for k in keys])


def trace(clips, model, key, euro=EYE_FILTER, settled=False, blink_gate=True):
    """(seconds, raw points, filtered points) in screen pixels for one clip; None where the capture
    would have dropped the frame (eyes not found, or closed by the eyelid-opening gate). Follows
    Capture.eye_point and Capture.on_gaze."""
    z = clips.z
    index, _ = clips.frames(key, settled)
    usable = clips.seen[index] & (z["opening"][index] >= (getattr(model, "eye_open", EYE_OPEN_DEFAULT) if blink_gate else 0.))
    filters = [OneEuroFilter(*euro) for _ in range(2)]
    raw_points, smooth = [], []
    for i, ok in zip(index, usable):
        if not ok:
            for f in filters:
                f.reset()
            raw_points.append(None), smooth.append(None)
            continue
        raw = np.array([z["raw_x"][i], z["raw_y"][i]])
        x, y = model.predict(raw, raw)[1]
        if model.vertical is not None:
            y = model.vertical.predict(raw, (z["iris"][i], z["pitch"][i]))
        point = np.clip(np.multiply((x, y), SCREEN), 0, np.subtract(SCREEN, 1))
        raw_points.append(point)
        smooth.append(np.array([f.update(v, z["t_ms"][i] / 1000) for f, v in zip(filters, point)]))
    return z["t_ms"][index] / 1000, raw_points, smooth


def join(*traces):
    """Clips played one after another on one clock. The switch between clips is a cut, not a
    recorded eye movement. Returns the joined trace and the start time of each clip."""
    times, raws, smooths, starts, clock = [], [], [], [], 0.
    for t, raw, smooth in traces:
        starts.append(clock)
        times += list(t - t[0] + clock)
        raws += raw
        smooths += smooth
        clock = times[-1] + float(np.median(np.diff(t)))
    return (np.array(times), raws, smooths), starts


def refilter(joined, euro):
    times, raws, _ = joined
    filters, smooth = [OneEuroFilter(*euro) for _ in range(2)], []
    for t, point in zip(times, raws):
        if point is None:
            for f in filters:
                f.reset()
        smooth.append(None if point is None else np.array([f.update(v, t) for f, v in zip(filters, point)]))
    return times, raws, smooth


def scroll_blocks(clips, model, key):
    indices, _ = clips.frames(key, settled=False)
    limit = getattr(model, "scroll_iris_limit", None)
    return [ScrollPacer.below_calibrated_screen(iris, limit) for iris in clips.z["iris"][indices]]


def paced(times, smooth, blocked=None, dwell=None):
    """Scroll steps [(seconds, pixels)] and the time spent scrolling, with the controller's loop:
    ticks every TICK seconds on the latest filtered sample; a lost frame (a blink included) or a
    sample older than 120 ms ends the run."""
    pacer, steps, active = ScrollPacer(dwell), [], []
    for k, (t, point) in enumerate(zip(times, smooth)):
        end = times[k + 1] if k + 1 < len(times) else t + .05
        now = t
        while now < end:
            if point is None or now - t > .12:
                pacer.reset()
            else:
                pixels = pacer.update(point[1] / SCREEN[1], now, below_screen=blocked is not None and blocked[k])
                if pixels:
                    steps.append((now, pixels))
            active.append((now, pacer.direction))
            now += TICK
    return steps, active


def summary(steps, active, since=0.):
    steps = [(t, p) for t, p in steps if t >= since]
    on = [d for t, d in active if t >= since]
    return dict(pixels_down=-sum(p for _, p in steps if p < 0), pixels_up=sum(p for _, p in steps if p > 0),
                first=min((t for t, _ in steps), default=None), last=max((t for t, _ in steps), default=None),
                share=float(np.mean([d != 0 for d in on])) if on else 0.)


def compare_scroll(agent, clips, calibration, test):
    if calibration == test:
        raise ValueError("calibration and test sessions must differ")
    anchors = [key for key in clips.keys if key[0] == calibration]
    keys = [key for key in clips.keys if key[0] == test]
    if not anchors or not keys:
        raise ValueError("calibration and test must both name recorded sessions")
    model = calibrate(agent, clips, anchors)
    cases = (("center", "any"), ("down", "any"), ("bottom", "pixels_down"), ("top", "pixels_up"))
    rows = {name: [] for name, _ in cases}
    for key in keys:
        times, _, smooth = trace(clips, model, key)
        blocked = scroll_blocks(clips, model, key)
        row = row_of(key[1])
        direction = {"top": 1, "bottom": -1}.get(row)
        results = []
        for steps, active in (paced(times, smooth, dwell=.22), paced(times, smooth, blocked)):
            result = summary(steps, active)
            result["first"] = next((ts - times[0] for ts, px in steps if direction is None or px * direction > 0), None)
            result["any"] = result["pixels_up"] + result["pixels_down"]
            results.append(result)
        rows[row].append(results)
    print(f"Cross-session {calibration} -> {test} · down dwell: 220 -> {ScrollPacer.DWELL * 1000:.0f} ms + iris guard · "
          f"up dwell: {ScrollPacer.UP_DWELL * 1000:.0f} ms")
    report = {"calibration": calibration, "test": test, "rows": {}}
    for row, wanted in cases:
        counts = []
        for column in zip(*rows[row]):
            active = [result for result in column if result[wanted]]
            counts.append(dict(clips=len(column), triggered=len(active), pixels=sum(result[wanted] for result in active),
                               first_ms=round(float(np.median([r["first"] for r in active])) * 1000) if active else None,
                               wrong_direction=sum(bool(r["pixels_up" if row == "bottom" else "pixels_down"])
                                                   for r in column) if wanted != "any" else None))
        print(f"{row:>6}  " + " -> ".join(f"{c['triggered']}/{c['clips']} clips, {c['pixels']} px, "
              f"first step {c['first_ms']} ms" for c in counts))
        report["rows"][row] = dict(zip(("before", "after"), counts))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="read the clips again instead of using the cache")
    ap.add_argument("--compare-scroll", nargs=2, metavar=("CALIBRATION", "TEST"),
                    help="compare previous and current scroll rules on an explicitly held-out session")
    ap.add_argument("--json", type=Path, help="save the comparison to a new file")
    args = ap.parse_args()
    if args.json is not None and not args.compare_scroll:
        ap.error("--json requires --compare-scroll")
    z = signals(args.refresh)
    import handsfree_mac_agent as agent
    import gaze_follower_test as gaze

    clips = Clips(z, gaze.DefaultConfig().eye_blink_threshold)
    if args.compare_scroll:
        report = compare_scroll(agent, clips, *args.compare_scroll)
        if args.json is not None:
            with args.json.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        return
    sessions = sorted({k[0] for k in clips.keys})
    splits = {"cross-session (calibrated on one session, replayed on another)":
                  [(f"{a}->{b}", [k for k in clips.keys if k[0] == a], [k for k in clips.keys if k[0] == b])
                   for a in sessions for b in sessions if a != b],
              "within-session (calibrated on repetition 0, replayed on repetitions 1-2)":
                  [(a, [k for k in clips.keys if k[0] == a and k[2] == 0], [k for k in clips.keys if k[0] == a and k[2] > 0])
                   for a in sessions]}
    report = {}
    print(f"{len(clips.keys)} clips, {len(z['t_ms'])} frames, {int(clips.usable.sum())} usable; "
          f"pacer: down >= {ScrollPacer.DOWN}, up <= {ScrollPacer.UP}, quiet {ScrollPacer.QUIET}, "
          f"down/up dwell {ScrollPacer.DWELL * 1000:.0f}/{ScrollPacer.UP_DWELL * 1000:.0f} ms")

    fitted = {name: [(tag, calibrate(agent, clips, train), calibrate(agent, clips, train, fused=False), test)
                     for tag, train, test in parts] for name, parts in splits.items()}

    # 1-3. Scrolling ----------------------------------------------------------------------------
    for name, parts in fitted.items():
        rows = {row: [] for row in ("center", "down", "bottom", "top")}
        stops = []
        for tag, model, _, test in parts:
            for key in test:
                t, _, smooth = trace(clips, model, key)
                rows[row_of(key[1])].append(summary(*paced(t, smooth, scroll_blocks(clips, model, key)))
                                            | dict(seconds=float(t[-1] - t[0]), start=float(t[0])))
            for key in test:
                if row_of(key[1]) != "bottom":
                    continue
                for center in ("neg_blink", "neg_swallow"):
                    other = (key[0], center, key[2])
                    if other not in test:
                        continue
                    joined, starts = join(trace(clips, model, key), trace(clips, model, other))
                    blocked = scroll_blocks(clips, model, key) + scroll_blocks(clips, model, other)
                    steps, active = paced(joined[0], joined[2], blocked)
                    if not any(t < starts[1] and p < 0 for t, p in steps):
                        continue  # never scrolled on the bottom clip: nothing to stop
                    middle = next((t for t, p in zip(joined[0], joined[2]) if t >= starts[1] and p is not None
                                   and ScrollPacer.QUIET[0] < p[1] / SCREEN[1] < ScrollPacer.QUIET[1]), None)
                    after = [t - starts[1] for t, p in steps if t >= starts[1]]
                    carried = [a for a in after if middle is None or a <= middle - starts[1]]
                    stops.append((max(carried, default=0.), None if middle is None else middle - starts[1],
                                  len(after) - len(carried)))
        print(f"\n== {name} ==")
        quiet = rows["center"]
        print(f"[1] centre clips with any scroll step: {sum(bool(r['pixels_down'] + r['pixels_up']) for r in quiet)}/{len(quiet)}"
              f"   pixels down/up over {sum(r['seconds'] for r in quiet):.0f} s: {sum(r['pixels_down'] for r in quiet)}/{sum(r['pixels_up'] for r in quiet)}")
        down = rows["down"]
        print(f"    look-down-at-keyboard clips with any scroll step: {sum(bool(r['pixels_down'] + r['pixels_up']) for r in down)}/{len(down)}"
              f"   pixels down/up: {sum(r['pixels_down'] for r in down)}/{sum(r['pixels_up'] for r in down)}")
        for row, wanted, wrong in (("bottom", "pixels_down", "pixels_up"), ("top", "pixels_up", "pixels_down")):
            got = rows[row]
            started = [r for r in got if r[wanted]]
            delays = [r["first"] - r["start"] for r in started]
            print(f"[2] {row} clips that scrolled {'down' if row == 'bottom' else 'up'}: {len(started)}/{len(got)}"
                  f"   first step after {np.median(delays) * 1000 if delays else float('nan'):.0f} ms (median from clip start)"
                  f"   time spent scrolling {np.mean([r['share'] for r in got]):.0%}"
                  f"   wrong direction in {sum(bool(r[wrong]) for r in got)} clips")
        if stops:
            reached = [s for s in stops if s[1] is not None]
            print(f"[3] bottom clip followed by a centre clip ({len(stops)} pairs that were scrolling): gaze first read as middle "
                  f"{np.median([s[1] for s in reached]) * 1000:.0f} ms after the cut (median), {max(s[1] for s in reached) * 1000:.0f} ms worst; "
                  f"last scroll step before that {np.median([s[0] for s in reached]) * 1000:.0f} ms after the cut (median), "
                  f"{max(s[0] for s in reached) * 1000:.0f} ms worst; never read as middle in {len(stops) - len(reached)} pairs; "
                  f"scrolled again later in the centre clip in {sum(bool(s[2]) for s in reached)} pairs")
        report[name] = dict(rows={k: v for k, v in rows.items()}, stops=stops)

    # 4. Vertical error ---------------------------------------------------------------------------
    print("\n[4] Height error per clip (median over frames, fractions of the capture view height)")
    for name, parts in fitted.items():
        table = {kind: {row: [] for row in TARGET_Y | {"center": .5}} for kind in ("quadratic", "with iris")}
        chosen = []
        for tag, model, plain, test in parts:
            chosen.append("pitch" if model.vertical is not None and model.vertical.uses_pitch
                          else "iris" if model.vertical is not None else "none")
            for key in test:
                if target_of(key[1]) is None:
                    continue
                for kind, m in (("quadratic", plain), ("with iris", model)):
                    _, raw, _ = trace(clips, m, key, settled=True)
                    heights = [p[1] / SCREEN[1] for p in raw if p is not None]
                    table[kind][row_of(key[1])].append((np.median(heights), target_of(key[1])[1]))
        print(f"  {name}; fusion chosen by the held-out check: {chosen}")
        for kind, by_row in table.items():
            errors = {row: np.mean([abs(h - t) for h, t in pairs]) for row, pairs in by_row.items()}
            bottom = [h for h, _ in by_row["bottom"]]
            print(f"    {kind:10s} top {errors['top']:.3f}  centre {errors['center']:.3f}  bottom {errors['bottom']:.3f}"
                  f"   bottom-row height {np.mean(bottom):.3f}, >= 0.85 in {np.mean([b >= .85 for b in bottom]):.0%}, "
                  f">= 0.75 in {np.mean([b >= .75 for b in bottom]):.0%} of {len(bottom)} clips")
        report[f"height:{name}"] = dict(chosen=chosen, table={k: {r: [list(map(float, x)) for x in v] for r, v in t.items()}
                                                              for k, t in table.items()})

    # 5. Filter -----------------------------------------------------------------------------------
    print("\n[5] Eye filter on within-session traces (pixels on a 1352 x 878 screen, clips at 20 frames/s)")
    results = {}
    for label, euro in (("unfiltered", None), ("previous", PREVIOUS_FILTER), ("new", EYE_FILTER)):
        steps, lags, locks = [], [], []
        for tag, model, _, test in fitted[list(fitted)[1]]:
            for key in test:
                if row_of(key[1]) == "down":
                    continue
                t, raw, smooth = trace(clips, model, key, euro or EYE_FILTER, settled=True)
                points = [p for p in (raw if euro is None else smooth) if p is not None]
                steps.append(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            if euro is None:
                continue
            for first in test:
                for second in test:
                    if {row_of(first[1]), row_of(second[1])} != {"top", "bottom"} or first[2] != second[2]:
                        continue
                    joined, starts = join(trace(clips, model, first, settled=True), trace(clips, model, second, settled=True))
                    times, raws, smooth = refilter(joined, euro)
                    before = np.median([p[1] for t, p in zip(times, raws) if p is not None and t < starts[1]])
                    after_points = np.array([p for t, p in zip(times, raws) if p is not None and t >= starts[1]])
                    goal = np.median(after_points, axis=0)
                    if abs(goal[1] - before) < 150:
                        continue
                    reached = [t - starts[1] for t, p in zip(times, smooth)
                               if p is not None and t >= starts[1] and (p[1] - before) / (goal[1] - before) >= .8]
                    lags.append(min(reached, default=np.nan))
                    lock = FixationLock()
                    for t, r, f in zip(times, raws, smooth):
                        if r is None:
                            lock.reset()
                            continue
                        fixed = lock.update(tuple(r), tuple(f), t, gaze.FIX_SPEED_PX_S, dwell=gaze.FIX_DWELL_MS / 1000)
                        if t >= starts[1] + .1 and fixed is not None:
                            locks.append((t - starts[1], abs(fixed[1] - goal[1])))
                            break
        results[label] = dict(step=float(np.mean(steps)), lag=lags, locks=locks)
        line = f"  {label:10s} frame-to-frame movement while fixating {np.mean(steps):5.1f} px"
        if lags:
            line += (f"   80% of a top<->bottom step after {np.nanmedian(lags) * 1000:.0f} ms (median), {np.nanpercentile(lags, 90) * 1000:.0f} ms (p90)"
                     f"   first fixation lock after the step: {np.median([l[0] for l in locks]) * 1000:.0f} ms, "
                     f"{np.median([l[1] for l in locks]):.0f} px from where gaze settled (n={len(locks)})")
        print(line)
    report["filter"] = results

    out = ROOT / "logs" / f"gaze_scroll_eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
