"""Offline evaluation on real captured clips: gaze quadrant + open-mouth trigger.

Mouth trigger: tuned with leave-one-(session, rep)-out CV over --tune-sessions (default session_2 session_3),
jawOpen threshold never below 0.28; false triggers checked on every negative clip / fixate window in all sessions. Gaze: within-session LORO + cross-session.
See docs/plan.md.  Run:  .venv/bin/python eval_offline.py [--sessions ... --tune-sessions ...]
Exit code 2 = run INVALID (integrity / coverage failure).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]
NEG_LABELS = ["neg_blink", "neg_swallow", "neg_look_down"]
REQUIRED = CORNERS + NEG_LABELS
MIN_TRAIN_FRAMES_PER_CLASS = 10

DEBOUNCE_MS = 100  # min hold, on timestamps_ms -> independent of 17-20 fps
JAW_MIN_FLOOR = 0.28  # biomechanical minimum: never select a hair-trigger jawOpen threshold below this
DEFAULT_SESSIONS = ["session_1", "session_2", "session_3"]
DEFAULT_TUNE_SESSIONS = ["session_2", "session_3"]  # sessions that contain real open-mouth gestures
TRIGGER_CONFIG_PATH = ROOT / "logs" / "trigger_config.json"  # read by live_app.py
# Gaze feature order: h, v, yaw, pitch. Floors set from measured spreads (docs/plan.md): iris pooled std is
# ~0.025-0.05 (h) and ~0.010-0.013 (v), so the iris floors sit at ~half of that and never mute the real
# top/bottom signal; head pose floor 2 deg (calibration-window pitch std was 0.6-1.3 deg).
FEATURE_STD_FLOOR = np.array([0.01, 0.005, 2.0, 2.0])
FEATURE_WEIGHT = np.array([1.0, 1.0, 0.3, 0.3])


def invalid(msg):
    print(f"\n!! RUN INVALID: {msg}")
    sys.exit(2)


# ---------- loading + integrity ----------

def mp4_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def load_session(session):
    sdir = ROOT / "data" / session
    files = sorted(sdir.glob("*.npz"))
    if not files:
        invalid(f"no .npz files in {sdir}")
    clips, problems = [], []
    for f in files:
        d = np.load(f)
        n = len(d["timestamps_ms"])
        mp4 = f.with_suffix(".mp4")
        vn = mp4_frame_count(mp4) if mp4.exists() else -1
        lens = {k: len(d[k]) for k in ["landmarks", "blendshapes", "head_matrix", "face_present", "cue"]}
        if vn != n or any(v != n for v in lens.values()):
            problems.append(f"{session}/{f.name}: npz={n} mp4={vn} arrays={lens}")
            continue
        names = list(d["blendshape_names"])
        if "jawOpen" not in names:
            problems.append(f"{session}/{f.name}: no jawOpen blendshape (face never detected?)")
            continue
        clips.append(dict(
            session=session, name=f"{session}/{f.stem}", label=str(d["label"]), rep=int(d["rep"]),
            ts=d["timestamps_ms"].astype(np.int64), cue=d["cue"].astype(str),
            lm=d["landmarks"], face=d["face_present"].astype(bool),
            jaw=d["blendshapes"][:, names.index("jawOpen")], hm=d["head_matrix"],
            aspect=float(d["width"]) / float(d["height"]),
        ))
    if problems:
        for p in problems:
            print("  BAD", p)
        invalid(f"{len(problems)} clip(s) in {session} failed integrity checks")
    # coverage: every required label present for every rep in the session
    reps = sorted({c["rep"] for c in clips})
    have = {(c["label"], c["rep"]) for c in clips}
    missing = [(lab, r) for lab in REQUIRED for r in reps if (lab, r) not in have]
    if missing:
        invalid(f"{session} missing label/rep combos: {missing}")
    if len(reps) < 2:
        invalid(f"{session} has {len(reps)} rep(s); LORO-CV needs >= 2")
    return clips, reps


# ---------- features ----------

def head_angles(hm):
    R = hm[:, :3, :3]
    pitch = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    yaw = np.degrees(np.arctan2(-R[:, 2, 0], np.hypot(R[:, 2, 1], R[:, 2, 2])))
    return yaw, pitch


def gaze_features(c):
    """[h, v, yaw, pitch]. Iris position in an eye-aligned frame, in pixel-proportional units (x scaled by
    width/height), so it is invariant to face scale (distance) and head roll:
      h = iris projected onto the corner->corner axis / eye width   (0 = corner a, 1 = corner b)
      v = iris offset perpendicular to that axis from the corner midpoint / eye width (+ = down)
    Normalized by eye-corner width, never by eyelid height, so lid droop doesn't compress v."""
    P = c["lm"][:, :, :2] * np.array([c.get("aspect", 16 / 9), 1.0], np.float32)

    def eye(iris, a, b):
        e = P[:, b] - P[:, a]
        ew = np.linalg.norm(e, axis=1)
        u = e / ew[:, None]
        n = np.stack([-u[:, 1], u[:, 0]], 1)  # u rotated +90 deg: points down in image for a level eye
        h = ((P[:, iris] - P[:, a]) * u).sum(1) / ew
        v = ((P[:, iris] - (P[:, a] + P[:, b]) / 2) * n).sum(1) / ew
        return h, v

    with np.errstate(invalid="ignore", divide="ignore"):
        h1, v1 = eye(468, 33, 133)
        h2, v2 = eye(473, 362, 263)
        yaw, pitch = head_angles(c["hm"])
    return np.column_stack([(h1 + h2) / 2, (v1 + v2) / 2, yaw, pitch])


def lip_gap(c):
    L = c["lm"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.linalg.norm(L[:, 13, :2] - L[:, 14, :2], axis=1)
                / np.linalg.norm(L[:, 78, :2] - L[:, 308, :2], axis=1))


def gesture_start_ms(c):
    g = np.flatnonzero(c["cue"] == "gesture")
    return int(c["ts"][g[0]]) if len(g) else None


# ---------- gaze ----------

def valid_gaze(c, cols):
    X = gaze_features(c)[:, cols]
    ok = c["face"] & np.isfinite(X).all(1)
    return X[ok], ok


def nearest_centroid_fit(X, y, cols):
    """z-score with a per-feature std floor (a still calibration can't blow up z-scores later),
    then weight head-pose dims down relative to iris dims."""
    floor, w = FEATURE_STD_FLOOR[cols], FEATURE_WEIGHT[cols]
    mu, sd = X.mean(0), np.maximum(X.std(0), floor)
    return mu, sd, w, {k: ((X[y == k] - mu) / sd * w).mean(0) for k in CORNERS}


def centroid_distances(model, X):
    mu, sd, w, cents = model
    Z = (X - mu) / sd * w
    return np.stack([np.linalg.norm(Z - cents[k], axis=1) for k in CORNERS], 1)  # columns in CORNERS order


def nearest_centroid_predict(model, X):
    return np.array(CORNERS)[centroid_distances(model, X).argmin(1)]


def confusion(true, pred):
    m = np.zeros((4, 4), int)
    for t, p in zip(true, pred):
        m[CORNERS.index(t), CORNERS.index(p)] += 1
    return m


def print_confusion(title, m):
    print(f"\n{title}  (rows = true, cols = predicted)")
    print(" " * 14 + "".join(f"{k:>14s}" for k in CORNERS) + "    recall")
    for i, k in enumerate(CORNERS):
        r = m[i, i] / m[i].sum() if m[i].sum() else float("nan")
        print(f"{k:>14s}" + "".join(f"{v:14d}" for v in m[i]) + f"    {r:6.1%}")
    tot = m.sum()
    print(f"accuracy {np.trace(m) / tot:.1%} ({np.trace(m)}/{tot})" if tot else "accuracy n/a (0 frames)")


def gaze_split(train, test, cols):
    """Fit on train clips, score test clips. Returns (frame_m, clip_m, info) or None if invalid."""
    Xs, ys = [], []
    for c in train:
        X, _ = valid_gaze(c, cols)
        Xs.append(X); ys += [c["label"]] * len(X)
    X, y = np.vstack(Xs), np.array(ys)
    per_class = {k: int((y == k).sum()) for k in CORNERS}
    if min(per_class.values()) < MIN_TRAIN_FRAMES_PER_CLASS:
        return None, f"too few valid train frames per class: {per_class}"
    model = nearest_centroid_fit(X, y, cols)
    fm, cm, lines = np.zeros((4, 4), int), np.zeros((4, 4), int), []
    for c in test:
        Xt, ok = valid_gaze(c, cols)
        lines.append(f"{c['name']}: valid {ok.sum()}/{len(ok)}")
        if not len(Xt):
            continue  # no valid frames -> not scored; shows up in the valid counts
        p = nearest_centroid_predict(model, Xt)
        fm += confusion([c["label"]] * len(p), p)
        vals, cnt = np.unique(p, return_counts=True)
        cm += confusion([c["label"]], [vals[cnt.argmax()]])
    return (fm, cm, lines, len(y)), None


def eval_gaze(sessions, cols, tag):
    print(f"\n### Gaze [{tag}]")
    out = {}
    for s, (clips, reps) in sessions.items():
        corner = [c for c in clips if c["label"] in CORNERS]
        fm, cm, n_ok = np.zeros((4, 4), int), np.zeros((4, 4), int), 0
        print(f"\n{s}: within-session LORO-CV")
        for r in reps:
            res, err = gaze_split([c for c in corner if c["rep"] != r], [c for c in corner if c["rep"] == r], cols)
            if err:
                print(f"  test rep {r}: FOLD INVALID ({err})"); continue
            f, c_, lines, ntr = res
            n_ok += 1
            fm += f; cm += c_
            wrong = f.sum() - np.trace(f)
            print(f"  test rep {r}: {wrong}/{f.sum()} valid frames wrong "
                  f"({wrong / max(f.sum(), 1):.1%}); train valid frames {ntr}; " + "; ".join(lines))
        print_confusion(f"{s} frame-level [{tag}]", fm)
        print_confusion(f"{s} clip-level majority [{tag}]", cm)
        out[f"{s}_within"] = dict(frame=fm.tolist(), clip=cm.tolist(), valid_folds=n_ok)
    names = list(sessions)
    for a, b in itertools.permutations(names, 2):
        tr = [c for c in sessions[a][0] if c["label"] in CORNERS]
        te = [c for c in sessions[b][0] if c["label"] in CORNERS]
        res, err = gaze_split(tr, te, cols)
        print(f"\nCROSS-SESSION train {a} -> test {b}")
        if err:
            print(f"  INVALID ({err})"); continue
        fm, cm, lines, ntr = res
        print(f"  train valid frames {ntr}; test valid: " + "; ".join(lines))
        print_confusion(f"cross {a}->{b} frame-level [{tag}]", fm)
        print_confusion(f"cross {a}->{b} clip-level majority [{tag}]", cm)
        out[f"cross_{a}_to_{b}"] = dict(frame=fm.tolist(), clip=cm.tolist())
    return out


# ---------- mouth trigger ----------

def triggers(ts, cond, debounce_ms):
    """Fire once per episode after cond held >= debounce_ms (timestamps, not frames). [(t_fire, t_onset)]"""
    out, onset, fired = [], None, False
    for t, c in zip(ts, cond):
        if not c:  # includes face-absent frames (NaN compares False): episode broken
            onset, fired = None, False
            continue
        if onset is None:
            onset = t
        if not fired and t - onset >= debounce_ms:
            out.append((int(t), int(onset)))
            fired = True
    return out


def eval_trigger(clips, cfg, feats):
    tj, tg, d = cfg
    fps, tp, lat_cue, lat_onset, n_pos, n_neg = [], 0, [], [], 0, 0
    for c in clips:
        jaw, gap = feats[c["name"]]
        with np.errstate(invalid="ignore"):
            cond = c["face"] & (jaw >= tj) & (gap >= tg)
        fires = triggers(c["ts"], cond, d)
        if c["label"] in NEG_LABELS:
            n_neg += 1
            fps += [(c["name"], t, "negative") for t, _ in fires]
            continue
        n_pos += 1
        g0 = gesture_start_ms(c)
        fps += [(c["name"], t, "fixate") for t, _ in fires if t < g0]
        hits = [(t, o) for t, o in fires if t >= g0]
        if hits:
            tp += 1
            lat_cue.append(hits[0][0] - g0)
            lat_onset.append(hits[0][0] - hits[0][1])
    return dict(fp=fps, tp=tp, n_pos=n_pos, n_neg=n_neg, lat_cue=lat_cue, lat_onset=lat_onset)


def held_level(ts, x, d):
    """Highest value x stays >= for a span of >= d ms (same semantics as the debounce in triggers())."""
    x = np.where(np.isfinite(x), x, -np.inf)
    best = -np.inf
    for i in range(len(ts)):
        k = np.searchsorted(ts, ts[i] + d, "left")
        if k >= len(ts):
            break
        best = max(best, x[i:k + 1].min())
    return best


def select(clips, feats, d=DEBOUNCE_MS):
    """Max-margin thresholds: per signal, midway between the highest level held >= d ms in any negative clip
    or fixate window (ceiling) and the lowest per-clip held peak in gesture windows (floor)."""
    ceil, floor = [-np.inf, -np.inf], [np.inf, np.inf]
    for c in clips:
        for s in (0, 1):
            x = np.where(c["face"], feats[c["name"]][s], np.nan)
            if c["label"] in NEG_LABELS:
                ceil[s] = max(ceil[s], held_level(c["ts"], x, d))
            else:
                for cue, agg in (("fixate", "ceil"), ("gesture", "floor")):
                    m = c["cue"] == cue
                    v = held_level(c["ts"][m], x[m], d)
                    if agg == "ceil":
                        ceil[s] = max(ceil[s], v)
                    else:
                        floor[s] = min(floor[s], v)
    jaw_mid = float((ceil[0] + floor[0]) / 2)
    jaw_t = max(jaw_mid, JAW_MIN_FLOOR)
    cfg = (round(jaw_t, 4), round(float((ceil[1] + floor[1]) / 2), 4), d)
    margin = dict(jaw_ceiling=float(ceil[0]), jaw_floor=float(floor[0]), gap_ceiling=float(ceil[1]),
                  gap_floor=float(floor[1]), separable=bool(floor[0] > ceil[0] and floor[1] > ceil[1]),
                  jaw_midpoint=round(jaw_mid, 4), jaw_min_floor_applied=bool(jaw_mid < JAW_MIN_FLOOR),
                  jaw_threshold_below_gesture_floor=bool(jaw_t < floor[0]))
    return cfg, eval_trigger(clips, cfg, feats), margin


def fmt_margin(m):
    floor_note = (f" [midpoint {m['jaw_midpoint']:.3f} < {JAW_MIN_FLOOR} -> raised to {JAW_MIN_FLOOR}]"
                  if m["jaw_min_floor_applied"] else "")
    miss_note = "" if m["jaw_threshold_below_gesture_floor"] else " [!! jaw threshold >= weakest gesture peak]"
    return (f"jaw ceil {m['jaw_ceiling']:.3f} / floor {m['jaw_floor']:.3f}{floor_note}{miss_note}, "
            f"gap ceil {m['gap_ceiling']:.3f} / floor {m['gap_floor']:.3f}, separable={m['separable']}")


def fmt_cfg(cfg):
    return f"jawOpen>={cfg[0]}, lip_gap>={cfg[1]}, debounce={cfg[2]}ms"


def print_fp(res, indent="      "):
    for name, t, w in res["fp"]:
        print(f"{indent}FP  {name} @ {t} ms ({w})")


def signal_table(clips, feats):
    rows = {}
    for grp, sel in [("negatives", None), ("fixate", "fixate"), ("gesture", "gesture")]:
        cs = [c for c in clips if (c["label"] in NEG_LABELS) == (sel is None)]
        j, g = [], []
        for c in cs:
            m = c["face"] if sel is None else c["face"] & (c["cue"] == sel)
            j.append(feats[c["name"]][0][m]); g.append(feats[c["name"]][1][m])
        j, g = np.concatenate(j), np.concatenate(g)
        rows[grp] = (j, g)
        print(f"    {grp:10s} valid frames {len(j):5d} | jawOpen max={j.max():.4f} p99={np.percentile(j, 99):.4f}"
              f" | lip_gap max={g.max():.4f} p99={np.percentile(g, 99):.4f}")
    return rows


def eval_mouth(sessions, tune):
    print("\n## 1. Mouth trigger — false positives first")
    all_clips = [c for clips, _ in sessions.values() for c in clips]
    feats = {c["name"]: (c["jaw"], lip_gap(c)) for c in all_clips}
    print("valid (face-present) frames per clip:")
    for c in all_clips:
        if not c["face"].all():
            print(f"    {c['name']}: {c['face'].sum()}/{len(c['face'])}")
    print("    (all other clips 100%)")
    for s, (clips, _) in sessions.items():
        print(f"  signal ceilings, {s}:")
        signal_table(clips, feats)

    tclips = [c for s in tune for c in sessions[s][0]]
    groups = [(s, r) for s in tune for r in sessions[s][1]]  # one fold per (session, rep)
    neg_ceiling = max(np.nanmax(feats[c["name"]][0][c["face"]]) for c in all_clips if c["label"] in NEG_LABELS)
    separable = True
    print(f"\n  separation check per tune session (gesture jawOpen max vs all-negative ceiling {neg_ceiling:.4f}):")
    for s in tune:
        ges = np.concatenate([feats[c["name"]][0][c["face"] & (c["cue"] == "gesture")]
                              for c in sessions[s][0] if c["label"] in CORNERS])
        ok = bool(len(ges) and ges.max() > neg_ceiling)
        separable &= ok
        print(f"    {s}: gesture max {ges.max():.4f} -> "
              f"{'OK' if ok else 'FAIL — gesture-window signal does not exceed negatives'}")

    print(f"\n  Leave-one-(session, rep)-out CV over {tune} ({len(groups)} folds; thresholds chosen on the other "
          f"{len(groups) - 1} groups, jawOpen floor {JAW_MIN_FLOOR}, hold {DEBOUNCE_MS} ms):")
    held_tp = held_pos = held_fp = 0
    folds = []
    held_lat, fold_unsep = [], []
    for s, r in groups:
        in_group = lambda c: c["session"] == s and c["rep"] == r  # noqa: E731
        cfg, tr, mg = select([c for c in tclips if not in_group(c)], feats)
        te = eval_trigger([c for c in tclips if in_group(c)], cfg, feats)
        r = f"{s}/rep{r}"
        held_tp += te["tp"]; held_pos += te["n_pos"]; held_fp += len(te["fp"]); held_lat += te["lat_cue"]
        if not mg["separable"]:
            fold_unsep.append(r)
        folds.append(dict(test_rep=r, cfg=cfg, margin=mg, fp=te["fp"], tp=te["tp"], n_pos=te["n_pos"],
                          lat_cue=te["lat_cue"], lat_onset=te["lat_onset"]))
        neg_fp = sum(1 for x in te["fp"] if x[2] == "negative")
        print(f"    test {r}: [{fmt_cfg(cfg)}] FP on {te['n_neg']} negative clips={neg_fp}, "
              f"FP on {te['n_pos']} fixate windows={len(te['fp']) - neg_fp}; HELD-OUT TP {te['tp']}/{te['n_pos']} "
              f"(train TP {tr['tp']}/{tr['n_pos']})"
              + (f"; cue->trigger {te['lat_cue']} ms" if te["lat_cue"] else ""))
        print(f"      train margins: {fmt_margin(mg)}")
        print_fp(te)
    print(f"    held-out totals: FP {held_fp}, TP {held_tp}/{held_pos}"
          + (f"; cue->trigger median {np.median(held_lat):.0f} ms (range {min(held_lat)}-{max(held_lat)}, "
             f"includes human reaction)" if held_lat else ""))

    cfg, trn, mg = select(tclips, feats)
    print(f"\n  Candidate config, fit on ALL reps of {'+'.join(tune)} [TRAINING-ONLY metrics]: {fmt_cfg(cfg)}")
    print(f"    margins: {fmt_margin(mg)}")
    print(f"    train FP {len(trn['fp'])}, train TP {trn['tp']}/{trn['n_pos']}")
    if trn["lat_cue"]:
        print(f"    train latency cue->trigger ms: median {np.median(trn['lat_cue']):.0f} "
              f"(range {min(trn['lat_cue'])}-{max(trn['lat_cue'])}, includes human reaction); "
              f"onset->trigger median {np.median(trn['lat_onset']):.0f} ms")
    if len(tune) > 1:
        print("\n  Cross-session trigger check among tune sessions (fit on one, test on another) [reported, not gated]:")
        for a in tune:
            for b in tune:
                if a == b:
                    continue
                xc, _, xm = select(sessions[a][0], feats)
                xr = eval_trigger(sessions[b][0], xc, feats)
                print(f"    {a} -> {b}: [{fmt_cfg(xc)}] FP {len(xr['fp'])}, TP {xr['tp']}/{xr['n_pos']}"
                      + (f", cue->trigger median {np.median(xr['lat_cue']):.0f} ms" if xr["lat_cue"] else ""))
                print_fp(xr, indent="        ")
    others = [c for s, (cl, _) in sessions.items() if s not in tune for c in cl]
    ext = eval_trigger(others, cfg, feats) if others else None
    if ext:
        print(f"  Candidate applied to OTHER sessions (independent): "
              f"FP {len(ext['fp'])} over {ext['n_neg']} negative clips + {ext['n_pos']} fixate windows; "
              f"TP {ext['tp']}/{ext['n_pos']}")
        print_fp(ext)

    reasons = []
    if not separable:
        reasons.append("gesture signal does not separate from negatives")
    if not mg["separable"]:
        reasons.append("no positive margin (gesture floor <= negative/fixate ceiling) on all reps")
    if fold_unsep:
        reasons.append(f"no positive margin in training folds for held-out groups {fold_unsep}")
    if held_tp == 0:
        reasons.append("0 held-out true positives")
    if held_fp:
        reasons.append(f"{held_fp} held-out false positive(s)")
    if ext and ext["fp"]:
        reasons.append(f"{len(ext['fp'])} false positive(s) on other sessions")
    deployable = not reasons
    print(f"\n  DEPLOYABLE: {'YES' if deployable else 'NO — ' + '; '.join(reasons)}")
    trigger_config = (dict(jawOpen=cfg[0], lip_gap=cfg[1], debounce_ms=cfg[2], tune_sessions=tune,
                           margins=mg, heldout_tp=held_tp, heldout_n_pos=held_pos,
                           heldout_cue_latency_median_ms=(float(np.median(held_lat)) if held_lat else None))
                      if deployable else None)
    if deployable:
        TRIGGER_CONFIG_PATH.parent.mkdir(exist_ok=True)
        TRIGGER_CONFIG_PATH.write_text(json.dumps(trigger_config, indent=1))
        print(f"  saved {TRIGGER_CONFIG_PATH.relative_to(ROOT)}")
    elif TRIGGER_CONFIG_PATH.exists():
        TRIGGER_CONFIG_PATH.unlink()  # never leave a stale validated config behind
        print(f"  removed stale {TRIGGER_CONFIG_PATH.relative_to(ROOT)}")
    return dict(
        tune_sessions=tune, separable=separable, deployable=deployable, refusal_reasons=reasons,
        trigger_config=trigger_config,
        training_only=dict(cfg=cfg, margins=mg, fp=trn["fp"], tp=trn["tp"], n_pos=trn["n_pos"],
                           lat_cue=trn["lat_cue"]),
        heldout=dict(folds=folds, tp=held_tp, n_pos=held_pos, fp=held_fp),
        other_sessions=(dict(fp=ext["fp"], tp=ext["tp"], n_pos=ext["n_pos"], n_neg=ext["n_neg"]) if ext else None),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--tune-sessions", nargs="+", default=DEFAULT_TUNE_SESSIONS,
                    help="sessions with real open-mouth gestures; trigger is tuned on these")
    args = ap.parse_args()
    tune = args.tune_sessions
    if not set(tune) <= set(args.sessions):
        invalid(f"--tune-sessions {tune} must be a subset of --sessions {args.sessions}")

    print(f"# eval_offline — sessions {args.sessions}, mouth tuned on {tune} "
          f"(jawOpen floor {JAW_MIN_FLOOR}, hold {DEBOUNCE_MS} ms)")
    sessions = {}
    for s in args.sessions:
        clips, reps = load_session(s)
        sessions[s] = (clips, reps)
        n_c = sum(c["label"] in CORNERS for c in clips)
        n_n = sum(c["label"] in NEG_LABELS for c in clips)
        face = np.mean(np.concatenate([c["face"] for c in clips]))
        print(f"{s}: {len(clips)} clips ({n_c} corner, {n_n} negative), reps {reps}, face-present {face:.1%} — integrity OK")
    if len(sessions) < 2:
        print("cross-SESSION error: NOT MEASURABLE (only one session)")

    mouth = eval_mouth(sessions, tune)
    print("\n## 2. Gaze quadrant (nearest centroid; face-absent / non-finite frames excluded)")
    gaze_iris = eval_gaze(sessions, [0, 1], "iris only: h, v")
    gaze_head = eval_gaze(sessions, [0, 1, 2, 3], "iris + head: h, v, yaw, pitch")

    out = ROOT / "logs" / f"eval_{'+'.join(args.sessions)}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(sessions=args.sessions, mouth=mouth, labels=CORNERS,
                                   gaze_iris=gaze_iris, gaze_iris_head=gaze_head), indent=1, default=int))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
