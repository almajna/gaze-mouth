"""Offline, open-vocabulary evaluation of a completed silent_speech_bench session.

Setup: clone https://github.com/mpc001/auto_avsr into models/auto_avsr and put
the official vsr_trlrs2lrs3vox2avsp_base.pth there as vsr.pth. Download the
mlx-community models named below into matching local models/ directories.
Run with .venv-gf/bin/python -m eval.eval_silent_speech [SESSION].
"""
import argparse
import gc
import json
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter
import wave

import numpy as np
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WHISPER = ("whisper-small.en-mlx", "whisper-large-v3-turbo")
DEFAULT = ROOT / "data/silent_speech/20261119-211044-711080"


# Midpoint between the largest silent and smallest whispered band contrast in
# session 20261119-211044-711080. Validation on another session is pending.
LOW_BAND_DB = 6.6
PROMPT = "Short macOS commands and spoken questions."


def band_peak_snr(samples, meta):
    """P95 of 20 ms speech-band frame powers relative to median quiet-frame power."""
    band = sosfilt(butter(3, [300, 3400], btype="bandpass", fs=16000, output="sos"), samples)
    frames = np.lib.stride_tricks.sliding_window_view(band, 320)[::160]
    power = np.mean(frames**2, axis=1)
    centers = meta["audio_start_ms"] + np.arange(len(frames)) * 10 + 10
    active = np.zeros(len(frames), bool)
    for onset, release in meta["vad_segments_ms"]:
        # The collector confirms release only after 350 ms of stillness.
        active |= (centers >= onset) & (centers < release - 350)
    if not active.any() or active.all():
        raise ValueError("Need speech and quiet frames for band SNR")
    floor = np.finfo(float).tiny
    return float(10 * (np.log10(max(floor, np.percentile(power[active], 95)))
                       - np.log10(max(floor, np.median(power[~active])))))


def audio_input(folder, meta):
    with wave.open(str(folder / "audio_16k.wav"), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{folder}: expected 16 kHz mono PCM16")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
    mask = np.zeros(samples.size, dtype=bool)
    origin = meta["audio_start_ms"]
    for start, end in meta["vad_segments_ms"]:
        mask[max(0, (start-origin-meta.get("pre_roll_ms", 150))*16):max(0, (end-origin+150)*16)] = True
    filtered = sosfilt(butter(1, 120, fs=16000, btype="highpass", output="sos"), samples)
    emphasized = filtered.copy()
    emphasized[1:] -= 0.85 * filtered[:-1]
    x = emphasized[mask]
    if not len(x):
        raise ValueError(f"{folder}: no samples overlap the padded V-VAD intervals")
    peak = np.max(np.abs(x))
    if peak:
        x *= min(0.1 / np.sqrt(np.mean(x*x)), 0.98 / peak)
    return x.astype(np.float32), band_peak_snr(samples, meta)


def resample_video(video, meta):
    times = np.asarray(meta["source_frame_times_ms"], dtype=float)
    if len(times) != len(video) or np.any(np.diff(times) < 0):
        raise ValueError("Missing or unordered source frame timestamps")
    times, indices = np.unique(times, return_index=True)
    # Stored NPY slots span the original segment at video_fps. Interpolate
    # the actual camera observations without stretching that segment.
    ticks = meta["audio_start_ms"] + np.arange(0, len(video)*1000/meta["video_fps"], 40)
    position = np.interp(ticks, times, np.arange(len(times)))
    left = position.astype(int)
    right = np.minimum(left+1, len(times)-1)
    weight = (position-left).astype(np.float32)[:, None, None]
    x = video[indices[left]] * (1-weight) + video[indices[right]] * weight
    return np.pad(x, ((4, 4), (0, 0), (0, 0)), mode="edge")


def visual_decoder(device):
    import torch
    repo = MODELS / "auto_avsr"
    sys.path.insert(0, str(repo))
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

    units = (repo / "spm/unigram/unigram5000_units.txt").read_text().splitlines()
    tokens = ["<blank>", *[line.split()[0] for line in units], "<eos>"]
    model = E2E(len(tokens), "video")
    model.load_state_dict(torch.load(repo / "vsr.pth", map_location="cpu", weights_only=True))
    model.eval().to(device)
    beam = BatchBeamSearch(beam_size=8, vocab_size=len(tokens), weights={"decoder": .9, "ctc": .1},
                           scorers=model.scorers(), sos=model.sos, eos=model.eos, token_list=tokens)

    @torch.inference_mode()
    def decode(video):
        # Recorder already aligned frames to 25 fps. Match Auto-AVSR's test transform.
        x = torch.from_numpy(video[:, 4:92, 4:92]).to(device, dtype=torch.float32)
        x = ((x / 255 - .421) / .165)[None, :, None]
        encoded, _ = model.encoder(model.proj_encoder(model.frontend(x)), None)
        ids = beam(encoded[0])[0].yseq.tolist()
        return "".join(tokens[i] for i in ids if i not in (model.sos, model.eos, 0)).replace("▁", " ").strip()
    return decode


def timed(fn, *args):
    start = perf_counter()
    value = fn(*args)
    return value, (perf_counter() - start) * 1000


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower().replace("’", "'"))


def gate(result, snr):
    accepted = []
    confident = True
    for segment in result["segments"]:
        text = segment["text"].strip()
        normalized = " ".join(words(text))
        logprob = segment["avg_logprob"]
        no_speech = segment["no_speech_prob"]
        if normalized in {"", "thank you", "thanks for watching", "you", " ".join(words(PROMPT))}:
            continue
        if segment["compression_ratio"] > 2.2:
            continue
        if snr < LOW_BAND_DB and (no_speech > .65 or logprob < -1.0):
            continue
        accepted.append(text)
        # In this session the weakest genuine turbo hypothesis scored -1.005.
        confident &= snr >= LOW_BAND_DB and no_speech <= .65 and logprob >= -1.01
    return " ".join(accepted), bool(accepted and confident)


def errors(reference, prediction):
    ref, hyp = words(reference), words(prediction)
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        nxt = [i]
        for j, guess in enumerate(hyp, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(word != guess)))
        row = nxt
    return row[-1], len(ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="Auto-AVSR device")
    args = ap.parse_args()
    rows = []
    for i in range(1, 21):
        mode = "silent" if i <= 10 else "whisper"
        folder = args.session / f"{i:02}_{mode}"
        meta = json.loads((folder / "meta.json").read_text())
        video = np.load(folder / "mouth_96x96.npy", allow_pickle=False)
        if (meta["trial"] != i or meta["mode"] != mode or meta["video_fps"] <= 0 or
                video.dtype != np.uint8 or video.shape[1:] != (96, 96) or not len(video)):
            raise ValueError(f"Unexpected clip metadata or crop format: {folder}")
        video, video_ms = timed(resample_video, video, meta)
        (audio, snr), prep_ms = timed(audio_input, folder, meta)
        rows.append(dict(id=folder.name, mode=mode, ref=meta["phrase"], video=video, audio=audio, snr=snr, prep_ms=prep_ms, video_ms=video_ms))

    import torch
    torch.set_num_threads(4)
    visual = visual_decoder(args.device)
    visual(rows[0]["video"])  # Untimed real-clip warmup
    for r in rows:
        r["vsr"], r["vsr_ms"] = timed(visual, r.pop("video"))
        r["vsr_ms"] += r["video_ms"]
    del visual
    gc.collect()

    import mlx.core as mx
    from mlx_whisper.transcribe import transcribe, ModelHolder
    for name, key in zip(WHISPER, ("small", "turbo")):
        path = MODELS / name
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path)
        def decode(x, snr):
            result = transcribe(x, path_or_hf_repo=str(path), language="en", temperature=0.,
                                condition_on_previous_text=False, verbose=None,
                                initial_prompt=PROMPT)
            mx.synchronize()
            return gate(result, snr)
        decode(rows[0]["audio"], rows[0]["snr"])
        for r in rows:
            (r[key], r[key+"_confident"]), ms = timed(decode, r["audio"], r["snr"])
            r[key+"_ms"] = r["prep_ms"] + ms
        ModelHolder.model = ModelHolder.model_path = None
        mx.clear_cache()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model = tokenizer = None
    sampler = make_sampler(temp=0.0)

    def correct(small, turbo, visual):
        messages = [
            {"role": "system", "content": "Restore one short natural English utterance or macOS command "
             "from three noisy phonetic transcripts of the same utterance. Prefer shared evidence. "
             "Repair phonetic near-misses without adding extra words. Output only the transcript."},
            {"role": "user", "content": json.dumps({"audio_1": small, "audio_2": turbo, "lips": visual})},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        output = generate(model, tokenizer, prompt=prompt, sampler=sampler, max_tokens=32, verbose=False)
        mx.synchronize()
        return output.strip()


    keys = ("small", "turbo", "vsr", "raw", "corrected")
    print("clip | reference | small | turbo | VSR | gated turbo | final | band dB | route | latency ms | WER %")
    print("Latency and WER order: small / turbo / VSR / raw hybrid / corrected")
    for r in rows:
        r["raw"] = r["turbo"]
        r["raw_ms"] = r["turbo_ms"]
        route = "turbo" if r["turbo"] else "reject"
        r["corrected"], r["corrected_ms"] = r["raw"], r["raw_ms"]
        if r["turbo"] and not r["turbo_confident"]:
            route = "Qwen"
            if model is None:
                model, tokenizer = load(str(MODELS / "Qwen2.5-3B-Instruct-4bit"))
                correct(r["small"], r["turbo"], r["vsr"])  # warmup
            r["corrected"], ms = timed(correct, r["small"], r["turbo"], r["vsr"])
            r["corrected_ms"] += r["small_ms"] - r["prep_ms"] + r["vsr_ms"] + ms
        lat = "/".join(f"{r[k+'_ms']:.1f}" for k in keys)
        scores = []
        for k in keys:
            e, n = errors(r["ref"], r[k])
            scores.append(f"{100*e/n:.0f}")
        print(" | ".join([r["id"], r["ref"], *[r[k] or "∅" for k in keys], f"{r['snr']:.2f}", route, lat, "/".join(scores)]))
    print("Silent audio false positives (out of 10):", {k: sum(bool(r[k]) for r in rows[:10])
                                                    for k in ("small", "turbo", "corrected")})
    for key, subset in [("vsr", rows[:10]), ("vsr", rows[10:]), ("small", rows[10:]),
                        ("turbo", rows[10:]), ("raw", rows), ("corrected", rows[:10]),
                        ("corrected", rows[10:]), ("corrected", rows)]:
        e, n = map(sum, zip(*(errors(r["ref"], r[key]) for r in subset)))
        print(f"{key} ({subset[0]['mode'] if len(subset)==10 else 'all'}): {e}/{n} word errors, "
              f"WER {100*e/n:.1f}%, median {statistics.median(r[key+'_ms'] for r in subset):.1f} ms")


if __name__ == "__main__":
    main()
