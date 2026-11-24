"""Leave each recording session out of acoustic calibration and replay its clips."""
import json
from pathlib import Path

from handsfree.audio_arbitration import AudioSpeechDetector, FRAME_SAMPLES, audio_speech_ratios, recorded_audio
from training.silent_speech_bench import OFF_MS

ROOT = Path(__file__).resolve().parents[1] / "data" / "silent_speech"


def onset_times(pcm, meta, ratios):
    detector = AudioSpeechDetector(*ratios, OFF_MS)
    for offset in range(0, len(pcm) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        end_ms = meta["audio_start_ms"] + (offset + FRAME_SAMPLES) / 16
        visual = any(start <= end_ms <= end for start, end in meta["vad_segments_ms"])
        if detector.update(pcm[offset:offset + FRAME_SAMPLES], end_ms, visual) is True:
            yield end_ms


def replay(pcm, meta, ratios):
    return sum(1 for _ in onset_times(pcm, meta, ratios))


def main():
    rows = [row for mode in ("silent", "whisper") for row in recorded_audio(ROOT, mode)]
    sessions = sorted({folder.parent.name for folder, _, _ in rows})
    folds = []
    false_onsets = []
    for session in sessions:
        ratios = audio_speech_ratios(ROOT, exclude_session=session)
        silent = false = whispers = recalled = 0
        for folder, meta, pcm in rows:
            if folder.parent.name != session:
                continue
            starts = list(onset_times(pcm, meta, ratios))
            if meta["mode"] == "silent":
                silent += 1
                false += bool(starts)
                if starts:
                    false_onsets.append({"clip": str(folder.relative_to(ROOT)), "onsets_ms": starts})
            else:
                whispers += 1
                recalled += any(meta["speech_start_ms"] <= t <= meta["speech_end_ms"] for t in starts)
        folds.append({"session": session, "silent_clips": silent, "clips_with_audio_onset": false,
                      "whisper_clips": whispers, "whisper_clips_recalled": recalled})

    total = sum(f["silent_clips"] for f in folds)
    whispered = sum(f["whisper_clips"] for f in folds)
    hits = sum(f["whisper_clips_recalled"] for f in folds)
    print(json.dumps({
        "silent_clips_with_audio_onset": len(false_onsets), "silent_clips": total,
        "false_onsets": false_onsets, "leave_one_session_out": folds,
        "whisper_clips_with_audio_onset_during_speech": hits, "whisper_clips": whispered,
        "whisper_recall": hits / whispered if whispered else None,
        "note": "Each fold trains exclusively on other sessions' silent audio. Any acoustic onset counts against a silent clip.",
    }, indent=2))
    # The supplied corpus has 25 silent and 20 whisper clips. Its prior
    # sustained-RMS detector recognized 10 of the 20 recorded whispers.
    return int(total != 25 or bool(false_onsets) or whispered != 20 or hits < 10)


if __name__ == "__main__":
    raise SystemExit(main())
