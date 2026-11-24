"""Exercise custom spelling preferences and replay recorded silent and whispered utterances."""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import AppKit as A
import objc
from objc import super
from PyObjCTools import AppHelper

from cursormix import memory
from cursormix.app import ClickyDelegate, end_run, main
from cursormix.inputs.voice import VoiceInput
from cursormix.settings import preferences
from cursormix.voice import Recording, apply_spellings, transcribe

from .onboarding_check import buttons

WORDS = "echo = Echo\necho india = EchoIndia"


class WordsCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.words-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_({"didCompleteOnboarding": True, "shortcut": "Command Shift Space"})
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        self.error = None
        timer = A.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            0.15, self, "readDialog:", None, True
        )
        A.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, A.NSModalPanelRunLoopMode)
        try:
            settings = self.menu.itemWithTitle_("Settings").submenu()
            languages = settings.itemWithTitle_("Dictation Language").submenu()
            assert self.dictation_languages["default"].state()
            for code in ("fr", "auto", "default"):
                choice = self.dictation_languages[code]
                assert choice.menu() == languages
                A.NSApp.sendAction_to_from_(choice.action(), choice.target(), choice)
                assert preferences(self.suite).stringForKey_("dictationLanguage") == code
                assert [k for k, v in self.dictation_languages.items() if v.state()] == [code]
            command = settings.itemWithTitle_("Custom Dictation Words…")
            for before, edits, expected in (
                ("", [("Save", "echo ="), ("Save", WORDS)], WORDS),
                (WORDS, [("Cancel", "")], WORDS),
                (WORDS, [("Save", "")], ""),
            ):
                self.previous = before
                self.edits = iter(edits)
                A.NSApp.sendAction_to_from_(command.action(), command.target(), command)
                if self.error:
                    raise self.error
                assert preferences(self.suite).stringForKey_("dictationSpellings") == expected
        except Exception as error:  # noqa: BLE001 — relay native callback failures
            WordsCheck.failure = error
        finally:
            timer.invalidate()
            self.store.removePersistentDomainForName_(self.suite)
            AppHelper.callAfter(end_run)

    def readDialog_(self, timer):
        window = A.NSApp.modalWindow()
        if window is None:
            return
        try:
            field = window.initialFirstResponder()
            assert isinstance(field, A.NSTextView) and field.string() == self.previous
            action, text = next(self.edits)
            self.previous = text
            field.setString_(text)
            next(b for b in buttons(window.contentView()) if b.title() == action).performClick_(None)
        except Exception as error:  # noqa: BLE001
            self.error = error
            A.NSApp.abortModal()


def replay(data):
    by_session = defaultdict(Counter)
    examples = []
    for meta in sorted(data.glob("*/*/meta.json")):
        details = json.loads(meta.read_text())
        mode = details["mode"]
        path = meta.with_name("audio_16k.wav")
        if mode not in ("silent", "whisper") or not path.is_file():
            continue
        plain = transcribe(path, "Local")
        written = apply_spellings(plain, WORDS)
        stats = by_session[meta.parent.parent.name]
        stats[mode] += 1
        if mode == "silent":
            stats["false_transcripts"] += bool(re.search(r"\w", written))
            assert bool(plain.strip()) == bool(written.strip())
        else:
            phrase = details["phrase"]
            if phrase == "echo india" and "EchoIndia" in written:
                examples.append(path)
            expected = apply_spellings(phrase, WORDS)
            stats["errors"] += re.findall(r"\w+", written.casefold()) != re.findall(r"\w+", expected.casefold())
    # This recording previously produced a long transcript despite Whisper's
    # no-speech prediction on every segment.
    regression = by_session.get("20261119-210913-272809", {})
    assert regression.get("silent") == 1 and regression["false_transcripts"] == 0
    assert examples, "No recorded phrase yielded the preferred spelling."
    for path in examples:
        recording = Recording()
        shutil.copyfile(path, recording.path)
        assert "EchoIndia" in VoiceInput(recording, "Local", WORDS).read()
        assert not recording.path.exists()
    for session, stats in by_session.items():
        print(f"{session}: {stats['false_transcripts']}/{stats['silent']} silent false transcripts, "
              f"{stats['errors']}/{stats['whisper']} whispered phrase errors.")
    print(f"Recorded utterances with the custom spelling through VoiceInput: {len(examples)}.")


def replay_languages(folder, data):
    # Human pronunciation recordings from Wikimedia Commons, identified by their source filenames.
    examples = (
        ("es", "hola", "Es-hola.oga", "f574df9002dea6b8e0fde26ccb81be1b20ae601ad46898a7df2a91eea6e40b88"),
        ("fr", "bonjour", "Fr-bonjour.ogg", "9f60a623dd5ea98f1a57f81ef93f220e2d4e0193ae4275d596453a3428d8d925"),
    )
    sessions = defaultdict(list)
    for meta in sorted(data.glob("*/*/meta.json")):
        if json.loads(meta.read_text())["mode"] == "silent":
            sessions[meta.parent.parent.name].append(meta.with_name("audio_16k.wav"))
    assert sessions, "No real silent recordings were found."
    for session, paths in sessions.items():
        transcripts = [transcribe(path, "Local", "auto") for path in paths]
        print(f"{session}: {sum(bool(t) for t in transcripts)}/{len(paths)} silent clips produced text.")
    print("One recording per language. Cross-session recognition error is unmeasured.")
    correct = 0
    for code, expected, name, digest in examples:
        original = folder / name
        assert sha256(original.read_bytes()).hexdigest() == digest, name
        for language in (code, "auto"):
            recording = Recording()
            try:
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(original),
                                "-ar", "16000", "-ac", "1", str(recording.path)], check=True)
                heard = VoiceInput(recording, "Local", language=language).read()
            finally:
                recording.close()
            assert not recording.path.exists()
            matches = re.findall(r"\w+", heard.casefold()) == [expected]
            print(f"{name} ({language}): {heard!r}; exact words {matches}")
            correct += language == code and matches
    assert correct == len(examples), "An explicitly selected language missed its recorded word."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/silent_speech"))
    parser.add_argument("--languages", type=Path, help="Replay the original French and Spanish audio in this folder.")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        memory.DATABASE = Path(tmp) / "history.sqlite3"
        WordsCheck.failure = None
        main(delegate_type=WordsCheck, argv=[])
        if WordsCheck.failure:
            raise WordsCheck.failure
    print("Native Settings rejected an incomplete entry, saved, cancelled edits and cleared the vocabulary.", flush=True)
    if args.languages:
        replay_languages(args.languages, args.data)
    else:
        replay(args.data)
