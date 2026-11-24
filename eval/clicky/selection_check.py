"""Read selections from an app-owned Cocoa editor loaded with the real workspace document."""

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import AppKit as A
import ApplicationServices as AX

from clicky import agent, ax, memory, teach
from clicky.capture import Context
from clicky.memory import observed_window
from clicky.models import answer

from .corpus import ROOT


def live_textedit(provider):
    source = (ROOT / "data/clicky/task-workspace/Clicky-source.txt").resolve()
    passage = "FITNESS FOR A PARTICULAR PURPOSE"
    if ax.session_locked():
        raise SystemExit("The desktop is locked. No window was read.")
    print(f"Switch to {source.name} in TextEdit and highlight “{passage}”. Checking in 12 seconds.", flush=True)
    time.sleep(12)
    if ax.session_locked():
        raise SystemExit("The desktop locked before the check.")
    pid, app = ax.frontmost()
    running = A.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if pid else None
    if ax.session_locked() or not running or running.bundleIdentifier() != "com.apple.TextEdit":
        raise SystemExit("The unlocked foreground app must be TextEdit.")
    root = AX.AXUIElementCreateApplication(pid)
    doc = ax.attr(root, "AXFocusedWindow")
    document_url = ax.attr(doc, "AXDocument")
    url = A.NSURL.URLWithString_(str(document_url or ""))
    if not url or not url.isFileURL() or Path(url.path()).resolve() != source:
        raise SystemExit("Open the specified workspace document in TextEdit.")
    window = ax.read_window(pid, app)
    if (ax.session_locked() or window.ref != doc or ax.frontmost()[0] != pid
            or ax.attr(doc, "AXDocument") != document_url
            or window.selected_text != passage or not window.typing_anchor):
        raise SystemExit("The expected document and native text selection were not observed.")
    reply = "".join(answer("Quote only the passage I have highlighted.", provider, Context(window)))
    if passage.casefold() not in reply.casefold():
        raise SystemExit("The answer missed the selected passage.")
    print("Cross-app Accessibility read the selected passage and the model quoted it. No screenshot or edit was needed.")


def check(provider):
    A.NSApplication.sharedApplication()
    source = ROOT / "data/clicky/task-workspace/Clicky-source.txt"
    document = source.read_text()
    editor = A.NSTextView.alloc().initWithFrame_(((0, 0), (680, 420)))
    editor.setString_(document)
    app = A.NSProcessInfo.processInfo().processName()
    window = A.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (680, 420)), A.NSWindowStyleMaskTitled, A.NSBackingStoreBuffered, False
    )
    window.setTitle_(source.name)
    window.setContentView_(editor)
    window.makeFirstResponder_(editor)

    def anchor():
        return ax.typing_anchor(editor.accessibilitySelectedText(), tuple(editor.selectedRange()))

    def select(passage):
        value = editor.string()
        editor.setSelectedRange_(value.rangeOfString_(passage) if passage else (0, 0))
        assert window.firstResponder() == editor
        element = ax.Element(editor.accessibilityRole(), str(editor.accessibilityValue())[:280], None, focused=True)
        selected = ax.selection_excerpt(editor.accessibilitySelectedText())
        return Context(ax.Window(0, str(app), str(window.title()), elements=[element], selected_text=selected))

    first = "copies or substantial portions of the Software"
    second = "FITNESS FOR A PARTICULAR PURPOSE"
    cleared, before, after = select(""), select(first), select(second)
    assert first not in after.window.elements[0].label and second not in after.window.elements[0].label
    assert json.loads(cleared.description())["selected_text"] == ""
    assert "selected_text" not in json.loads(Context(ax.Window(0, str(app))).description())
    assert before.window.selected_text == first and after.window.selected_text == second
    assert before.window.state() != after.window.state()
    request = 'Guide me through highlighting the exact words "MIT License" at the top using the mouse.'
    instruction, _, _ = teach.next_instruction(request, [], [], before, provider, starting=observed_window(before.window))
    planned = bool(instruction and instruction.highlight_text == "MIT License")
    if planned:
        assert teach.verify_transition(instruction, before, select("MIT License"), provider)
    print("Planner supplied the exact highlight criterion:", planned, flush=True)
    editor.setSelectedRange_((0, 0))
    original = anchor()
    assert original == anchor()
    editor.setSelectedRange_((1, 0))
    assert original != anchor(), "Moving an empty caret preserved typing approval."
    word = "Software"
    editor.setSelectedRange_((document.index(word), len(word)))
    original = anchor()
    editor.setSelectedRange_((document.rindex(word), len(word)))
    assert editor.accessibilitySelectedText() == word and original != anchor()
    # Use a real, longer document for the excerpt boundary.
    recording = next((ROOT / "data/clicky/development").glob("finder-roundtrip-*/start.json"))
    text = recording.read_text()
    editor.setString_(text)
    editor.selectAll_(None)
    assert len(text) > 1600 and ax.selection_excerpt(editor.accessibilitySelectedText()) == text[:1600] + "…"
    original = anchor()
    last = editor.selectedRange().length - 1
    editor.textStorage().replaceCharactersInRange_withString_((last, 1), document[0])
    editor.selectAll_(None)
    assert ax.selection_excerpt(editor.accessibilitySelectedText()) == text[:1600] + "…"
    assert original[0] == anchor()[0] and original != anchor(), "An edit beyond the excerpt preserved approval."
    print("A moved caret, a second occurrence and an edit beyond the excerpt each changed the typing destination.")

    goal = teach.Step("Select the fitness clause.", second, f"The highlighted words are {second}.", False, highlight_text=second)
    assert not teach.verify_transition(goal, cleared, before, provider), "The guide accepted an unrelated highlight."
    assert not teach.verify_transition(goal, before, cleared, provider), "Clearing the selection advanced the guide."
    changed = teach.verify_transition(goal, before, after, provider)
    assert changed and (changed["highlight_before"], changed["highlight_after"]) == (first, second)
    task = f"Leave the words {second} highlighted in the document."
    for ctx in (before, cleared, after):
        status, _ = agent.verify_result(task, observed_window(cleared.window), [], ctx, provider)
        assert (status == "verified") == (ctx is after), "The agent misread the current highlight."
    question = "Quote only the words I have highlighted."
    history = []
    for ctx, expected, other in ((before, first, second), (after, second, first)):
        reply = "".join(answer(question, provider, ctx, conversation=history))
        assert expected.casefold() in reply.casefold(), "The answer missed the selected passage."
        assert other.casefold() not in reply.casefold(), "The answer included an unselected passage."
        history.append({"request": question, "reply": reply})
    response = "".join(answer("Is any text currently selected? Answer yes or no.", provider, cleared, conversation=history))
    assert response.casefold().strip().startswith("no"), "The answer carried an old highlight into the cleared editor."
    print("The guide and agent rejected the wrong and cleared highlights, then verified the requested selection. "
          "Answers identified both passages and the empty selection.")

    restore = teach.Step("Select the original passage again.", first, f"The highlighted text is {first}.", False, highlight_text=first)
    task = "Select the fitness clause and then highlight the passage that was originally selected."
    state = {
        "goal": task, "starting": observed_window(before.window), "plan": [asdict(goal), asdict(restore)],
        "verified": [changed], "questions": [], "pending": {**asdict(restore), "before": observed_window(after.window)},
    }
    normal_db = memory.DATABASE
    with tempfile.TemporaryDirectory() as folder:
        try:
            memory.DATABASE = Path(folder) / "history.sqlite3"
            key = "walkthrough/selected-passage"
            memory.save_walkthrough(key, state, 30, create=True)
            saved = memory.walkthrough(key, 30)
        finally:
            memory.DATABASE = normal_db
    assert saved["starting"]["highlight_excerpt"] == first
    editor.setString_(document)
    assert teach.reconcile_step(saved["pending"], select(second), provider) is None, "Resume credited an unrestored selection."
    restored = select(first)
    progress = teach.reconcile_step(saved["pending"], restored, provider)
    assert progress and (progress["highlight_before"], progress["highlight_after"]) == (second, first)
    done, _ = teach.check_goal(saved["goal"], saved["starting"], [*saved["verified"], progress], restored, provider)
    assert done, "The restored selection was insufficient to finish the recorded round trip."
    print("A saved walkthrough kept the original excerpt and finished after the original selection returned.")
    if not planned:
        raise SystemExit("The planner did not propose a literal highlight step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("Anthropic", "Local"), default="Anthropic", nargs="?")
    parser.add_argument("--live-textedit", action="store_true", help="Read a highlight from the workspace license open in TextEdit.")
    args = parser.parse_args()
    if args.live_textedit:
        live_textedit(args.provider)
    else:
        check(args.provider)
