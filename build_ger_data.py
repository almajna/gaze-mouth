#!/usr/bin/env python3
"""Build the GER LoRA dataset (mlx-lm chat format) from out-of-fold beams of the training clips.

Each training clip gives one row with an empty screen context and one with a context drawn by the
sentence's domain. Contexts are picked by domain only and are never written from the reference.
"""
import argparse
import json
from pathlib import Path
import random

from context_decoder.ger import ger_messages, guess_lines
from silent_avsr.features import ROOT

TERMINAL = [
    dict(app_name="Terminal", ax_role="AXTextArea", ax_text="$ git status On branch main nothing to commit, working tree clean $", nearby_labels=["zsh", "main", "~/Projects/api"]),
    dict(app_name="iTerm2", ax_role="AXTextArea", ax_text="$ npm test PASS src/app.test.ts Tests: 14 passed, 14 total $", nearby_labels=["zsh", "node", "~/Projects/web"]),
    dict(app_name="Visual Studio Code", ax_role="AXTextArea", ax_text="def load_config(path): with open(path) as handle: return json.load(handle)", nearby_labels=["Explorer", "Search", "Source Control", "Run and Debug", "Terminal"]),
    dict(app_name="Cursor", ax_role="AXTextField", ax_text="Ask the agent to change this file", nearby_labels=["Chat", "Composer", "Terminal", "Problems", "Output"]),
    dict(app_name="Warp", ax_role="AXTextArea", ax_text="$ docker ps CONTAINER ID IMAGE STATUS PORTS $", nearby_labels=["zsh", "docker", "~/Projects/infra"]),
    dict(app_name="Xcode", ax_role="AXTextArea", ax_text="struct ContentView: View { var body: some View { Text(title) } }", nearby_labels=["Build", "Run", "Stop", "Navigator", "Debug area"]),
    dict(app_name="Claude", ax_role="AXTextArea", ax_text="Reply to Claude", nearby_labels=["New chat", "Projects", "Attach", "Send"]),
    dict(app_name="Ghostty", ax_role="AXTextArea", ax_text="$ python train.py --epochs 10 epoch 3 loss 0.412 $", nearby_labels=["zsh", "python", "~/Projects/ml"]),
]
INTERFACE = [
    dict(app_name="Safari", ax_role="AXWebArea", ax_text="Search or enter website name", nearby_labels=["Back", "Forward", "Reload", "Share", "New Tab", "Sidebar"]),
    dict(app_name="Google Chrome", ax_role="AXTextField", ax_text="Search Google or type a URL", nearby_labels=["Back", "Forward", "Reload", "Bookmarks", "Extensions", "New Tab"]),
    dict(app_name="Finder", ax_role="AXWindow", ax_text="Documents - 24 items", nearby_labels=["Recents", "Applications", "Desktop", "Documents", "Downloads"]),
    dict(app_name="System Settings", ax_role="AXButton", ax_text="Displays", nearby_labels=["Wi-Fi", "Bluetooth", "Sound", "Displays", "Keyboard", "Accessibility"]),
    dict(app_name="Preview", ax_role="AXScrollArea", ax_text="Page 2 of 14", nearby_labels=["Zoom In", "Zoom Out", "Share", "Markup", "Rotate", "Search"]),
    dict(app_name="Arc", ax_role="AXWebArea", ax_text="New Tab", nearby_labels=["Back", "Forward", "Reload", "Split View", "Library"]),
]
MESSAGING = [
    dict(app_name="Messages", ax_role="AXTextField", ax_text="iMessage", nearby_labels=["Compose", "Search", "Details", "Emoji", "Send"]),
    dict(app_name="Mail", ax_role="AXTextArea", ax_text="Inbox - 8 unread", nearby_labels=["Get Mail", "New Message", "Reply", "Forward", "Archive", "Flag"]),
    dict(app_name="Slack", ax_role="AXTextArea", ax_text="Message #general", nearby_labels=["Threads", "Mentions", "Drafts", "Huddle", "Send"]),
    dict(app_name="Calendar", ax_role="AXGroup", ax_text="Week of the 12th", nearby_labels=["Today", "Day", "Week", "Month", "Year", "New Event"]),
    dict(app_name="Reminders", ax_role="AXTextField", ax_text="New Reminder", nearby_labels=["Today", "Scheduled", "All", "Flagged", "Add List"]),
    dict(app_name="WhatsApp", ax_role="AXTextArea", ax_text="Type a message", nearby_labels=["Chats", "Calls", "Status", "Attach", "Voice message"]),
]
WRITING = [
    dict(app_name="Pages", ax_role="AXTextArea", ax_text="Untitled", nearby_labels=["Insert", "Table", "Chart", "Text", "Format", "Document"]),
    dict(app_name="Notes", ax_role="AXTextArea", ax_text="New Note", nearby_labels=["Folders", "New Note", "Checklist", "Table", "Share"]),
    dict(app_name="Google Chrome", ax_role="AXTextArea", ax_text="Untitled document - Google Docs", nearby_labels=["File", "Edit", "Insert", "Format", "Tools", "Share"]),
    dict(app_name="Microsoft Word", ax_role="AXTextArea", ax_text="Document1", nearby_labels=["Home", "Insert", "Layout", "References", "Review"]),
    dict(app_name="ChatGPT", ax_role="AXTextArea", ax_text="Ask anything", nearby_labels=["New chat", "Search chats", "Attach", "Send"]),
    dict(app_name="Keynote", ax_role="AXTextArea", ax_text="Slide 3 of 12", nearby_labels=["Play", "Table", "Chart", "Text", "Shape", "Format"]),
]
DICTATION = [
    dict(app_name="Notes", ax_role="AXTextArea", ax_text="New Note", nearby_labels=["Folders", "New Note", "Checklist", "Table", "Share"]),
    dict(app_name="Messages", ax_role="AXTextField", ax_text="iMessage", nearby_labels=["Compose", "Search", "Details", "Emoji", "Send"]),
    dict(app_name="Pages", ax_role="AXTextArea", ax_text="Untitled", nearby_labels=["Insert", "Table", "Chart", "Text", "Format", "Document"]),
    dict(app_name="TextEdit", ax_role="AXTextArea", ax_text="Untitled", nearby_labels=["Format", "Font", "Alignment", "Spacing"]),
    dict(app_name="Mail", ax_role="AXTextArea", ax_text="New Message", nearby_labels=["Send", "Attach", "Format", "To", "Subject"]),
    dict(app_name="Journal", ax_role="AXTextArea", ax_text="New Entry", nearby_labels=["Done", "Photos", "Location", "Bookmark"]),
]
CONTEXTS = dict(dev_terminal=TERMINAL, ui_navigation=INTERFACE, messaging_calendar=MESSAGING,
                document_writing=WRITING, everyday_statement=DICTATION, everyday_command=DICTATION,
                question=DICTATION)


def row(beam, context, variant, **extra):
    return dict(messages=ger_messages(beam["hypotheses"], context, beam["reference"]), clip=beam["clip"],
                variant=variant, reference=beam["reference"], screen_context=context,
                hypotheses=beam["hypotheses"], **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beams", type=Path, default=ROOT / "models/usr2_speaker_lora_v2_folds/oof_beams.json",
                    help="eval_wer.py rows for the training clips, each decoded by a model that never saw it")
    ap.add_argument("--valid-beams", type=Path,
                    help="eval_wer.py rows for the validation clips (default: their rows in --beams)")
    ap.add_argument("--split", type=Path, default=ROOT / "models/usr2_speaker_lora/training.json")
    ap.add_argument("--domains", type=Path, default=ROOT / "train_sentence_domains.json")
    ap.add_argument("--out", type=Path, default=ROOT / "models/ger/data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    split = json.loads(args.split.read_text())
    beams = {Path(beam["clip"]).name: beam for beam in json.loads(args.beams.read_text())}
    missing = (set(split["train_clips"]) | set(split["val_clips"])) - set(beams)
    if missing:
        ap.error(f"{args.beams} has no row for {sorted(missing)[:5]}")
    domains = {f"{entry['index']:03d}.npz": entry["domain"] for entry in json.loads(args.domains.read_text())}
    rng = random.Random(args.seed)
    train = []
    for name in split["train_clips"]:
        beam = beams[name]
        if any("eval_holdout" in str(value) for value in (beam["clip"], beam.get("session", ""))):
            raise ValueError("holdout rows do not belong in the GER training data")
        if not guess_lines(beam["hypotheses"]):
            continue
        train.append(row(beam, None, "no_context"))
        train.append(row(beam, rng.choice(CONTEXTS[domains[name]]), "domain_context", domain=domains[name]))
    rng.shuffle(train)
    source = {Path(beam["clip"]).name: beam for beam in json.loads(args.valid_beams.read_text())} \
        if args.valid_beams else beams
    valid = [row(source[name], None, "no_context") for name in split["val_clips"]]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(train)} training rows from {len(train) // 2} clips, {len(valid)} validation rows, in {args.out}")


if __name__ == "__main__":
    main()
