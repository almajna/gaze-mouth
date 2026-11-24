"""Attach real window and display metadata to a reference image already in the corpus."""

import argparse
import json
import time

from clicky import ax
from clicky.capture import display_for, on_summon

from . import corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metadata", help="new JSON path inside data/clicky/frozen")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-image", help="save one real display JPEG to a new private fixture path")
    source.add_argument("--image", help="existing reference image relative to the same directory")
    p.add_argument("--delay", type=float, default=5)
    args = p.parse_args()
    if args.image and not corpus.inside(args.image).is_file():
        p.error("Place the reference image in data/clicky/frozen first.")
    path = corpus.inside(args.metadata)
    if path.exists() or (args.record_image and corpus.inside(args.record_image).exists()):
        p.error("Choose new paths for this recording.")
    print("Focus the reference window and keep its layout fixed.", flush=True)
    time.sleep(max(0, args.delay))
    if args.record_image:
        context = on_summon(*ax.frontmost())
        if not context.jpeg or not context.window.elements:
            p.error(context.issue or "The window has no readable Accessibility elements.")
        target = corpus.inside(args.record_image)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(context.jpeg)
        window = context.window
    else:
        window = ax.read_window(*ax.frontmost())
    _, display, scale = display_for(window)
    metadata = {
        "app": window.app,
        "title": window.title,
        "frame": window.frame,
        **({"selected_text": window.selected_text} if window.selected_text is not None else {}),
        "display": display,
        "scale": scale,
        "image": args.record_image or args.image,
        "elements": [{"frame": e.frame, **e.description()} for e in window.elements],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("Saved reference-window metadata at", path)


if __name__ == "__main__":
    main()
