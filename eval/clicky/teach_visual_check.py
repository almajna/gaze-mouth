"""Check visual progress and returning to a starting window with real Finder recordings."""

import argparse
from dataclasses import asdict, replace

from clicky import ax, capture, memory, teach
from clicky.overlay import Target, locate, visual_target
from clicky.vision import ScreenshotRegions, match_icon, region_pixels

from . import corpus
from .annotation_check import recorded_finder


def check_image_changes():
    # These are independent public screenshots, not a recorded interactive session.
    before, unrelated, info = (corpus.context(f"public-screens/qa-0-{index}.json") for index in (1, 4, 2))
    assert all(ctx.jpeg and not ctx.window.elements for ctx in (before, unrelated, info))
    assert before.window.state() == unrelated.window.state() == info.window.state()
    step = teach.Step(
        "Open Get Info for the Amiga file.",
        "Get Info",
        "Finder shows the Amiga Info window with Kind and Size.",
        visual=False,
    )
    assert not teach.completed(step, before, before, "Anthropic"), "An unchanged view advanced the guide."
    assert not teach.completed(step, before, unrelated, "Anthropic"), "An unrelated view advanced the guide."
    evidence = teach.verify_transition(step, before, info, "Anthropic")
    assert evidence and "amiga" in evidence["observed_change"].casefold(), "The visible file was missed."
    print("Unchanged and unrelated views rejected. The observed change identified the visible file.")



def check_visual_targets(start, inside):
    # Replay the real display image with app-name-only context. The independently
    # recorded AX row supplies the reference bounds solely for scoring.
    screen_only = capture.Context(
        ax.Window(0, start.window.app), start.display, start.jpeg
    )
    destination = next(
        e
        for e in start.window.elements
        if e.label == "Clicky material" and e.role == "AXTextField"
    )
    request = "Walk me through opening the Clicky material folder by double-clicking its name."
    opening, target, _ = teach.next_instruction(
        request, [], [], screen_only, "Anthropic"
    )
    assert opening and target and target.element_id is None, (
        "The screenshot-only step gave no pointer."
    )
    x, y, w, h = target.frame
    left, top, width, height = destination.frame
    assert left <= x + w / 2 < left + width and top <= y + h / 2 < top + height, (
        "The arrow missed the recorded name."
    )
    shortcut = "Guide me through opening Get Info for the selected file using only the Command-I shortcut."
    keys, target, _ = teach.next_instruction(shortcut, [], [], start, "Anthropic")
    assert keys and target is None, (
        "A keyboard-only step pointed at an unrelated control."
    )
    print(
        "The screenshot-only step pointed inside the recorded folder name. The keyboard step showed no arrow."
    )

    # The Search and Back icons have no printed label. Compare the screenshot-only
    # result with independently recorded control bounds at two window positions.
    point = locate("Point to the magnifying glass in the Finder toolbar.", screen_only, "Anthropic")
    back_screen = capture.Context(ax.Window(0, inside.window.app), inside.display, inside.jpeg)
    _, back_arrow, _ = teach.next_instruction(
        "Walk me through using the back arrow in the toolbar to return to the previous folder.",
        [], [], back_screen, "Anthropic",
    )
    for arrow, ctx, name in ((point, start, "Search"), (back_arrow, inside, "back")):
        reference = next(e.frame for e in ctx.window.elements if e.role == "AXButton" and e.label == name)
        assert arrow is not None and arrow.element_id is None, f"The {name} icon had no arrow."
        x, y, w, h = arrow.frame
        left, top, width, height = reference
        assert left <= x + w/2 < left + width and top <= y + h/2 < top + height, f"The arrow missed {name}."
    regions = ScreenshotRegions(start.jpeg, start.display)
    # A real failed Search prediction landed on this featureless patch above Finder.
    missed_search = (1048.026, 224.004, 15.334, 9.344)
    assert regions.uniform(missed_search) and not regions.uniform(point.frame)
    blank = locate("Point to the middle of the empty white space above the Finder window.", screen_only, "Anthropic")
    assert regions.uniform(blank.frame) and blank.frame[1] + blank.frame[3] / 2 < start.window.frame[1], (
        "The intentionally blank target moved onto another control."
    )
    print("Point and Teach both landed inside the recorded Finder toolbar icons.")
    return point



def check_text_tracking(start, moved, inside, returned):
    label = "Clicky-terminal.txt"
    reference = next(e.frame for e in start.window.elements if e.label == label)
    # The first image has a matching AX text field. The second reference is the
    # filename's visible bounds in moved.jpg, which came after its AX snapshot.
    left, top, width, height = reference
    dx, dy, dw, dh = start.display
    box = [1000 * (x - origin) / span for x, origin, span in (
        (left, dx, dw), (top, dy, dh), (left + width, dx, dw), (top + height, dy, dh)
    )]
    anchor = visual_target(start, "Local", label, label, box, label)
    assert anchor and anchor.visible_text == label
    for current, bounds in ((start, reference), (moved, (595, 363, 117, 18)), (returned, (595, 363, 117, 18))):
        target = teach.find_step_text(anchor, start.window, current)
        assert target, "The visible filename lost its pointer."
        x, y, w, h = target.frame
        left, top, width, height = bounds
        assert left <= x + w/2 < left + width and top <= y + h/2 < top + height
    assert teach.find_step_text(anchor, start.window, inside) is None
    duplicate = replace(anchor, visible_text="Clicky-source.txt")
    assert teach.find_step_text(duplicate, start.window, moved) is None
    print("The pointer followed the printed filename, hid in the subfolder, returned and rejected a duplicate.")


def check_icon_tracking(start, moved, inside, returned, anchor=None):
    # Withhold window geometry and AX controls to exercise the screenshot-only
    # path. The window moved between the AX read and the later JPEG in moved.
    def screen_only(context):
        return capture.Context(replace(context.window, elements=[], frame=None), context.display, context.jpeg)

    original = screen_only(start)
    moved_screen = screen_only(moved)
    duplicate = Target((550, 418, 12, 16), "Finder document icon")
    assert teach.refresh_visual_target(duplicate, original, moved_screen) is None
    close = region_pixels(start.jpeg, start.display, (365, 329, 11, 11), 1)
    minimize = region_pixels(start.jpeg, start.display, (381, 318, 35, 35), 1)
    assert match_icon(close, minimize) is None, "The minimize button substituted for the close button."
    anchor = anchor or Target((1233, 325, 22, 22), "Finder Search")
    assert teach.refresh_visual_target(anchor, original, screen_only(inside)) is None
    assert teach.refresh_visual_target(anchor, original, capture.Context(moved_screen.window)) is None
    # Independently read from the Search control in each recorded image.
    for state, bounds in ((original, (1233, 325, 22, 22)), (moved_screen, (1262, 171, 22, 22)),
                          (screen_only(returned), (1262, 171, 22, 22))):
        tracked = teach.refresh_visual_target(anchor, original, state)
        assert tracked, "Finder's Search icon lost its pointer."
        x, y, w, h = tracked.frame
        left, top, width, height = bounds
        assert left <= x+w/2 <= left+width and top <= y+h/2 <= top+height, tracked
    print("Rejected repeated icons, a different folder and missing pixels. Tracked Search at both positions.")


def check_same_view():
    """Supply the annotated control frame and measure reacquisition in the same real image."""
    for collection in ("point-1x", "point"):
        corpus.verify(collection)
        rows = corpus.rows(collection)
        missing = withheld = inside = outside = 0
        for row in rows:
            context = corpus.context(row["context"])
            frame = context.control_frame(row["target"])
            if not frame:
                missing += 1
                continue
            target = teach.refresh_visual_target(Target(frame, "reference"), context, context)
            if not target:
                withheld += 1
                continue
            x, y, w, h = target.frame
            left, top, width, height = frame
            if left <= x+w/2 <= left+width and top <= y+h/2 <= top+height:
                inside += 1
            else:
                outside += 1
        print(f"{collection}: {outside} wrong centers, {inside} returned inside, {withheld} withheld, "
              f"{missing} unavailable reference frames on {len(rows)} unchanged views.", flush=True)


def check_round_trip(start, moved, inside, returned):
    goal = (
        "Guide me into the Clicky material folder, then back to my original folder. "
        "Finish when the file I originally had selected is selected again."
    )
    itinerary = teach.plan(goal, start, "Anthropic")
    opening, _, _ = teach.next_instruction(goal, itinerary, [], start, "Anthropic")
    assert opening and not teach.completed(opening, start, moved, "Anthropic")
    selected = next(e.label for e in start.window.elements if e.selected and e.role == "AXTextField")
    question = "Which file is currently selected? Give its name."
    reply = teach.answer_step(question, goal, opening, [], start, "Anthropic")
    assert selected.casefold() in reply.casefold(), "The step answer missed the selected file."
    prior = [{"request": question, "reply": reply, "screen_at_answer": memory.screen_excerpt(start.window)}]
    assert not teach.completed(opening, start, start, "Anthropic"), "A question advanced the walkthrough."
    entered = teach.verify_transition(opening, start, inside, "Anthropic")
    assert entered and entered["starting_window"] == start.window.title != inside.window.title
    assert entered["resulting_window"] == inside.window.title
    assert "Clicky-source.txt" in entered["starting_selection"]
    back, _, _ = teach.next_instruction(goal, itinerary, [entered], inside, "Anthropic")
    assert back and not teach.completed(back, inside, inside, "Anthropic")
    reply = teach.answer_step(
        "Which folder will I return to, and what filename did you tell me earlier?",
        goal,
        back,
        [entered],
        inside,
        "Anthropic",
        conversation=prior,
    )
    assert start.window.title.casefold() in reply.casefold(), "The step answer lost the starting folder."
    assert selected.casefold() in reply.casefold(), "The step answer lost the earlier exchange."
    restored = teach.verify_transition(back, inside, returned, "Anthropic")
    assert restored and restored["resulting_window"] == entered["starting_window"]
    assert restored["resulting_selection"] == entered["starting_selection"]
    starting = teach.observed_window(start.window)
    for history, screen in (([], start), ([], moved), ([entered], inside)):
        achieved, _ = teach.check_goal(goal, starting, history, screen, "Anthropic")
        assert not achieved, "An incomplete round trip passed final verification."
    two_visits = goal + " Make that round trip twice before finishing."
    achieved, _ = teach.check_goal(two_visits, starting, [entered, restored], returned, "Anthropic")
    assert not achieved, "One verified visit counted as two."
    achieved, _ = teach.check_goal(f"Select {selected} in this folder.", starting, [], start, "Anthropic")
    assert achieved, "A visible, already-achieved selection unnecessarily required an action."
    _, _, conclusion = teach.next_instruction(goal, itinerary, [entered, restored], returned, "Anthropic")
    assert conclusion, "The guide continued after the original folder was restored."
    print(
        "Two step questions identified the selected file and original folder. The recorded return finished the guide."
    )


def check_wrong_folder(start, moved, inside, returned, provider="Anthropic"):
    selected = next(e.label for e in start.window.elements if e.selected and e.role == "AXTextField")
    goal = (
        f"Guide me through opening Get Info for {selected} with Command-I. "
        "If I accidentally leave this folder first, guide me back with the toolbar Back button."
    )
    starting = memory.observed_window(start.window)
    plan = teach.plan(goal, start, provider)
    opening, _, _ = teach.next_instruction(goal, plan, [], start, provider, starting=starting)
    assert opening and "info" in (opening.instruction + opening.expected).casefold()
    known_dialogs = teach.dialogs(start.window)
    assert teach.step_observation(opening, start, moved, provider, known_dialogs) == (None, False)
    wrong_folder = teach.step_observation(opening, start, inside, provider, known_dialogs)
    assert wrong_folder == (None, True), (opening, wrong_folder)
    pending = {**asdict(opening), "before": starting}
    assert teach.reconcile_step(pending, inside, provider) is None, "Relaunch credited the wrong folder."
    recovery, pointer, _ = teach.next_instruction(
        goal, plan, [], inside, provider, starting=starting,
        interrupted=pending,
    )
    assert recovery and pointer, ("The guide did not point back from the unexpected folder.", recovery)
    left, top, width, height = next(
        e.frame for e in inside.window.elements if e.role == "AXButton" and e.label == "back"
    )
    x, y, w, h = pointer.frame
    assert left <= x + w / 2 < left + width and top <= y + h / 2 < top + height
    progress, detour = teach.step_observation(recovery, inside, returned, provider, teach.dialogs(inside.window))
    assert progress and not detour and progress["resulting_window"] == starting["window"]
    retry, _, conclusion = teach.next_instruction(goal, plan, [progress], returned, provider, starting=starting)
    assert retry and not conclusion and "info" in (retry.instruction + retry.expected).casefold()
    print("Moving the window left the step pending. Entering the wrong folder pointed Back. Returning resumed Get Info.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detour", choices=("Anthropic", "OpenAI", "Local"), help="Run the wrong-folder replay.")
    parser.add_argument("--track-text", action="store_true", help="Replay the moving screenshot label offline.")
    parser.add_argument("--track-icons", action="store_true", help="Replay the moving screenshot icon offline.")
    parser.add_argument("--same-view", action="store_true", help="Check target reacquisition on 1× and Retina images.")
    args = parser.parse_args()
    if args.same_view:
        check_same_view()
        return
    start, moved, inside, returned = (
        recorded_finder(name) for name in ("start", "moved", "inside", "returned")
    )
    if args.detour:
        check_wrong_folder(start, moved, inside, returned, args.detour)
        return
    if args.track_icons:
        check_icon_tracking(start, moved, inside, returned)
        return
    check_text_tracking(start, moved, inside, returned)
    if args.track_text:
        return
    check_image_changes()
    search_icon = check_visual_targets(start, inside)
    check_icon_tracking(start, moved, inside, returned, search_icon)
    check_round_trip(start, moved, inside, returned)
    check_wrong_folder(start, moved, inside, returned)


if __name__ == "__main__":
    main()
