"""Submit and continue a real recorded screen question through the native Ask panel."""

import tempfile
import time
import uuid
from pathlib import Path

import objc
from objc import super
from PyObjCTools import AppHelper

from clicky import memory, router
from clicky.app import ClickyDelegate, end_run, main
from clicky.models import answer
from clicky.settings import preferences

from . import corpus
from .annotation_check import recorded_finder


class FollowupCheck(ClickyDelegate):
    @objc.python_method
    def settings_store(self):
        self.suite = "local.clicky.followup-check." + uuid.uuid4().hex
        store = preferences(self.suite)
        store.registerDefaults_(
            {
                "didCompleteOnboarding": True,
                "provider": "Anthropic",
                "voice": False,
                "agentTasks": False,
                "teaching": False,
            }
        )
        return store

    def applicationDidFinishLaunching_(self, note):
        super().applicationDidFinishLaunching_(note)
        self.stage = "first answer"
        self.context = corpus.context("public-screens/qa-0-2.json")
        self.panel.show()
        self.panel.field.setStringValue_("What kind of file is Amiga in this Finder Info window?")
        AppHelper.callLater(75, self.timeout)
        AppHelper.callAfter(self.submit_, None)

    @objc.python_method
    def finished(self, request, prompt, error):
        super().finished(request, prompt, error)
        if error:
            self.finish_check(RuntimeError(error))

    @objc.python_method
    def show_suggestions(self, ticket, choices):
        super().show_suggestions(ticket, choices)
        try:
            if self.stage == "first answer":
                assert "jpeg" in self.panel.output.string().lower()
                assert not self.continue_item.isHidden()
                assert self.conversation[-1]["screen_at_answer"]["app"] == "Finder"
                assert (
                    memory.turns(self.conversation_id, 30)[-1]["screen_at_answer"]
                    == self.conversation[-1]["screen_at_answer"]
                )
                visible = [b for b in self.panel.suggestions if not b.isHidden()]
                assert visible and len({str(b.title()).casefold() for b in visible}) == len(visible)
                self.choice = next(
                    b
                    for b in visible
                    if router.classify(str(b.title()), "Anthropic") in ("answer", "screen_question")
                )
                self.prompt = str(self.choice.title())
                self.old_ticket = ticket
                self.old_choices = choices
                self.key = self.conversation_id
                self.stage = "clicked follow-up"
                self.choice.performClick_(None)
                assert self.panel.field.stringValue() == self.prompt
                assert all(b.isHidden() for b in self.panel.suggestions)
                super().show_suggestions(self.old_ticket, self.old_choices)
                assert all(b.isHidden() for b in self.panel.suggestions)
            elif self.stage == "clicked follow-up":
                assert len(self.conversation) == 2 and self.conversation[-1]["request"] == self.prompt
                assert len(memory.turns(self.key, 30)) == 2 and self.conversation_id == self.key
                assert self.panel.output.string().strip() and not self.continue_item.isHidden()
                assert any(not b.isHidden() for b in self.panel.suggestions)
                assert self.context.jpeg
                self.store.setInteger_forKey_(7, "followupsIgnored")
                self.store.setBool_forKey_(False, "screenContext")
                self.stage = "new summon"
                self.continue_item.menu().performActionForItemAtIndex_(
                    self.continue_item.menu().indexOfItem_(self.continue_item)
                )
                assert (
                    self.context is None and self.conversation_id == self.key and len(self.conversation) == 2
                )
                assert self.store.doubleForKey_("followupsSnoozedUntil") > time.time() + 3500
        except Exception as exc:  # noqa: BLE001 — bring AppKit callback failures back to the shell
            self.finish_check(exc)

    @objc.python_method
    def show_context(self, ticket, future, pid, name, shortcut):
        super().show_context(ticket, future, pid, name, shortcut)
        if self.stage != "new summon":
            return
        try:
            assert self.context and not self.context.jpeg and self.context.issue
            assert len(self.conversation) == 2 and self.conversation_id == self.key
            assert self.panel.field.stringValue() == ""
            self.flag_(self.flags["suggestFollowups"])
            assert not self.store.boolForKey_("suggestFollowups")
            assert not self.store.doubleForKey_("followupsSnoozedUntil")
            super().show_suggestions(self.old_ticket, self.old_choices)
            assert all(b.isHidden() for b in self.panel.suggestions)
            self.dismiss_(None)
            assert self.continue_item.isHidden() and not self.conversation
        except Exception as exc:  # noqa: BLE001
            self.finish_check(exc)
            return
        self.finish_check(
            "Native follow-up submitted, history retained, old capture released and snooze verified."
        )

    @objc.python_method
    def timeout(self):
        self.finish_check(RuntimeError("Timed out during " + self.stage))

    @objc.python_method
    def finish_check(self, result):
        if not outcomes:
            outcomes.append(result)
            self.store.removePersistentDomainForName_(self.suite)
            AppHelper.callAfter(end_run)


if __name__ == "__main__":
    outcomes = []
    with tempfile.TemporaryDirectory() as tmp:
        memory.DATABASE = Path(tmp) / "history.sqlite3"
        main(delegate_type=FollowupCheck, argv=[])
    if not outcomes:
        raise RuntimeError("The follow-up check ended without a result.")
    if isinstance(outcomes[0], Exception):
        raise outcomes[0]
    print(outcomes[0])
    first, second = recorded_finder(), recorded_finder("inside")
    earlier = "What is the folder name in this Finder window? Give only its name."
    reply = "".join(answer(earlier, "Anthropic", first, visual=True))
    assert first.window.title.casefold() in reply.casefold()
    turns = [{"request": earlier, "reply": reply, "screen_at_answer": memory.screen_excerpt(first.window)}]
    current = "".join(
        answer(
            "Which folder is open now? Give only the current name.",
            "Anthropic",
            second,
            conversation=turns,
            visual=True,
        )
    )
    assert (
        second.window.title.casefold() in current.casefold()
        and first.window.title.casefold() not in current.casefold()
    )
    prior = "".join(
        answer(
            "Which folder was open before this one? Give only its name.",
            "Anthropic",
            second,
            conversation=turns,
            visual=True,
        )
    )
    assert (
        first.window.title.casefold() in prior.casefold()
        and second.window.title.casefold() not in prior.casefold()
    )
    print("Claude distinguished the current and previous folders in the recorded Finder transition.")
