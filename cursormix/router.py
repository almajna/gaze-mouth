"""Dispatch explicit request forms immediately and ask the fast model for ambiguous intent."""

import re

from .models import structured

ROUTES = ("answer", "point_to", "teach", "agent_task")

RULES = (
    (
        "point_to",
        r"^(where (?:is|are|do|should|can) |point (?:me )?(?:to|at)|highlight|locate the|show me where)\b",
    ),
    (
        "teach",
        r"^(walk me through|teach me|guide me|show me how|step.by.step|how to|how (?:do|can|should) (?:i|we|one)"
        r"|help me (?:with|adjust|change|set ?up|configure|connect|pair|turn|enable|disable|fix|find|use|install|add|remove))\b",
    ),
    (
        "agent_task",
        r"^(please )?(open|launch|switch|create|rename|move|send|compose|click|type|run|select|press|delete|set|change"
        r"|turn (?:on|off)|enable|disable|find and open|do it for me)\b",
    ),
)


def classify(prompt, provider):
    text = prompt.strip().casefold()
    for route, pattern in RULES:
        if re.search(pattern, text):
            return route
    # A conventional factual question can start streaming immediately.
    if re.search(r"^(?:what|who|why|when|which|whose|is|are|does|do)\b", text) and not re.search(
        r"\b(?:click|tap|find|show|teach|guide|steps)\b", text
    ):
        return "answer"
    result = structured(
        prompt,
        provider,
        "Classify what the person wants from CursorMix, an assistant on their Mac that can see the screen. "
        f'{{"route":"{"|".join(ROUTES)}"}}. '
        "answer: a question answered in words, including questions about what is on screen. "
        "point_to: asks where a control or feature is. "
        "teach: the person wants to do, change, set up or fix something on their Mac themselves and wants "
        "to be guided through it, such as 'help me with bluetooth settings' or 'how do I share my screen'. "
        "agent_task: the person tells CursorMix to carry out an operation for them, such as 'open Notes' or "
        "'turn on dark mode'. A request for information must never become agent_task.",
        limit=42,
        schema={"type": "object", "properties": {"route": {"enum": list(ROUTES)}}, "required": ["route"]},
    )
    route = result.get("route")
    return route if route in ROUTES else "answer"
