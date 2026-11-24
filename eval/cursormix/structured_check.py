"""Exercise the provider's actual output limit with a recorded, selected Finder file."""

import argparse

from cursormix.models import OutputLimitReached, generate, object_schema, structured

from .annotation_check import recorded_finder


def check(provider):
    context = recorded_finder()
    prompt = context.prompt("Which Finder folder is open, and which file is selected?")
    instruction = "Give the exact names from the current Finder window."
    schema = object_schema(folder={"type": "string"}, file={"type": "string"})
    try:
        list(generate(prompt, provider, instruction, limit=8, json_mode=schema))
    except OutputLimitReached:
        pass
    else:
        raise AssertionError("The provider did not report the deliberately short output limit.")
    result = structured(prompt, provider, instruction, limit=8, schema=schema)
    assert result["folder"] == context.window.title
    assert result["file"] in {e.label for e in context.window.elements if e.selected}
    print(f"{provider} reported truncation. A complete second response identified the recorded folder and selected file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", nargs="?", choices=("Anthropic", "OpenAI", "Local"), default="Anthropic")
    check(parser.parse_args().provider)
