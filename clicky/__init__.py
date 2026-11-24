"""Load optional checkout-local provider keys before the app or evaluators select a model."""

import os
from pathlib import Path


def key_file():
    return Path(os.environ.get("CLICKY_ENV_FILE") or Path(__file__).resolve().parent.parent / ".env.local")


def save_local_key(name, value):
    """Store a provider key in the gitignored key file, replacing any earlier value for that name."""
    path = key_file()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    kept = [line for line in lines if line.strip().removeprefix("export ").partition("=")[0].strip() != name]
    path.write_text("\n".join([*kept, f"{name}={value}"]) + "\n", encoding="utf-8")
    path.chmod(0o600)


def load_local_keys():
    path = key_file()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        name, equals, value = line.strip().removeprefix("export ").partition("=")
        name = name.strip()
        if not equals or name not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value:
            os.environ.setdefault(name, value)


load_local_keys()
