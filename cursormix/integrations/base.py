from typing import Protocol


class Integration(Protocol):
    """Prepare a service action without performing it."""

    def prepare(self, arguments: dict) -> str: ...
