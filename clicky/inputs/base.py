from typing import Protocol


class RequestInput(Protocol):
    def read(self) -> str: ...
