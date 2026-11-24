"""Screen-grounded phoneme-lattice -> text reconstruction with a local Qwen2.5-3B (mlx-lm).

Exports resolve lazily so the data generator and prompt code import without mlx.
"""
_EXPORTS = {"ScreenGroundedReconstructor": "reconstructor"}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        from importlib import import_module
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
