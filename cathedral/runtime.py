from __future__ import annotations

from cathedral.Memory import get_memory
from cathedral.conversation import ConversationAdapter


class _LazyProxy:
    """Lazy proxy that defers instantiation until first use."""

    def __init__(self, factory, name: str):
        self._factory = factory
        self._name = name
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, item):
        return getattr(self._get(), item)

    def __repr__(self) -> str:
        if self._instance is None:
            return f"<LazyProxy {self._name} (uninitialized)>"
        return repr(self._instance)


def _build_loom() -> ConversationAdapter:
    return ConversationAdapter()


def _build_memory():
    return get_memory()


loom = _LazyProxy(_build_loom, "loom")
memory = _LazyProxy(_build_memory, "memory")

# Thread-personality associations (in-memory, could be persisted)
thread_personalities: dict[str, str] = {}


__all__ = ["loom", "memory", "thread_personalities"]
