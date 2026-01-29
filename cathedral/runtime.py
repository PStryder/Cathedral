from __future__ import annotations

from loom import Loom

from cathedral.Memory import get_memory


loom = Loom()
memory = get_memory()

# Thread-personality associations (in-memory, could be persisted)
thread_personalities: dict[str, str] = {}


__all__ = ["loom", "memory", "thread_personalities"]
