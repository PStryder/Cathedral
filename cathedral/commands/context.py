from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from cathedral.services import ServiceRegistry


@dataclass(slots=True)
class CommandContext:
    loom: Any
    memory: Any
    thread_uid: str
    thread_personalities: Dict[str, str]
    services: ServiceRegistry


__all__ = ["CommandContext"]
