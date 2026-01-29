from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Any


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@dataclass(slots=True)
class ServiceRegistry:
    """Explicit service hooks for side-effects (events, agent updates)."""

    emit_event: Callable[..., Awaitable[None]] = _noop
    record_agent_update: Callable[..., Awaitable[None]] = _noop


__all__ = ["ServiceRegistry"]
