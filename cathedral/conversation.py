from __future__ import annotations

from typing import Any, List, Dict

from cathedral.MemoryGate.conversation import (
    get_conversation_service,
    enable_discovery,
    disable_discovery,
)


class ConversationAdapter:
    """
    Loom-compatible adapter backed by MemoryGate conversation service.

    This preserves the legacy "loom" method surface while using the
    MemoryGate implementation under the hood.
    """

    def __init__(self):
        self._svc = get_conversation_service()

    # ---- Thread management ----
    def list_all_threads(self) -> List[Dict[str, Any]]:
        return self._svc.list_threads()

    def create_new_thread(self, thread_name: str | None = None) -> str:
        return self._svc.create_thread(thread_name)

    def switch_to_thread(self, thread_uid: str) -> bool:
        return self._svc.switch_thread(thread_uid)

    # ---- History ----
    def recall(self, thread_uid: str | None = None) -> List[Dict[str, Any]]:
        return self._svc.recall(thread_uid)

    async def recall_async(self, thread_uid: str | None = None) -> List[Dict[str, Any]]:
        return await self._svc.recall_async(thread_uid)

    def clear(self, thread_uid: str | None = None) -> int:
        return self._svc.clear(thread_uid)

    # ---- Messages ----
    def append(self, role: str, content: str, thread_uid: str | None = None) -> str:
        return self._svc.append(role, content, thread_uid)

    async def append_async(self, role: str, content: str, thread_uid: str | None = None) -> str:
        return await self._svc.append_async(role, content, thread_uid)

    # ---- Context ----
    async def compose_prompt_context_async(
        self, user_input: str, thread_uid: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        return await self._svc.compose_context(
            user_input=user_input,
            thread_uid=thread_uid,
            semantic_limit=top_k,
        )

    def compose_prompt_context(
        self, user_input: str, thread_uid: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        # Sync context builds only use summarized history.
        return self._svc.recall_with_summary(thread_uid=thread_uid)

    # ---- Search ----
    async def semantic_search(
        self,
        query: str,
        thread_uid: str | None = None,
        limit: int = 5,
        include_all_threads: bool = False,
    ) -> List[Dict[str, Any]]:
        return await self._svc.semantic_search(
            query=query,
            thread_uid=thread_uid,
            limit=limit,
            include_all_threads=include_all_threads,
        )

    async def backfill_embeddings(self, batch_size: int = 50) -> int:
        return await self._svc.backfill_embeddings(batch_size=batch_size)

    # ---- Discovery wiring ----
    def enable_discovery(self, queue_func) -> None:
        enable_discovery(queue_func)

    def disable_discovery(self) -> None:
        disable_discovery()


__all__ = ["ConversationAdapter"]
