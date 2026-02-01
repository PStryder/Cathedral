from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse


class UserInput(BaseModel):
    user_input: str
    thread_uid: str
    enable_tools: bool = False  # Enable ToolGate tool calling
    enabled_gates: list[str] = []  # List of enabled gates (e.g., ["MemoryGate", "ShellGate"])
    enable_context: bool = True  # Enable context injection (RAG/memory context)


class ThreadRequest(BaseModel):
    thread_uid: str | None = None
    thread_name: str | None = None


class RenameThreadRequest(BaseModel):
    thread_name: str


def create_router(templates, process_input_stream, loom, services) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main chat UI."""
        return templates.TemplateResponse("index.html", {"request": request})

    @router.get("/api/threads")
    async def api_list_threads():
        """List all threads for the sidebar."""
        return {"threads": loom.list_all_threads()}

    @router.post("/api/thread")
    async def api_create_or_switch_thread(request: ThreadRequest):
        """Create a new thread or switch to an existing one."""
        if request.thread_uid:
            loom.switch_to_thread(request.thread_uid)
            return {"status": "switched", "thread_uid": request.thread_uid}
        else:
            thread_uid = loom.create_new_thread(request.thread_name)
            return {"status": "created", "thread_uid": thread_uid}

    @router.get("/api/thread/{thread_uid}/history")
    async def api_get_thread_history(thread_uid: str):
        """Get chat history for a specific thread."""
        history = await loom.recall_async(thread_uid)
        return {"history": history if history else []}

    @router.put("/api/thread/{thread_uid}/rename")
    async def api_rename_thread(thread_uid: str, request: RenameThreadRequest):
        """Rename a thread."""
        success = services.conversation.rename_thread(thread_uid, request.thread_name)
        if success:
            return {"status": "renamed", "thread_uid": thread_uid, "thread_name": request.thread_name}
        else:
            return {"status": "error", "message": "Thread not found"}

    @router.post("/api/chat/stream")
    async def api_chat_stream(user_input: UserInput):
        """
        Stream chat response using Server-Sent Events.

        Args:
            user_input: User message
            thread_uid: Thread identifier
            enable_tools: Enable ToolGate tool calling (default: False)
        """

        async def generate():
            try:
                async for token in process_input_stream(
                    user_input.user_input,
                    user_input.thread_uid,
                    services=services,
                    enable_tools=user_input.enable_tools,
                    enabled_gates=user_input.enabled_gates if user_input.enabled_gates else None,
                    enable_context=user_input.enable_context,
                ):
                    yield {"data": json.dumps({"token": token})}
                yield {"data": json.dumps({"done": True})}
            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(generate())

    return router


__all__ = ["create_router"]
