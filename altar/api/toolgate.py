"""
ToolGate API - Tool Protocol Configuration.

Exposes the tool protocol system prompt for UI viewing/editing.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class ToolPromptUpdate(BaseModel):
    """Request to update the tool protocol prompt."""
    prompt: str
    acknowledge_risk: bool = False


def create_router() -> APIRouter:
    """Create the ToolGate API router."""
    from cathedral import ToolGate

    router = APIRouter(prefix="/api/toolgate", tags=["toolgate"])

    @router.get("/prompt")
    async def get_tool_prompt_config():
        """
        Get the current tool protocol prompt configuration.

        Returns:
            - prompt: The current prompt text
            - is_custom: Whether a custom prompt is in use
            - is_using_fallback: Whether fallback is active due to broken custom prompt
            - base_version: Version the prompt was based on
            - current_version: Current default version
            - needs_update: Whether custom prompt is outdated
            - custom_created_at: When custom prompt was created (if applicable)
        """
        ToolGate.initialize()
        return ToolGate.get_prompt_config()

    @router.get("/prompt/default")
    async def get_default_prompt():
        """Get the default tool protocol prompt (for reference/comparison)."""
        from cathedral.ToolGate import DEFAULT_TOOL_PROTOCOL_PROMPT, PROMPT_VERSION

        return {
            "prompt": DEFAULT_TOOL_PROTOCOL_PROMPT,
            "version": PROMPT_VERSION,
        }

    @router.get("/prompt/warning")
    async def get_edit_warning():
        """
        Get the warning message to display before editing the prompt.

        This should be shown in a modal before allowing edits.
        FUCKING THIS UP BREAKS EVERYTHING.
        """
        return {
            "warning": ToolGate.get_edit_warning(),
            "required_markers": [
                '"type"',
                "tool_call",
                '"id"',
                '"tool"',
                '"args"',
                "tool_result",
            ],
        }

    @router.post("/prompt")
    async def update_tool_prompt(request: ToolPromptUpdate):
        """
        Update the tool protocol prompt.

        WARNING: Breaking this prompt will disable ALL tool calling.

        Args:
            prompt: The new prompt content
            acknowledge_risk: Must be True to proceed (confirms user saw warning)

        Returns:
            Success status and message
        """
        ToolGate.initialize()

        success, message = ToolGate.set_custom_prompt(
            prompt=request.prompt,
            acknowledge_risk=request.acknowledge_risk,
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return {"status": "saved", "message": message}

    @router.post("/prompt/restore")
    async def restore_default_prompt():
        """
        Restore the default tool protocol prompt.

        Use this to recover if a custom prompt breaks tool calling.
        """
        ToolGate.initialize()

        success, message = ToolGate.restore_default_prompt()

        if not success:
            raise HTTPException(status_code=500, detail=message)

        return {"status": "restored", "message": message}

    @router.get("/prompt/validate")
    async def validate_prompt(prompt: str):
        """
        Validate a prompt without saving it.

        Args:
            prompt: The prompt text to validate (as query param)

        Returns:
            - valid: Whether the prompt is valid
            - missing: List of missing required markers
            - functional: Whether tool calling would work
        """
        from cathedral.ToolGate import validate_prompt, is_prompt_functional

        valid, missing = validate_prompt(prompt)

        return {
            "valid": valid,
            "missing": missing,
            "functional": is_prompt_functional(prompt),
        }

    @router.get("/tools")
    async def list_available_tools(policy: Optional[str] = None):
        """
        List available tools, optionally filtered by policy class.

        Args:
            policy: Filter by policy class (read_only, write, destructive, privileged, network)
        """
        from cathedral.ToolGate import PolicyClass

        ToolGate.initialize()

        policy_filter = None
        if policy:
            try:
                policy_filter = {PolicyClass(policy)}
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid policy: {policy}. Valid: read_only, write, destructive, privileged, network"
                )

        tools = ToolGate.list_tools(policy_filter=policy_filter)

        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "gate": t.gate,
                    "method": t.method,
                    "policy_class": t.policy_class.value,
                    "args": {
                        name: {
                            "type": schema.type,
                            "required": schema.required,
                            "description": schema.description,
                        }
                        for name, schema in t.args_schema.items()
                    },
                }
                for t in tools
            ],
            "count": len(tools),
        }

    @router.get("/status")
    async def get_toolgate_status():
        """Get ToolGate health and status information."""
        ToolGate.initialize()
        return ToolGate.get_health_status()

    return router


__all__ = ["create_router"]
