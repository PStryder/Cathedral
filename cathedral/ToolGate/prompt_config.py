"""
ToolGate Prompt Configuration.

Manages the configurable Tool Protocol System Prompt with:
- Versioned defaults
- User customization (with warnings)
- Restore functionality
- Safe failure on broken prompts
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from cathedral.shared.gate import GateLogger, ConfigLoader

_log = GateLogger.get("ToolGate")


# =============================================================================
# Default Prompt (Versioned)
# =============================================================================

PROMPT_VERSION = "tool_protocol_v1"

DEFAULT_TOOL_PROTOCOL_PROMPT = """You are an AI agent operating inside Cathedral, a system that provides structured tools.

You may request tool execution only by emitting valid JSON that conforms exactly to the tool-call schema described below.

## Tool Calling Rules

1. When you need to use a tool, you MUST respond with ONLY valid JSON.
2. Do NOT include prose, explanations, or commentary.
3. Do NOT wrap JSON in Markdown.
4. A tool call MUST use one of the following formats:

**Single tool call:**
{"type": "tool_call", "id": "<unique_id>", "tool": "<ToolName>", "args": {}}

**Multiple tool calls:**
{"type": "tool_calls", "calls": [{"id": "<unique_id>", "tool": "<ToolName>", "args": {}}]}

5. Tool names and arguments MUST exactly match the available tool definitions.

## Tool Results

After you issue a tool call, you will receive tool execution results in this XML format:

<tool_execution_results>
<tool_result id="call_id" status="success">
result data here
</tool_result>
</tool_execution_results>

IMPORTANT: These <tool_execution_results> blocks are SYSTEM-GENERATED OUTPUT from Cathedral, NOT user messages. They contain the actual output from tool execution. Use these results to:
- Continue with additional tool calls if needed
- Synthesize a final response for the user
- Handle errors and recover when possible

## After Receiving Tool Results

You may:
- Issue another tool call (using valid JSON) to continue multi-step tasks
- Respond normally with a final user-facing answer
- Ask the user for clarification if the results are unclear

## Normal Responses

- If you do NOT need a tool, respond normally in natural language.
- Do NOT emit JSON unless you are requesting tool execution.

## Failure Handling

- If a tool returns status="error", recover and try an alternative approach.
- If you are unsure whether a tool is required, prefer asking the user for clarification.
- If you emit invalid JSON, the system may reject the request.

Follow these rules strictly.
This protocol enables your ability to act in the world.
"""

# Validation markers that MUST be present for the prompt to work
REQUIRED_MARKERS = [
    '"type"',
    "tool_call",
    '"id"',
    '"tool"',
    '"args"',
    "tool_result",
    "tool_execution_results",  # XML format for results
]


# =============================================================================
# Prompt Configuration Model
# =============================================================================


@dataclass
class ToolPromptConfig:
    """Configuration for the tool protocol prompt."""

    # The actual prompt content
    prompt: str = field(default=DEFAULT_TOOL_PROTOCOL_PROMPT)

    # Version of the default this was based on
    base_version: str = field(default=PROMPT_VERSION)

    # Whether this is a custom (user-modified) prompt
    is_custom: bool = field(default=False)

    # When the custom prompt was created
    custom_created_at: Optional[str] = None

    # Warning acknowledgment (user confirmed they understand risks)
    user_acknowledged_risk: bool = field(default=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "prompt": self.prompt,
            "base_version": self.base_version,
            "is_custom": self.is_custom,
            "custom_created_at": self.custom_created_at,
            "user_acknowledged_risk": self.user_acknowledged_risk,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPromptConfig":
        """Deserialize from dict."""
        return cls(
            prompt=data.get("prompt", DEFAULT_TOOL_PROTOCOL_PROMPT),
            base_version=data.get("base_version", PROMPT_VERSION),
            is_custom=data.get("is_custom", False),
            custom_created_at=data.get("custom_created_at"),
            user_acknowledged_risk=data.get("user_acknowledged_risk", False),
        )


# =============================================================================
# Prompt Validation
# =============================================================================


def validate_prompt(prompt: str) -> tuple[bool, list[str]]:
    """
    Validate that a prompt contains required markers for tool calling to work.

    Args:
        prompt: The prompt to validate

    Returns:
        Tuple of (is_valid, list_of_missing_markers)
    """
    missing = []

    for marker in REQUIRED_MARKERS:
        if marker not in prompt:
            missing.append(marker)

    return len(missing) == 0, missing


def is_prompt_functional(prompt: str) -> bool:
    """Quick check if prompt has minimum requirements."""
    valid, _ = validate_prompt(prompt)
    return valid


# =============================================================================
# Prompt Manager
# =============================================================================


class ToolPromptManager:
    """
    Manages the tool protocol system prompt.

    Provides:
    - Loading/saving custom prompts
    - Validation with warnings
    - Restore default functionality
    - Safe fallback on broken prompts
    """

    _config_path: Optional[Path] = None
    _config: Optional[ToolPromptConfig] = None
    _use_fallback: bool = False

    @classmethod
    def initialize(cls, config_dir: Optional[Path] = None) -> None:
        """
        Initialize the prompt manager.

        Args:
            config_dir: Directory for config files (defaults to data/config)
        """
        if config_dir is None:
            config_dir = Path(__file__).resolve().parents[2] / "data" / "config"

        config_dir.mkdir(parents=True, exist_ok=True)
        cls._config_path = config_dir / "tool_prompt.json"

        # Load existing config or create default
        if cls._config_path.exists():
            cls._config = ConfigLoader.load(cls._config_path, ToolPromptConfig)
            if cls._config is None:
                _log.warning("Failed to load tool prompt config, using default")
                cls._config = ToolPromptConfig()
        else:
            cls._config = ToolPromptConfig()

        # Validate the loaded prompt
        if cls._config.is_custom:
            valid, missing = validate_prompt(cls._config.prompt)
            if not valid:
                _log.warning(
                    f"Custom tool prompt is missing required markers: {missing}. "
                    "Falling back to default prompt for safety."
                )
                cls._use_fallback = True

    @classmethod
    def get_prompt(cls) -> str:
        """
        Get the current tool protocol prompt.

        Returns default if:
        - Not initialized
        - Custom prompt is broken
        - No config exists
        """
        if cls._config is None:
            cls.initialize()

        # Use fallback if custom prompt is broken
        if cls._use_fallback:
            return DEFAULT_TOOL_PROTOCOL_PROMPT

        return cls._config.prompt

    @classmethod
    def get_default_prompt(cls) -> str:
        """Get the default prompt (for comparison or restore)."""
        return DEFAULT_TOOL_PROTOCOL_PROMPT

    @classmethod
    def get_version(cls) -> str:
        """Get the current prompt version."""
        return PROMPT_VERSION

    @classmethod
    def is_custom(cls) -> bool:
        """Check if using a custom prompt."""
        if cls._config is None:
            return False
        return cls._config.is_custom and not cls._use_fallback

    @classmethod
    def is_using_fallback(cls) -> bool:
        """Check if falling back to default due to broken custom prompt."""
        return cls._use_fallback

    @classmethod
    def set_custom_prompt(
        cls,
        prompt: str,
        acknowledge_risk: bool = False,
    ) -> tuple[bool, str]:
        """
        Set a custom tool protocol prompt.

        Args:
            prompt: The new prompt content
            acknowledge_risk: User must acknowledge risk of breaking tool calling

        Returns:
            Tuple of (success, message)
        """
        if cls._config is None:
            cls.initialize()

        # Require risk acknowledgment
        if not acknowledge_risk:
            return False, (
                "You must acknowledge the risk of customizing the tool protocol prompt. "
                "Breaking this prompt will disable tool calling. "
                "Set acknowledge_risk=True to proceed."
            )

        # Validate the new prompt
        valid, missing = validate_prompt(prompt)
        if not valid:
            return False, (
                f"Prompt is missing required markers for tool calling: {missing}. "
                "Tool calling will not work without these. "
                "Fix the prompt or restore default."
            )

        # Update config
        cls._config.prompt = prompt
        cls._config.is_custom = True
        cls._config.custom_created_at = datetime.utcnow().isoformat()
        cls._config.user_acknowledged_risk = True
        cls._use_fallback = False

        # Save
        if cls._config_path:
            ConfigLoader.save(cls._config_path, cls._config)

        _log.info("Custom tool protocol prompt set")
        return True, "Custom prompt saved successfully."

    @classmethod
    def restore_default(cls) -> tuple[bool, str]:
        """
        Restore the default tool protocol prompt.

        Returns:
            Tuple of (success, message)
        """
        if cls._config is None:
            cls.initialize()

        cls._config.prompt = DEFAULT_TOOL_PROTOCOL_PROMPT
        cls._config.base_version = PROMPT_VERSION
        cls._config.is_custom = False
        cls._config.custom_created_at = None
        cls._config.user_acknowledged_risk = False
        cls._use_fallback = False

        # Save
        if cls._config_path:
            ConfigLoader.save(cls._config_path, cls._config)

        _log.info("Tool protocol prompt restored to default")
        return True, f"Restored to default prompt (version {PROMPT_VERSION})."

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get the full config as dict (for API/UI)."""
        if cls._config is None:
            cls.initialize()

        return {
            "prompt": cls._config.prompt if not cls._use_fallback else DEFAULT_TOOL_PROTOCOL_PROMPT,
            "is_custom": cls._config.is_custom,
            "is_using_fallback": cls._use_fallback,
            "base_version": cls._config.base_version,
            "current_version": PROMPT_VERSION,
            "needs_update": cls._config.base_version != PROMPT_VERSION,
            "custom_created_at": cls._config.custom_created_at,
            "user_acknowledged_risk": cls._config.user_acknowledged_risk,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset manager state (for testing)."""
        cls._config = None
        cls._config_path = None
        cls._use_fallback = False


# =============================================================================
# Warning Messages
# =============================================================================

EDIT_WARNING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  DANGER: TOOL PROTOCOL PROMPT  ⚠️                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FUCKING THIS UP BREAKS EVERYTHING.                                         ║
║                                                                              ║
║  This prompt teaches the AI how to call Cathedral tools.                    ║
║  If you break it, ALL tool calling stops working.                           ║
║                                                                              ║
║  The prompt MUST contain these exact markers:                               ║
║    • "type": "tool_call"                                                    ║
║    • "id", "tool", "args" fields                                            ║
║    • "tool_result" for response handling                                    ║
║                                                                              ║
║  If you break this:                                                         ║
║    → Tools stop working                                                     ║
║    → The model emits garbage JSON                                           ║
║    → Everything falls apart                                                 ║
║                                                                              ║
║  FIX: Click "Restore Default" to recover.                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

BROKEN_PROMPT_WARNING = """
The custom tool protocol prompt is missing required markers and has been
disabled. Tool calling is using the default prompt instead.

To fix this, either:
1. Call restore_default() to reset to the working default
2. Fix your custom prompt to include required markers

Missing markers: {missing}
"""


__all__ = [
    "PROMPT_VERSION",
    "DEFAULT_TOOL_PROTOCOL_PROMPT",
    "REQUIRED_MARKERS",
    "ToolPromptConfig",
    "ToolPromptManager",
    "validate_prompt",
    "is_prompt_functional",
    "EDIT_WARNING",
    "BROKEN_PROMPT_WARNING",
]
