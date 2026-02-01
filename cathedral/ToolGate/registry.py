"""
ToolGate Registry.

Central registry of all available tools with their schemas and metadata.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

from cathedral.shared.gate import GateLogger
from cathedral.ToolGate.models import (
    ArgSchema,
    PolicyClass,
    ToolDefinition,
)

_log = GateLogger.get("ToolGate")


# =============================================================================
# Tool Definitions by Gate
# =============================================================================

# MemoryGate tools
MEMORYGATE_TOOLS: List[Dict[str, Any]] = [
    {
        "method": "get_info",
        "description": "Get comprehensive documentation for MemoryGate including all tools, call formats, and best practices",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
    {
        "method": "search",
        "description": "Semantic search across memory observations, patterns, and concepts",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "query": ArgSchema(type="string", description="Search query text", required=True),
            "limit": ArgSchema(type="integer", description="Max results", required=False, default=5),
            "min_confidence": ArgSchema(type="number", description="Min confidence 0-1", required=False, default=0.0),
            "domain": ArgSchema(type="string", description="Filter by domain", required=False),
            "include_cold": ArgSchema(type="boolean", description="Include archived", required=False, default=False),
        },
    },
    {
        "method": "recall",
        "description": "List recent observations from memory",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "domain": ArgSchema(type="string", description="Filter by domain", required=False),
            "limit": ArgSchema(type="integer", description="Max results", required=False, default=10),
            "min_confidence": ArgSchema(type="number", description="Min confidence", required=False, default=0.0),
        },
    },
    {
        "method": "store_observation",
        "description": "Store a new observation in memory",
        "policy": PolicyClass.WRITE,
        "is_async": False,
        "args": {
            "text": ArgSchema(type="string", description="Observation text", required=True),
            "confidence": ArgSchema(type="number", description="Confidence 0-1", required=False, default=0.8),
            "domain": ArgSchema(type="string", description="Domain/category", required=False),
            "evidence": ArgSchema(type="array", description="Supporting evidence", required=False, items={"type": "string"}),
        },
    },
    {
        "method": "get_concept",
        "description": "Look up a concept by name",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "name": ArgSchema(type="string", description="Concept name", required=True),
        },
    },
    {
        "method": "get_related",
        "description": "Get items related to a reference",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "ref": ArgSchema(type="string", description="Reference (e.g., obs:123, con:456)", required=True),
            "rel_type": ArgSchema(type="string", description="Relationship type filter", required=False),
            "limit": ArgSchema(type="integer", description="Max results", required=False, default=50),
        },
    },
    {
        "method": "add_relationship",
        "description": "Create a relationship between two items",
        "policy": PolicyClass.WRITE,
        "is_async": False,
        "args": {
            "from_ref": ArgSchema(type="string", description="Source reference", required=True),
            "to_ref": ArgSchema(type="string", description="Target reference", required=True),
            "rel_type": ArgSchema(type="string", description="Relationship type", required=True),
            "weight": ArgSchema(type="number", description="Weight 0-1", required=False, default=0.5),
            "description": ArgSchema(type="string", description="Relationship description", required=False),
        },
    },
    {
        "method": "get_stats",
        "description": "Get memory system statistics",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
]

# FileSystemGate tools
FILESYSTEMGATE_TOOLS: List[Dict[str, Any]] = [
    {
        "method": "get_info",
        "description": "Get comprehensive documentation for FileSystemGate including all tools, call formats, and security info",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
    {
        "method": "list_dir",
        "description": "List contents of a directory",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "folder_id": ArgSchema(type="string", description="Registered folder ID", required=True),
            "relative_path": ArgSchema(type="string", description="Path within folder", required=False, default=""),
            "show_hidden": ArgSchema(type="boolean", description="Show hidden files", required=False, default=False),
        },
    },
    {
        "method": "read_file",
        "description": "Read contents of a file",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "folder_id": ArgSchema(type="string", description="Registered folder ID", required=True),
            "relative_path": ArgSchema(type="string", description="Path to file", required=True),
            "encoding": ArgSchema(type="string", description="Text encoding", required=False, default="utf-8"),
            "binary": ArgSchema(type="boolean", description="Read as binary", required=False, default=False),
        },
    },
    {
        "method": "write_file",
        "description": "Write content to a file",
        "policy": PolicyClass.WRITE,
        "is_async": False,
        "args": {
            "folder_id": ArgSchema(type="string", description="Registered folder ID", required=True),
            "relative_path": ArgSchema(type="string", description="Path to file", required=True),
            "content": ArgSchema(type="string", description="Content to write", required=True),
            "encoding": ArgSchema(type="string", description="Text encoding", required=False, default="utf-8"),
            "create_dirs": ArgSchema(type="boolean", description="Create parent dirs", required=False, default=True),
        },
    },
    {
        "method": "mkdir",
        "description": "Create a directory",
        "policy": PolicyClass.WRITE,
        "is_async": False,
        "args": {
            "folder_id": ArgSchema(type="string", description="Registered folder ID", required=True),
            "relative_path": ArgSchema(type="string", description="Directory path", required=True),
            "parents": ArgSchema(type="boolean", description="Create parent dirs", required=False, default=True),
        },
    },
    {
        "method": "delete",
        "description": "Delete a file or directory",
        "policy": PolicyClass.DESTRUCTIVE,
        "is_async": False,
        "args": {
            "folder_id": ArgSchema(type="string", description="Registered folder ID", required=True),
            "relative_path": ArgSchema(type="string", description="Path to delete", required=True),
            "recursive": ArgSchema(type="boolean", description="Delete recursively", required=False, default=False),
        },
    },
    {
        "method": "info",
        "description": "Get file/directory information",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "folder_id": ArgSchema(type="string", description="Registered folder ID", required=True),
            "relative_path": ArgSchema(type="string", description="Path to inspect", required=True),
        },
    },
    {
        "method": "list_folders",
        "description": "List all registered folders",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
]

# ShellGate tools
SHELLGATE_TOOLS: List[Dict[str, Any]] = [
    {
        "method": "get_info",
        "description": "Get comprehensive documentation for ShellGate including all tools, call formats, and security policies",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
    {
        "method": "execute",
        "description": "Execute a shell command (subject to security policy)",
        "policy": PolicyClass.PRIVILEGED,
        "is_async": False,
        "args": {
            "command": ArgSchema(type="string", description="Command to execute", required=True),
            "working_dir": ArgSchema(type="string", description="Working directory", required=False),
            "timeout": ArgSchema(type="integer", description="Timeout in seconds", required=False),
        },
    },
    {
        "method": "validate_command",
        "description": "Check if a command is allowed by security policy",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "command": ArgSchema(type="string", description="Command to validate", required=True),
        },
    },
    {
        "method": "estimate_risk",
        "description": "Estimate the risk level of a command",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "command": ArgSchema(type="string", description="Command to assess", required=True),
        },
    },
    {
        "method": "get_history",
        "description": "Get command execution history",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "limit": ArgSchema(type="integer", description="Max entries", required=False, default=20),
            "success_only": ArgSchema(type="boolean", description="Only successful", required=False, default=False),
        },
    },
]

# ScriptureGate tools
SCRIPTUREGATE_TOOLS: List[Dict[str, Any]] = [
    {
        "method": "get_info",
        "description": "Get comprehensive documentation for ScriptureGate including all tools, call formats, and RAG usage",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
    {
        "method": "search",
        "description": "Semantic search across stored documents",
        "policy": PolicyClass.READ_ONLY,
        "is_async": True,
        "args": {
            "query": ArgSchema(type="string", description="Search query", required=True),
            "limit": ArgSchema(type="integer", description="Max results", required=False, default=5),
            "file_type": ArgSchema(type="string", description="Filter by type", required=False),
            "min_similarity": ArgSchema(type="number", description="Min similarity 0-1", required=False, default=0.0),
        },
    },
    {
        "method": "build_context",
        "description": "Build RAG context from relevant documents",
        "policy": PolicyClass.READ_ONLY,
        "is_async": True,
        "args": {
            "query": ArgSchema(type="string", description="Query for context", required=True),
            "limit": ArgSchema(type="integer", description="Max docs", required=False, default=3),
            "min_similarity": ArgSchema(type="number", description="Min similarity", required=False, default=0.3),
        },
    },
    {
        "method": "store_text",
        "description": "Store text content as scripture",
        "policy": PolicyClass.WRITE,
        "is_async": True,
        "args": {
            "content": ArgSchema(type="string", description="Text content", required=True),
            "title": ArgSchema(type="string", description="Document title", required=True),
            "description": ArgSchema(type="string", description="Description", required=False),
            "tags": ArgSchema(type="array", description="Tags", required=False, items={"type": "string"}),
        },
    },
    {
        "method": "get",
        "description": "Get scripture by UID",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "scripture_uid": ArgSchema(type="string", description="Scripture UID", required=True),
        },
    },
    {
        "method": "list_scriptures",
        "description": "List stored scriptures",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "file_type": ArgSchema(type="string", description="Filter by type", required=False),
            "limit": ArgSchema(type="integer", description="Max results", required=False, default=20),
        },
    },
    {
        "method": "stats",
        "description": "Get scripture storage statistics",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
]

# BrowserGate tools
BROWSERGATE_TOOLS: List[Dict[str, Any]] = [
    {
        "method": "get_info",
        "description": "Get comprehensive documentation for BrowserGate including all tools, call formats, and providers",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
    {
        "method": "search",
        "description": "Search the web",
        "policy": PolicyClass.NETWORK,
        "is_async": True,
        "args": {
            "query": ArgSchema(type="string", description="Search query", required=True),
            "provider": ArgSchema(
                type="string",
                description="Search provider",
                required=False,
                default="duckduckgo",
                enum=["duckduckgo", "google", "bing"],
            ),
            "max_results": ArgSchema(type="integer", description="Max results", required=False, default=5),
        },
    },
    {
        "method": "fetch",
        "description": "Fetch and parse a web page",
        "policy": PolicyClass.NETWORK,
        "is_async": True,
        "args": {
            "url": ArgSchema(type="string", description="URL to fetch", required=True),
            "mode": ArgSchema(
                type="string",
                description="Fetch mode",
                required=False,
                default="readable",
                enum=["readable", "raw", "screenshot"],
            ),
        },
    },
]

# ToolGate tools (meta-documentation)
TOOLGATE_TOOLS: List[Dict[str, Any]] = [
    {
        "method": "get_info",
        "description": "Get comprehensive documentation for the entire tool system including protocol format, policy classes, available gates, and how to discover tool documentation",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
]

# SubAgentGate tools
SUBAGENTGATE_TOOLS: List[Dict[str, Any]] = [
    {
        "method": "get_info",
        "description": "Get comprehensive documentation for SubAgentGate including all tools, call formats, and agent types",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {},
    },
    {
        "method": "spawn",
        "description": "Spawn a sub-agent to handle a task. Use agent_type='claude_code' for autonomous coding agents with file/tool access.",
        "policy": PolicyClass.WRITE,
        "is_async": False,
        "args": {
            "task": ArgSchema(type="string", description="Task instructions for the agent", required=True),
            "context": ArgSchema(type="object", description="Context dict to pass to agent", required=False),
            "agent_type": ArgSchema(
                type="string",
                description="Agent backend: 'llm' (simple completion), 'claude_code' (autonomous with file/tool access), 'codex'",
                required=False,
                default="llm",
                enum=["llm", "claude_code", "codex"],
            ),
            "working_dir": ArgSchema(
                type="string",
                description="Working directory for CLI agents (claude_code/codex). Defaults to project root.",
                required=False,
            ),
            "personality": ArgSchema(type="string", description="Personality ID (for llm type only)", required=False),
        },
    },
    {
        "method": "status",
        "description": "Get sub-agent status",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "agent_id": ArgSchema(type="string", description="Agent ID", required=True),
        },
    },
    {
        "method": "result",
        "description": "Get sub-agent result",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "agent_id": ArgSchema(type="string", description="Agent ID", required=True),
        },
    },
    {
        "method": "list_agents",
        "description": "List all sub-agents",
        "policy": PolicyClass.READ_ONLY,
        "is_async": False,
        "args": {
            "include_completed": ArgSchema(type="boolean", description="Include completed", required=False, default=True),
        },
    },
    {
        "method": "cancel",
        "description": "Cancel a running sub-agent",
        "policy": PolicyClass.WRITE,
        "is_async": False,
        "args": {
            "agent_id": ArgSchema(type="string", description="Agent ID", required=True),
        },
    },
    {
        "method": "prompt",
        "description": "Send a follow-up prompt to a completed sub-agent, continuing the conversation",
        "policy": PolicyClass.WRITE,
        "is_async": False,
        "args": {
            "agent_id": ArgSchema(type="string", description="ID of the completed agent to continue", required=True),
            "message": ArgSchema(type="string", description="Follow-up message/prompt", required=True),
        },
    },
]


# =============================================================================
# Tool Registry
# =============================================================================


class ToolRegistry:
    """Central registry of all available tools."""

    _tools: Dict[str, ToolDefinition] = {}
    _external_tools: Dict[str, ToolDefinition] = {}
    _external_tool_servers: Dict[str, str] = {}  # tool_name -> server_id
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize registry with all Gate tools."""
        if cls._initialized:
            return

        cls._register_gate_tools("ToolGate", TOOLGATE_TOOLS)
        cls._register_gate_tools("MemoryGate", MEMORYGATE_TOOLS)
        cls._register_gate_tools("FileSystemGate", FILESYSTEMGATE_TOOLS)
        cls._register_gate_tools("ShellGate", SHELLGATE_TOOLS)
        cls._register_gate_tools("ScriptureGate", SCRIPTUREGATE_TOOLS)
        cls._register_gate_tools("BrowserGate", BROWSERGATE_TOOLS)
        cls._register_gate_tools("SubAgentGate", SUBAGENTGATE_TOOLS)

        cls._initialized = True
        _log.info(f"Tool registry initialized with {len(cls._tools)} tools")

    @classmethod
    def _register_gate_tools(cls, gate_name: str, tools: List[Dict[str, Any]]) -> None:
        """Register tools for a gate."""
        for tool_config in tools:
            method_name = tool_config["method"]
            tool_name = f"{gate_name}.{method_name}"

            # Convert args dict to ArgSchema instances if needed
            args_schema = {}
            for arg_name, arg_def in tool_config.get("args", {}).items():
                if isinstance(arg_def, ArgSchema):
                    args_schema[arg_name] = arg_def
                elif isinstance(arg_def, dict):
                    args_schema[arg_name] = ArgSchema(**arg_def)

            tool = ToolDefinition(
                name=tool_name,
                description=tool_config["description"],
                gate=gate_name,
                method=method_name,
                policy_class=tool_config.get("policy", PolicyClass.READ_ONLY),
                is_async=tool_config.get("is_async", False),
                args_schema=args_schema,
            )

            cls._tools[tool_name] = tool

    @classmethod
    def get_tool(cls, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        cls.initialize()
        # Check built-in tools first, then external
        return cls._tools.get(name) or cls._external_tools.get(name)

    @classmethod
    def list_tools(
        cls,
        policy_filter: Optional[Set[PolicyClass]] = None,
        gate_filter: Optional[str] = None,
        gates_filter: Optional[List[str]] = None,
        include_external: bool = True,
    ) -> List[ToolDefinition]:
        """
        List tools, optionally filtered.

        Args:
            policy_filter: Filter by policy classes
            gate_filter: Filter by single gate name (deprecated, use gates_filter)
            gates_filter: Filter by list of gate names
            include_external: Include external (MCP) tools

        Returns:
            List of tool definitions
        """
        cls.initialize()

        # Combine built-in and external tools
        tools = list(cls._tools.values())
        if include_external:
            tools.extend(cls._external_tools.values())

        if policy_filter:
            tools = [t for t in tools if t.policy_class in policy_filter]

        # Single gate filter (legacy)
        if gate_filter:
            tools = [t for t in tools if t.gate == gate_filter]

        # Multi-gate filter
        if gates_filter:
            tools = [t for t in tools if t.gate in gates_filter]

        return tools

    @classmethod
    def list_tool_names(cls, policy_filter: Optional[Set[PolicyClass]] = None) -> List[str]:
        """List tool names, optionally filtered."""
        return [t.name for t in cls.list_tools(policy_filter)]

    @classmethod
    def get_tools_for_prompt(
        cls,
        enabled_policies: Set[PolicyClass],
        gates_filter: Optional[List[str]] = None,
    ) -> str:
        """
        Generate tool documentation for system prompt.

        Args:
            enabled_policies: Set of enabled policy classes
            gates_filter: Optional list of gate names to include

        Returns:
            Formatted tool documentation for system prompt
        """
        tools = cls.list_tools(enabled_policies, gates_filter=gates_filter)

        if not tools:
            return ""

        sections = []
        for tool in sorted(tools, key=lambda t: t.name):
            sections.append(tool.to_prompt_schema())

        return "\n\n".join(sections)

    @classmethod
    def reset(cls) -> None:
        """Reset registry (for testing)."""
        cls._tools = {}
        cls._external_tools = {}
        cls._external_tool_servers = {}
        cls._initialized = False

    # =========================================================================
    # External Tool Registration (for MCP servers)
    # =========================================================================

    @classmethod
    def register_external_tool(
        cls,
        server_id: str,
        server_name: str,
        tool: Any,  # MCPTool
        policy_class: str = "network",
    ) -> None:
        """
        Register an external tool from an MCP server.

        Args:
            server_id: MCP server identifier
            server_name: Human-readable server name
            tool: MCPTool instance with name, description, input_schema
            policy_class: Policy class string (read_only, network, write, privileged)
        """
        # Build full tool name: MCP.{server_id}.{tool_name}
        full_name = f"MCP.{server_id}.{tool.name}"

        # Map policy string to PolicyClass
        policy_map = {
            "read_only": PolicyClass.READ_ONLY,
            "network": PolicyClass.NETWORK,
            "write": PolicyClass.WRITE,
            "privileged": PolicyClass.PRIVILEGED,
            "destructive": PolicyClass.DESTRUCTIVE,
        }
        policy = policy_map.get(policy_class.lower(), PolicyClass.NETWORK)

        # Build args schema from MCP input schema
        args_schema = {}
        input_schema = getattr(tool, "input_schema", None)
        if input_schema:
            properties = getattr(input_schema, "properties", {}) or {}
            required_list = getattr(input_schema, "required", []) or []

            for prop_name, prop_def in properties.items():
                if isinstance(prop_def, dict):
                    args_schema[prop_name] = ArgSchema(
                        type=prop_def.get("type", "string"),
                        description=prop_def.get("description", ""),
                        required=prop_name in required_list,
                        default=prop_def.get("default"),
                        enum=prop_def.get("enum"),
                    )

        tool_def = ToolDefinition(
            name=full_name,
            description=f"[{server_name}] {tool.description}" if tool.description else f"[{server_name}] MCP tool",
            gate="MCPClient",
            method=tool.name,
            policy_class=policy,
            is_async=True,  # MCP tools are always async
            args_schema=args_schema,
        )

        cls._external_tools[full_name] = tool_def
        cls._external_tool_servers[full_name] = server_id

        _log.debug(f"Registered external tool: {full_name}")

    @classmethod
    def unregister_server_tools(cls, server_id: str) -> int:
        """
        Unregister all tools from a specific MCP server.

        Args:
            server_id: MCP server identifier

        Returns:
            Number of tools unregistered
        """
        # Find tools belonging to this server
        tools_to_remove = [
            name for name, sid in cls._external_tool_servers.items()
            if sid == server_id
        ]

        # Remove them
        for tool_name in tools_to_remove:
            cls._external_tools.pop(tool_name, None)
            cls._external_tool_servers.pop(tool_name, None)

        if tools_to_remove:
            _log.debug(f"Unregistered {len(tools_to_remove)} tools from server {server_id}")

        return len(tools_to_remove)

    @classmethod
    def get_external_tool_server(cls, tool_name: str) -> Optional[str]:
        """Get the server ID for an external tool."""
        return cls._external_tool_servers.get(tool_name)

    @classmethod
    def list_external_tools(cls) -> List[ToolDefinition]:
        """List all external (MCP) tools."""
        return list(cls._external_tools.values())


__all__ = [
    "ToolRegistry",
    "TOOLGATE_TOOLS",
    "MEMORYGATE_TOOLS",
    "FILESYSTEMGATE_TOOLS",
    "SHELLGATE_TOOLS",
    "SCRIPTUREGATE_TOOLS",
    "BROWSERGATE_TOOLS",
    "SUBAGENTGATE_TOOLS",
]
